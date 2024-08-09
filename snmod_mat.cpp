#include "sparse_mat.h"
#include <cstring>
#include "thread_pool.hpp"

// Row x - a*Row y

inline void snmod_mat_xmay(snmod_mat_t mat, slong x, slong y, ulong a,
	nmod_t p) {
	snmod_vec_sub_mul(mat->rows + x, mat->rows + y, a, p);
}

// first look for rows with only one nonzero value and eliminate them
// we assume that mat is canonical, i.e. each index is sorted
// and the result is also canonical
ulong eliminate_row_with_one_nnz(snmod_mat_t mat,
	sparse_mat_t<ulong*> tranmat,
	slong* donelist) {
	auto localcounter = 0;
	slong* pivlist = (slong*)malloc(mat->nrow * sizeof(slong));
	slong* collist = (slong*)malloc(mat->ncol * sizeof(slong));
	for (size_t i = 0; i < mat->nrow; i++)
		pivlist[i] = -1;
	for (size_t i = 0; i < mat->ncol; i++)
		collist[i] = -1;
	for (size_t i = 0; i < mat->nrow; i++) {
		if (donelist[i] != -1)
			continue;
		if (mat->rows[i].nnz == 1) {
			if (collist[mat->rows[i].indices[0]] == -1) {
				localcounter++;
				pivlist[i] = mat->rows[i].indices[0];
				collist[mat->rows[i].indices[0]] = i;
			}
		}
	}

	if (localcounter == 0) {
		free(pivlist);
		free(collist);
		return localcounter;
	}

	sparse_mat_transpose_pointer(tranmat, mat);
	for (size_t i = 0; i < mat->nrow; i++) {
		if (pivlist[i] == -1)
			continue;
		auto thecol = tranmat->rows + pivlist[i];
		ulong colone = 0;
		for (size_t j = 0; j < thecol->nnz; j++) {
			if (thecol->indices[j] == i) {
				colone = j;
				*(thecol->entries[j]) = 1;
			}
			else
				*(thecol->entries[j]) = 0;
		}
	}

	for (size_t i = 0; i < mat->nrow; i++) {
		sparse_vec_canonicalize(mat->rows + i);
	}

	for (size_t i = 0; i < mat->nrow; i++)
		if (pivlist[i] != -1)
			donelist[i] = pivlist[i];

	free(pivlist);
	free(collist);
	return localcounter;
}

ulong eliminate_row_with_one_nnz_rec(snmod_mat_t mat,
	sparse_mat_t<ulong*> tranmat,
	slong* donelist, bool verbose,
	slong max_depth = INT_MAX) {
	slong depth = 0;
	ulong oldnnz = 0;
	ulong localcounter = 0;
	ulong count = 0;

	do {
		oldnnz = sparse_mat_nnz(mat);
		localcounter = eliminate_row_with_one_nnz(mat, tranmat, donelist);
		if (verbose) {
			std::cout << "\r-- eliminating rows with only one element, depth: "
				<< depth << ", eliminated row: " << count << std::flush;
		}
		count += localcounter;
		depth++;
	} while (localcounter > 0 && depth < max_depth);
	return count;
}

inline slong rowcolpart(std::vector<slong> conp, slong nrow) {
	slong i;
	for (i = 0; i < conp.size(); i++)
		if (conp[i] >= nrow)
			break;
	return i;
}

slong findrowpivot(snmod_mat_t mat, slong col, slong* rowpivs,
	std::vector<slong>& colparts, std::vector<slong>& kkparts,
	std::vector<std::vector<slong>>& rowparts, slong* dolist,
	ulong** entrylist, slong& dolist_len,
	BS::thread_pool& pool) {

	auto mm = colparts[col];
	auto rowlist = rowparts[mm];
	// parallelize this loop

	auto loop = [&](slong i) {
		if (rowlist[i] == -1)
			return;
		auto entry = sparse_mat_entry(mat, rowlist[i], col, true);
		if (entry == NULL || scalar_is_zero(entry))
			return;
		// add one to make it nonzero
		entrylist[i] = entry;
		dolist[i] = 1;
		};

	pool.detach_loop<slong>(0, kkparts[mm], loop);
	pool.wait();

	slong rowi = -1; // choose the row with the minimal nnz
	ulong mininnz = ULLONG_MAX;
	for (size_t i = 0; i < kkparts[mm]; i++) {
		if (dolist[i] == -1)
			continue;

		auto row = rowparts[mm][i];

		// do not choose rows that have been used to eliminate
		if (rowpivs[row] != -1)
			continue;

		dolist[dolist_len] = row;
		entrylist[dolist_len] = entrylist[i];
		dolist_len++;

		if (mat->rows[row].nnz < mininnz) { // check the nnz of the row
			rowi = row;
			mininnz = mat->rows[row].nnz;
		}
	}
	return rowi;
}

auto findmanypivots_c(snmod_mat_t mat, sparse_mat_t<ulong*> tranmat,
	slong* rowpivs, std::vector<slong>& colperm,
	std::vector<slong>::iterator start,
	size_t max_depth = ULLONG_MAX) {

	std::vector<std::pair<slong, std::vector<slong>::iterator>> pivots;
	std::unordered_set<slong> prows;
	prows.reserve(std::min((size_t)4096, max_depth));
	for (auto col = start; col < colperm.end(); col++) {
		if (col - start > max_depth)
			break;
		bool flag = true;
		auto thecol = tranmat->rows + *col;
		auto indices = thecol->indices;
		for (size_t i = 0; i < thecol->nnz; i++) {
			flag = (prows.find(indices[i]) == prows.end());
			if (!flag)
				break;
		}
		if (!flag)
			continue;

		if (thecol->nnz == 0)
			continue;
		slong row;
		ulong mnnz = ULLONG_MAX;
		for (size_t i = 0; i < thecol->nnz; i++) {
			if (rowpivs[indices[i]] != -1)
				continue;
			if (mat->rows[indices[i]].nnz < mnnz) {
				row = indices[i];
				mnnz = mat->rows[row].nnz;
			}
		}
		if (mnnz != ULLONG_MAX) {
			pivots.push_back(std::make_pair(row, col));
			prows.insert(row);
		}
	}
	return pivots;
}

template <typename T>
auto findmanypivots_r(snmod_mat_t mat, sparse_mat_t<T> tranmat,
	slong* colpivs, std::vector<slong>& rowperm,
	std::vector<slong>::iterator start,
	size_t max_depth = ULLONG_MAX) {

	std::vector<std::pair<slong, std::vector<slong>::iterator>> pivots;
	std::unordered_set<slong> pcols;
	pcols.reserve(std::min((size_t)4096, max_depth));
	for (auto row = start; row != rowperm.end(); row++) {
		if (row - start > max_depth)
			break;

		auto therow = mat->rows + *row;
		if (therow->nnz == 0)
			continue;
		auto indices = therow->indices;

		slong col;
		ulong mnnz = ULLONG_MAX;
		bool flag = true;

		for (size_t i = 0; i < therow->nnz; i++) {
			flag = (pcols.find(indices[i]) == pcols.end());
			if (!flag)
				break;
			if (colpivs[indices[i]] != -1)
				continue;
			if (tranmat->rows[indices[i]].nnz < mnnz) {
				col = indices[i];
				mnnz = tranmat->rows[col].nnz;
			}
		}
		if (!flag)
			continue;
		if (mnnz != ULLONG_MAX) {
			pivots.push_back(std::make_pair(col, row));
			pcols.insert(col);
		}
	}
	return pivots;
}

// first write a stupid one
// TODO: Gilbert-Peierls algorithm for parallel computation 
// see https://hal.science/hal-01333670/document
void schur_complete(snmod_mat_t mat, slong row, std::vector<std::pair<slong, slong>>& pivots, int ordering, nmod_t p, ulong* tmpvec) {
	auto therow = mat->rows + row;

	if (therow->nnz == 0)
		return;

	// snmod_vec_t tmpvec;
	// sparse_vec_init(tmpvec, mat->ncol);
	// sparse_vec_set(tmpvec, therow);

	// ulong* tmpvec = (ulong*)malloc(mat->ncol * sizeof(ulong));
	memset(tmpvec, 0, mat->ncol * sizeof(ulong));
	for (size_t i = 0; i < therow->nnz; i++)
		tmpvec[therow->indices[i]] = therow->entries[i];

	if (ordering == 1) {
		for (auto& [r, c] : pivots) {
			auto entry = tmpvec[c];
			if (entry == 0)
				continue;
			auto row = mat->rows + r;
			ulong e_pr = n_mulmod_precomp_shoup(entry, p.n);

			for (size_t i = 0; i < row->nnz; i++) {
				tmpvec[row->indices[i]] = nmod_sub(tmpvec[row->indices[i]],
					n_mulmod_shoup(entry, row->entries[i], e_pr, p.n), p);
			}
			// snmod_vec_sub_mul(tmpvec, mat->rows + r, *entry, p);
		}
	}
	else if (ordering == -1) {
		for (auto ii = pivots.rbegin(); ii != pivots.rend(); ii++) {
			auto& [r, c] = *ii;
			auto entry = tmpvec[c];
			if (entry == 0)
				continue;
			auto row = mat->rows + r;
			ulong e_pr = n_mulmod_precomp_shoup(entry, p.n);

			for (size_t i = 0; i < row->nnz; i++) {
				tmpvec[row->indices[i]] = nmod_sub(tmpvec[row->indices[i]],
					n_mulmod_shoup(entry, row->entries[i], e_pr, p.n), p);
			}
			// snmod_vec_sub_mul(tmpvec, mat->rows + r, *entry, p);
		}
	}

	// sparse_vec_swap(therow, tmpvec);
	// sparse_vec_clear(tmpvec);

	therow->nnz = 0;
	for (size_t i = 0; i < mat->ncol; i++) {
		if (tmpvec[i] != 0)
			_sparse_vec_set_entry(therow, i, tmpvec[i]);
	}
	// free(tmpvec);
}

// upper solver : ordering = -1
// lower solver : ordering = 1
void triangular_solver(snmod_mat_t mat, std::vector<std::pair<slong, slong>>& pivots,
	rref_option_t opt, int ordering, nmod_t p, BS::thread_pool& pool) {
	bool verbose = opt->verbose;
	auto printstep = opt->print_step;

	std::vector<std::vector<slong>> tranmat(mat->ncol);

	// we only need to compute the transpose of the submatrix involving pivots

	for (size_t i = 0; i < pivots.size(); i++) {
		auto therow = mat->rows + pivots[i].first;
		for (size_t j = 0; j < therow->nnz; j++) {
			if (scalar_is_zero(therow->entries + j))
				continue;
			auto col = therow->indices[j];
			tranmat[col].push_back(pivots[i].first);
		}
	}

	for (size_t i = 0; i < pivots.size(); i++) {
		size_t index;
		if (ordering == 1)
			index = i;
		else if (ordering == -1)
			index = pivots.size() - 1 - i;
		else
			break; // do nothing
		auto pp = pivots[index];
		auto thecol = tranmat[pp.second];
		auto start = clocknow();
		auto loop = [&](slong j) {
			auto r = thecol[j];
			if (r == pp.first)
				return;
			auto entry = sparse_mat_entry(mat, r, pp.second, true);
			snmod_mat_xmay(mat, r, pp.first, *entry, p);
			};
		if (thecol.size() > 1) {
			pool.detach_loop<slong>(0, thecol.size(), loop);
			pool.wait();
		}
		auto end = clocknow();

		if ((i % printstep == 0 || i == pivots.size() - 1) && verbose) {
			end = clocknow();
			auto now_nnz = sparse_mat_nnz(mat);
			std::cout << "\r-- Row: " << (i + 1) << "/" << pivots.size()
				<< "  " << "row to eliminate: " << thecol.size() - 1
				<< "  " << "nnz: " << now_nnz << "  " << "density: "
				<< (double)100 * now_nnz / (mat->nrow * mat->ncol)
				<< "%  " << "speed: " << printstep / usedtime(start, end)
				<< " row/s" << std::flush;
			start = clocknow();
		}
	}
}

std::vector<std::pair<slong, slong>> snmod_mat_rref_c(snmod_mat_t mat, nmod_t p, BS::thread_pool& pool,
	rref_option_t opt) {
	// first canonicalize, sort and compress the matrix
	sparse_mat_compress(mat);
	for (size_t i = 0; i < mat->nrow; i++) {
		sparse_vec_sort_indices(mat->rows + i);
	}

	ulong rank = 0;

	ulong init_nnz = sparse_mat_nnz(mat);
	ulong now_nnz = init_nnz;

	ulong scalar;

	// store the pivots that have been used
	// -1 is not used
	slong* rowpivs = (slong*)malloc(mat->nrow * sizeof(slong));
	for (size_t i = 0; i < mat->nrow; i++)
		rowpivs[i] = -1;
	// perm the col
	std::vector<slong> colperm(mat->ncol);
	for (size_t i = 0; i < mat->ncol; i++) {
		colperm[i] = i;
	}

	// look for row with only one non-zero entry

	// compute the transpose of pointers of the matrix
	sparse_mat_t<ulong*> tranmat;
	sparse_mat_init(tranmat, mat->ncol, mat->nrow);

	bool verbose = opt->verbose;
	ulong count =
		eliminate_row_with_one_nnz_rec(mat, tranmat, rowpivs, verbose);
	rank += count;
	now_nnz = sparse_mat_nnz(mat);
	if (verbose) {
		std::cout << "\n** eliminated " << count
			<< " rows, and reduce nnz: " << init_nnz << " -> " << now_nnz
			<< std::endl;
	}

	sparse_mat_transpose_pointer(tranmat, mat);

	// sort pivots by nnz, it will be faster
	std::stable_sort(colperm.begin(), colperm.end(),
		[&tranmat](slong a, slong b) {
			return tranmat->rows[a].nnz < tranmat->rows[b].nnz;
		});

	ulong** entrylist = (ulong**)malloc(mat->nrow * sizeof(ulong*));
	slong* dolist = (slong*)malloc(mat->nrow * sizeof(slong));

	// look for pivot cols with only one nonzero element
	slong countone = 0;
	for (size_t i = 0; i < mat->ncol; i++) {
		if (tranmat->rows[colperm[i]].nnz == 1) {
			auto row = tranmat->rows[colperm[i]].indices[0];
			if (rowpivs[row] != -1)
				continue;
			rowpivs[row] = colperm[i];
			dolist[countone] = row;
			entrylist[countone] =
				sparse_vec_entry(mat->rows + row, rowpivs[row], true);
			countone++;
			rank++;
		}
		else if (tranmat->rows[colperm[i]].nnz > 1)
			break; // since it's sorted
	}

	// eliminate the rows with only one nonzero element
	// we can do this in parallel

	pool.detach_loop<slong>(0, countone, [&](slong i) {
		ulong scalar;
		scalar = nmod_inv(*(entrylist[i]), p);
		snmod_vec_rescale(mat->rows + dolist[i], scalar, p);
		});
	pool.wait();

	if (verbose) {
		std::cout << ">> there are " << countone
			<< " columns that has already been eliminated" << std::endl;
	}

	init_nnz = sparse_mat_nnz(mat);

	countone = 0;
	for (size_t i = 0; i < mat->ncol; i++) {
		if (tranmat->rows[colperm[i]].nnz <= 1)
			countone++;
		else
			break; // since it's sorted
	}

	// we cut the matrix as
	// (*********| A A A ... A A A )
	// (*********| A A A ... A A A )
	// (*********| A A A ... A A A )
	//  countone |   submatrix B
	// countone part is eliminated

	if (verbose)
		std::cout << "-- computing connected components" << std::endl;

	Graph adjgraph((mat->ncol) + (mat->nrow));

	for (size_t i = countone; i < mat->ncol; i++) {
		auto thecol = tranmat->rows + colperm[i];
		for (size_t j = 0; j < thecol->nnz; j++) {
			adjgraph.addEdge(thecol->indices[j], mat->nrow + colperm[i]);
		}
	}

	auto connected_components = adjgraph.findMaximalConnectedComponents();
	std::sort(connected_components.begin(), connected_components.end(),
		[](std::vector<slong>& a, std::vector<slong>& b) {
			return a.size() < b.size();
		});
	for (auto& g : connected_components)
		std::sort(g.begin(), g.end());

	// use colparts to label components
	std::vector<slong> colparts(mat->ncol);
	std::vector<slong> kkparts(connected_components.size());
	std::vector<std::vector<slong>> rowparts(connected_components.size());
	for (size_t i = 0; i < mat->ncol; i++)
		colparts[i] = -1;
	for (auto mm = 0; mm < connected_components.size(); mm++) {
		auto g = connected_components[mm];
		kkparts[mm] = rowcolpart(g, mat->nrow);
		rowparts[mm] = std::vector<slong>(g.begin(), g.begin() + kkparts[mm]);
		for (size_t i = kkparts[mm]; i < g.size(); i++)
			colparts[g[i] - mat->nrow] = mm;
	}
	adjgraph.clear();

	if (verbose) {
		std::cout << "** found " << connected_components.size() << " components"
			<< std::endl;
	}

	std::stable_sort(
		colperm.begin() + countone, colperm.end(),
		[&colparts](slong a, slong b) { return colparts[a] < colparts[b]; });

	// upper triangle (with respect to row and col perm)

	sparse_mat_transpose_pointer(tranmat, mat);

	// upper triangle (with respect to row and col perm)
	while (countone < mat->ncol) {
		auto start = clocknow();

		if (colparts[colperm[countone]] == -1) {
			countone++;
			continue;
		}

		auto ps = findmanypivots_c(mat, tranmat, rowpivs, colperm,
			colperm.begin() + countone, opt->search_depth);
		if (ps.size() == 0)
			break;

		// std::vector<std::pair<slong, slong>> n_pivots;
		// for (auto& [r, cp] : ps) 
		//     n_pivots.push_back(std::make_pair(r, *cp));

		for (auto kk = 0; kk < ps.size(); kk++) {
			auto start = clocknow();

			auto pp = ps[kk];
			auto row = pp.first;
			auto col = *pp.second;
			rowpivs[row] = col;
			rank++;
			scalar = nmod_inv(*sparse_mat_entry(mat, row, col, true), p);
			snmod_vec_rescale(mat->rows + row, scalar, p);
			auto thecol = tranmat->rows + col;
			pool.detach_loop<slong>(0, thecol->nnz, [&](slong i) {
				if (thecol->indices[i] == row)
					return;
				auto entry =
					sparse_mat_entry(mat, thecol->indices[i], col, true);
				// auto this_id = BS::this_thread::get_index().value();
				// snmod_mat_xmay_cached(mat, thecol->indices[i], row,
				// cachemat->rows + this_id, *entry, p);
				snmod_mat_xmay(mat, thecol->indices[i], row, *entry, p);
				});
			pool.wait();
			auto end = clocknow();

			if (verbose && (countone + kk + 1) % opt->print_step == 0) {
				now_nnz = sparse_mat_nnz(mat);
				std::cout << "\r-- Col: " << countone + kk + 1 << "/"
					<< mat->ncol << "  " << "row to eliminate: "
					<< tranmat->rows[*pp.second].nnz - 1 << "  "
					<< "rank: " << rank << "  " << "nnz: " << now_nnz
					<< "  " << "density: "
					<< 100 * (double)now_nnz / (mat->nrow * mat->ncol)
					<< "%  " << "speed: " << 1 / usedtime(start, end)
					<< " col/s" << std::flush;
			}
		}

		std::vector<slong> tempcolperm(colperm.begin() + countone,
			colperm.end());
		std::vector<slong> diffs(ps.size());
		for (size_t i = 0; i < diffs.size(); i++)
			diffs[i] = ps[i].second - (colperm.begin() + countone);
		remove_indices(tempcolperm, diffs);

		for (auto i = 0; i < ps.size(); i++)
			colperm[countone + i] = *ps[i].second;
		for (auto i = 0; i < tempcolperm.size(); i++)
			colperm[countone + ps.size() + i] = tempcolperm[i];

		count = eliminate_row_with_one_nnz_rec(mat, tranmat, rowpivs, false, 3);
		rank += count;
		now_nnz = sparse_mat_nnz(mat);

		countone += ps.size();

		sparse_mat_transpose_pointer(tranmat, mat);
		// sort pivots by nnz, it will be faster
		std::stable_sort(colperm.begin() + countone, colperm.end(),
			[&tranmat, &colparts](slong a, slong b) {
				if (colparts[a] < colparts[b]) {
					return true;
				}
				else if (colparts[a] == colparts[b]) {
					return tranmat->rows[a].nnz <
						tranmat->rows[b].nnz;
				}
				else
					return false;
			});

		if (ps.size() < opt->search_min) {
			break;
		}
	}

	for (auto kk = countone; kk < mat->ncol; kk++) {
		auto start = clocknow();

		// first scan to get the list of rows to eliminate
		// and determine the the pivot with minimal nnz

		auto pivot = colperm[kk];

		if (colparts[pivot] == -1) {
			continue;
		}

		slong dolist_len = 0;
		for (auto i = 0; i < mat->nrow; i++)
			dolist[i] = -1;
		slong rowi =
			findrowpivot(mat, pivot, rowpivs, colparts, kkparts, rowparts,
				dolist, entrylist, dolist_len, pool);

		if (rowi == -1)
			continue;

		rowpivs[rowi] = pivot;
		rank++;

		scalar = nmod_inv(*sparse_vec_entry(mat->rows + rowi, pivot, true), p);
		snmod_vec_rescale(mat->rows + rowi, scalar, p);

		pool.detach_loop<slong>(0, dolist_len, [&](slong i) {
			if (dolist[i] == rowi)
				return;
			snmod_mat_xmay(mat, dolist[i], rowi, *entrylist[i], p);
			});
		pool.wait();

		if (kk % opt->sort_step == 0 || kk == mat->ncol - 1) {
			// auto oldalloc = sparse_mat_alloc(mat);

			count =
				eliminate_row_with_one_nnz_rec(mat, tranmat, rowpivs, false);
			rank += count;
			now_nnz = sparse_mat_nnz(mat);
			// if (verbose) {
			//     std::cout << "\n** eliminated " << count << " rows, and nnz is "
			//         << now_nnz << std::endl;
			// }

			// sort pivots by nnz, it will be faster
			std::stable_sort(colperm.begin() + kk + 1, colperm.end(),
				[&tranmat, &colparts](slong a, slong b) {
					if (colparts[a] < colparts[b]) {
						return true;
					}
					else if (colparts[a] == colparts[b]) {
						return tranmat->rows[a].nnz <
							tranmat->rows[b].nnz;
					}
					else
						return false;
				});
		}

		auto end = clocknow();

		if (verbose) {
			now_nnz = sparse_mat_nnz(mat);
			if (kk % opt->print_step == 0 || kk == mat->ncol - 1) {
				std::cout << "\r-- Col: " << kk + 1 << "/" << mat->ncol << "  "
					<< "row to eliminate: " << dolist_len << "  "
					<< "rank: " << rank << "  " << "nnz: " << now_nnz
					<< "  " << "density: "
					<< (double)100 * now_nnz / (mat->nrow * mat->ncol)
					<< "%  " << "speed: " << 1 / usedtime(start, end)
					<< " col/s" << std::flush;
			}
		}
	}

	if (verbose) {
		std::cout << "\n** Rank: " << rank << " nnz: " << sparse_mat_nnz(mat)
			<< "  " << std::endl;
	}

	if (verbose) {
		std::cout << "\n>> Reverse solving: " << std::endl;
	}

	std::vector<slong> colpivs(mat->ncol, -1);
	for (size_t i = 0; i < mat->nrow; i++) {
		if (rowpivs[i] != -1)
			colpivs[rowpivs[i]] = i;
	}

	// the ordering of pivots is important
	// which makes the submatrix is upper triangular
	// i.e. mat(row(p[j]), col(p[i])) = 0 for i < j
	std::vector<std::pair<slong, slong>> pivots;
	for (slong i = 0; i < mat->ncol; i++) {
		auto col = colperm[i];
		auto row = colpivs[col];
		if (row != -1) {
			pivots.push_back(std::make_pair(row, col));
		}
	}

	// the matrix is upper triangular
	triangular_solver(mat, pivots, opt, -1, p, pool);

	if (verbose) {
		std::cout << std::endl;
	}

	sparse_mat_clear(tranmat);

	free(entrylist);
	free(dolist);
	free(rowpivs);

	if (verbose) {
		std::cout << std::endl;
	}

	return pivots;
}

std::vector<std::pair<slong, slong>> snmod_mat_rref_r(snmod_mat_t mat, nmod_t p, BS::thread_pool& pool,
	rref_option_t opt) {
	// first canonicalize, sort and compress the matrix

	for (size_t i = 0; i < mat->nrow; i++) {
		sparse_vec_sort_indices(mat->rows + i);
	}
	// sparse_mat_compress(mat);

	std::vector<slong> rowperm(mat->nrow);
	for (size_t i = 0; i < mat->nrow; i++)
		rowperm[i] = i;

	auto printstep = opt->print_step;
	bool verbose = opt->verbose;

	ulong rank = 0;

	ulong init_nnz = sparse_mat_nnz(mat);
	ulong now_nnz = init_nnz;

	// store the pivots that have been used
	// -1 is not used
	slong* rowpivs = (slong*)malloc(mat->nrow * sizeof(slong));
	slong* colpivs = (slong*)malloc(mat->ncol * sizeof(slong));
	memset(colpivs, -1, mat->ncol * sizeof(slong));
	memset(rowpivs, -1, mat->nrow * sizeof(slong));

	sparse_mat_t<ulong*> tranmatp;
	sparse_mat_init(tranmatp, mat->ncol, mat->nrow);
	ulong count =
		eliminate_row_with_one_nnz_rec(mat, tranmatp, rowpivs, verbose);
	now_nnz = sparse_mat_nnz(mat);
	if (verbose) {
		std::cout << "\n** eliminated " << count
			<< " rows, and reduce nnz: " << init_nnz << " -> " << now_nnz
			<< std::endl;
	}
	init_nnz = now_nnz;
	sparse_mat_clear(tranmatp);

	// sort rows by nnz
	std::stable_sort(rowperm.begin(), rowperm.end(),
		[&mat](slong a, slong b) {
			if (mat->rows[a].nnz < mat->rows[b].nnz) {
				return true;
			}
			else if (mat->rows[a].nnz == mat->rows[b].nnz) {
				auto ri1 = mat->rows[a].indices;
				auto ri2 = mat->rows[b].indices;
				auto nnz = mat->rows[a].nnz;
				return std::lexicographical_compare(ri1, ri1 + nnz, ri2, ri2 + nnz);
			}
			else
				return false;
		});

	std::vector<std::pair<slong, slong>> pivots;

	sparse_mat_t<ulong> tranmat;
	sparse_mat_init(tranmat, mat->ncol, mat->nrow);

	ulong* cachedensedmat = (ulong*)malloc(mat->ncol * pool.get_thread_count() * sizeof(ulong));

	// skip the rows with only one/zero nonzero element
	slong kk;
	for (kk = 0; kk < mat->nrow; kk++) {
		auto row = rowperm[kk];
		auto therow = sparse_mat_row(mat, row);
		if (therow->nnz == 0)
			continue;
		else if (therow->nnz == 1) {
			auto col = therow->indices[0];
			pivots.push_back(std::make_pair(row, col));
			rank++;
			rowpivs[row] = col;
			colpivs[col] = row;
		}
		else
			break;
	}

	//if (verbose)
	//    std::cout << "-- computing connected components" << std::endl;

	//Graph adjgraph((mat->nrow) + (mat->ncol));

	//for (size_t i = kk; i < mat->nrow; i++) {
	//    auto therow = mat->rows + rowperm[i];
	//    for (size_t j = 0; j < therow->nnz; j++) {
	//        adjgraph.addEdge(rowperm[i], mat->nrow + therow->indices[j]);
	//    }
	//}

	//auto connected_components = adjgraph.findMaximalConnectedComponents();
	//std::sort(connected_components.begin(), connected_components.end(),
	//    [](std::vector<slong>& a, std::vector<slong>& b) {
	//        return a.size() < b.size();
	//    });
	//for (auto& g : connected_components)
	//    std::sort(g.begin(), g.end());

	//// use rowparts to label components
	//std::vector<slong> rowparts(mat->nrow);
	//std::vector<slong> kkparts(connected_components.size());
	//for (size_t i = 0; i < mat->nrow; i++)
	//    rowparts[i] = -1;
	//for (auto mm = 0; mm < connected_components.size(); mm++) {
	//    auto g = connected_components[mm];
	//    kkparts[mm] = rowcolpart(g, mat->nrow);
	//    for (size_t i = 0; i < kkparts[mm]; i++)
	//        rowparts[g[i]] = mm;
	//}
	//adjgraph.clear();

	//if (verbose) {
	//    std::cout << "** found " << connected_components.size() << " components"
	//        << std::endl;
	//}

	//// sort rows by connected_components and nnz
	//std::stable_sort(rowperm.begin() + kk, rowperm.end(),
	//    [&mat,&rowparts](slong a, slong b) {
	//        if (rowparts[a] < rowparts[b])
	//            return true;
	//        else if (mat->rows[a].nnz < mat->rows[b].nnz) 
	//            return true;
	//        else if (mat->rows[a].nnz == mat->rows[b].nnz) {
	//            auto ri1 = mat->rows[a].indices;
	//            auto ri2 = mat->rows[b].indices;
	//            auto nnz = mat->rows[a].nnz;
	//            return std::lexicographical_compare(ri1, ri1 + nnz, ri2, ri2 + nnz);
	//        }
	//        else
	//            return false;
	//    });

	while (kk < mat->nrow) {
		auto start = clocknow();
		auto row = rowperm[kk];

		if (mat->rows[row].nnz == 0) {
			kk++;
			continue;
		}

		std::vector<slong> leftrows(rowperm.begin() + kk, rowperm.end());
		sparse_mat_transpose_part(tranmat, mat, leftrows);
		auto ps = findmanypivots_r(mat, tranmat, colpivs,
			rowperm, rowperm.begin() + kk, opt->search_depth);

		if (ps.size() == 0)
			break;

		std::vector<std::pair<slong, slong>> n_pivots;

		for (auto& [c, rp] : ps) {
			pivots.push_back(std::make_pair(*rp, c));
			n_pivots.push_back(std::make_pair(*rp, c));
			colpivs[c] = *rp;
			rowpivs[*rp] = c;
			ulong scalar = nmod_inv(*sparse_mat_entry(mat, *rp, c, true), p);
			snmod_vec_rescale(mat->rows + *rp, scalar, p);
		}

		std::vector<slong> temprowperm(rowperm.begin() + kk, rowperm.end());
		std::vector<slong> diffs(ps.size());
		for (size_t i = 0; i < diffs.size(); i++)
			diffs[i] = ps[i].second - (rowperm.begin() + kk);
		remove_indices(temprowperm, diffs);

		for (auto i = 0; i < ps.size(); i++)
			rowperm[kk + i] = *ps[i].second;
		for (auto i = 0; i < temprowperm.size(); i++)
			rowperm[kk + ps.size() + i] = temprowperm[i];

		kk += ps.size();
		rank += ps.size();

		auto end = clocknow();
		double oldstatus = 0;
		std::atomic<int> count(0);
		pool.detach_loop<slong>(kk, mat->nrow, [&](slong i) {
			if (rowpivs[rowperm[i]] != -1)
				return;
			auto id = BS::this_thread::get_index().value();
			schur_complete(mat, rowperm[i], n_pivots, 1, p,
				cachedensedmat + id * mat->ncol);
			count++;
			});
		while (count < mat->nrow - kk) {
			auto status = (kk - ps.size() + 1) + ((double)count / (mat->nrow - kk)) * ps.size();
			if (verbose && count % printstep == 0) {
				end = clocknow();
				now_nnz = sparse_mat_nnz(mat);
				std::cout << "\r-- Row: " << (int)std::floor(status) << "/" << mat->nrow
					<< "  rank: " << rank
					<< "  nnz: " << now_nnz << "  " << "density: "
					<< (double)100 * now_nnz / (mat->nrow * mat->ncol) << "%"
					<< "  speed: " << (status - oldstatus) / usedtime(start, end)
					<< "  row/s" << std::flush;
				start = clocknow();
				oldstatus = status;
			}
			//std::this_thread::sleep_for(std::chrono::milliseconds(50));
		}
		pool.wait();

		if (ps.size() < 0) {
			break;
		}
	}

	free(cachedensedmat);

	if (verbose) {
		std::cout << "\n** Rank: " << rank
			<< " nnz: " << sparse_mat_nnz(mat) << std::endl
			<< "\n>> Reverse solving: " << std::endl;
	}

	// the matrix is upper triangular
	triangular_solver(mat, pivots, opt, -1, p, pool);

	if (verbose) {
		std::cout << std::endl;
	}

	sparse_mat_clear(tranmat);
	free(colpivs);
	free(rowpivs);

	return pivots;
}


std::vector<std::pair<slong, slong>> snmod_mat_rref(snmod_mat_t mat, nmod_t p, BS::thread_pool& pool, rref_option_t opt) {
	if (opt->pivot_dir)
		return snmod_mat_rref_r(mat, p, pool, opt);
	else
		return snmod_mat_rref_c(mat, p, pool, opt);
}

ulong snmod_mat_rref_kernel(snmod_mat_t K, const snmod_mat_t M, const std::vector<std::pair<slong, slong>>& pivots, nmod_t p, BS::thread_pool& pool) {
	auto rank = pivots.size();
	if (rank == M->ncol)
		return 0; // full rank, no kernel

	ulong m1 = 1;

	if (rank == 0) {
		sparse_mat_init(K, M->ncol, M->ncol);
		for (size_t i = 0; i < M->ncol; i++)
			_sparse_vec_set_entry(sparse_mat_row(K, i), i, m1);
		return M->ncol;
	}
	m1 = nmod_neg(m1, p);

	snmod_mat_t rows, trows;
	sparse_mat_init(rows, rank, M->ncol);
	sparse_mat_init(trows, M->ncol, rank);
	for (size_t i = 0; i < rank; i++) {
		sparse_vec_set(sparse_mat_row(rows, i), sparse_mat_row(M, pivots[i].first));
	}
	sparse_mat_transpose(trows, rows);
	sparse_mat_clear(rows);

	sparse_mat_init(K, M->ncol - rank, M->ncol);
	for (size_t i = 0; i < K->nrow; i++)
		sparse_mat_row(K, i)->nnz = 0;

	std::vector<slong> colpivs(M->ncol, -1);
	std::vector<slong> nonpivs;
	for (size_t i = 0; i < rank; i++)
		colpivs[pivots[i].second] = pivots[i].first;

	for (auto i = 0; i < M->ncol; i++)
		if (colpivs[i] == -1)
			nonpivs.push_back(i);

	pool.detach_loop<size_t>(0, nonpivs.size(), [&](size_t i) {
		auto thecol = sparse_mat_row(trows, nonpivs[i]);
		auto k_vec = sparse_mat_row(K, i);
		sparse_vec_realloc(k_vec, thecol->nnz + 1);
		for (size_t j = 0; j < thecol->nnz; j++) {
			_sparse_vec_set_entry(k_vec,
				pivots[thecol->indices[j]].second,
				thecol->entries + j);
		}
		_sparse_vec_set_entry(k_vec, nonpivs[i], m1);
		sparse_vec_sort_indices(k_vec); // sort the indices
		});
	pool.wait();

	sparse_mat_clear(trows);
	return M->ncol - rank;
}