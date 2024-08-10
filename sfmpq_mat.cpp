#include "sparse_mat.h"
#include <cstring>

using namespace std::chrono_literals;

// Row x - a*Row y
inline void sfmpq_mat_xmay(sfmpq_mat_t mat, slong x, slong y, fmpq_t a) {
	sfmpq_vec_sub_mul(mat->rows + x, mat->rows + y, a);
}

// first look for rows with only one nonzero value and eliminate them
// we assume that mat is canonical, i.e. each index is sorted
// and the result is also canonical
ulong eliminate_row_with_one_nnz(sfmpq_mat_t mat,
	sparse_mat_t<fmpq*> tranmat,
	slong* donelist) {
	auto localcounter = 0;
	slong* pivlist = (slong*)malloc(mat->nrow * sizeof(slong));
	slong* collist = (slong*)malloc(mat->ncol * sizeof(slong));
	for (auto i = 0; i < mat->nrow; i++)
		pivlist[i] = -1;
	for (auto i = 0; i < mat->ncol; i++)
		collist[i] = -1;
	for (auto i = 0; i < mat->nrow; i++) {
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
		return 0;
	}

	sparse_mat_transpose_pointer(tranmat, mat);
	for (auto i = 0; i < mat->nrow; i++) {
		if (pivlist[i] == -1)
			continue;
		auto thecol = tranmat->rows + pivlist[i];
		ulong colone = 0;
		for (auto j = 0; j < thecol->nnz; j++) {
			if (thecol->indices[j] == i) {
				colone = j;
				fmpq_one(thecol->entries[j]);
			}
			else
				fmpq_zero(thecol->entries[j]);
		}
	}

	for (auto i = 0; i < mat->nrow; i++) {
		sparse_vec_canonicalize(mat->rows + i);
	}

	for (auto i = 0; i < mat->nrow; i++)
		if (pivlist[i] != -1)
			donelist[i] = pivlist[i];

	free(pivlist);
	free(collist);
	return localcounter;
}

ulong eliminate_row_with_one_nnz_rec(sfmpq_mat_t mat,
	sparse_mat_t<fmpq*> tranmat,
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

// lowerbound?
inline slong rowcolpart(std::vector<slong> conp, ulong nrow) {
	slong i;
	for (i = 0; i < conp.size(); i++)
		if (conp[i] >= nrow)
			break;
	return i;
}

// 0    1    -1   2
// v a  v 0  v a  v 0
// b w  b w  0 w  0 w

inline int pair_irr(std::pair<slong, slong> v, std::pair<slong, slong> w,
	sfmpq_mat_t mat) {
	if (v.first == w.first || v.second == w.second)
		return 0;
	auto a = sparse_mat_entry(mat, v.first, w.second, true);
	auto b = sparse_mat_entry(mat, w.first, v.second, true);
	if (a == NULL && b == NULL)
		return 2;
	if (a == NULL)
		return 1;
	if (b == NULL)
		return -1;
	return 0;
}

auto findmanypivots(sfmpq_mat_t mat, sparse_mat_t<fmpq*> tranmat,
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
			flag = flag && (prows.find(indices[i]) == prows.end());
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

slong findrowpivot(sfmpq_mat_t mat, slong col, slong* rowpivs,
	std::vector<slong>& colparts, std::vector<slong>& kkparts,
	std::vector<std::vector<slong>>& rowparts, slong* dolist,
	fmpq** entrylist, slong& dolist_len, BS::thread_pool& pool) {

	auto pivot = col;
	auto mm = colparts[col];
	auto rowlist = rowparts[mm];

	// parallelize this loop

	const auto loop = [&](slong i) {
		if (rowlist[i] == -1)
			return;
		auto entry = sparse_mat_entry(mat, rowlist[i], pivot, true);
		if (entry == NULL || fmpq_is_zero(entry))
			return;
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

// first write a stupid one
// TODO: Gilbert-Peierls algorithm for parallel computation 
// see https://hal.science/hal-01333670/document
void schur_complete(sfmpq_mat_t mat, slong row, std::vector<std::pair<slong, slong>>& pivots, int ordering, fmpq* tmpvec) {
	auto therow = mat->rows + row;

	if (therow->nnz == 0)
		return;

	// if pivots size is small, we can use the sparse vector
	// to save to cost of converting between sparse and dense
	// vectors, otherwise we use dense vector
	if (pivots.size() < 100) {
		sfmpq_vec_t tmpsvec;
		sparse_vec_init(tmpsvec, mat->ncol);
		sparse_vec_set(tmpsvec, therow);

		if (ordering == 1) {
			for (auto& [r, c] : pivots) {
				auto entry = sparse_vec_entry(tmpsvec, c);
				if (entry == NULL)
					continue;
				auto row = mat->rows + r;
				sfmpq_vec_sub_mul(tmpsvec, row, entry);
			}
		}
		else if (ordering == -1) {
			for (auto ii = pivots.rbegin(); ii != pivots.rend(); ii++) {
				auto& [r, c] = *ii;
				auto entry = sparse_vec_entry(tmpsvec, c);
				if (entry == NULL)
					continue;
				auto row = mat->rows + r;
				sfmpq_vec_sub_mul(tmpsvec, row, entry);
			}
		}

		sparse_vec_swap(therow, tmpsvec);
		sparse_vec_clear(tmpsvec);
		return;
	}

	for (size_t i = 0; i < mat->ncol; i++)
		fmpq_zero(tmpvec + i);
	for (size_t i = 0; i < therow->nnz; i++)
		fmpq_set(tmpvec + therow->indices[i], therow->entries + i);

	fmpq_t entry;
	fmpq_init(entry);

	if (ordering == 1) {
		for (auto& [r, c] : pivots) {
			fmpq_set(entry, tmpvec + c);
			if (fmpq_is_zero(entry))
				continue;

			auto row = mat->rows + r;

			for (size_t i = 0; i < row->nnz; i++)
				fmpq_submul(tmpvec + row->indices[i], entry, row->entries + i);
		}
	}
	else if (ordering == -1) {
		for (auto ii = pivots.rbegin(); ii != pivots.rend(); ii++) {
			auto& [r, c] = *ii;
			auto entry = tmpvec + c;
			if (fmpq_is_zero(entry))
				continue;

			auto row = mat->rows + r;
			for (size_t i = 0; i < row->nnz; i++)
				fmpq_submul(tmpvec + row->indices[i], entry, row->entries + i);
		}
	}

	therow->nnz = 0;
	for (size_t i = 0; i < mat->ncol; i++) {
		if (!fmpq_is_zero(tmpvec + i))
			_sparse_vec_set_entry(therow, i, tmpvec + i);
	}

	fmpq_clear(entry);
}

// upper solver : ordering = -1
// lower solver : ordering = 1
void triangular_solver(sfmpq_mat_t mat, std::vector<std::pair<slong, slong>>& pivots,
	rref_option_t opt, int ordering, BS::thread_pool& pool) {
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
			sfmpq_mat_xmay(mat, r, pp.first, entry);
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

auto sfmpq_mat_rref_c(sfmpq_mat_t mat, BS::thread_pool& pool,
	rref_option_t opt) {
	// first canonicalize, sort and compress the matrix
	for (size_t i = 0; i < mat->nrow; i++) {
		sparse_vec_sort_indices(mat->rows + i);
	}
	sparse_mat_compress(mat);

	auto printstep = opt->print_step;
	bool verbose = opt->verbose;

	ulong rank = 0;

	ulong init_nnz = sparse_mat_nnz(mat);
	ulong now_nnz = init_nnz;

	fmpq_t scalar;
	fmpq_init(scalar);

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
	sparse_mat_t<fmpq*> tranmat;
	sparse_mat_init(tranmat, mat->ncol, mat->nrow);

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

	fmpq** entrylist = (fmpq**)malloc(mat->nrow * sizeof(fmpq*));
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
		fmpq_t scalar;
		fmpq_init(scalar);
		fmpq_inv(scalar, entrylist[i]);
		sfmpq_vec_rescale(mat->rows + dolist[i], scalar);
		fmpq_clear(scalar);
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

	if (verbose) {
		std::cout << "** found " << connected_components.size() << " components"
			<< std::endl;
	}

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

	std::stable_sort(
		colperm.begin() + countone, colperm.end(),
		[&colparts](slong a, slong b) { return colparts[a] < colparts[b]; });

	// int num_cache = pool.get_thread_count();
	// sfmpq_mat_t cachemat;
	// sparse_mat_init(cachemat, num_cache, mat->ncol);

	sparse_mat_transpose_pointer(tranmat, mat);

	// upper triangle (with respect to row and col perm)
	while (countone < mat->ncol) {
		auto start = clocknow();

		if (colparts[colperm[countone]] == -1) {
			countone++;
			continue;
		}

		auto ps = findmanypivots(mat, tranmat, rowpivs, colperm,
			colperm.begin() + countone, opt->search_depth);

		for (auto kk = 0; kk < ps.size(); kk++) {
			auto start = clocknow();

			auto pp = ps[kk];
			auto row = pp.first;
			auto col = *pp.second;
			rowpivs[row] = col;
			rank++;
			fmpq_inv(scalar, sparse_mat_entry(mat, row, col, true));
			sfmpq_vec_rescale(mat->rows + row, scalar);
			auto thecol = tranmat->rows + col;
			pool.detach_loop<slong>(0, thecol->nnz, [&](slong i) {
				if (thecol->indices[i] == row)
					return;
				auto entry =
					sparse_mat_entry(mat, thecol->indices[i], col, true);
				auto this_id = BS::this_thread::get_index().value();
				// sfmpq_mat_xmay_cached(mat, thecol->indices[i], row,
				// cachemat->rows + this_id, entry);
				sfmpq_mat_xmay(mat, thecol->indices[i], row, entry);
				});
			pool.wait();
			auto end = clocknow();

			if (verbose && (countone + kk + 1) % printstep == 0) {
				now_nnz = sparse_mat_nnz(mat);
				std::cout << "\r-- Col: " << countone + kk + 1 << "/"
					<< mat->ncol << "  " << "row to eliminate: "
					<< tranmat->rows[*pp.second].nnz - 1 << "  "
					<< "rank: " << rank << "  " << "nnz: " << now_nnz
					<< "  " << "density: "
					<< (double)100 * now_nnz / (mat->nrow * mat->ncol)
					<< "%  " << "speed: " << 1 / usedtime(start, end)
					<< " col/s" << std::flush;
			}
		}

		// reorder the cols, move ps to the front
		std::unordered_set<slong> indices(ps.size());
		for (size_t i = 0; i < ps.size(); i++)
			indices.insert(ps[i].second - colperm.begin());
		std::vector<slong> result(colperm.begin(), colperm.begin() + countone);
		result.reserve(colperm.size());
		for (auto ind : ps) {
			result.push_back(*ind.second);
		}
		for (auto it = countone; it < mat->ncol; it++) {
			if (indices.find(it) == indices.end()) {
				result.push_back(colperm[it]);
			}
		}
		colperm = std::move(result);

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

	for (slong kk = countone; kk < mat->ncol; kk++) {
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

		fmpq_inv(scalar, sparse_vec_entry(mat->rows + rowi, pivot, true));
		sfmpq_vec_rescale(mat->rows + rowi, scalar);

		pool.detach_loop<slong>(0, dolist_len, [&](slong i) {
			if (dolist[i] == rowi)
				return;
			sfmpq_mat_xmay(mat, dolist[i], rowi, entrylist[i]);
			});
		pool.wait();

		if (kk % opt->sort_step == 0 || kk == mat->ncol - 1) {
			if (verbose)
				std::cout << std::endl;
			count =
				eliminate_row_with_one_nnz_rec(mat, tranmat, rowpivs, false);
			rank += count;
			now_nnz = sparse_mat_nnz(mat);
			if (verbose) {
				std::cout << "** eliminated " << count << " rows, and nnz is "
					<< now_nnz << std::endl;
			}

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
			if (kk % printstep == 0 || kk == mat->ncol - 1) {
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

	triangular_solver(mat, pivots, opt, -1, pool);

	if (verbose) {
		std::cout << std::endl;
	}

	// sparse_mat_clear(cachemat);
	sparse_mat_clear(tranmat);

	fmpq_clear(scalar);
	free(entrylist);
	free(dolist);
	free(rowpivs);

	return pivots;
}

auto sfmpq_mat_rref_r(sfmpq_mat_t mat, BS::thread_pool& pool, rref_option_t opt) {
	// first canonicalize, sort and compress the matrix

	for (size_t i = 0; i < mat->nrow; i++) {
		sparse_vec_sort_indices(mat->rows + i);
	}
	sparse_mat_compress(mat);

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

	sparse_mat_t<fmpq*> tranmatp;
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

	sparse_mat_t<bool> tranmat;
	sparse_mat_init(tranmat, mat->ncol, mat->nrow);

	fmpq* cachedensedmat = (fmpq*)malloc(mat->ncol * pool.get_thread_count() * sizeof(fmpq));
	for (size_t i = 0; i < mat->ncol * pool.get_thread_count(); i++)
		fmpq_init(cachedensedmat + i);

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

		fmpq_t scalar;
		fmpq_init(scalar);

		for (auto& [c, rp] : ps) {
			pivots.push_back(std::make_pair(*rp, c));
			n_pivots.push_back(std::make_pair(*rp, c));
			colpivs[c] = *rp;
			rowpivs[*rp] = c;
			fmpq_inv(scalar, sparse_mat_entry(mat, *rp, c, true));
			sfmpq_vec_rescale(mat->rows + *rp, scalar);
		}

		fmpq_clear(scalar);

		// reorder the rows, move ps to the front
		std::unordered_set<slong> indices(ps.size());
		for (size_t i = 0; i < ps.size(); i++)
			indices.insert(ps[i].second - rowperm.begin());
		std::vector<slong> result(rowperm.begin(), rowperm.begin() + kk);
		result.reserve(rowperm.size());
		for (auto ind : ps) {
			result.push_back(*ind.second);
		}
		for (auto it = kk; it < mat->nrow; it++) {
			if (indices.find(it) == indices.end()) {
				result.push_back(rowperm[it]);
			}
		}
		rowperm = std::move(result);

		kk += ps.size();
		rank += ps.size();

		auto end = clocknow();
		double oldstatus = 0;
		std::atomic<int> count(0);
		pool.detach_loop<slong>(kk, mat->nrow, [&](slong i) {
			if (rowpivs[rowperm[i]] != -1)
				return;
			auto id = BS::this_thread::get_index().value();
			schur_complete(mat, rowperm[i], n_pivots, 1, cachedensedmat + id * mat->ncol);
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
			// std::this_thread::sleep_for(std::chrono::milliseconds(50));
		}
		pool.wait();

		if (ps.size() < 0) {
			break;
		}
	}

	for (size_t i = 0; i < mat->ncol * pool.get_thread_count(); i++)
		fmpq_clear(cachedensedmat + i);
	free(cachedensedmat);

	if (verbose) {
		std::cout << "\n** Rank: " << rank
			<< " nnz: " << sparse_mat_nnz(mat) << std::endl
			<< "\n>> Reverse solving: " << std::endl;
	}

	// the matrix is upper triangular
	triangular_solver(mat, pivots, opt, -1, pool);

	if (verbose) {
		std::cout << std::endl;
	}

	sparse_mat_clear(tranmat);
	free(colpivs);
	free(rowpivs);

	return pivots;
}


std::vector<std::pair<slong, slong>> sfmpq_mat_rref(sfmpq_mat_t mat, BS::thread_pool& pool, rref_option_t opt) {
	if (opt->pivot_dir)
		return sfmpq_mat_rref_r(mat, pool, opt);
	else
		return sfmpq_mat_rref_c(mat, pool, opt);
}

ulong sfmpq_mat_rref_kernel(sfmpq_mat_t K, const sfmpq_mat_t M, const std::vector<std::pair<slong, slong>>& pivots, BS::thread_pool& pool) {
	auto rank = pivots.size();
	if (rank == M->ncol)
		return 0; // full rank, no kernel

	fmpq_t m1;
	fmpq_init(m1);
	fmpq_one(m1);

	if (rank == 0) {
		sparse_mat_init(K, M->ncol, M->ncol);
		for (size_t i = 0; i < M->ncol; i++)
			_sparse_vec_set_entry(sparse_mat_row(K, i), i, m1);
		fmpq_clear(m1);
		return M->ncol;
	}
	fmpq_neg(m1, m1);

	sfmpq_mat_t rows, trows;
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
	fmpq_clear(m1);
	return M->ncol - rank;
}