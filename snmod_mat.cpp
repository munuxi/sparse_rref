#include "sparse_mat.h"

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
	std::vector<slong> pivlist(mat->nrow, -1);
	std::vector<slong> collist(mat->ncol, -1);
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

	if (localcounter == 0) 
		return localcounter;

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

	for (size_t i = 0; i < mat->nrow; i++) 
		sparse_vec_canonicalize(mat->rows + i);

	for (size_t i = 0; i < mat->nrow; i++)
		if (pivlist[i] != -1)
			donelist[i] = pivlist[i];

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

	using iter = std::vector<slong>::iterator;
	auto end = colperm.end();

	std::list<std::pair<slong, iter>> pivots;
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

	// leftlook then
	// now prows will be used as pcols to store the cols that have been used
	prows.clear();
	// make a table to help to look for row pointers
	std::vector<iter> colptrs(mat->ncol, end);
	for (auto it = start; it != end; it++)
		colptrs[*it] = it;

	for (auto p : pivots) 
		prows.insert(*(p.second));

	for (size_t i = 0; i < mat->nrow; i++) {
		if (pivots.size() > max_depth)
			break;
		auto row = i; 
		if (rowpivs[row] != -1)
			continue;
		bool flag = true;
		slong col = 0;
		ulong mnnz = ULLONG_MAX;
		auto tc = mat->rows + row;
		for (size_t j = 0; j < tc->nnz; j++) {
			if (colptrs[tc->indices[j]] == end)
				continue;
			flag = (prows.find(tc->indices[j]) == prows.end());
			if (!flag)
				break;
			if (tranmat->rows[tc->indices[j]].nnz < mnnz) {
				mnnz = tranmat->rows[tc->indices[j]].nnz;
				col = tc->indices[j];
			}
		}
		if (!flag)
			continue;
		if (mnnz != ULLONG_MAX) {
			pivots.push_front(std::make_pair(row, colptrs[col]));
			prows.insert(col);
		}
	}

	std::vector<std::pair<slong, iter>> result(pivots.begin(), pivots.end());
	return result;
}

// TODO: add a DFS algorithm to find a maximal compatible set
// //     |    a2 b2 |    |    a2 b2 |    |    a2 b2|    |    a2 b2 |   
// //  11 | a1  *  * | 10 | a1  *  0 | 01 | a1  *  *| 00 | a1  *  0 |   
// //     | b1  *  * |    | b1  *  * |    | b1  0  *|    | b1  0  * |   
// //   3               2               1              0 
// 
// int compatible_degree(std::unordered_set<std::pair<slong, slong>>& adjmat,
// 	std::pair<slong, slong>& a, std::pair<slong, slong>& b) {
// 	auto [a1,a2] = a;
// 	auto [b1,b2] = b;
// 	if (a1 == b1 || a2 == b2) // same row or same col
// 		return 3;
// 	bool test1 = adjmat.find(std::make_pair(a1, b2)) == adjmat.end();
// 	bool test2 = adjmat.find(std::make_pair(b1, a2)) == adjmat.end();
// 	if (test1 && test2)
// 		return 0;
// 	if (test1)
// 		return 2;
// 	if (test2)
// 		return 1;
// 	return 3;
// }

// first write a stupid one
// TODO: Gilbert-Peierls algorithm for parallel computation 
// see https://hal.science/hal-01333670/document
void schur_complete(snmod_mat_t mat, slong row, std::vector<std::pair<slong, slong>>& pivots, int ordering, nmod_t p, ulong* tmpvec) {
	auto therow = mat->rows + row;

	if (therow->nnz == 0)
		return;

	// if pivots size is small, we can use the sparse vector
	// to save to cost of converting between sparse and dense
	// vectors, otherwise we use dense vector
	if (pivots.size() < 100) {
		snmod_vec_t tmpsvec;
		sparse_vec_init(tmpsvec, mat->ncol);
		sparse_vec_set(tmpsvec, therow);

		// auto& tmpsvec = therow;

		if (ordering == 1) {
			for (auto& [r, c] : pivots) {
				auto entry = sparse_vec_entry(tmpsvec, c);
				if (entry == NULL)
					continue;
				auto row = mat->rows + r;
				snmod_vec_sub_mul(tmpsvec, row, *entry, p);
			}
		}
		else if (ordering == -1) {
			for (auto ii = pivots.rbegin(); ii != pivots.rend(); ii++) {
				auto& [r, c] = *ii;
				auto entry = sparse_vec_entry(tmpsvec, c);
				if (entry == NULL)
					continue;
				auto row = mat->rows + r;
				snmod_vec_sub_mul(tmpsvec, row, *entry, p);
			}
		}

		sparse_vec_swap(therow, tmpsvec);
		sparse_vec_clear(tmpsvec);
		return;
	}

	memset(tmpvec, 0, mat->ncol * sizeof(ulong));
	for (size_t i = 0; i < therow->nnz; i++)
		tmpvec[therow->indices[i]] = therow->entries[i];

	if (ordering > 0) {
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
		}
	}
	else {
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

			//if (row->nnz < 4) {
			//	for (size_t i = 0; i < row->nnz; i++) {
			//		tmpvec[row->indices[i]] = nmod_sub(tmpvec[row->indices[i]],
			//			n_mulmod_shoup(entry, row->entries[i], e_pr, p.n), p);
			//	}
			//}
			//else {
			//	for (size_t i = 0; i < (row->nnz) / 4; i++) {
			//		tmpvec[row->indices[4 * i]] = nmod_sub(tmpvec[row->indices[4 * i]],
			//			n_mulmod_shoup(entry, row->entries[4 * i], e_pr, p.n), p);
			//		tmpvec[row->indices[4 * i + 1]] = nmod_sub(tmpvec[row->indices[4 * i + 1]],
			//			n_mulmod_shoup(entry, row->entries[4 * i + 1], e_pr, p.n), p);
			//		tmpvec[row->indices[4 * i + 2]] = nmod_sub(tmpvec[row->indices[4 * i + 2]],
			//			n_mulmod_shoup(entry, row->entries[4 * i + 2], e_pr, p.n), p);
			//		tmpvec[row->indices[4 * i + 3]] = nmod_sub(tmpvec[row->indices[4 * i + 3]],
			//			n_mulmod_shoup(entry, row->entries[4 * i + 3], e_pr, p.n), p);
			//	}
			//	int left = row->nnz % 4;
			//	for (size_t i = 0; i < left; i++) {
			//		auto index = row->nnz - left + i;
			//		tmpvec[row->indices[index]] = nmod_sub(tmpvec[row->indices[index]],
			//			n_mulmod_shoup(entry, row->entries[index], e_pr, p.n), p);
			//	}
			//}
		}
	}
	therow->nnz = 0;
	for (size_t i = 0; i < mat->ncol; i++) {
		if (tmpvec[i] != 0)
			_sparse_vec_set_entry(therow, i, tmpvec[i]);
	}
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
			auto now_nnz = sparse_mat_nnz(mat);
			std::cout << "\r-- Row: " << (i + 1) << "/" << pivots.size()
				<< "  " << "row to eliminate: " << thecol.size() - 1
				<< "  " << "nnz: " << now_nnz << "  " << "density: "
				<< (double)100 * now_nnz / (mat->nrow * mat->ncol)
				<< "%  " << "speed: " << 1 / usedtime(start, end)
				<< " row/s" << std::flush;
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
	memset(rowpivs, -1, mat->nrow * sizeof(slong));
	std::vector<slong> colpivs(mat->ncol, -1);
	std::vector<std::pair<slong, slong>> pivots;
	// perm the col
	std::vector<slong> colperm(mat->ncol);
	for (size_t i = 0; i < mat->ncol; i++) 
		colperm[i] = i;

	// look for row with only one non-zero entry

	// compute the transpose of pointers of the matrix
	sparse_mat_t<ulong*> tranmat;
	sparse_mat_init(tranmat, mat->ncol, mat->nrow);

	bool verbose = opt->verbose;
	ulong count =
		eliminate_row_with_one_nnz_rec(mat, tranmat, rowpivs, verbose);
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

	// look for pivot cols with only one nonzero element
	slong kk = 0;
	memset(rowpivs, -1, mat->nrow * sizeof(slong));
	for (; kk < mat->ncol; kk++) {
		auto nnz = tranmat->rows[colperm[kk]].nnz;
		if (nnz == 0)
			continue;
		if (nnz == 1) {
			auto row = tranmat->rows[colperm[kk]].indices[0];
			if (rowpivs[row] != -1)
				continue;
			rowpivs[row] = colperm[kk];
			colpivs[colperm[kk]] = row;
			auto e = sparse_vec_entry(mat->rows + row, rowpivs[row], true);
			auto scalar = nmod_inv(*e, p);
			snmod_vec_rescale(mat->rows + row, scalar, p);
			pivots.push_back(std::make_pair(row, colperm[kk]));
			rank++;
		}
		else if (nnz > 1)
			break; // since it's sorted
	}

	init_nnz = sparse_mat_nnz(mat);

	ulong* cachedensedmat = (ulong*)malloc(mat->ncol * pool.get_thread_count() * sizeof(ulong));
	slong* donelist = (slong*)malloc(mat->nrow * sizeof(slong));
	sparse_mat_transpose_pointer(tranmat, mat);

	// upper triangle (with respect to row and col perm)
	while (kk < mat->ncol) {
		auto start = clocknow();

		auto ps = findmanypivots_c(mat, tranmat, rowpivs, colperm,
			colperm.begin() + kk, opt->search_depth);
		if (ps.size() == 0)
			break;

		std::vector<std::pair<slong, slong>> n_pivots;
		for (auto i = ps.rbegin(); i != ps.rend(); i++){
			auto [r, cp] = *i;
			rowpivs[r] = *cp;
			colpivs[*cp] = r;
			scalar = nmod_inv(*sparse_mat_entry(mat, r, *cp), p);
			snmod_vec_rescale(mat->rows + r, scalar, p);
			n_pivots.push_back(std::make_pair(r, *cp));
			pivots.push_back(std::make_pair(r, *cp));
		}
		rank += ps.size();

		pool.detach_loop<slong>(0, mat->nrow, [&](slong i) {
			if (rowpivs[i] != -1)
				return;
			auto id = BS::this_thread::get_index().value();
			schur_complete(mat, i, n_pivots, 1, p,
				cachedensedmat + id * mat->ncol);
			});
		
		// reorder the cols, move ps to the front
		std::unordered_set<slong> indices(ps.size());
		for (size_t i = 0; i < ps.size(); i++)
			indices.insert(ps[i].second - colperm.begin());
		std::vector<slong> result(colperm.begin(), colperm.begin() + kk);
		result.reserve(colperm.size());
		for (auto ind : ps) {
			result.push_back(*ind.second);
		}
		for (auto it = kk; it < mat->ncol; it++) {
			if (indices.find(it) == indices.end()) {
				result.push_back(colperm[it]);
			}
		}
		colperm = std::move(result);

		pool.wait();

		auto end = clocknow();
		now_nnz = sparse_mat_nnz(mat);
		std::cout << "\r-- Col: " << kk + ps.size() << "/"
			<< mat->ncol
			<< " rank: " << rank << "  " << "nnz: " << now_nnz
			<< "  " << "density: "
			<< 100 * (double)now_nnz / (mat->nrow * mat->ncol)
			<< "%  " << "speed: " << ps.size() / usedtime(start, end)
			<< " col/s" << std::flush;
		
		memcpy(donelist, rowpivs, mat->nrow * sizeof(slong));
		count = eliminate_row_with_one_nnz_rec(mat, tranmat, donelist, false, 0);
		rank += count;
		kk += ps.size();

		sparse_mat_transpose_pointer(tranmat, mat);
		// sort pivots by nnz, it will be faster
		std::stable_sort(colperm.begin() + kk, colperm.end(),
			[&tranmat](slong a, slong b) {
				return tranmat->rows[a].nnz < tranmat->rows[b].nnz;
			});
	}

	if (verbose) {
		std::cout << "\n** Rank: " << rank << " nnz: " << sparse_mat_nnz(mat)
			<< "  " << std::endl;
		std::cout << "\n>> Reverse solving: " << std::endl;
	}

	// the matrix is upper triangular
	triangular_solver(mat, pivots, opt, -1, p, pool);

	if (verbose) {
		std::cout << '\n' << std::endl;
	}

	sparse_mat_clear(tranmat);

	free(donelist);
	free(rowpivs);
	free(cachedensedmat);

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

	sparse_mat_t<bool> tranmat;
	sparse_mat_init(tranmat, mat->ncol, mat->nrow);

	//sparse_mat_struct<bool>* tranmat_vec = (sparse_mat_struct<bool>*)
	//	malloc(pool.get_thread_count() * sizeof(sparse_mat_struct<bool>));
	//for (size_t i = 0; i < pool.get_thread_count(); i++)
	//	sparse_mat_init(tranmat_vec + i, mat->ncol, mat->nrow);

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

	sparse_mat_transpose_part(tranmat, mat, rowperm);

	while (kk < mat->nrow) {
		auto start = clocknow();
		auto row = rowperm[kk];
		std::vector<std::pair<slong, slong>> n_pivots;

		if (mat->rows[row].nnz == 0) {
			kk++;
			continue;
		}

		pool.wait();
		std::vector<std::pair<slong, std::vector<slong>::iterator>> ps;
		ps = findmanypivots_r(mat, tranmat, colpivs,
			rowperm, rowperm.begin() + kk, opt->search_depth);

		for (auto& [c, rp] : ps) {
			pivots.push_back(std::make_pair(*rp, c));
			n_pivots.push_back(std::make_pair(*rp, c));
			colpivs[c] = *rp;
			rowpivs[*rp] = c;
			ulong scalar = nmod_inv(*sparse_mat_entry(mat, *rp, c), p);
			snmod_vec_rescale(mat->rows + *rp, scalar, p);
		}

		// reorder the rows, move ps to the front
		std::unordered_set<slong> indices(ps.size());
		for (size_t i = 0; i < ps.size(); i++)
			indices.insert(ps[i].second - rowperm.begin());
		std::vector<slong> result(rowperm.begin(), rowperm.begin() + kk);
		result.reserve(rowperm.size());
		for (auto ind : ps){
			result.push_back(*ind.second);
		}
		for (auto it = kk; it < mat->nrow; it++) {
			if (indices.find(it) == indices.end()){
				result.push_back(rowperm[it]);
			}
		}
		rowperm = std::move(result);

		kk += ps.size();
		rank += ps.size();
		slong newpiv = ps.size();

		auto end = clocknow();
		double oldstatus = 0;
		int oldcout = 0;
		std::atomic<int> count(0);
		ulong tran_count = 0;
		// flags[i] is true if the i-th row has been computed
		std::vector<std::atomic<bool>> flags(mat->nrow - kk);
		for (size_t i = 0; i < mat->nrow - kk; i++)
			flags[i] = false;
		// and then compute the elimination of the rows asynchronizely
		pool.detach_loop<slong>(kk, mat->nrow, [&](slong i) {
			if (rowpivs[rowperm[i]] != -1)
				return;
			auto id = BS::this_thread::get_index().value();
			schur_complete(mat, rowperm[i], n_pivots, 1, p,
				cachedensedmat + id * mat->ncol);
			count++;
			flags[i - kk] = true;
			});
		std::vector<slong> leftrows(rowperm.begin() + kk, rowperm.end());
		for (size_t i = 0; i < tranmat->nrow; i++)
			tranmat->rows[i].nnz = 0;
		// compute the transpose of the submatrix and print the status asynchronizely
		while (tran_count < leftrows.size()) {
			for (size_t i = 0; i < leftrows.size(); i++) {
				if (flags[i]) {
					auto row = leftrows[i];
					auto therow = mat->rows + row;
					for (size_t j = 0; j < therow->nnz; j++) {
						auto col = therow->indices[j];
						_sparse_vec_set_entry(tranmat->rows + col, row, therow->entries + j);
					}
					tran_count++;
					flags[i] = false;
				}
			}
			if (verbose && count - oldcout > printstep) {
				auto status = (kk - newpiv + 1) + ((double)count / (mat->nrow - kk)) * newpiv;
				end = clocknow();
				now_nnz = sparse_mat_nnz(mat);
				std::cout << "\r-- Row: " << (int)std::floor(status) << "/" << mat->nrow
					<< "  rank: " << rank
					<< "  nnz: " << now_nnz << "  " << "density: "
					<< (double)100 * now_nnz / (mat->nrow * mat->ncol) << "%"
					<< "  speed: " << (status - oldstatus) / usedtime(start, end)
					<< "  row/s" << std::flush;
				oldstatus = status;
				oldcout = count;
				start = clocknow();
			}
		}
		// wait for the completion of the computation
		pool.wait();
		pool.detach_loop<slong>(0, mat->ncol, [&](slong i) {
			sparse_vec_sort_indices(tranmat->rows + i);
			});
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
				thecol->entries[j]);
		}
		_sparse_vec_set_entry(k_vec, nonpivs[i], m1);
		sparse_vec_sort_indices(k_vec); // sort the indices
		});
	pool.wait();

	sparse_mat_clear(trows);
	return M->ncol - rank;
}