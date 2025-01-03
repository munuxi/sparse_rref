/*
	Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

	This file is part of Sparse_rref. The Sparse_rref is free software: 
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/


#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include "sparse_vec.h"

template <typename T> struct sparse_mat_struct {
	ulong nrow;
	ulong ncol;
	sparse_vec_struct<T>* rows;
};

template <typename T> using sparse_mat_t = struct sparse_mat_struct<T>[1];

typedef sparse_mat_t<ulong> snmod_mat_t;
typedef sparse_mat_t<fmpq> sfmpq_mat_t;

typedef std::pair<slong, slong> pivot_t;

#define sparse_mat_row(mat, ind) ((mat)->rows + (ind))

template <typename T>
inline void _sparse_mat_init(sparse_mat_t<T> mat, ulong nrow, ulong ncol,
	ulong alloc) {
	mat->nrow = nrow;
	mat->ncol = ncol;
	mat->rows = s_malloc<sparse_vec_struct<T>>(nrow);
	for (size_t i = 0; i < nrow; i++)
		sparse_vec_init(sparse_mat_row(mat, i), alloc);
}

template <typename T>
inline void sparse_mat_init(sparse_mat_t<T> mat, ulong nrow,
	ulong ncol) {
	_sparse_mat_init(mat, nrow, ncol, 1ULL);
}

template <typename T>
inline void sparse_mat_clear(sparse_mat_t<T> mat) {
	for (size_t i = 0; i < mat->nrow; i++)
		sparse_vec_clear(sparse_mat_row(mat, i));
	s_free(mat->rows);
	mat->nrow = 0;
	mat->ncol = 0;
	mat->rows = NULL;
}

// deep copy, we assume that their sizes are the same, and des is already initialized
template <typename T>
inline void sparse_mat_set(sparse_mat_t<T> des, sparse_mat_t<T> mat) {
	for (size_t i = 0; i < mat->nrow; i++)
		sparse_vec_set(sparse_mat_row(des, i), sparse_mat_row(mat, i));
}

template <typename T>
inline ulong sparse_mat_nnz(const sparse_mat_t<T> mat) {
	ulong nnz = 0;
	for (size_t i = 0; i < mat->nrow; i++)
		nnz += sparse_mat_row(mat, i)->nnz;
	return nnz;
}

template <typename T>
inline ulong sparse_mat_alloc(const sparse_mat_t<T> mat) {
	ulong alloc = 0;
	for (size_t i = 0; i < mat->nrow; i++)
		alloc += sparse_mat_row(mat, i)->alloc;
	return alloc;
}

template <typename T>
inline void sparse_mat_compress(sparse_mat_t<T> mat) {
	for (size_t i = 0; i < mat->nrow; i++) {
		auto row = sparse_mat_row(mat, i);
		sparse_vec_canonicalize(row);
		sparse_vec_sort_indices(row);
		sparse_vec_realloc(row, row->nnz);
	}
}

template <typename T>
inline T* sparse_mat_entry(sparse_mat_t<T> mat, ulong row, ulong col, bool isbinary = true) {
	return sparse_vec_entry(sparse_mat_row(mat, row), col, isbinary);
}

template <typename T>
inline void sparse_mat_clear_zero_row(sparse_mat_t<T> mat) {
	ulong new_nrow = 0;
	for (size_t i = 0; i < mat->nrow; i++) {
		if (mat->rows[i].nnz != 0) {
			sparse_vec_swap(sparse_mat_row(mat, new_nrow), sparse_mat_row(mat, i));
			new_nrow++;
		}
		else 
			sparse_vec_clear(sparse_mat_row(mat, i));
	}
	mat->nrow = new_nrow;
	s_realloc(mat->rows, new_nrow);
}

template <typename T>
inline void sparse_mat_transpose(sparse_mat_t<T> mat2, const sparse_mat_t<T> mat) {
	for (size_t i = 0; i < mat2->nrow; i++)
		sparse_mat_row(mat2, i)->nnz = 0;

	for (size_t i = 0; i < mat->nrow; i++) {
		auto therow = sparse_mat_row(mat, i);
		for (size_t j = 0; j < therow->nnz; j++) {
			auto col = therow->indices[j];
			_sparse_vec_set_entry(sparse_mat_row(mat2, col), i, therow->entries + j);
		}
	}
}

template <typename T>
inline void sparse_mat_transpose(sparse_mat_t<T*> mat2, const sparse_mat_t<T> mat) {
	for (size_t i = 0; i < mat2->nrow; i++)
		sparse_mat_row(mat2, i)->nnz = 0;

	for (size_t i = 0; i < mat->nrow; i++) {
		auto therow = sparse_mat_row(mat, i);
		for (size_t j = 0; j < therow->nnz; j++) {
			auto col = therow->indices[j];
			auto entry = therow->entries + j;
			_sparse_vec_set_entry(sparse_mat_row(mat2, col), i, &entry);
		}
	}
}

template <typename T>
inline void sparse_mat_transpose(sparse_mat_t<bool> mat2, const sparse_mat_t<T> mat) {
	for (size_t i = 0; i < mat2->nrow; i++)
		sparse_mat_row(mat2, i)->nnz = 0;

	for (size_t i = 0; i < mat->nrow; i++) {
		auto therow = sparse_mat_row(mat, i);
		for (size_t j = 0; j < therow->nnz; j++) {
			auto col = therow->indices[j];
			_sparse_vec_set_entry(sparse_mat_row(mat2, col), i, (bool*)NULL);
		}
	}
}

// tranpose only part of the rows
template <typename T>
inline void sparse_mat_transpose_part(sparse_mat_t<T> mat2, const sparse_mat_t<T> mat, const std::vector<slong>& rows) {
	for (size_t i = 0; i < mat2->nrow; i++)
		sparse_mat_row(mat2, i)->nnz = 0;

	for (size_t i = 0; i < rows.size(); i++) {
		auto row = rows[i];
		auto therow = sparse_mat_row(mat, row);
		for (size_t j = 0; j < therow->nnz; j++) {
			auto col = therow->indices[j];
			_sparse_vec_set_entry(sparse_mat_row(mat2, col), row, therow->entries + j);
		}
	}
}

template <typename T>
inline void sparse_mat_transpose_part(sparse_mat_t<bool> mat2, const sparse_mat_t<T> mat, const std::vector<slong>& rows) {
	for (size_t i = 0; i < mat2->nrow; i++)
		sparse_mat_row(mat2, i)->nnz = 0;

	for (size_t i = 0; i < rows.size(); i++) {
		auto row = rows[i];
		auto therow = sparse_mat_row(mat, row);
		for (size_t j = 0; j < therow->nnz; j++) {
			auto col = therow->indices[j];
			_sparse_vec_set_entry(sparse_mat_row(mat2, col), row, (bool*)NULL);
		}
	}
}

template <typename T>
inline void sparse_mat_transpose_part(sparse_mat_t<T*> mat2, const sparse_mat_t<T> mat, const std::vector<slong>& rows) {
	for (size_t i = 0; i < mat2->nrow; i++)
		sparse_mat_row(mat2, i)->nnz = 0;

	for (size_t i = 0; i < rows.size(); i++) {
		auto row = rows[i];
		auto therow = sparse_mat_row(mat, row);
		for (size_t j = 0; j < therow->nnz; j++) {
			auto col = therow->indices[j];
			auto ptr = therow->entries + j;
			_sparse_vec_set_entry(sparse_mat_row(mat2, col), row, &ptr);
		}
	}
}

// dot product
template <typename T>
inline int sparse_mat_dot_sparse_vec(sparse_vec_t<T> result, const sparse_mat_t<T> mat, const sparse_vec_t<T> vec, field_t F) {
	sparse_vec_zero(result);
	if (vec->nnz == 0 || sparse_mat_nnz(mat) == 0) 
		return 0;
	T tmp[1];
	scalar_init(tmp);

	for (size_t i = 0; i < mat->nrow; i++) {
		auto therow = sparse_mat_row(mat, i);
		scalar_zero(tmp);
		if (!sparse_vec_dot(tmp, therow, vec, F))
			_sparse_vec_set_entry(result, i, tmp);
	}
	scalar_clear(tmp);
	return 1;
}

// A = B * C
template <typename T>
inline int sparse_mat_dot_sparse_mat(sparse_mat_t<T> A, sparse_mat_t<T> B, sparse_mat_t<T> C, field_t F) {
	if (B->ncol != C->nrow)
		return -1;

	sparse_mat_t<T> Ct;
	sparse_mat_init(Ct, C->ncol, C->nrow);
	sparse_mat_transpose(Ct, C);

	for (size_t i = 0; i < B->nrow; i++)
		sparse_mat_dot_sparse_vec(sparse_mat_row(A, i), B, sparse_mat_row(Ct, i), F);

	sparse_mat_clear(Ct);
	return 0;
}

// rref staffs

// first look for rows with only one nonzero value and eliminate them
// we assume that mat is canonical, i.e. each index is sorted
// and the result is also canonical
template <typename T>
ulong eliminate_row_with_one_nnz(sparse_mat_t<T> mat,
	sparse_mat_t<T*> tranmat, std::vector<slong>& donelist, bool is_tran = false) {
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

	if (!is_tran)
		sparse_mat_transpose(tranmat, mat);
	for (size_t i = 0; i < mat->nrow; i++) {
		if (pivlist[i] == -1)
			continue;
		auto thecol = sparse_mat_row(tranmat, pivlist[i]);
		for (size_t j = 0; j < thecol->nnz; j++) {
			if (thecol->indices[j] == i) {
				scalar_one(thecol->entries[j]);
			}
			else
				scalar_zero(thecol->entries[j]);
		}
	}

	for (size_t i = 0; i < mat->nrow; i++)
		sparse_vec_canonicalize(sparse_mat_row(mat, i));

	for (size_t i = 0; i < mat->nrow; i++)
		if (pivlist[i] != -1)
			donelist[i] = pivlist[i];

	return localcounter;
}

template <typename T>
ulong eliminate_row_with_one_nnz_rec(sparse_mat_t<T> mat,
	sparse_mat_t<T*> tranmat,
	std::vector<slong>& donelist, bool verbose,
	slong max_depth = INT_MAX) {
	slong depth = 0;
	ulong oldnnz = 0;
	ulong localcounter = 0;
	ulong count = 0;

	do {
		if (verbose) {
			oldnnz = sparse_mat_nnz(mat);
		}
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

template <typename T, typename S>
std::vector<std::pair<slong, std::vector<slong>::iterator>> findmanypivots(const sparse_mat_t<T> mat, const sparse_mat_t<S> tranmat,
	std::vector<slong>& rdivpivs, std::vector<slong>& dirperm,
	std::vector<slong>::iterator start,
	bool dir, size_t max_depth = ULLONG_MAX) {

	if (!dir)
		return findmanypivots(tranmat, mat, rdivpivs, dirperm, start, true, max_depth);

	using iter = std::vector<slong>::iterator;
	auto end = dirperm.end();

	auto matdir = mat;
	auto matrdir = tranmat;
	auto ndir = matdir->nrow;
	auto nrdir = matrdir->nrow;

	std::list<std::pair<slong, iter>> pivots;
	std::unordered_set<slong> pdirs;
	pdirs.reserve(std::min((size_t)4096, max_depth));

	// rightlook first
	for (auto dir = start; dir < end; dir++) {
		if ((ulong)(dir - start) > max_depth)
			break;

		auto thedir = sparse_mat_row(matdir, *dir);
		if (thedir->nnz == 0)
			continue;
		auto indices = thedir->indices;

		slong rdiv;
		ulong mnnz = ULLONG_MAX;
		bool flag = true;

		for (size_t i = 0; i < thedir->nnz; i++) {
			flag = (pdirs.count(indices[i]) == 0);
			if (!flag)
				break;
			if (rdivpivs[indices[i]] != -1)
				continue;
			ulong newnnz = matrdir->rows[indices[i]].nnz;
			if (newnnz < mnnz) {
				rdiv = indices[i];
				mnnz = newnnz;
			}
			// make the result stable
			else if (newnnz == mnnz && indices[i] < rdiv) {
				rdiv = indices[i];
			}
		}
		if (!flag)
			continue;
		if (mnnz != ULLONG_MAX) {
			pivots.push_back(std::make_pair(rdiv, dir));
			pdirs.insert(rdiv);
		}
	}

	// leftlook then
	pdirs.clear();
	// make a table to help to look for dir pointers
	std::vector<iter> dirptrs(ndir, end);
	for (auto it = start; it != end; it++)
		dirptrs[*it] = it;

	for (auto p : pivots) 
		pdirs.insert(*(p.second));

	for (size_t i = 0; i < nrdir; i++) {
		if (pivots.size() > max_depth)
			break;
		auto rdir = i;
		// auto rdir = nrdir - i - 1; // reverse ordering
		if (rdivpivs[rdir] != -1)
			continue;
		
		slong dir = 0;
		ulong mnnz = ULLONG_MAX;
		bool flag = true;

		auto tc = sparse_mat_row(matrdir, rdir);

		for (size_t j = 0; j < tc->nnz; j++) {
			if (dirptrs[tc->indices[j]] == end)
				continue;
			flag = (pdirs.count(tc->indices[j]) == 0);
			if (!flag)
				break;
			if (matdir->rows[tc->indices[j]].nnz < mnnz) {
				mnnz = matdir->rows[tc->indices[j]].nnz;
				dir = tc->indices[j];
			}
			// make the result stable
			else if (matdir->rows[tc->indices[j]].nnz == mnnz && tc->indices[j] < dir) {
				dir = tc->indices[j];
			}
		}
		if (!flag)
			continue;
		if (mnnz != ULLONG_MAX) {
			pivots.push_front(std::make_pair(rdir, dirptrs[dir]));
			pdirs.insert(dir);
		}
	}

	std::vector<std::pair<slong, iter>> result(pivots.begin(), pivots.end());
	return result;
}

// upper solver : ordering = -1
// lower solver : ordering = 1
template <typename T>
void triangular_solver(sparse_mat_t<T> mat, std::vector<pivot_t>& pivots,
	field_t F, rref_option_t opt, int ordering, BS::thread_pool& pool) {
	bool verbose = opt->verbose;
	auto printstep = opt->print_step;

	std::vector<std::vector<slong>> tranmat(mat->ncol);

	// we only need to compute the transpose of the submatrix involving pivots

	for (size_t i = 0; i < pivots.size(); i++) {
		auto therow = sparse_mat_row(mat, pivots[i].first);
		for (size_t j = 0; j < therow->nnz; j++) {
			if (scalar_is_zero(therow->entries + j))
				continue;
			auto col = therow->indices[j];
			tranmat[col].push_back(pivots[i].first);
		}
	}

	size_t count = 0;
	for (size_t i = 0; i < pivots.size(); i++) {
		size_t index = i;
		if (ordering < 0)
			index = pivots.size() - 1 - i;
		auto pp = pivots[index];
		auto thecol = tranmat[pp.second];
		auto start = sparse_base::clocknow();
		if (thecol.size() > 1) {
			pool.detach_loop<slong>(0, thecol.size(), [&](slong j) {
				auto r = thecol[j];
				if (r == pp.first)
					return;
				auto entry = sparse_mat_entry(mat, r, pp.second);
				sparse_vec_sub_mul(sparse_mat_row(mat, r), sparse_mat_row(mat, pp.first), entry, F);
				});
		}
		pool.wait();
		
		if (verbose && (i % printstep == 0 || i == pivots.size() - 1) && thecol.size() > 1) {
			count++;
			auto end = sparse_base::clocknow();
			auto now_nnz = sparse_mat_nnz(mat);
			std::cout << "\r-- Row: " << (i + 1) << "/" << pivots.size()
				<< "  " << "row to eliminate: " << thecol.size() - 1
				<< "  " << "nnz: " << now_nnz << "  " << "density: "
				<< (double)100 * now_nnz / (mat->nrow * mat->ncol)
				<< "%  " << "speed: " << count / sparse_base::usedtime(start, end)
				<< " row/s" << std::flush;
			start = sparse_base::clocknow();
			count = 0;
		}
	}
	if (opt->verbose)
		std::cout << std::endl;
}

template <typename T>
void triangular_solver(sparse_mat_t<T> mat, std::vector<std::vector<pivot_t>>& pivots,
	field_t F, rref_option_t opt, int ordering, BS::thread_pool& pool) {
	std::vector<pivot_t> n_pivots;
	for (auto p : pivots)
		n_pivots.insert(n_pivots.end(), p.begin(), p.end());
	triangular_solver(mat, n_pivots, F, opt, ordering, pool);
}

template <typename T>
size_t apart_pivots(sparse_mat_t<T> mat, std::vector<pivot_t>& pivots, size_t index) {
	auto [sr, sc] = pivots[index];
	std::unordered_set<slong> colset;
	colset.reserve(mat->ncol);
	colset.insert(sc);
	size_t i = index + 1;
	for (; i < pivots.size(); i++) {
		auto [r, c] = pivots[i];
		bool flag = true;
		auto therow = sparse_mat_row(mat, r);
		for (auto j = 0; flag && (j < therow->nnz); j++) {
			flag = (colset.count(therow->indices[j]) == 0);
		}
		if (!flag)
			break;
		colset.insert(c);
	}
	return i;
}

// SLOW!!!
template <typename T>
std::pair<std::vector<pivot_t>, std::vector<pivot_t>> apart_pivots_2(sparse_mat_t<T> mat, std::vector<pivot_t>& pivots) {
	if (pivots.size() == 0)
		return std::make_pair(std::vector<pivot_t>(), std::vector<pivot_t>());
	auto [sr, sc] = pivots[0];
	std::unordered_set<slong> colset;
	colset.reserve(mat->ncol);
	colset.insert(sc);
	std::vector<pivot_t> n_pivots;
	std::vector<pivot_t> left_pivots;
	n_pivots.push_back(pivots[0]);
	size_t i = 1;
	for (; i < pivots.size(); i++) {
		auto [r, c] = pivots[i];
		bool flag = true;
		auto therow = sparse_mat_row(mat, r);
		for (auto j = 0; flag && (j < therow->nnz); j++) {
			flag = (colset.count(therow->indices[j]) == 0);
		}
		if (!flag) {
			left_pivots.push_back(pivots[i]);
			continue;
		}
		colset.insert(c);
		n_pivots.push_back(pivots[i]);
	}
	return std::make_pair(n_pivots, left_pivots);
}

// TODO: CHECK!!!
template <typename T>
void triangular_solver_2(sparse_mat_t<T> mat, std::vector<pivot_t>& pivots,
	field_t F, rref_option_t opt, int ordering, BS::thread_pool& pool) {

	if (ordering < 0) {
		std::vector<pivot_t> npivots(pivots.rbegin(), pivots.rend());
		triangular_solver_2(mat, npivots, F, opt, -ordering, pool);
	}

	bool verbose = opt->verbose;
	auto printstep = opt->print_step;

	// then do the elimination parallelly
	int nthreads = pool.get_thread_count();
	T* cachedensedmat = s_malloc<T>(mat->ncol * nthreads);
	for (size_t i = 0; i < mat->ncol * nthreads; i++)
		scalar_init(cachedensedmat + i);

	size_t index = 0;
	while (index < pivots.size()) {
		size_t end = apart_pivots(mat, pivots, index);
		std::vector<pivot_t> n_pivots(pivots.begin() + index, pivots.begin() + end);

		for (auto [r, c] : n_pivots) {
			scalar_inv(cachedensedmat, sparse_mat_entry(mat, r, c), F);
			sparse_vec_rescale(sparse_mat_row(mat, r), cachedensedmat, F);
		}

		pool.detach_blocks<ulong>(end, pivots.size(), [&](const ulong s, const ulong e) {
			auto id = BS::this_thread::get_index().value();
			for (ulong j = s; j < e; j++) {
				schur_complete(mat, pivots[j].first, n_pivots, 1,
					F, cachedensedmat + id * mat->ncol, true);
			}
			}, (((pivots.size() - end) < 20 * nthreads) ? 0 : 10 * nthreads));
		pool.wait();

		if (opt->verbose) {
			std::cout << "\r-- Row: " << end << "/" << pivots.size() << std::flush;
		}
		index = end;
	}

	if (opt->verbose)
		std::cout << std::endl;

	// clear tmp array
	for (size_t i = 0; i < mat->ncol * nthreads; i++)
		scalar_clear(cachedensedmat + i);
	s_free(cachedensedmat);
}

template <typename T>
void triangular_solver_2(sparse_mat_t<T> mat, std::vector<std::vector<pivot_t>>& pivots,
	field_t F, rref_option_t opt, int ordering, BS::thread_pool& pool) {
	std::vector<pivot_t> n_pivots;
	if (ordering < 0) {
		for (auto i = pivots.rbegin(); i != pivots.rend(); i++) {
			auto& p = *i;
			n_pivots.insert(n_pivots.end(), p.rbegin(), p.rend());
		}
	}
	else {
		for (auto p : pivots)
			n_pivots.insert(n_pivots.end(), p.begin(), p.end());
	}
	triangular_solver_2(mat, n_pivots, F, opt, 1, pool);
}

// TODO: CHECK!!!
template <typename T>
void triangular_solver_3(sparse_mat_t<T> mat, std::vector<pivot_t>& pivots,
	field_t F, rref_option_t opt, int ordering, BS::thread_pool& pool, T* cachedensedmat) {

	if (ordering < 0) {
		std::vector<pivot_t> npivots(pivots.rbegin(), pivots.rend());
		triangular_solver_3(mat, npivots, F, opt, -ordering, pool, cachedensedmat);
	}

	int nthreads = pool.get_thread_count();
	if (pivots.size() == 0) {
		// clear tmp array
		if (cachedensedmat != NULL) {
			for (size_t i = 0; i < mat->ncol * nthreads; i++)
				scalar_clear(cachedensedmat + i);
			s_free(cachedensedmat);
		}
		return;
	}
	
	if (cachedensedmat == NULL) {
		cachedensedmat = s_malloc<T>(mat->ncol * nthreads);
		for (size_t i = 0; i < mat->ncol * nthreads; i++)
			scalar_init(cachedensedmat + i);
	}

	auto [n_pivots, left_pivots] = apart_pivots_2(mat, pivots);
	for (auto [r, c] : n_pivots) {
		scalar_inv(cachedensedmat, sparse_mat_entry(mat, r, c), F);
		sparse_vec_rescale(sparse_mat_row(mat, r), cachedensedmat, F);
	}

	pool.detach_blocks<ulong>(0, left_pivots.size(), [&](const ulong s, const ulong e) {
		auto id = BS::this_thread::get_index().value();
		for (ulong j = s; j < e; j++) {
			schur_complete(mat, left_pivots[j].first, n_pivots, 1,
				F, cachedensedmat + id * mat->ncol, true);
		}
		}, ((left_pivots.size() < 20 * nthreads) ? 0 : 10 * nthreads));
	pool.wait();

	if (opt->verbose) {
		std::cout << "\r-- Row: " << n_pivots.size() << "/" << pivots.size() << std::flush;
	}

	triangular_solver_3(mat, left_pivots, F, opt, 1, pool, cachedensedmat);
}

template <typename T>
void triangular_solver_3(sparse_mat_t<T> mat, std::vector<std::vector<pivot_t>>& pivots,
	field_t F, rref_option_t opt, int ordering, BS::thread_pool& pool, T* cachedensedmat) {
	std::vector<pivot_t> n_pivots;
	if (ordering < 0) {
		for (auto i = pivots.rbegin(); i != pivots.rend(); i++) {
			auto& p = *i;
			n_pivots.insert(n_pivots.end(), p.rbegin(), p.rend());
		}
	}
	else {
		for (auto p : pivots)
			n_pivots.insert(n_pivots.end(), p.begin(), p.end());
	}
	triangular_solver_3(mat, n_pivots, F, opt, 1, pool, cachedensedmat);
}

// first write a stupid one
// TODO: Gilbert-Peierls algorithm for parallel computation 
// see https://hal.science/hal-01333670/document
// mode : true: very sparse < SPARSE_BOUND%
template <typename T>
void schur_complete(sparse_mat_t<T> mat, slong row, std::vector<pivot_t>& pivots,
	int ordering, field_t F, T* tmpvec, sparse_base::uset& nonzero_c) {
	if (ordering < 0) {
		std::vector<pivot_t> npivots(pivots.rbegin(), pivots.rend());
		schur_complete(mat, row, npivots, -ordering, F, tmpvec, nonzero_c);
	}

	auto therow = sparse_mat_row(mat, row);

	if (therow->nnz == 0)
		return;

	// sparse_base::uset nonzero_c(mat->ncol);
	nonzero_c.clear();

	for (size_t i = 0; i < therow->nnz; i++) {
		nonzero_c.insert(therow->indices[i]);
		scalar_set(tmpvec + therow->indices[i], therow->entries + i);
	}
	T entry[1];
	ulong e_pr;
	scalar_init(entry);
	for (auto [r, c] : pivots) {
		if (nonzero_c.count(c) == 0)
			continue;
		scalar_set(entry, tmpvec + c);
		if (scalar_is_zero(entry)) {
			nonzero_c.erase(c);
			continue;
		}
		auto row = sparse_mat_row(mat, r);
		if constexpr (std::is_same_v<T, ulong>) {
			e_pr = n_mulmod_precomp_shoup(*entry, F->pvec[0].n);
		}
		for (size_t i = 0; i < row->nnz; i++) {
			if (!nonzero_c.count(row->indices[i])) {
				nonzero_c.insert(row->indices[i]);
				scalar_zero(tmpvec + row->indices[i]);
			}
			if constexpr (std::is_same_v<T, ulong>) {
				tmpvec[row->indices[i]] = _nmod_sub(tmpvec[row->indices[i]],
					n_mulmod_shoup(*entry, row->entries[i], e_pr, F->pvec[0].n), F->pvec[0]);
			}
			else if constexpr (std::is_same_v<T, fmpq>) {
				fmpq_submul(tmpvec + row->indices[i], entry, row->entries + i);
			}
			if (scalar_is_zero(tmpvec + row->indices[i]))
				nonzero_c.erase(row->indices[i]);
		}
	}
	scalar_clear(entry);
	
	therow->nnz = 0;
	for (size_t i = 0; i < nonzero_c.size(); i++) {
		if (nonzero_c[i].none())
			continue;
		auto size = nonzero_c.bitset_size;
		for (size_t j = 0; j < size; j++) {
			if (nonzero_c[i][j] && !scalar_is_zero(tmpvec + i * size + j))
				_sparse_vec_set_entry(therow, i * size + j, tmpvec + i * size + j);
		}
	}
}

// TODO: TEST!!! 
// TODO: add ordering
// if already know the pivots, we can directly do the rref
template <typename T>
void sparse_mat_direct_rref(sparse_mat_t<T>mat,
	std::vector<std::vector<pivot_t>>& pivots,
	field_t F, BS::thread_pool& pool, rref_option_t opt) {
	T scalar[1];
	scalar_init(scalar);

	// first set rows not in pivots to zero
	std::vector<slong> rowset(mat->nrow, -1);
	for (auto p : pivots)
		for (auto [r, c] : p)
			rowset[r] = c;
	for (size_t i = 0; i < mat->nrow; i++)
		if (rowset[i] == -1)
			sparse_vec_zero(sparse_mat_row(mat, i));

	sparse_mat_compress(mat);

	sparse_mat_t<T*> tranmatp;
	std::vector<slong> tmplist(mat->nrow, -1);
	sparse_mat_init(tranmatp, mat->ncol, mat->nrow);
	eliminate_row_with_one_nnz_rec(mat, tranmatp, tmplist, false);
	sparse_mat_clear(tranmatp);

	// then do the elimination parallelly
	int nthreads = pool.get_thread_count();
	T* cachedensedmat = s_malloc<T>(mat->ncol * nthreads);
	for (size_t i = 0; i < mat->ncol * nthreads; i++)
		scalar_init(cachedensedmat + i);

	for (auto i = 0; i < pivots.size(); i++) {
		auto n_pivots = pivots[i];
		if (n_pivots.size() == 0)
			continue;

		// rescale the pivots
		for (auto [r, c] : n_pivots) {
			scalar_inv(scalar, sparse_mat_entry(mat, r, c), F);
			sparse_vec_rescale(sparse_mat_row(mat, r), scalar, F);
			rowset[r] = -1;
		}

		// the first is done by eliminate_row_with_one_nnz_rec
		if (i == 0)
			continue;

		std::vector<slong> leftrows;
		for (size_t j = 0; j < mat->nrow; j++) {
			if (rowset[j] != -1)
				leftrows.push_back(j);
		}

		// upper solver
		// TODO: check mode
		pool.detach_blocks<ulong>(0, leftrows.size(), [&](const ulong s, const ulong e) {
			auto id = BS::this_thread::get_index().value();
			for (ulong j = s; j < e; j++) {
				schur_complete(mat, leftrows[j], n_pivots, 1,
					F, cachedensedmat + id * mat->ncol, true);
			}
			}, ((leftrows.size() < 20 * nthreads) ? 0 : 10 * nthreads));
		pool.wait();
	}

	// clear tmp array
	scalar_clear(scalar);
	for (size_t i = 0; i < mat->ncol * nthreads; i++)
		scalar_clear(cachedensedmat + i);
	s_free(cachedensedmat);
}

template <typename T>
std::vector<std::vector<pivot_t>> sparse_mat_rref_c(sparse_mat_t<T> mat, field_t F,
	BS::thread_pool& pool, rref_option_t opt) {
	// first canonicalize, sort and compress the matrix
	sparse_mat_compress(mat);

	T scalar[1];
	scalar_init(scalar);

	// perm the col
	std::vector<slong> colperm(mat->ncol);
	for (size_t i = 0; i < mat->ncol; i++)
		colperm[i] = i;

	auto printstep = opt->print_step;
	bool verbose = opt->verbose;

	ulong init_nnz = sparse_mat_nnz(mat);
	ulong now_nnz = init_nnz;

	// store the pivots that have been used
	// -1 is not used
	std::vector<slong> rowpivs(mat->nrow, -1);
	std::vector<std::vector<pivot_t>> pivots;

	// look for row with only one non-zero entry

	// compute the transpose of pointers of the matrix
	sparse_mat_t<T*> tranmatp;
	sparse_mat_init(tranmatp, mat->ncol, mat->nrow);
	ulong count =
		eliminate_row_with_one_nnz_rec(mat, tranmatp, rowpivs, verbose);
	now_nnz = sparse_mat_nnz(mat);
	if (verbose) {
		std::cout << "\n** eliminated " << count
			<< " rows, and reduce nnz: " << init_nnz << " -> " << now_nnz
			<< std::endl;
	}

	sparse_mat_transpose(tranmatp, mat);

	// sort pivots by nnz, it will be faster
	std::stable_sort(colperm.begin(), colperm.end(),
		[&tranmatp](slong a, slong b) {
			return tranmatp->rows[a].nnz < tranmatp->rows[b].nnz;
		});

	// look for pivot cols with only one nonzero element
	ulong kk = 0;
	std::fill(rowpivs.begin(), rowpivs.end(), -1);
	std::vector<pivot_t> n_pivots;
	for (; kk < mat->ncol; kk++) {
		auto nnz = tranmatp->rows[colperm[kk]].nnz;
		if (nnz == 0)
			continue;
		if (nnz == 1) {
			auto row = tranmatp->rows[colperm[kk]].indices[0];
			if (rowpivs[row] != -1)
				continue;
			rowpivs[row] = colperm[kk];
			scalar_inv(scalar, sparse_mat_entry(mat, row, rowpivs[row]), F);
			sparse_vec_rescale(sparse_mat_row(mat, row), scalar, F);
			n_pivots.push_back(std::make_pair(row, colperm[kk]));
		}
		else if (nnz > 1)
			break; // since it's sorted
	}
	pivots.push_back(std::move(n_pivots));
	sparse_mat_clear(tranmatp);
	auto rank = pivots[0].size();

	int nthreads = pool.get_thread_count();
	T* cachedensedmat = s_malloc<T>(mat->ncol * nthreads);
	std::vector<sparse_base::uset> nonzero_c(nthreads);
	for (size_t i = 0; i < mat->ncol * nthreads; i++) 
		scalar_init(cachedensedmat + i);
	for (size_t i = 0; i < nthreads; i++)
		nonzero_c[i].resize(mat->ncol);

	sparse_mat_t<bool> tranmat;
	sparse_mat_init(tranmat, mat->ncol, mat->nrow);
	sparse_mat_transpose(tranmat, mat);

	std::vector<slong> leftrows;
	leftrows.reserve(mat->nrow);
	for (size_t i = 0; i < mat->nrow; i++) {
		if (rowpivs[i] != -1 || mat->rows[i].nnz == 0)
			continue;
		leftrows.push_back(i);
	}

	// for printing
	double oldpr = 0;
	int bitlen_nnz = (int)std::floor(std::log(now_nnz) / std::log(10)) + 3;
	int bitlen_ncol = (int)std::floor(std::log(mat->ncol) / std::log(10)) + 1;

	while (kk < mat->ncol) {
		auto start = sparse_base::clocknow();

		auto ps = findmanypivots(mat, tranmat, rowpivs, colperm,
			colperm.begin() + kk, false, opt->search_depth);
		if (ps.size() == 0)
			break;

		n_pivots.clear();
		for (auto i = ps.rbegin(); i != ps.rend(); i++) {
			auto [r, cp] = *i;
			rowpivs[r] = *cp;
			n_pivots.push_back(std::make_pair(r, *cp));
			scalar_inv(scalar, sparse_mat_entry(mat, r, *cp), F);
			sparse_vec_rescale(sparse_mat_row(mat, r), scalar, F);
		}
		pivots.push_back(n_pivots);
		rank += n_pivots.size();

		ulong n_leftrows = 0;
		for (size_t i = 0; i < leftrows.size(); i++) {
			auto row = leftrows[i];
			if (rowpivs[row] != -1 || mat->rows[row].nnz == 0)
				continue;
			leftrows[n_leftrows] = row;
			n_leftrows++;
		}
		leftrows.resize(n_leftrows);

		std::vector<uint8_t> flags(leftrows.size(), 0);

		now_nnz = 0;
		for (auto i : leftrows)
			now_nnz += mat->rows[i].nnz;
		pool.detach_blocks<ulong>(0, leftrows.size(), [&](const ulong s, const ulong e) {
			auto id = BS::this_thread::get_index().value();
			for (ulong i = s; i < e; i++) {
				schur_complete(mat, leftrows[i], n_pivots, 1, F, cachedensedmat + id * mat->ncol, nonzero_c[id]);
				flags[i] = 1;
			}
			}, (leftrows.size() < 20 * nthreads ? 0 : 10 * nthreads));

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
			if (indices.count(it) == 0) {
				result.push_back(colperm[it]);
			}
		}
		colperm = std::move(result);
		std::vector<slong> donelist(rowpivs);

		bool print_once = true; // print at least once
		// we need first clear the transpose matrix
		for (auto i = 0; i < tranmat->nrow; i++)
			tranmat->rows[i].nnz = 0;

		ulong localcount = 0;
		while (localcount < leftrows.size()) {
			for (size_t i = 0; i < leftrows.size(); i++) {
				if (flags[i]) {
					auto row = leftrows[i];
					auto therow = sparse_mat_row(mat, row);
					for (size_t j = 0; j < therow->nnz; j++) {
						auto col = therow->indices[j];
						_sparse_vec_set_entry(sparse_mat_row(tranmat, col), row, (bool*)NULL);
					}
					flags[i] = 0;
					localcount++;
				}
			}

			double pr = kk + (1.0 * ps.size() * localcount) / leftrows.size();
			if (verbose && (print_once || pr - oldpr > printstep)) {
				auto end = sparse_base::clocknow();
				now_nnz = sparse_mat_nnz(mat);
				std::cout << "-- Col: " << std::setw(bitlen_ncol) 
					<< (int)pr << "/" << mat->ncol
					<< "  rank: " << std::setw(bitlen_ncol) << rank
					<< "  nnz: " << std::setw(bitlen_nnz) << now_nnz
					<< "  density: " << std::setprecision(6) << std::setw(8)
					<< 100 * (double)now_nnz / (mat->nrow * mat->ncol) << "%" 
					<< "  speed: " << std::setprecision(2) << std::setw(8) <<
					((pr - oldpr) / sparse_base::usedtime(start, end))
					<< " col/s    \r" << std::flush;
				oldpr = pr;
				start = end;
				print_once = false;
			}
		}
		pool.wait();

		kk += ps.size();
	}

	if (verbose) {
		std::cout << "\n** Rank: " << rank
			<< " nnz: " << sparse_mat_nnz(mat) << std::endl;
	}

	scalar_clear(scalar);
	for (size_t i = 0; i < mat->ncol * nthreads; i++)
		scalar_clear(cachedensedmat + i);

	s_free(cachedensedmat);
	sparse_mat_clear(tranmat);

	return pivots;
}

// TODO: unify sparse_mat_rref_c and sparse_mat_rref_r
template <typename T>
std::vector<std::vector<pivot_t>> sparse_mat_rref_r(sparse_mat_t<T> mat, field_t F,
	BS::thread_pool& pool, rref_option_t opt) {
	// first canonicalize, sort and compress the matrix
	sparse_mat_compress(mat);

	T scalar[1];
	scalar_init(scalar);

	std::vector<slong> rowperm(mat->nrow);
	for (size_t i = 0; i < mat->nrow; i++)
		rowperm[i] = i;

	auto printstep = opt->print_step;
	bool verbose = opt->verbose;

	ulong init_nnz = sparse_mat_nnz(mat);
	ulong now_nnz = init_nnz;

	// store the pivots that have been used
	// -1 is not used
	std::vector<slong> rowpivs(mat->nrow, -1);
	std::vector<slong> colpivs(mat->ncol, -1);

	sparse_mat_t<T*> tranmatp;
	sparse_mat_init(tranmatp, mat->ncol, mat->nrow);
	ulong count =
		eliminate_row_with_one_nnz_rec(mat, tranmatp, rowpivs, verbose);
	now_nnz = sparse_mat_nnz(mat);
	if (verbose) {
		std::cout << "\n** eliminated " << count
			<< " rows, and reduce nnz: " << init_nnz 
			<< " -> " << now_nnz << std::endl;
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

	std::vector<std::vector<pivot_t>> pivots;

	sparse_mat_t<bool> tranmat;
	sparse_mat_init(tranmat, mat->ncol, mat->nrow);

	int nthreads = pool.get_thread_count();
	T* cachedensedmat = s_malloc<T>(mat->ncol * nthreads);
	std::vector<sparse_base::uset> nonzero_c(nthreads);
	for (size_t i = 0; i < mat->ncol * nthreads; i++)
		scalar_init(cachedensedmat + i);
	for (size_t i = 0; i < nthreads; i++)
		nonzero_c[i].resize(mat->ncol);

	// skip the rows with only one/zero nonzero element
	std::vector<pivot_t> n_pivots;
	ulong kk;
	for (kk = 0; kk < mat->nrow; kk++) {
		auto row = rowperm[kk];
		auto therow = sparse_mat_row(mat, row);
		if (therow->nnz == 0)
			continue;
		else if (therow->nnz == 1) {
			auto col = therow->indices[0];
			n_pivots.push_back(std::make_pair(row, col));
			rowpivs[row] = col;
			colpivs[col] = row;
		}
		else
			break;
	}

	auto rank = n_pivots.size();
	pivots.push_back(std::move(n_pivots));
	sparse_mat_transpose_part(tranmat, mat, rowperm);

	// for printing
	double oldstatus = 0;
	int bitlen_nnz = (int)std::floor(std::log(now_nnz) / std::log(10)) + 3;
	int bitlen_nrow = (int)std::floor(std::log(mat->nrow) / std::log(10)) + 1;

	while (kk < mat->nrow) {
		auto start = sparse_base::clocknow();
		auto row = rowperm[kk];

		if (mat->rows[row].nnz == 0) {
			kk++;
			continue;
		}

		pool.wait();
		auto ps = findmanypivots(mat, tranmat, colpivs,
			rowperm, rowperm.begin() + kk, true, opt->search_depth);

		if (ps.size() == 0)
			break;

		n_pivots.clear();
		for (auto& [c, rp] : ps) {
			n_pivots.push_back(std::make_pair(*rp, c));
			colpivs[c] = *rp;
			rowpivs[*rp] = c;
			scalar_inv(scalar, sparse_mat_entry(mat, *rp, c, true), F);
			sparse_vec_rescale(sparse_mat_row(mat, *rp), scalar, F);
		}
		pivots.push_back(n_pivots);
		rank += n_pivots.size();

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
			if (indices.count(it) == 0) {
				result.push_back(rowperm[it]);
			}
		}
		rowperm = std::move(result);

		kk += ps.size();
		slong newpiv = ps.size();

		ulong tran_count = 0;
		// flags[i] is true if the i-th row has been computed
		std::vector<uint8_t> flags(mat->nrow - kk, 0);
		// and then compute the elimination of the rows asynchronizely
		now_nnz = 0;
		for (auto i = rowperm.begin() + kk; i != rowperm.end(); i++)
			now_nnz += mat->rows[*i].nnz;
		pool.detach_blocks<ulong>(kk, mat->nrow, [&](const ulong s, const ulong e) {
			auto id = BS::this_thread::get_index().value();
			for (ulong i = s; i < e; i++) {
				if (rowpivs[rowperm[i]] != -1)
					continue;
				schur_complete(mat, rowperm[i], n_pivots, 1, F, cachedensedmat + id * mat->ncol, nonzero_c[id]);
				flags[i - kk] = 1;
			}
			}, ((mat->nrow - kk) < 20 * nthreads ? 0 : 10 * nthreads));
		std::vector<slong> leftrows(rowperm.begin() + kk, rowperm.end());
		for (size_t i = 0; i < tranmat->nrow; i++)
			tranmat->rows[i].nnz = 0;
		// compute the transpose of the submatrix and print the status asynchronizely
		while (tran_count < leftrows.size()) {
			for (size_t i = 0; i < leftrows.size(); i++) {
				if (flags[i]) {
					auto row = leftrows[i];
					auto therow = sparse_mat_row(mat, row);
					for (size_t j = 0; j < therow->nnz; j++) {
						auto col = therow->indices[j];
						_sparse_vec_set_entry(sparse_mat_row(tranmat, col), row, (bool*)NULL);
					}
					tran_count++;
					flags[i] = 0;
				}
			}
			auto status = (kk - newpiv + 1) + ((double)tran_count / (mat->nrow - kk)) * newpiv;
			if (verbose && status - oldstatus > printstep) {
				auto end = sparse_base::clocknow();
				now_nnz = sparse_mat_nnz(mat);
				std::cout << "-- Row: " << std::setw(bitlen_nrow) 
					<< (int)std::floor(status) << "/" << mat->nrow
					<< "  rank: " << std::setw(bitlen_nrow) << rank
					<< "  nnz: " << std::setw(bitlen_nnz) << now_nnz
					<< "  density: " << std::setprecision(6) << std::setw(8)
					<< 100 * (double)now_nnz / (mat->nrow * mat->ncol) << "%"
					<< "  speed: " << std::setprecision(2) << std::setw(8) <<
					(status - oldstatus) / sparse_base::usedtime(start, end)
					<< " row/s    \r" << std::flush;
				oldstatus = status;
				start = end;
			}
		}
	}

	if (verbose) {
		std::cout << "\n** Rank: " << rank
			<< " nnz: " << sparse_mat_nnz(mat) << std::endl;
	}

	scalar_clear(scalar);
	for (size_t i = 0; i < mat->ncol * nthreads; i++)
		scalar_clear(cachedensedmat + i);

	s_free(cachedensedmat);
	sparse_mat_clear(tranmat);

	return pivots;
}

////TODO
//void sparse_mat_rref_uplift(sparse_mat_t<fmpq> mat) {
//	return;
//}

template <typename T>
std::vector<std::vector<pivot_t>> sparse_mat_rref(sparse_mat_t<T> mat, field_t F,
	BS::thread_pool& pool, rref_option_t opt) {
	std::vector<std::vector<pivot_t>> pivots;
	if (opt->pivot_dir)
		pivots = sparse_mat_rref_r(mat, F, pool, opt);
	else
		pivots = sparse_mat_rref_c(mat, F, pool, opt);

	if (opt->is_back_sub) {
		if (opt->verbose)
			std::cout << "\n>> Reverse solving: " << std::endl;
		triangular_solver(mat, pivots, F, opt, -1, pool);
	}
	return pivots;
}

template <typename T>
ulong sparse_mat_rref_kernel(sparse_mat_t<T> K, const sparse_mat_t<T> M,
	const std::vector<pivot_t>& pivots, field_t F, BS::thread_pool& pool) {
	auto rank = pivots.size();
	if (rank == M->ncol) 
		return 0; // full rank, no kernel

	T m1[1];
	scalar_init(m1);
	scalar_one(m1);

	if (rank == 0) {
		sparse_mat_init(K, M->ncol, M->ncol);
		for (size_t i = 0; i < M->ncol; i++)
			_sparse_vec_set_entry(sparse_mat_row(K, i), i, m1);
		scalar_clear(m1);
		return M->ncol;
	}
	scalar_neg(m1, m1, F);

	sparse_mat_t<T> rows, trows;
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
	scalar_clear(m1);
	return M->ncol - rank;
}

template <typename T>
ulong sparse_mat_rref_kernel(sparse_mat_t<T> K, const sparse_mat_t<T> M,
	const std::vector<std::vector<pivot_t>>& pivots, field_t F, BS::thread_pool& pool) {
	std::vector<pivot_t> n_pivots;
	for (auto& p : pivots)
		n_pivots.insert(n_pivots.end(), p.begin(), p.end());
	return sparse_mat_rref_kernel(K, M, n_pivots, F, pool);
}

// convert
static inline void snmod_mat_from_sfmpq(snmod_mat_t mat, const sfmpq_mat_t src,
	nmod_t p) {
	for (size_t i = 0; i < src->nrow; i++) 
		snmod_vec_from_sfmpq(sparse_mat_row(mat, i), sparse_mat_row(src, i), p);
}

// IO
template <typename T> void sfmpq_mat_read(sfmpq_mat_t mat, T& st) {
	if (!st.is_open())
		return;
	std::string strLine;

	bool is_size = true;
	fmpq_t val;
	fmpq_init(val);

	while (getline(st, strLine)) {
		if (strLine[0] == '%')
			continue;

		auto tokens = sparse_base::SplitString(strLine, " ");
		if (is_size) {
			ulong nrow = std::stoul(tokens[0]);
			ulong ncol = std::stoul(tokens[1]);
			// ulong nnz = std::stoul(tokens[2]);
			// here we alloc 1, or alloc nnz/ncol ?
			sparse_mat_init(mat, nrow, ncol);
			is_size = false;
		}
		else {
			if (tokens.size() != 3) {
				std::cerr << "Error: wrong format in the matrix file" << std::endl;
				std::exit(-1);
			}
			slong row = std::stoll(tokens[0]) - 1;
			slong col = std::stoll(tokens[1]) - 1;
			// SMS stop at 0 0 0
			if (row < 0 || col < 0)
				break;
			sparse_base::DeleteSpaces(tokens[2]);
			fmpq_set_str(val, tokens[2].c_str(), 10);
			_sparse_vec_set_entry(sparse_mat_row(mat, row), col, val);
		}
	}
}

template <typename T, typename S> void sparse_mat_write(sparse_mat_t<T> mat, S& st) {
	if constexpr (std::is_same_v<T, fmpq>) {
		st << "%%MatrixMarket matrix coordinate rational general" << '\n';
	}
	else {
		st << "%%MatrixMarket matrix coordinate integer general" << '\n';
	}
	st << mat->nrow << ' ' << mat->ncol << ' ' << sparse_mat_nnz(mat) << '\n';
	for (size_t i = 0; i < mat->nrow; i++) {
		auto therow = sparse_mat_row(mat, i);
		for (size_t j = 0; j < therow->nnz; j++) {
			if (scalar_is_zero(therow->entries + j))
				continue;
			st << i + 1 << ' '
				<< therow->indices[j] + 1 << ' '
				<< scalar_to_str(therow->entries + j) << '\n';
		}
	}
}

static std::pair<size_t, char*> snmod_mat_to_binary(sparse_mat_t<ulong> mat) {
	auto ratio = sizeof(ulong) / sizeof(char);
	auto nnz = sparse_mat_nnz(mat);
	auto len = (3 + mat->nrow + 2 * nnz) * ratio;
	char* buffer = s_malloc<char>(len);
	char* ptr = buffer;
	std::memcpy(ptr, &(mat->nrow), sizeof(ulong)); ptr += ratio;
	std::memcpy(ptr, &(mat->ncol), sizeof(ulong)); ptr += ratio;
	std::memcpy(ptr, &nnz, sizeof(ulong)); ptr += ratio;
	for (size_t i = 0; i < mat->nrow; i++) {
		auto therow = sparse_mat_row(mat, i);
		std::memcpy(ptr, &(therow->nnz), sizeof(ulong)); ptr += ratio;
		std::memcpy(ptr, therow->indices, therow->nnz * sizeof(ulong)); ptr += therow->nnz * ratio;
		std::memcpy(ptr, therow->entries, therow->nnz * sizeof(ulong)); ptr += therow->nnz * ratio;
	}
	return std::make_pair(len, buffer);
}

static void snmod_mat_from_binary(sparse_mat_t<ulong> mat, char* buffer) {
	auto ratio = sizeof(ulong) / sizeof(char);
	char* ptr = buffer;
	ulong nnz;
	std::memcpy(&(mat->nrow), ptr, sizeof(ulong)); ptr += ratio;
	std::memcpy(&(mat->ncol), ptr, sizeof(ulong)); ptr += ratio;
	std::memcpy(&nnz, ptr, sizeof(ulong)); ptr += ratio;
	sparse_mat_init(mat, mat->nrow, mat->ncol);
	for (size_t i = 0; i < mat->nrow; i++) {
		auto therow = sparse_mat_row(mat, i);
		std::memcpy(&(therow->nnz), ptr, sizeof(ulong)); ptr += ratio;
		std::memcpy(therow->indices, ptr, therow->nnz * sizeof(ulong)); ptr += therow->nnz * ratio;
		std::memcpy(therow->entries, ptr, therow->nnz * sizeof(ulong)); ptr += therow->nnz * ratio;
	}
}

#endif
