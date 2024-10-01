#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include "sparse_vec.h"
#include <set>

constexpr double SPARSE_BOUND = 0.1;

template <typename T> struct sparse_mat_struct {
	ulong nrow;
	ulong ncol;
	sparse_vec_struct<T>* rows;
};

template <typename T> using sparse_mat_t = struct sparse_mat_struct<T>[1];

typedef sparse_mat_t<ulong> snmod_mat_t;
typedef sparse_mat_t<fmpq> sfmpq_mat_t;

template <typename T>
inline sparse_vec_struct<T>* sparse_mat_row(const sparse_mat_t<T> mat, const slong i) {
	return mat->rows + i;
}

template <typename T>
inline void _sparse_mat_init(sparse_mat_t<T> mat, ulong nrow, ulong ncol,
	ulong alloc) {
	mat->nrow = nrow;
	mat->ncol = ncol;
	mat->rows = s_malloc<sparse_vec_struct<T>>(nrow);
	for (size_t i = 0; i < nrow; i++)
		sparse_vec_init(mat->rows + i, alloc);
}

template <typename T>
inline void sparse_mat_init(sparse_mat_t<T> mat, ulong nrow,
	ulong ncol) {
	_sparse_mat_init(mat, nrow, ncol, 1ULL);
}

template <typename T>
inline void sparse_mat_clear(sparse_mat_t<T> mat) {
	for (size_t i = 0; i < mat->nrow; i++)
		sparse_vec_clear(mat->rows + i);
	s_free(mat->rows);
	mat->nrow = 0;
	mat->ncol = 0;
	mat->rows = NULL;
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
	for (size_t i = 0; i < mat->nrow; i++)
		sparse_vec_realloc(sparse_mat_row(mat, i), sparse_mat_row(mat, i)->nnz);
}

template <typename T>
inline T* sparse_mat_entry(sparse_mat_t<T> mat, slong row, slong col, bool isbinary = true) {
	return sparse_vec_entry(sparse_mat_row(mat, row), col, isbinary);
}

template <typename T>
inline void sparse_mat_clear_zero_row(sparse_mat_t<T> mat) {
	ulong newnrow = 0;
	for (size_t i = 0; i < mat->nrow; i++) {
		if (mat->rows[i].nnz != 0) {
			mat->rows[newnrow] = mat->rows[i];
			newnrow++;
		}
		else {
			sparse_vec_clear(sparse_mat_row(mat, i));
		}
	}
	mat->nrow = newnrow;
}

template <typename T>
inline void sparse_mat_transpose(sparse_mat_t<T> mat2, const sparse_mat_t<T> mat) {
	for (size_t i = 0; i < mat2->nrow; i++)
		sparse_mat_row(mat2, i)->nnz = 0;

	for (size_t i = 0; i < mat->nrow; i++) {
		auto therow = sparse_mat_row(mat, i);
		for (size_t j = 0; j < therow->nnz; j++) {
			auto col = therow->indices[j];
			_sparse_vec_set_entry(mat2->rows + col, i, therow->entries + j);
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
			_sparse_vec_set_entry(mat2->rows + col, i, &entry);
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
			_sparse_vec_set_entry(mat2->rows + col, i, (bool*)NULL);
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
		auto therow = mat->rows + row;
		for (size_t j = 0; j < therow->nnz; j++) {
			auto col = therow->indices[j];
			_sparse_vec_set_entry(mat2->rows + col, row, therow->entries + j);
		}
	}
}

template <typename T>
inline void sparse_mat_transpose_part(sparse_mat_t<bool> mat2, const sparse_mat_t<T> mat, const std::vector<slong>& rows) {
	for (size_t i = 0; i < mat2->nrow; i++)
		sparse_mat_row(mat2, i)->nnz = 0;

	for (size_t i = 0; i < rows.size(); i++) {
		auto row = rows[i];
		auto therow = mat->rows + row;
		for (size_t j = 0; j < therow->nnz; j++) {
			auto col = therow->indices[j];
			_sparse_vec_set_entry(mat2->rows + col, row, (bool*)NULL);
		}
	}
}

template <typename T>
inline void sparse_mat_transpose_part(sparse_mat_t<T*> mat2, const sparse_mat_t<T> mat, const std::vector<slong>& rows) {
	for (size_t i = 0; i < mat2->nrow; i++)
		sparse_mat_row(mat2, i)->nnz = 0;

	for (size_t i = 0; i < rows.size(); i++) {
		auto row = rows[i];
		auto therow = mat->rows + row;
		for (size_t j = 0; j < therow->nnz; j++) {
			auto col = therow->indices[j];
			auto ptr = therow->entries + j;
			_sparse_vec_set_entry(mat2->rows + col, row, &ptr);
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
		auto therow = mat->rows + i;
		scalar_zero(tmp);
		if (!sparse_vec_dot(tmp, therow, vec, F))
			_sparse_vec_set_entry(result, i, tmp);
	}
	scalar_clear(tmp);
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
		sparse_mat_dot_sparse_vec(A->rows + i, B, Ct->rows + i, F);

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
		auto thecol = tranmat->rows + pivlist[i];
		for (size_t j = 0; j < thecol->nnz; j++) {
			if (thecol->indices[j] == i) {
				scalar_one(thecol->entries[j]);
			}
			else
				scalar_zero(thecol->entries[j]);
		}
	}

	for (size_t i = 0; i < mat->nrow; i++)
		sparse_vec_canonicalize(mat->rows + i);

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
auto findmanypivots_r(sparse_mat_t<T> mat, const sparse_mat_struct<S>* tranmat_vec,
	std::vector<slong>& colpivs, std::vector<slong>& rowperm,
	std::vector<slong>::iterator start,
	size_t max_depth = ULLONG_MAX, int vec_len = 1) {

	auto end = rowperm.end();
	using iter = std::vector<slong>::iterator;

	std::list<std::pair<slong, iter>> pivots;
	std::unordered_set<slong> pcols;
	pcols.reserve(std::min((size_t)4096, max_depth));

	// rightlook first
	for (auto row = start; row != end; row++) {
		if (pivots.size() > max_depth)
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
			ulong newnnz = 0;
			for (size_t j = 0; j < vec_len; j++)
				newnnz += tranmat_vec[j].rows[indices[i]].nnz;
			if (newnnz < mnnz) {
				col = indices[i];
				mnnz = newnnz;
			}
			// make the result stable
			else if (newnnz == mnnz && indices[i] < col) {
				col = indices[i];
			}
		}
		if (!flag)
			continue;
		if (mnnz != ULLONG_MAX) {
			pivots.push_back(std::make_pair(col, row));
			pcols.insert(col);
		}
	}
	// leftlook then
	// now pcols will be used as prows to store the rows that have been used
	pcols.clear();
	// make a table to help to look for row pointers
	std::vector<iter> rowptrs(mat->nrow, end);
	for (auto it = start; it != end; it++)
		rowptrs[*it] = it;

	for (auto p : pivots) {
		pcols.insert(*(p.second));
	}

	for (size_t i = 0; i < mat->ncol; i++) {
		if (pivots.size() > max_depth)
			break;
		auto col = mat->ncol - i - 1; // reverse ordering
		if (colpivs[col] != -1)
			continue;
		bool flag = true;
		slong row = 0;
		ulong mnnz = ULLONG_MAX;
		for (auto it = 0; it < vec_len; it++) {
			auto tc = tranmat_vec[it].rows + col;
			for (size_t j = 0; j < tc->nnz; j++) {
				if (rowptrs[tc->indices[j]] == end)
					continue;
				flag = (pcols.find(tc->indices[j]) == pcols.end());
				if (!flag)
					break;
				if (mat->rows[tc->indices[j]].nnz < mnnz) {
					mnnz = mat->rows[tc->indices[j]].nnz;
					row = tc->indices[j];
				}
				// make the result stable
				else if (mat->rows[tc->indices[j]].nnz == mnnz && tc->indices[j] < row) {
					row = tc->indices[j];
				}
			}
			if (!flag)
				break;
		}
		if (!flag)
			continue;
		if (mnnz != ULLONG_MAX) {
			pivots.push_front(std::make_pair(col, rowptrs[row]));
			pcols.insert(row);
		}
	}

	std::vector<std::pair<slong, iter>> result(pivots.begin(), pivots.end());

	return result;
}

template <typename T, typename S>
auto findmanypivots_c(sparse_mat_t<T> mat, sparse_mat_t<S> tranmat,
	std::vector<slong>& rowpivs, std::vector<slong>& colperm,
	std::vector<slong>::iterator start,
	size_t max_depth = ULLONG_MAX) {

	using iter = std::vector<slong>::iterator;
	auto end = colperm.end();

	std::list<std::pair<slong, iter>> pivots;
	std::unordered_set<slong> prows;
	prows.reserve(std::min((size_t)4096, max_depth));
	for (auto col = start; col < colperm.end(); col++) {
		if ((ulong)(col - start) > max_depth)
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
			// make the result stable
			else if (mat->rows[indices[i]].nnz == mnnz && indices[i] < row) {
				row = indices[i];
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
		auto tc = sparse_mat_row(mat, row);
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
			// make the result stable
			else if (tranmat->rows[tc->indices[j]].nnz == mnnz && tc->indices[j] < col) {
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

// upper solver : ordering = -1
// lower solver : ordering = 1
template <typename T>
void triangular_solver(sparse_mat_t<T> mat, std::vector<std::pair<slong, slong>>& pivots,
	field_t F, rref_option_t opt, int ordering, BS::thread_pool& pool) {
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
		size_t index = i;
		if (ordering < 0)
			index = pivots.size() - 1 - i;
		auto pp = pivots[index];
		auto thecol = tranmat[pp.second];
		auto start = clocknow();
		auto loop = [&](slong j) {
			auto r = thecol[j];
			if (r == pp.first)
				return;
			auto entry = sparse_mat_entry(mat, r, pp.second, true);
			sparse_vec_sub_mul(mat->rows + r, mat->rows + pp.first, entry, F);
			};
		if (thecol.size() > 1) {
			pool.detach_loop<slong>(0, thecol.size(), loop);
			pool.wait();
		}
		auto end = clocknow();

		if ((i % printstep == 0 || i == pivots.size() - 1) && verbose && thecol.size() > 1) {
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

// first write a stupid one
// TODO: Gilbert-Peierls algorithm for parallel computation 
// see https://hal.science/hal-01333670/document
// mode : true: very sparse < SPARSE_BOUND%
template <typename T>
void schur_complete(sparse_mat_t<T> mat, slong row, std::vector<std::pair<slong, slong>>& pivots,
	int ordering, field_t F, T* tmpvec, bool mode) {
	if (ordering < 0) {
		std::vector<std::pair<slong, slong>> npivots(pivots.rbegin(), pivots.rend());
		schur_complete(mat, row, npivots, -ordering, F, tmpvec, mode);
	}

	auto therow = sparse_mat_row(mat, row);

	if (therow->nnz == 0)
		return;

	// if pivots size is small, we can use the sparse vector
	// to save to cost of converting between sparse and dense
	// vectors, otherwise we use dense vector
	if (pivots.size() < 100) {
		for (auto [r, c] : pivots) {
			auto entry = sparse_vec_entry(therow, c);
			if (entry == NULL)
				continue;
			auto row = mat->rows + r;
			sparse_vec_sub_mul(therow, row, entry, F);
		}
		return;
	}

	auto rrefonerow = [&](auto& nonzero_c) {
		for (size_t i = 0; i < therow->nnz; i++) {
			nonzero_c.insert(therow->indices[i]);
			scalar_set(tmpvec + therow->indices[i], therow->entries + i);
		}
		T entry[1];
		ulong e_pr;
		scalar_init(entry);
		for (auto [r, c] : pivots) {
			if (nonzero_c.find(c) == nonzero_c.end())
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
				auto old_len = nonzero_c.size();
				nonzero_c.insert(row->indices[i]);
				if (nonzero_c.size() != old_len)
					scalar_zero(tmpvec + row->indices[i]);
				if constexpr (std::is_same_v<T, ulong>) {
					tmpvec[row->indices[i]] = _nmod_sub(tmpvec[row->indices[i]],
						n_mulmod_shoup(*entry, row->entries[i], e_pr, F->pvec[0].n), F->pvec[0]);
				}
				else if constexpr (std::is_same_v<T, fmpq>) {
					fmpq_submul(tmpvec + row->indices[i], entry, row->entries + i);
				}
			}
			nonzero_c.erase(c);
		}
		scalar_clear(entry);
		};

	if (mode) {
		std::set<slong> nonzero_c;
		rrefonerow(nonzero_c);
		therow->nnz = 0;
		for (auto i : nonzero_c) {
			if (!scalar_is_zero(tmpvec + i))
				_sparse_vec_set_entry(therow, i, tmpvec + i);
		}
	}
	else {
		std::unordered_set<slong> nonzero_c;
		nonzero_c.reserve(5 * therow->nnz);
		rrefonerow(nonzero_c);
		std::vector<slong> indices(nonzero_c.begin(), nonzero_c.end());
		std::sort(indices.begin(), indices.end());
		therow->nnz = 0;
		for (auto i : indices) {
			if (!scalar_is_zero(tmpvec + i))
				_sparse_vec_set_entry(therow, i, tmpvec + i);
		}
	}
}

std::vector<std::pair<slong, slong>> sparse_mat_rref(sfmpq_mat_t mat, field_t F, BS::thread_pool& pool, rref_option_t opt);
std::vector<std::pair<slong, slong>> sparse_mat_rref(snmod_mat_t mat, field_t F, BS::thread_pool& pool, rref_option_t opt);

template <typename T>
ulong sparse_mat_rref_kernel(sparse_mat_t<T> K, const sparse_mat_t<T> M,
	const std::vector<std::pair<slong, slong>>& pivots, field_t F, BS::thread_pool& pool) {
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

// convert
static inline void snmod_mat_from_sfmpq(snmod_mat_t mat, const sfmpq_mat_t src,
	nmod_t p) {
	for (size_t i = 0; i < src->nrow; i++) {
		auto row = src->rows + i;
		snmod_vec_from_sfmpq(mat->rows + i, row, p);
	}
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

		auto tokens = SplitString(strLine, " ");
		if (is_size) {
			ulong nrow = std::stoul(tokens[0]);
			ulong ncol = std::stoul(tokens[1]);
			// ulong nnz = std::stoul(tokens[2]);
			// here we alloc 1, or alloc nnz/ncol ?
			sparse_mat_init(mat, nrow, ncol);
			is_size = false;
		}
		else {
			slong row = std::stoll(tokens[0]) - 1;
			slong col = std::stoll(tokens[1]) - 1;
			// SMS stop at 0 0 0
			if (row < 0 || col < 0)
				break;
			DeleteSpaces(tokens[2]);
			fmpq_set_str(val, tokens[2].c_str(), 10);
			_sparse_vec_set_entry(mat->rows + row, col, val);
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
	st << mat->nrow << " " << mat->ncol << " " << sparse_mat_nnz(mat) << '\n';
	for (size_t i = 0; i < mat->nrow; i++) {
		auto therow = mat->rows + i;
		for (size_t j = 0; j < therow->nnz; j++) {
			if (scalar_is_zero(therow->entries + j))
				continue;
			st << i + 1 << " "
				<< therow->indices[j] + 1 << " "
				<< scalar_to_str(therow->entries + j) << '\n';
		}
	}
}

std::pair<size_t, char*> snmod_mat_to_binary(sparse_mat_t<ulong> mat);
void snmod_mat_from_binary(sparse_mat_t<ulong> mat, char* buffer);

#endif
