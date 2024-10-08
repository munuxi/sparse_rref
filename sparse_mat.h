#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include "sparse_vec.h"

constexpr double SPARSE_BOUND = 0.1;

template <typename T> struct sparse_mat_struct {
	ulong nrow;
	ulong ncol;
	sparse_vec_struct<T>* rows;
};

template <typename T> using sparse_mat_t = struct sparse_mat_struct<T>[1];

typedef sparse_mat_t<ulong> snmod_mat_t;
typedef sparse_mat_t<fmpq> sfmpq_mat_t;

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

		auto therow = sparse_mat_row(mat, *row);
		if (therow->nnz == 0)
			continue;
		auto indices = therow->indices;

		slong col;
		ulong mnnz = ULLONG_MAX;
		bool flag = true;

		for (size_t i = 0; i < therow->nnz; i++) {
			flag = (pcols.count(indices[i]) == 0);
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
			auto tc = sparse_mat_row(tranmat_vec + it, col);
			for (size_t j = 0; j < tc->nnz; j++) {
				if (rowptrs[tc->indices[j]] == end)
					continue;
				flag = (pcols.count(tc->indices[j]) == 0);
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
		auto thecol = sparse_mat_row(tranmat, *col);
		auto indices = thecol->indices;
		for (size_t i = 0; i < thecol->nnz; i++) {
			flag = (prows.count(indices[i]) == 0);
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
			flag = (prows.count(tc->indices[j]) == 0);
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
		auto start = clocknow();
		if (thecol.size() > 1) {
			pool.detach_loop<slong>(0, thecol.size(), [&](slong j) {
				auto r = thecol[j];
				if (r == pp.first)
					return;
				auto entry = sparse_mat_entry(mat, r, pp.second, true);
				sparse_vec_sub_mul(sparse_mat_row(mat, r), sparse_mat_row(mat, pp.first), entry, F);
				});
		}
		pool.wait();
		
		if (verbose && (i % printstep == 0 || i == pivots.size() - 1) && thecol.size() > 1) {
			count++;
			auto end = clocknow();
			auto now_nnz = sparse_mat_nnz(mat);
			std::cout << "\r-- Row: " << (i + 1) << "/" << pivots.size()
				<< "  " << "row to eliminate: " << thecol.size() - 1
				<< "  " << "nnz: " << now_nnz << "  " << "density: "
				<< (double)100 * now_nnz / (mat->nrow * mat->ncol)
				<< "%  " << "speed: " << count / usedtime(start, end)
				<< " row/s" << std::flush;
			start = clocknow();
			count = 0;
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
			auto row = sparse_mat_row(mat, r);
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

template <typename T>
std::vector<std::pair<slong, slong>> sparse_mat_rref_c(sparse_mat_t<T> mat, field_t F, 
	BS::thread_pool& pool, rref_option_t opt) {
	// first canonicalize, sort and compress the matrix
	sparse_mat_compress(mat);

	T scalar[1];
	scalar_init(scalar);

	ulong init_nnz = sparse_mat_nnz(mat);
	ulong now_nnz = init_nnz;

	// store the pivots that have been used
	// -1 is not used
	std::vector<slong> rowpivs(mat->nrow, -1);
	std::vector<std::pair<slong, slong>> pivots;
	// perm the col
	std::vector<slong> colperm(mat->ncol);
	for (size_t i = 0; i < mat->ncol; i++)
		colperm[i] = i;

	// look for row with only one non-zero entry

	// compute the transpose of pointers of the matrix
	sparse_mat_t<T*> tranmatp;
	sparse_mat_init(tranmatp, mat->ncol, mat->nrow);

	bool verbose = opt->verbose;
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
	for (; kk < mat->ncol; kk++) {
		auto nnz = tranmatp->rows[colperm[kk]].nnz;
		if (nnz == 0)
			continue;
		if (nnz == 1) {
			auto row = tranmatp->rows[colperm[kk]].indices[0];
			if (rowpivs[row] != -1)
				continue;
			rowpivs[row] = colperm[kk];
			auto e = sparse_mat_entry(mat, row, rowpivs[row], true);
			scalar_inv(scalar, e, F);
			sparse_vec_rescale(sparse_mat_row(mat, row), scalar, F);
			pivots.push_back(std::make_pair(row, colperm[kk]));
		}
		else if (nnz > 1)
			break; // since it's sorted
	}
	sparse_mat_clear(tranmatp);

	int nthreads = pool.get_thread_count();
	T* cachedensedmat = s_malloc<T>(mat->ncol * nthreads);
	for (size_t i = 0; i < mat->ncol * nthreads; i++) 
		scalar_init(cachedensedmat + i);

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

	double oldpr = 0;
	// upper triangle (with respect to row and col perm)
	while (kk < mat->ncol) {
		auto start = clocknow();

		auto ps = findmanypivots_c(mat, tranmat, rowpivs, colperm,
			colperm.begin() + kk, opt->search_depth);
		if (ps.size() == 0)
			break;

		std::vector<std::pair<slong, slong>> n_pivots;
		for (auto i = ps.rbegin(); i != ps.rend(); i++) {
			auto [r, cp] = *i;
			rowpivs[r] = *cp;
			n_pivots.push_back(std::make_pair(r, *cp));
			pivots.push_back(std::make_pair(r, *cp));
			scalar_inv(scalar, sparse_mat_entry(mat, r, *cp), F);
			sparse_vec_rescale(sparse_mat_row(mat, r), scalar, F);
		}

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
		bool mode = ((double)100 * now_nnz / (leftrows.size() * mat->ncol) < SPARSE_BOUND);
		pool.detach_blocks<ulong>(0, leftrows.size(), [&](const ulong s, const ulong e) {
			auto id = BS::this_thread::get_index().value();
			for (ulong i = s; i < e; i++) {
				schur_complete(mat, leftrows[i], n_pivots, 1, F, cachedensedmat + id * mat->ncol, mode);
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
			if (verbose && (print_once || pr - oldpr > opt->print_step)) {
				auto end = clocknow();
				now_nnz = sparse_mat_nnz(mat);
				std::cout << "\r-- Col: " << (int)pr << "/"
					<< mat->ncol
					<< "  rank: " << pivots.size() << "  " << "nnz: " << now_nnz
					<< "  " << "density: "
					<< 100 * (double)now_nnz / (mat->nrow * mat->ncol)
					<< "%  " << "speed: " <<
					((pr - oldpr) / usedtime(start, end))
					<< " col/s" << std::flush;
				oldpr = pr;
				start = end;
				print_once = false;
			}
		}
		pool.wait();

		kk += ps.size();
	}

	if (verbose) {
		std::cout << "\n** Rank: " << pivots.size() << " nnz: " << sparse_mat_nnz(mat)
			<< "  " << std::endl;
		std::cout << "\n>> Reverse solving: " << std::endl;
	}

	// the matrix is upper triangular
	triangular_solver(mat, pivots, F, opt, -1, pool);

	if (verbose) {
		std::cout << std::endl;
	}

	scalar_clear(scalar);
	sparse_mat_clear(tranmat);

	for (size_t i = 0; i < mat->ncol * nthreads; i++) {
		scalar_clear(cachedensedmat + i);
	}
	s_free(cachedensedmat);

	return pivots;
}

template <typename T>
std::vector<std::pair<slong, slong>> sparse_mat_rref_r(sparse_mat_t<T> mat, field_t F,
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

	int nthreads = pool.get_thread_count();
	T* cachedensedmat = s_malloc<T>(mat->ncol * nthreads);
	for (size_t i = 0; i < mat->ncol * nthreads; i++)
		scalar_init(cachedensedmat + i);

	// skip the rows with only one/zero nonzero element
	ulong kk;
	for (kk = 0; kk < mat->nrow; kk++) {
		auto row = rowperm[kk];
		auto therow = sparse_mat_row(mat, row);
		if (therow->nnz == 0)
			continue;
		else if (therow->nnz == 1) {
			auto col = therow->indices[0];
			pivots.push_back(std::make_pair(row, col));
			rowpivs[row] = col;
			colpivs[col] = row;
		}
		else
			break;
	}

	sparse_mat_transpose_part(tranmat, mat, rowperm);

	double oldstatus = 0;
	while (kk < mat->nrow) {
		auto start = clocknow();
		auto row = rowperm[kk];

		if (mat->rows[row].nnz == 0) {
			kk++;
			continue;
		}

		pool.wait();
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
			scalar_inv(scalar, sparse_mat_entry(mat, *rp, c, true), F);
			sparse_vec_rescale(sparse_mat_row(mat, *rp), scalar, F);
		}

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
		bool mode = ((double)100 * now_nnz / ((mat->nrow - kk) * mat->ncol) < SPARSE_BOUND);
		pool.detach_blocks<ulong>(kk, mat->nrow, [&](const ulong s, const ulong e) {
			auto id = BS::this_thread::get_index().value();
			for (ulong i = s; i < e; i++) {
				if (rowpivs[rowperm[i]] != -1)
					continue;
				schur_complete(mat, rowperm[i], n_pivots, 1, F, cachedensedmat + id * mat->ncol, mode);
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
				auto end = clocknow();
				now_nnz = sparse_mat_nnz(mat);
				std::cout << "\r-- Row: " << (int)std::floor(status) << "/" << mat->nrow
					<< "  rank: " << pivots.size()
					<< "  nnz: " << now_nnz << "  " << "density: "
					<< (double)100 * now_nnz / (mat->nrow * mat->ncol) << "%"
					<< "  speed: " << (status - oldstatus) / usedtime(start, end)
					<< " row/s" << std::flush;
				oldstatus = status;
				start = end;
			}
		}
	}

	if (verbose) {
		std::cout << "\n** Rank: " << pivots.size()
			<< " nnz: " << sparse_mat_nnz(mat) << std::endl
			<< "\n>> Reverse solving: " << std::endl;
	}

	// the matrix is upper triangular
	triangular_solver(mat, pivots, F, opt, -1, pool);

	if (verbose) {
		std::cout << std::endl;
	}

	scalar_clear(scalar);
	for (size_t i = 0; i < mat->ncol * nthreads; i++)
		scalar_clear(cachedensedmat + i);

	s_free(cachedensedmat);
	sparse_mat_clear(tranmat);

	return pivots;
}

template <typename T>
std::vector<std::pair<slong, slong>> sparse_mat_rref(sparse_mat_t<T> mat, field_t F,
	BS::thread_pool& pool, rref_option_t opt) {
	if (opt->pivot_dir)
		return sparse_mat_rref_r(mat, F, pool, opt);
	else
		return sparse_mat_rref_c(mat, F, pool, opt);
}

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
