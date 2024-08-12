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

template <typename T>
inline sparse_vec_struct<T>* sparse_mat_row(const sparse_mat_t<T> mat, const slong i) {
	return mat->rows + i;
}

template <typename T>
inline void _sparse_mat_init(sparse_mat_t<T> mat, ulong nrow, ulong ncol,
	ulong alloc) {
	mat->nrow = nrow;
	mat->ncol = ncol;
	mat->rows =
		(sparse_vec_struct<T> *)malloc(nrow * sizeof(sparse_vec_struct<T>));
	for (size_t i = 0; i < nrow; i++)
		_sparse_vec_init(mat->rows + i, ncol, alloc);
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
	free(mat->rows);
	mat->nrow = 0;
	mat->ncol = 0;
	mat->rows = NULL;
}

template <typename T>
inline ulong sparse_mat_nnz(sparse_mat_t<T> mat) {
	ulong nnz = 0;
	for (size_t i = 0; i < mat->nrow; i++)
		nnz += mat->rows[i].nnz;
	return nnz;
}

template <typename T>
inline ulong sparse_mat_alloc(sparse_mat_t<T> mat) {
	ulong alloc = 0;
	for (size_t i = 0; i < mat->nrow; i++)
		alloc += mat->rows[i].alloc;
	return alloc;
}

template <typename T>
inline void sparse_mat_compress(sparse_mat_t<T> mat) {
	for (size_t i = 0; i < mat->nrow; i++)
		sparse_vec_realloc(mat->rows + i, mat->rows[i].nnz);
}

template <typename T>
inline T* sparse_mat_entry(sparse_mat_t<T> mat, slong row, slong col, bool isbinary = true) {
	if (row < 0 || col < 0 || (ulong)row >= mat->nrow ||
		(ulong)col >= mat->ncol)
		return NULL;
	return sparse_vec_entry(mat->rows + row, col, isbinary);
}

template <typename T, typename S>
inline void _sparse_mat_set_entry(sparse_mat_t<T> mat, slong row, slong col, S val) {
	if (row < 0 || col < 0 || (ulong)row >= mat->nrow ||
		(ulong)col >= mat->ncol)
		return;
	_sparse_vec_set_entry(mat->rows + row, col, val);
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
			sparse_vec_clear(mat->rows + i);
		}
	}
	mat->nrow = newnrow;
}

template <typename T>
inline void sparse_mat_transpose_pointer(sparse_mat_t<T*> mat2, const sparse_mat_t<T> mat) {
	for (size_t i = 0; i < mat2->nrow; i++)
		mat2->rows[i].nnz = 0;

	for (size_t i = 0; i < mat->nrow; i++) {
		auto therow = mat->rows + i;
		for (size_t j = 0; j < therow->nnz; j++) {
			// if (scalar_is_zero(therow->entries + j))
			// 	continue;
			auto col = therow->indices[j];
			_sparse_vec_set_entry(mat2->rows + col, i, therow->entries + j);
		}
	}
}

template <typename T>
inline void sparse_mat_transpose(sparse_mat_t<T> mat2, const sparse_mat_t<T> mat) {
	for (size_t i = 0; i < mat2->nrow; i++)
		mat2->rows[i].nnz = 0;

	for (size_t i = 0; i < mat->nrow; i++) {
		auto therow = mat->rows + i;
		for (size_t j = 0; j < therow->nnz; j++) {
			if (scalar_is_zero(therow->entries + j))
				continue;
			auto col = therow->indices[j];
			if constexpr (std::is_same_v<T, fmpq>) {
				_sparse_vec_set_entry(mat2->rows + col, i, therow->entries + j);
			}
			else {
				_sparse_vec_set_entry(mat2->rows + col, i, therow->entries[j]);
			}
		}
	}
}

// tranpose only part of the rows
template <typename T, typename S>
inline void sparse_mat_transpose_part(sparse_mat_t<S> mat2, const sparse_mat_t<T> mat, const std::vector<slong>& rows) {
	for (size_t i = 0; i < mat2->nrow; i++)
		mat2->rows[i].nnz = 0;

	for (size_t i = 0; i < rows.size(); i++) {
		auto row = rows[i];
		auto therow = mat->rows + row;
		for (size_t j = 0; j < therow->nnz; j++) {
			auto col = therow->indices[j];
			_sparse_vec_set_entry(mat2->rows + col, row, therow->entries + j);
		}
	}
}

// tranpose only part of the rows
template <typename T, typename S>
inline void sparse_mat_transpose_part_parallel(sparse_mat_struct<S>* mat_vec, const sparse_mat_t<T> mat, const std::vector<slong>& rows, 
	BS::thread_pool& pool) {

	pool.detach_loop<size_t>(0, pool.get_thread_count(), [&](size_t it) {
		auto mat2 = mat_vec + it;
		for (size_t i = 0; i < mat2->nrow; i++)
			mat2->rows[i].nnz = 0;
		});
	pool.wait();

	pool.detach_loop<size_t>(0, rows.size(), [&](size_t i) {
		auto row = rows[i];
		auto therow = mat->rows + row;
		auto id = BS::this_thread::get_index().value();
		auto mat2 = mat_vec + id;
		for (size_t j = 0; j < therow->nnz; j++) {
			auto col = therow->indices[j];
			_sparse_vec_set_entry(mat2->rows + col, row, therow->entries + j);
		}
		});
	pool.wait();
}

// dot product
// -1: dimension mismatch
// 0: result is zero
// 1: result is non-zero

template <typename T>
inline int sparse_vec_dot_sparse_vec(T* result, const sparse_vec_t<T> v1, const sparse_vec_t<T> v2) {
	if (v1->len != v2->len)
		return -1;

	if (v1->nnz == 0 || v2->nnz == 0) {
		scalar_zero(result);
		return 0;
	}
	slong ptr1 = 0, ptr2 = 0;
	while (ptr1 < v1->nnz && ptr2 < v2->nnz) {
		if (v1->indices[ptr1] == v2->indices[ptr2]) {
			scalar_add(result, result, scalar_mul(v1->entries + ptr1, v2->entries + ptr2));
			ptr1++;
			ptr2++;
		}
		else if (v1->indices[ptr1] < v2->indices[ptr2])
			ptr1++;
		else
			ptr2++;
	}
	if (scalar_is_zero(result))
		return 0;
	return 1;
}

template <typename T>
inline int sparse_mat_dot_sparse_vec(sparse_vec_t<T> result, const sparse_mat_t<T> mat, const sparse_vec_t<T> vec) {
	if (mat->nrow != vec->len)
		return -1;

	sparse_vec_init(result, mat->nrow);
	if (vec->nnz == 0 || sparse_mat_nnz(mat) == 0) {
		sparse_vec_zero(result);
		return 0;
	}
	T tmp[1];
	if constexpr (std::is_same_v<T, fmpq>) {
		fmpq_init(tmp);
	}
	else {
		*tmp = 0;
	}

	for (size_t i = 0; i < mat->nrow; i++) {
		auto therow = mat->rows + i;
		sparse_vec_dot_sparse_vec(result->entries + i, therow, vec);
	}
}

// rref 

template <typename T, typename S>
auto findmanypivots_r(sparse_mat_t<T> mat, const sparse_mat_struct<S>* tranmat_vec,
	slong* colpivs, std::vector<slong>& rowperm,
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


std::vector<std::pair<slong, slong>> sfmpq_mat_rref(sfmpq_mat_t mat, BS::thread_pool& pool, rref_option_t opt);
ulong sfmpq_mat_rref_kernel(sfmpq_mat_t K, const sfmpq_mat_t M, const std::vector<std::pair<slong, slong>>& pivots, BS::thread_pool& pool);
std::vector<std::pair<slong, slong>> snmod_mat_rref(snmod_mat_t mat, nmod_t p, BS::thread_pool& pool, rref_option_t opt);
ulong snmod_mat_rref_kernel(snmod_mat_t K, const snmod_mat_t M, const std::vector<std::pair<slong, slong>>& pivots, nmod_t p, BS::thread_pool& pool);

// convert
inline void snmod_mat_from_sfmpq(snmod_mat_t mat, const sfmpq_mat_t src,
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

	int totalprint = 0;

	while (getline(st, strLine)) {
		if (strLine[0] == '%')
			continue;

		auto tokens = SplitString(strLine, " ");
		if (is_size) {
			ulong nrow = std::stoul(tokens[0]);
			ulong ncol = std::stoul(tokens[1]);
			ulong nnz = std::stoul(tokens[2]);
			// here we alloc 1, or alloc nnz/ncol ?
			sparse_mat_init(mat, nrow, ncol);
			is_size = false;
		}
		else {
			ulong row = std::stoul(tokens[0]) - 1;
			ulong col = std::stoul(tokens[1]) - 1;
			DeleteSpaces(tokens[2]);
			fmpq_set_str(val, tokens[2].c_str(), 10);
			_sparse_vec_set_entry(mat->rows + row, col, val);
		}
	}
}

template <typename T> void sfmpq_mat_write(sfmpq_mat_t mat, T& st) {
	// sfmpq_mat_compress(mat);
	st << "%%MatrixMarket matrix coordinate rational general" << '\n';
	st << mat->nrow << " " << mat->ncol << " " << sparse_mat_nnz(mat) << '\n';
	for (size_t i = 0; i < mat->nrow; i++) {
		auto therow = mat->rows + i;
		for (size_t j = 0; j < therow->nnz; j++) {
			if (fmpq_is_zero(therow->entries + j))
				continue;
			auto thenum = fmpq_get_str(NULL, 10, therow->entries + j);
			st << i + 1 << " " << therow->indices[j] + 1 << " " << thenum
				<< '\n';
		}
	}
}

template <typename T> void sfmpq_mat_dense_write(sfmpq_mat_t mat, T& st) {
	// sfmpq_mat_compress(mat);
	for (size_t i = 0; i < mat->nrow; i++) {
		for (size_t j = 0; j < mat->ncol; j++) {
			auto entry = sparse_mat_entry(mat, i, j);
			if (entry == NULL)
				st << "0 ";
			else
				st << fmpq_get_str(NULL, 10, entry) << " ";
		}
		st << '\n';
	}
}

template <typename T> void snmod_mat_read(snmod_mat_t mat, nmod_t p, T& st) {
	if (!st.is_open())
		return;
	std::string strLine;

	bool is_size = true;
	ulong val;

	fmpq_t val;
	fmpq_init(val);

	int totalprint = 0;

	while (getline(st, strLine)) {
		if (strLine[0] == '%')
			continue;

		auto tokens = SplitString(strLine, " ");
		if (is_size) {
			ulong nrow = std::stoul(tokens[0]);
			ulong ncol = std::stoul(tokens[1]);
			ulong nnz = std::stoul(tokens[2]);
			// here we alloc 1, or alloc nnz/ncol ?
			sparse_mat_init(mat, nrow, ncol);
			is_size = false;
		}
		else {
			ulong row = std::stoul(tokens[0]) - 1;
			ulong col = std::stoul(tokens[1]) - 1;
			DeleteSpaces(tokens[2]);
			fmpq_set_str(val, tokens[2].c_str(), 10);
			fmpz_mod_ui(fmpq_numref(val), fmpq_numref(val), p.n);
			fmpz_mod_ui(fmpq_denref(val), fmpq_denref(val), p.n);
			ulong num = fmpz_get_ui(fmpq_numref(val));
			ulong den = fmpz_get_ui(fmpq_denref(val));
			ulong val = nmod_div(num, den, p);
			_sparse_vec_set_entry(mat->rows + row, col, val);
		}
	}

	fmpq_clear(val);
}

template <typename T> void snmod_mat_write(snmod_mat_t mat, T& st) {
	// snmod_mat_compress(mat);
	st << "%%MatrixMarket matrix coordinate integer general" << '\n';
	st << mat->nrow << " " << mat->ncol << " " << sparse_mat_nnz(mat) << '\n';
	for (size_t i = 0; i < mat->nrow; i++) {
		auto therow = mat->rows + i;
		for (size_t j = 0; j < therow->nnz; j++) {
			if (scalar_is_zero(therow->entries + j))
				continue;
			ulong thenum = therow->entries[j];
			st << i + 1 << " " << therow->indices[j] + 1 << " " << thenum
				<< '\n';
		}
	}
}

// never use it somehow
template <typename T> void snmod_mat_dense_write(snmod_mat_t mat, T& st) {
	// snmod_mat_compress(mat);
	slong shiftedentry = 0;
	for (size_t i = 0; i < mat->nrow; i++) {
		for (size_t j = 0; j < mat->ncol; j++) {
			auto entry = sparse_mat_entry(mat, i, j);
			if (entry == NULL)
				st << "0 ";
			else
				st << *entry << " ";
		}
		st << '\n';
	}
}

#endif
