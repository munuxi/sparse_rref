#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include "sparse_vec.h"

template<typename T>
struct sparse_mat_struct {
	ulong nrow;
	ulong ncol;
	sparse_vec_struct<T>* rows;
};

template<typename T>
using sparse_mat_t = struct sparse_mat_struct<T>[1];

typedef sparse_mat_t<ulong> snmod_mat_t;
typedef sparse_mat_t<fmpq> sfmpq_mat_t;

template<typename T>
static inline void _sparse_mat_init(sparse_mat_t<T> mat, ulong nrow, ulong ncol, ulong alloc) {
	mat->nrow = nrow;
	mat->ncol = ncol;
	mat->rows = (sparse_vec_struct<T>*)malloc(nrow * sizeof(sparse_vec_struct<T>));
	for (size_t i = 0; i < nrow; i++)
		_sparse_vec_init(mat->rows + i, ncol, alloc);
}

template<typename T>
static inline void sparse_mat_init(sparse_mat_t<T> mat, ulong nrow, ulong ncol) {
	_sparse_mat_init(mat, nrow, ncol, 1ULL);
}

template<typename T>
static inline void sparse_mat_clear(sparse_mat_t<T> mat) {
	for (size_t i = 0; i < mat->nrow; i++)
		sparse_vec_clear(mat->rows + i);
	free(mat->rows);
	mat->nrow = 0;
	mat->ncol = 0;
	mat->rows = NULL;
}

template<typename T>
static inline 
ulong sparse_mat_nnz(sparse_mat_t<T> mat) {
	ulong nnz = 0;
	for (size_t i = 0; i < mat->nrow; i++)
		nnz += mat->rows[i].nnz;
	return nnz;
}

template<typename T>
static inline 
ulong sparse_mat_alloc(sparse_mat_t<T> mat) {
	ulong alloc = 0;
	for (size_t i = 0; i < mat->nrow; i++)
		alloc += mat->rows[i].alloc;
	return alloc;
}

template <typename T>
static inline
void sparse_mat_compress(sparse_mat_t<T> mat) {
	for (size_t i = 0; i < mat->nrow; i++)
		sparse_vec_realloc(mat->rows + i, mat->rows[i].nnz);
}

template<typename T>
static inline 
T* sparse_mat_entry(sparse_mat_t<T> mat, slong row, slong col, bool isbinary = false) {
	if (row < 0 || col < 0 || (ulong)row >= mat->nrow || (ulong)col >= mat->ncol)
		return NULL;
	return sparse_vec_entry(mat->rows + row, col, isbinary);
}

template<typename T, typename S>
static inline
void _sparse_mat_set_entry(sparse_mat_t<T> mat, slong row, slong col, S val) {
	if (row < 0 || col < 0 || (ulong)row >= mat->nrow || (ulong)col >= mat->ncol)
		return;
	_sparse_vec_set_entry(mat->rows + row, col, val);
}

template<typename T>
static inline
void sparse_mat_clear_zero_row(sparse_mat_t<T> mat) {
	ulong newnrow = 0;
	for (size_t i = 0; i < mat->nrow; i++){
		if (mat->rows[i].nnz != 0){
			mat->rows[newnrow] = mat->rows[i];
			newnrow++;
		}
		else {
			sparse_vec_clear(mat->rows + i);
		}
	}
	mat->nrow = newnrow;
}

template<typename T>
static inline
void sparse_mat_transpose_pointer(sparse_mat_t<T*> mat2, sparse_mat_t<T> mat) {
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

template<typename T>
static inline
void sparse_mat_transpose(sparse_mat_t<T> mat2, sparse_mat_t<T> mat) {
	for (size_t i = 0; i < mat2->nrow; i++)
		mat2->rows[i].nnz = 0;

	for (size_t i = 0; i < mat->nrow; i++) {
		auto therow = mat->rows + i;
		for (size_t j = 0; j < therow->nnz; j++) {
			if (scalar_is_zero(therow->entries + j))
				continue;
			auto col = therow->indices[j];
			_sparse_vec_set_entry(mat2->rows + col, i, therow->entries + j);
		}
	}
}

#endif
