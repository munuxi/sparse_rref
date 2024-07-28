#ifndef SFMPQ_MAT_H
#define SFMPQ_MAT_H

#include <fstream>
#include <iostream>
#include <iomanip>
#include "sfmpq_vec.h"
#include "snmod_mat.h"
#include "sparse_mat.h"

slong* sfmpq_mat_rref(sfmpq_mat_t mat, BS::thread_pool& pool, rref_option_t opt);

static inline
void snmod_mat_from_sfmpq(snmod_mat_t mat, const sfmpq_mat_t src, nmod_t p) {
	for (size_t i = 0; i < src->nrow; i++) {
		auto row = src->rows + i;
		snmod_vec_from_sfmpq(mat->rows + i, row, p);
	}
}

// IO
template <typename T>
void sfmpq_mat_read(sfmpq_mat_t mat, T& st) {
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

template <typename T>
void sfmpq_mat_write(sfmpq_mat_t mat, T& st) {
	// sfmpq_mat_compress(mat);
	st << "%%MatrixMarket matrix coordinate rational general" << '\n';
	st << mat->nrow << " " << mat->ncol << " " << sparse_mat_nnz(mat) << '\n';
	for (size_t i = 0; i < mat->nrow; i++){
		auto therow = mat->rows + i;
		for (size_t j = 0; j < therow->nnz; j++){
			if (fmpq_is_zero(therow->entries + j))
				continue;
			auto thenum = fmpq_get_str(NULL, 10, therow->entries + j);
			st << i + 1 << " " << therow->indices[j] + 1 << " " << thenum << '\n';
		}
	}
}

template <typename T>
void sfmpq_mat_dense_write(sfmpq_mat_t mat, T& st) {
	// sfmpq_mat_compress(mat);
	for (size_t i = 0; i < mat->nrow; i++){
		for (size_t j = 0; j < mat->ncol; j++){
			auto entry = sparse_mat_entry(mat, i, j);
			if (entry == NULL)
				st << "0 ";
			else
				st << fmpq_get_str(NULL, 10, entry) << " ";
		}
		st << '\n';
	}
}

#endif