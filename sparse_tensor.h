#ifndef SPARSE_TENSOR_H
#define SPARSE_TENSOR_H

#include "sparse_vec.h"

// CRS format for sparse tensor
template <typename T> struct sparse_tensor_struct {
	uint8_t rank;
	ulong nnz;
	std::vector<ulong> dims;
	std::vector<ulong> rowptrs;
	sparse_vec_t<T> vec;
};

template <typename T> struct sparse_tensor_t {
	struct sparse_tensor_struct<T> data[1];
	void sparse_tensor_t<T>(uint8_t rank, const std::vector<ulong> dims) {
		data->rank = rank;
		data->nnz = 0;
		data->dims = dims;
		ulong max_len = 1;
		sparse_vec_init(data->vec, ULLONG_MAX);
	}
	void clear() {
		data->nnz = 0;
		data->rank = 0;
		std::vector<ulong>().swap(dims);
		std::vector<ulong>().swap(rowptrs);
		sparse_vec_clear(data->vec);
	}
};

#endif