#ifndef SPARSE_TENSOR_H
#define SPARSE_TENSOR_H

#include "sparse_vec.h"

// CRS format for sparse tensor
template <typename T> struct sparse_tensor_struct {
	ulong rank;
	ulong* dims;
	// TODO
};

template <typename T> using sparse_tensor_t = struct sparse_tensor_struct<T>[1];


#endif