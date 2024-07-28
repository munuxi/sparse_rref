#ifndef SPARSE_TENSOR_H
#define SPARSE_TENSOR_H

#include "sparse_vec.h"

template <typename T> struct sparse_tensor_struct {
    uint8_t rank;
    ulong *dims;
    slong **indices;
    T *entries;
};

template <typename T> using sparse_tensor_t = struct sparse_tensor_struct<T>[1];

#endif