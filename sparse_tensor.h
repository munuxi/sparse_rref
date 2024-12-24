#ifndef SPARSE_TENSOR_H
#define SPARSE_TENSOR_H

#include "sparse_vec.h"
#include <array>

// TODO: sparse tensor

// CRS format for sparse tensor
template <typename T, uint8_t rank> struct sparse_tensor_t {
	ulong dims[rank];
	ulong* rowptr;
	sparse_vec_t<T, rank - 1> data;
	slong* colptr;
	T* valptr;

	sparse_tensor_t(std::array<slong, rank> l) {
		std::copy(l.begin(), l.end(), dims);
		rowptr = s_malloc<ulong>(dims[0] + 1);
		for (ulong i = 0; i <= dims[0]; i++)
			rowptr[i] = 0;
		sparse_vec_init(data, 8);
		colptr = data->indices;
		valptr = data->entries;
	}

	~sparse_tensor_t() {
		s_free(rowptr);
		sparse_vec_clear(data);
	}

	std::pair<slong*, T*> row(ulong i) {
		return std::make_pair(colptr + rowptr[i] * (rank - 1), valptr + rowptr[i]);
	}

	slong* entry_ptr(std::array<slong, rank> l) {
		auto begin = row(l[0]).first;
		auto end = row(l[0] + 1).first;
		if (begin == end)
			return nullptr;
		auto ptr = sparse_base::binarysearch(begin, end, rank - 1, l.data() + 1);
		if (ptr == end)
			return nullptr;
		else
			return ptr;
	}

	slong* entry_ptr(slong* l) {
		auto begin = row(l[0]).first;
		auto end = row(l[0] + 1).first;
		if (begin == end)
			return nullptr;
		auto ptr = sparse_base::binarysearch(begin, end, rank - 1, l + 1);
		if (ptr == end)
			return nullptr;
		else
			return ptr;
	}

	slong* entry_lower_bound(std::array<slong, rank> l) {
		auto begin = row(l[0]).first;
		auto end = row(l[0] + 1).first;
		if (begin == end)
			return end;
		return sparse_base::lower_bound(begin, end, rank - 1, l.data() + 1);
	}

	void unordered_insert(std::array<slong, rank> l, T* val) {
		ulong row = l[0];
		if (data->nnz + 1 > data->alloc)
			sparse_vec_realloc(data, data->alloc * 2);
		size_t index = rowptr[row + 1];
		for (size_t i = data->nnz; i > index; i--) {
			for (size_t j = 0; j < rank - 1; j++)
				colptr[i * (rank - 1) + j] = colptr[(i - 1) * (rank - 1) + j];
			scalar_set(valptr + i, valptr + i - 1);
		}
		for (size_t i = 0; i < rank - 1; i++)
			colptr[index * (rank - 1) + i] = l[i + 1];
		scalar_set(valptr + index, val);
		data->nnz++;
		for (size_t i = row + 1; i <= dims[0]; i++)
			rowptr[i]++;
		return;
	}

	// ordered insert
	void insert(std::array<slong, rank> l, T* val) {
		ulong row = l[0];
		if (data->nnz + 1 > data->alloc)
			sparse_vec_realloc(data, data->alloc * 2);
		auto ptr = entry_lower_bound(l);
		size_t index = (ptr - colptr) / (rank - 1);
		for (size_t i = data->nnz; i > index; i--) {
			for (size_t j = 0; j < rank - 1; j++)
				colptr[i * (rank - 1) + j] = colptr[(i - 1) * (rank - 1) + j];
			scalar_set(valptr + i, valptr + i - 1);
		}
		for (size_t i = 0; i < rank - 1; i++)
			colptr[index * (rank - 1) + i] = l[i + 1];
		scalar_set(valptr + index, val);
		data->nnz++;
		for (size_t i = row + 1; i <= dims[0]; i++)
			rowptr[i]++;
		return;
	}

	// only for test
	void print_test() {
		for (ulong i = 0; i < dims[0]; i++) {
			for (ulong j = rowptr[i]; j < rowptr[i + 1]; j++) {
				std::cout << i << " ";
				for (ulong k = 0; k < rank - 1; k++)
					std::cout << colptr[j * (rank - 1) + k] << " ";
				std::cout << " : " << valptr[j] << std::endl;
			}
		}
	}

	sparse_tensor_t<T, rank> transpose(std::vector<slong> perm) {
		std::array<slong, rank> l;
		std::array<slong, rank> lperm;
		for (ulong i = 0; i < rank; i++)
			lperm[i] = dims[perm[i]];
		sparse_tensor_t<T, rank> B(lperm);
		for (ulong i = 0; i < dims[0]; i++) {
			for (ulong j = rowptr[i]; j < rowptr[i + 1]; j++) {
				l[0] = i;
				for (ulong k = 1; k < rank; k++)
					l[k] = colptr[j * (rank - 1) + k - 1];
				for (ulong k = 0; k < rank; k++)
					lperm[k] = l[perm[k]];
				B.insert(lperm, valptr + j);
			}
		}
		return B;
	}
};

#endif