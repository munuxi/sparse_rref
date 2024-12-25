#ifndef SPARSE_TENSOR_H
#define SPARSE_TENSOR_H

#include "sparse_vec.h"
#include <array>

// TODO: sparse tensor

// CSR format for sparse tensor
template <typename T> struct sparse_tensor_t {
	uint16_t rank;
	std::vector<ulong> dims;
	ulong* rowptr;
	ulong alloc;
	ulong* colptr;
	T* valptr;

	//empty constructor
	sparse_tensor_t() {
		rank = 0;
		rowptr = NULL;
		alloc = 0;
		colptr = NULL;
		valptr = NULL;
	}

	// Constructor with dimensions
	sparse_tensor_t(std::vector<ulong> l) {
		dims = l;
		rank = l.size();
		rowptr = s_malloc<ulong>(dims[0] + 1);
		for (ulong i = 0; i <= dims[0]; i++)
			rowptr[i] = 0;
		alloc = 8;
		colptr = s_malloc<ulong>(alloc * rank);
		valptr = s_malloc<T>(alloc);
		if constexpr (std::is_same<T, fmpq>::value) {
			for (ulong i = 0; i < alloc * rank; i++)
				fmpq_init(valptr + i);
		}
	}
	
	// Copy constructor
	sparse_tensor_t(const sparse_tensor_t& l) {
		dims = l.dims;
		rank = l.rank;
		rowptr = s_malloc<ulong>(dims[0] + 1);
		for (ulong i = 0; i <= dims[0]; i++)
			rowptr[i] = l.rowptr[i];
		alloc = l.alloc;
		colptr = s_malloc<ulong>(alloc * rank);
		valptr = s_malloc<T>(alloc);
		if constexpr (std::is_same<T, fmpq>::value) {
			for (ulong i = 0; i < alloc * rank; i++)
				fmpq_init(valptr + i);
		}
		for (ulong i = 0; i < alloc * rank; i++)
			colptr[i] = l.colptr[i];
		for (ulong i = 0; i < alloc; i++)
			scalar_set(valptr + i, l.valptr + i);
	}

	// Move constructor
	sparse_tensor_t(sparse_tensor_t&& l) {
		dims = l.dims;
		rank = l.rank;
		rowptr = l.rowptr;
		l.rowptr = NULL;
		alloc = l.alloc;
		colptr = l.colptr;
		l.colptr = NULL;
		valptr = l.valptr;
		l.valptr = NULL;
	}

	~sparse_tensor_t() {
		if (rowptr == NULL)
			return;
		s_free(rowptr);
		if constexpr (std::is_same<T, fmpq>::value) {
			for (ulong i = 0; i < alloc; i++)
				fmpq_clear(valptr + i);
		}
		s_free(colptr);
		s_free(valptr);
		alloc = 0;
	}

	// Copy assignment
	sparse_tensor_t& operator=(const sparse_tensor_t& l) {
		if (this == &l)
			return *this;
		if (rowptr != NULL)
			s_free(rowptr);
		if (colptr != NULL)
			s_free(colptr);
		if (valptr != NULL) {
			if constexpr (std::is_same<T, fmpq>::value) {
				for (ulong i = 0; i < alloc; i++)
					fmpq_clear(valptr + i);
			}
			s_free(valptr);
		}
		dims = l.dims;
		rank = l.rank;
		rowptr = s_malloc<ulong>(dims[0] + 1);
		for (ulong i = 0; i <= dims[0]; i++)
			rowptr[i] = l.rowptr[i];
		alloc = l.alloc;
		colptr = s_malloc<ulong>(alloc * rank);
		valptr = s_malloc<T>(alloc);
		if constexpr (std::is_same<T, fmpq>::value) {
			for (ulong i = 0; i < alloc * rank; i++)
				fmpq_init(valptr + i);
		}
		for (ulong i = 0; i < alloc * rank; i++)
			colptr[i] = l.colptr[i];
		for (ulong i = 0; i < alloc; i++)
			scalar_set(valptr + i, l.valptr + i);
		return *this;
	}

	// Move assignment
	sparse_tensor_t& operator=(sparse_tensor_t&& l) {
		if (this == &l)
			return *this;
		if (rowptr != NULL)
			s_free(rowptr);
		if (colptr != NULL)
			s_free(colptr);
		if (valptr != NULL) {
			if constexpr (std::is_same<T, fmpq>::value) {
				for (ulong i = 0; i < alloc; i++)
					fmpq_clear(valptr + i);
			}
			s_free(valptr);
		}
		dims = l.dims;
		rank = l.rank;
		rowptr = l.rowptr;
		l.rowptr = NULL;
		alloc = l.alloc;
		colptr = l.colptr;
		l.colptr = NULL;
		valptr = l.valptr;
		l.valptr = NULL;
		return *this;
	}

	inline ulong nnz() {
		return rowptr[dims[0]];
	}

	void reserve(ulong size) {
		colptr = s_realloc<ulong>(colptr, size * rank);
		if (size > alloc) {
			valptr = s_realloc<T>(valptr, size);
			if constexpr (std::is_same<T, fmpq>::value) {
				for (ulong i = alloc; i < size; i++)
					fmpq_init(valptr + i);
			}
		}
		else if (size < alloc) {
			if constexpr (std::is_same<T, fmpq>::value) {
				for (ulong i = size; i < alloc; i++)
					fmpq_clear(valptr + i);
			}
			valptr = s_realloc<T>(valptr, size);
		}
		alloc = size;
	}

	std::pair<ulong*, T*> row(ulong i) {
		return std::make_pair(colptr + rowptr[i] * (rank - 1), valptr + rowptr[i]);
	}

	std::pair<ulong*, T*> operator[](ulong i) {
		return row(i);
	}

	ulong* entry_ptr(std::vector<ulong> l) {
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

	ulong* operator[](std::vector<ulong> l) {
		return entry_ptr(l);
	}

	ulong* entry_ptr(ulong* l) {
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

	ulong* operator[](ulong* l) {
		return entry_ptr(l);
	}

	ulong* entry_lower_bound(std::vector<ulong> l) {
		auto begin = row(l[0]).first;
		auto end = row(l[0] + 1).first;
		if (begin == end)
			return end;
		return sparse_base::lower_bound(begin, end, rank - 1, l.data() + 1);
	}

	void unordered_insert(std::vector<ulong> l, T* val) {
		ulong row = l[0];
		ulong nnz = this->nnz();
		if (nnz + 1 > alloc)
			reserve(alloc * 2);
		size_t index = rowptr[row + 1];
		for (size_t i = nnz; i > index; i--) {
			for (size_t j = 0; j < rank - 1; j++)
				colptr[i * (rank - 1) + j] = colptr[(i - 1) * (rank - 1) + j];
			scalar_set(valptr + i, valptr + i - 1);
		}
		for (size_t i = 0; i < rank - 1; i++)
			colptr[index * (rank - 1) + i] = l[i + 1];
		scalar_set(valptr + index, val);
		for (size_t i = row + 1; i <= dims[0]; i++)
			rowptr[i]++;
		return;
	}

	// ordered insert
	void insert(std::vector<ulong> l, T* val) {
		ulong row = l[0];
		ulong nnz = this->nnz();
		if (nnz + 1 > alloc)
			reserve(alloc * 2);
		auto ptr = entry_lower_bound(l);
		size_t index = (ptr - colptr) / (rank - 1);
		for (size_t i = nnz; i > index; i--) {
			for (size_t j = 0; j < rank - 1; j++)
				colptr[i * (rank - 1) + j] = colptr[(i - 1) * (rank - 1) + j];
			scalar_set(valptr + i, valptr + i - 1);
		}
		for (size_t i = 0; i < rank - 1; i++)
			colptr[index * (rank - 1) + i] = l[i + 1];
		scalar_set(valptr + index, val);
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

	sparse_tensor_t<T> transpose(std::vector<ulong> perm) {
		std::vector<ulong> l(rank);
		std::vector<ulong> lperm(rank);
		for (ulong i = 0; i < rank; i++)
			lperm[i] = dims[perm[i]];
		sparse_tensor_t<T> B(lperm);
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