#ifndef SPARSE_TENSOR_H
#define SPARSE_TENSOR_H

#include "sparse_vec.h"
#include <array>

// TODO: sparse tensor

// CSR format for sparse tensor
template <typename T> struct sparse_tensor_t {
	ulong rank;
	std::vector<ulong> dims;
	std::vector<ulong> rowptr;
	ulong alloc;
	ulong* colptr;
	T* valptr;

	//empty constructor
	sparse_tensor_t() {
		rank = 0;
		alloc = 0;
		colptr = NULL;
		valptr = NULL;
	}

	// Constructor with dimensions

	void init(std::vector<ulong> l, ulong aoc = 8) {
		dims = l;
		rank = l.size();
		rowptr.resize(l[0] + 1);
		alloc = aoc;
		colptr = s_malloc<ulong>(rank * alloc);
		valptr = s_malloc<T>(alloc);
		if constexpr (std::is_same_v<T, fmpq>) {
			for (ulong i = 0; i < alloc; i++)
				fmpq_init(valptr + i);
		}
	}

	sparse_tensor_t(std::vector<ulong> l, ulong aoc = 8) {
		init(l, aoc);
	}
	
	// Copy constructor
	sparse_tensor_t(const sparse_tensor_t& l) {
		init(l.dims, l.alloc);
		std::copy(l.rowptr.begin(), l.rowptr.end(), rowptr.begin());
		std::copy(l.colptr, l.colptr + alloc * rank, colptr);
		for (ulong i = 0; i < alloc; i++)
			scalar_set(valptr + i, l.valptr + i);
	}

	// Move constructor
	sparse_tensor_t(sparse_tensor_t&& l) noexcept {
		dims = l.dims;
		rank = l.rank;
		rowptr = std::move(l.rowptr);
		alloc = l.alloc;
		colptr = l.colptr;
		l.colptr = NULL;
		valptr = l.valptr;
		l.valptr = NULL;
		l.alloc = 0; // important for no repeating clear
	}

	void clear() {
		if (alloc == 0)
			return;
		if constexpr (std::is_same<T, fmpq>::value) {
			for (ulong i = 0; i < alloc; i++)
				fmpq_clear(valptr + i);
		}
		s_free(valptr);
		s_free(colptr);
		alloc = 0;
	}

	~sparse_tensor_t() {
		clear();
	}

	void reserve(ulong size) {
		if (size == alloc)
			return;
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

	// Copy assignment
	sparse_tensor_t& operator=(const sparse_tensor_t& l) {
		if (this == &l)
			return *this;
		if (alloc == 0) {
			init(l.dims, l.alloc);
			std::copy(l.rowptr.begin(), l.rowptr.end(), rowptr.begin());
			std::copy(l.colptr, l.colptr + alloc * rank, colptr);
			for (ulong i = 0; i < alloc; i++)
				scalar_set(valptr + i, l.valptr + i);
			return *this;
		}
		dims = l.dims;
		rank = l.rank;
		rowptr.resize(dims[0] + 1);
		if (alloc < l.alloc)
			reserve(l.alloc);
		std::copy(l.rowptr.begin(), l.rowptr.end(), rowptr.begin());
		std::copy(l.colptr, l.colptr + alloc * rank, colptr);
		for (ulong i = 0; i < alloc; i++)
			scalar_set(valptr + i, l.valptr + i);
		return *this;
	}

	// Move assignment
	sparse_tensor_t& operator=(sparse_tensor_t&& l) noexcept {
		if (this == &l)
			return *this;
		clear();
		dims = l.dims;
		rank = l.rank;
		rowptr = std::move(l.rowptr);
		alloc = l.alloc;
		colptr = l.colptr;
		l.colptr = NULL;
		valptr = l.valptr;
		l.valptr = NULL;
		l.alloc = 0; // important for no repeating clear
		return *this;
	}

	inline ulong nnz() {
		return rowptr[dims[0]];
	}

	// remove zero entries
	void canonicalize() {
		ulong nnz = this->nnz();
		std::vector<ulong> rowptr_new(dims[0] + 1);
		ulong* colptr_new = s_malloc<ulong>(nnz * (rank - 1));
		T* valptr_new = s_malloc<T>(nnz);
		if constexpr (std::is_same<T, fmpq>::value) {
			for (ulong i = 0; i < nnz; i++)
				fmpq_init(valptr_new + i);
		}
		ulong index = 0;
		rowptr_new[0] = 0;
		for (ulong i = 0; i < dims[0]; i++) {
			for (ulong j = rowptr[i]; j < rowptr[i + 1]; j++) {
				if (!scalar_is_zero(valptr + j)) {
					for (ulong k = 0; k < rank - 1; k++)
						colptr_new[index * (rank - 1) + k] = colptr[j * (rank - 1) + k];
					scalar_set(valptr_new + index, valptr + j);
					index++;
				}
			}
			rowptr_new[i + 1] = index;
		}
		s_free(colptr);
		s_free(valptr);
		colptr = colptr_new;
		valptr = valptr_new;
		rowptr = std::move(rowptr_new);
		alloc = nnz;
	}

	std::pair<ulong*, T*> row(ulong i) {
		return std::make_pair(colptr + rowptr[i] * (rank - 1), valptr + rowptr[i]);
	}

	std::pair<ulong*, T*> operator[](ulong i) {
		return row(i);
	}

	ulong* entry_lower_bound(ulong* l) {
		auto begin = row(l[0]).first;
		auto end = row(l[0] + 1).first;
		if (begin == end)
			return end;
		return sparse_base::lower_bound(begin, end, rank - 1, l + 1);
	}

	ulong* entry_lower_bound(std::vector<ulong> l) {
		return entry_lower_bound(l.data());
	}

	ulong* entry_ptr(ulong* l) {
		auto ptr = entry_lower_bound(l);
		auto end = row(l[0] + 1).first;
		if (ptr == end || std::equal(ptr, ptr + rank - 1, l + 1))
			return ptr;
		else
			return end;
	}

	ulong* entry_ptr(std::vector<ulong> l) {
		return entry_ptr(l.data());
	}

	ulong* operator[](std::vector<ulong> l) {
		return entry_ptr(l);
	}

	ulong* operator[](ulong* l) {
		return entry_ptr(l);
	}

	// unordered, push back on the end of the row
	void push_back(std::vector<ulong> l, T* val) {
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
	// mode = false: insert anyway
	// mode = true: insert and replace if exist
	void insert(std::vector<ulong> l, T* val, bool mode = true) {
		ulong trow = l[0];
		ulong nnz = this->nnz();
		if (nnz + 1 > alloc)
			reserve(alloc * 2);
		ulong* ptr = entry_lower_bound(l);
		size_t index = (ptr - colptr) / (rank - 1);
		bool exist = (ptr != row(trow + 1).first && std::equal(ptr, ptr + rank - 1, l.data() + 1));
		if (!exist || !mode) {
			for (size_t i = nnz; i > index; i--) {
				for (size_t j = 0; j < rank - 1; j++)
					colptr[i * (rank - 1) + j] = colptr[(i - 1) * (rank - 1) + j];
				scalar_set(valptr + i, valptr + i - 1);
			}
			for (size_t i = 0; i < rank - 1; i++)
				colptr[index * (rank - 1) + i] = l[i + 1];
			scalar_set(valptr + index, val);
			for (size_t i = trow + 1; i <= dims[0]; i++)
				rowptr[i]++;
			return;
		}
		scalar_set(valptr + index, val);
	}

	// only for test
	void print_test() {
		for (ulong i = 0; i < dims[0]; i++) {
			for (ulong j = rowptr[i]; j < rowptr[i + 1]; j++) {
				std::cout << i << " ";
				for (ulong k = 0; k < rank - 1; k++)
					std::cout << colptr[j * (rank - 1) + k] << " ";
				std::cout << " : " << scalar_to_str(valptr + j) << std::endl;
			}
		}
	}

	sparse_tensor_t<T> transpose(const std::vector<ulong>& perm) {
		std::vector<ulong> l(rank);
		std::vector<ulong> lperm(rank);
		for (ulong i = 0; i < rank; i++)
			lperm[i] = dims[perm[i]];
		sparse_tensor_t<T> B(lperm, nnz());
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