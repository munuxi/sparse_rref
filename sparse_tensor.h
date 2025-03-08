/*
	Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

	This file is part of Sparse_rref. The Sparse_rref is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/


#ifndef SPARSE_TENSOR_H
#define SPARSE_TENSOR_H

#include "sparse_vec.h"
#include <array>

// TODO: sparse tensor

enum SPARSE_TYPE {
	SPARSE_CSR, // Compressed sparse row
	SPARSE_COO, // List of lists
	SPARSE_LR  // List of rows
};

namespace sparse_rref {

	template <typename T>
	int lexico_compare(const std::vector<T>& a, const std::vector<T>& b) {
		for (size_t i = 0; i < a.size(); i++) {
			if (a[i] < b[i])
				return -1;
			if (a[i] > b[i])
				return 1;
		}
		return 0;
	}

	template <typename T>
	int lexico_compare(T* a, T* b, size_t len) {
		for (size_t i = 0; i < len; i++) {
			if (a[i] < b[i])
				return -1;
			if (a[i] > b[i])
				return 1;
		}
		return 0;
	}

	using index_type = uint16_t;
	using index_t = std::vector<index_type>;
	using index_p = index_type*;

	// CSR format for sparse tensor
	template <typename T> struct sparse_tensor_struct {
		size_t rank;
		size_t alloc;
		index_type* colptr;
		T* valptr;
		std::vector<size_t> dims;
		std::vector<size_t> rowptr;

		//empty constructor
		sparse_tensor_struct() {
			rank = 0;
			alloc = 0;
			colptr = NULL;
			valptr = NULL;
		}

		// Constructor with dimensions

		void init(std::vector<size_t> l, size_t aoc = 8) {
			dims = l;
			rank = l.size();
			rowptr.resize(l[0] + 1);
			alloc = aoc;
			colptr = s_malloc<index_type>((rank - 1) * alloc);
			valptr = s_malloc<T>(alloc);
			for (size_t i = 0; i < alloc; i++)
				new (valptr + i) T();
		}

		sparse_tensor_struct(std::vector<size_t> l, size_t aoc = 8) {
			init(l, aoc);
		}

		// Copy constructor
		sparse_tensor_struct(const sparse_tensor_struct& l) {
			init(l.dims, l.alloc);
			std::copy(l.rowptr.begin(), l.rowptr.end(), rowptr.begin());
			std::copy(l.colptr, l.colptr + alloc * (rank - 1), colptr);
			for (size_t i = 0; i < alloc; i++)
				valptr[i] = l.valptr[i];
		}

		// Move constructor
		sparse_tensor_struct(sparse_tensor_struct&& l) noexcept {
			dims = l.dims;
			rank = l.rank;
			rowptr = l.rowptr;
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
			for (size_t i = 0; i < alloc; i++)
				valptr[i].~T();
			s_free(valptr);
			s_free(colptr);
			alloc = 0;
		}

		~sparse_tensor_struct() {
			clear();
		}

		void reserve(size_t size) {
			if (size == alloc)
				return;
			colptr = s_realloc<index_type>(colptr, size * (rank - 1));
			if (size > alloc) {
				valptr = s_realloc<T>(valptr, size);
				for (size_t i = alloc; i < size; i++)
					new (valptr + i) T();
			}
			else if (size < alloc) {
				for (size_t i = size; i < alloc; i++)
					valptr[i].~T();
				valptr = s_realloc<T>(valptr, size);
			}
			alloc = size;
		}

		void zero() {
			if (rank != 0)
				std::fill(rowptr.begin(), rowptr.end(), 0);
		}

		// Copy assignment
		sparse_tensor_struct& operator=(const sparse_tensor_struct& l) {
			if (this == &l)
				return *this;
			if (alloc == 0) {
				init(l.dims, l.alloc);
				std::copy(l.rowptr.begin(), l.rowptr.end(), rowptr.begin());
				std::copy(l.colptr, l.colptr + alloc * rank, colptr);
				for (size_t i = 0; i < alloc; i++)
					valptr[i] = l.valptr[i];
				return *this;
			}
			dims = l.dims;
			rank = l.rank;
			rowptr.resize(dims[0] + 1);
			if (alloc < l.alloc)
				reserve(l.alloc);
			std::copy(l.rowptr.begin(), l.rowptr.end(), rowptr.begin());
			std::copy(l.colptr, l.colptr + alloc * rank, colptr);
			for (size_t i = 0; i < alloc; i++)
				valptr[i] = l.valptr[i];
			return *this;
		}

		// Move assignment
		sparse_tensor_struct& operator=(sparse_tensor_struct&& l) noexcept {
			if (this == &l)
				return *this;
			clear();
			dims = l.dims;
			rank = l.rank;
			rowptr = l.rowptr;
			alloc = l.alloc;
			colptr = l.colptr;
			l.colptr = NULL;
			valptr = l.valptr;
			l.valptr = NULL;
			l.alloc = 0; // important for no repeating clear
			return *this;
		}

		inline size_t nnz() {
			return rowptr[dims[0]];
		}

		std::vector<size_t> row_nums() {
			return sparse_rref::difference(rowptr);
		}

		size_t row_nnz(size_t i) {
			return rowptr[i + 1] - rowptr[i];
		}

		// remove zero entries
		void canonicalize() {
			size_t nnz = this->nnz();
			std::vector<size_t> rowptr_new(dims[0] + 1);
			auto colptr_new = s_malloc<index_type>(nnz * (rank - 1));
			auto valptr_new = s_malloc<T>(nnz);
			for (size_t i = 0; i < nnz; i++)
				new (valptr_new + i) T();
			size_t index = 0;
			rowptr_new[0] = 0;
			for (size_t i = 0; i < dims[0]; i++) {
				for (size_t j = rowptr[i]; j < rowptr[i + 1]; j++) {
					if (valptr[j] != 0) {
						for (size_t k = 0; k < rank - 1; k++)
							colptr_new[index * (rank - 1) + k] = colptr[j * (rank - 1) + k];
						valptr_new[index] = valptr[j];
						index++;
					}
				}
				rowptr_new[i + 1] = index;
			}
			s_free(colptr);
			s_free(valptr);
			colptr = colptr_new;
			valptr = valptr_new;
			rowptr = rowptr_new;
			alloc = nnz;
		}

		std::pair<index_p, T*> row(size_t i) {
			return std::make_pair(colptr + rowptr[i] * (rank - 1), valptr + rowptr[i]);
		}

		index_p entry_lower_bound(const index_p l) {
			auto begin = row(l[0]).first;
			auto end = row(l[0] + 1).first;
			if (begin == end)
				return end;
			return sparse_rref::lower_bound(begin, end, rank - 1, l + 1);
		}

		index_p entry_lower_bound(const index_t& l) {
			return entry_lower_bound(l.data());
		}

		index_p entry_ptr(index_p l) {
			auto ptr = entry_lower_bound(l);
			auto end = row(l[0] + 1).first;
			if (ptr == end || std::equal(ptr, ptr + rank - 1, l + 1))
				return ptr;
			else
				return end;
		}

		index_p entry_ptr(const index_t& l) {
			return entry_ptr(l.data());
		}

		// unordered, push back on the end of the row
		void push_back(const index_t& l, const T& val) {
			index_type row = l[0];
			size_t nnz = this->nnz();
			if (nnz + 1 > alloc)
				reserve((alloc + 1) * 2);
			size_t index = rowptr[row + 1];
			for (size_t i = nnz; i > index; i--) {
				auto tmpptr = colptr + (i - 1) * (rank - 1);
				std::copy_backward(tmpptr, tmpptr + (rank - 1), tmpptr + 2 * (rank - 1));
				valptr[i] = valptr[i - 1];
			}
			for (size_t i = 0; i < rank - 1; i++)
				colptr[index * (rank - 1) + i] = l[i + 1];
			valptr[index] = val;
			for (size_t i = row + 1; i <= dims[0]; i++)
				rowptr[i]++;
		}

		// ordered insert
		// mode = false: insert anyway
		// mode = true: insert and replace if exist
		void insert(const index_t& l, const T& val, bool mode = true) {
			size_t trow = l[0];
			size_t nnz = this->nnz();
			if (nnz + 1 > alloc)
				reserve((alloc + 1) * 2);
			auto ptr = entry_lower_bound(l);
			size_t index = (ptr - colptr) / (rank - 1);
			bool exist = (ptr != row(trow + 1).first && std::equal(ptr, ptr + rank - 1, l.data() + 1));
			if (!exist || !mode) {
				for (size_t i = nnz; i > index; i--) {
					auto tmpptr = colptr + (i - 1) * (rank - 1);
					std::copy_backward(tmpptr, tmpptr + (rank - 1), tmpptr + 2 * (rank - 1));
					valptr[i] = valptr[i - 1];
				}
				std::copy(l.begin() + 1, l.begin() + rank, colptr + index * (rank - 1));
				valptr[index] = val;
				for (size_t i = trow + 1; i <= dims[0]; i++)
					rowptr[i]++;
				return;
			}
			valptr[index] = val;
		}

		// ordered add one value
		void insert_add(const index_t& l, const T& val) {
			size_t trow = l[0];
			size_t nnz = this->nnz();
			if (nnz + 1 > alloc)
				reserve((alloc + 1) * 2);
			auto ptr = entry_lower_bound(l);
			size_t index = (ptr - colptr) / (rank - 1);
			bool exist = (ptr != row(trow + 1).first && std::equal(ptr, ptr + rank - 1, l.data() + 1));
			if (!exist) {
				for (size_t i = nnz; i > index; i--) {
					auto tmpptr = colptr + (i - 1) * (rank - 1);
					std::copy_backward(tmpptr, tmpptr + (rank - 1), tmpptr + 2 * (rank - 1));
					valptr[i] = valptr[i - 1];
				}
				std::copy(l.begin() + 1, l.begin() + rank, colptr + index * (rank - 1));
				valptr[index] = val;
				for (size_t i = trow + 1; i <= dims[0]; i++)
					rowptr[i]++;
				return;
			}
			valptr[index] += val;
		}

		sparse_tensor_struct<T> transpose(const std::vector<size_t>& perm) {
			std::vector<size_t> l(rank);
			std::vector<size_t> lperm(rank);
			for (size_t i = 0; i < rank; i++)
				lperm[i] = dims[perm[i]];
			sparse_tensor_struct<T> B(lperm, nnz());
			for (size_t i = 0; i < dims[0]; i++) {
				for (size_t j = rowptr[i]; j < rowptr[i + 1]; j++) {
					l[0] = i;
					auto tmpptr = colptr + j * (rank - 1);
					for (size_t k = 1; k < rank; k++)
						l[k] = tmpptr[k - 1];
					for (size_t k = 0; k < rank; k++)
						lperm[k] = l[perm[k]];
					B.push_back(lperm, valptr[j]);
				}
			}
			return B;
		}

		void sort_indices() {
			for (size_t i = 0; i < dims[0]; i++) {
				size_t rownnz = rowptr[i + 1] - rowptr[i];
				std::vector<size_t> perm(rownnz);
				for (size_t j = 0; j < rownnz; j++)
					perm[j] = j;
				std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
					auto ptra = colptr + (rowptr[i] + a) * (rank - 1);
					auto ptrb = colptr + (rowptr[i] + b) * (rank - 1);
					return std::lexicographical_compare(
						ptra, ptra + rank - 1,
						ptrb, ptrb + rank - 1);
					});

				permute(perm, colptr + rowptr[i] * (rank - 1), rank - 1);
				permute(perm, valptr + rowptr[i]);
			}
		}
	};

	// define the default sparse tensor
	template <typename T, SPARSE_TYPE Type = SPARSE_COO> struct sparse_tensor_t;

	template <typename T> struct sparse_tensor_t<T, SPARSE_CSR> {
		sparse_tensor_struct<T> data;

		sparse_tensor_t() {}
		~sparse_tensor_t() {}
		sparse_tensor_t(std::vector<size_t> l, size_t aoc = 8) : data(l, aoc) {}
		sparse_tensor_t(const sparse_tensor_t& l) : data(l.data) {}
		sparse_tensor_t(sparse_tensor_t&& l) noexcept : data(std::move(l.data)) {}
		sparse_tensor_t& operator=(const sparse_tensor_t& l) { data = l.data; return *this; }
		sparse_tensor_t& operator=(sparse_tensor_t&& l) noexcept { data = std::move(l.data); return *this; }

		inline size_t nnz() { return data.rowptr[data.dims[0]]; }
		inline size_t rank() { return data.rank; }
		inline std::vector<size_t> dims() { return data.dims; }
		inline void zero() { data.zero(); }
		inline void insert(const index_t& l, const T& val, bool mode = true) { data.insert(l, val, mode); }
		inline void push_back(const index_t& l, const T& val) { data.push_back(l, val); }
		inline void canonicalize() { data.canonicalize(); }
		inline void sort_indices() { data.sort_indices(); }
		inline sparse_tensor_t transpose(const std::vector<size_t>& perm) {
			sparse_tensor_t B;
			B.data = data.transpose(perm);
			return B;
		}

		void convert_from_COO(const sparse_tensor_t<T, SPARSE_COO>& l) {
			std::vector<size_t> dims(l.data.dims.begin() + 1, l.data.dims.end()); // remove the first dimension
			size_t nnz = l.data.rowptr[1];
			size_t rank = dims.size();
			data.init(dims, nnz);
			std::vector<size_t> index(rank);
			for (size_t i = 0; i < nnz; i++) {
				for (size_t j = 0; j < rank; j++)
					index[j] = l.data.colptr[i * rank + j];
				data.push_back(index, l.data.valptr[i]);
			}
		}

		// constructor from COO
		sparse_tensor_t(const sparse_tensor_t<T, SPARSE_COO>& l) { convert_from_COO(l); }
		sparse_tensor_t& operator=(const sparse_tensor_t<T, SPARSE_COO>& l) {
			data.clear();
			convert_from_COO(l);
			return *this;
		}

		// only for test
		void print_test() {
			for (size_t i = 0; i < data.dims[0]; i++) {
				for (size_t j = data.rowptr[i]; j < data.rowptr[i + 1]; j++) {
					std::cout << i << " ";
					for (size_t k = 0; k < data.rank - 1; k++)
						std::cout << data.colptr[j * (data.rank - 1) + k] << " ";
					std::cout << " : " << data.valptr[j] << std::endl;
				}
			}
		}
	};

	template <typename T> struct sparse_tensor_t<T, SPARSE_COO> {
		sparse_tensor_struct<T> data;

		template <typename S>
		std::vector<S> prepend_num(const std::vector<S>& l, S num = 0) {
			std::vector<S> lp;
			lp.reserve(l.size() + 1);
			lp.push_back(num);
			lp.insert(lp.end(), l.begin(), l.end());
			return lp;
		}

		sparse_tensor_t() {}
		~sparse_tensor_t() {}
		sparse_tensor_t(const std::vector<size_t>& l, size_t aoc = 8) : data(prepend_num(l, 1ULL), aoc) {}
		sparse_tensor_t(const sparse_tensor_t& l) : data(l.data) {}
		sparse_tensor_t(sparse_tensor_t&& l) noexcept : data(std::move(l.data)) {}
		sparse_tensor_t& operator=(const sparse_tensor_t& l) { data = l.data; return *this; }
		sparse_tensor_t& operator=(sparse_tensor_t&& l) noexcept { data = std::move(l.data); return *this; }

		inline size_t nnz() const { return data.rowptr[1]; }
		inline size_t rank() const { return data.rank - 1; }
		inline std::vector<size_t> dims() const {
			std::vector<size_t> result(data.dims.begin() + 1, data.dims.end());
			return result;
		}
		inline void zero() { data.zero(); }
		inline void reserve(size_t size) { data.reserve(size); }
		inline void insert(const index_t& l, const T& val, bool mode = true) { data.insert(prepend_num(l), val, mode); }
		inline void insert_add(const index_t& l, const T& val) { data.insert_add(prepend_num(l), val); }
		inline void push_back(const index_t& l, const T& val) { data.push_back(prepend_num(l), val); }
		inline void canonicalize() { data.canonicalize(); }
		inline void sort_indices() { data.sort_indices(); }
		inline sparse_tensor_t transpose(const std::vector<size_t>& perm) {
			std::vector<size_t> perm_new(perm);
			for (auto& a : perm_new) { a++; }
			perm_new = prepend_num(perm_new, 0);
			sparse_tensor_t B;
			B.data = data.transpose(perm_new);
			return B;
		}

		void transpose_replace(const std::vector<size_t>& perm) {
			std::vector<size_t> new_dims(rank() + 1);
			new_dims[0] = data.dims[0];

			for (size_t i = 0; i < rank(); i++)
				new_dims[i + 1] = data.dims[perm[i] + 1];
			data.dims = new_dims;

			std::vector<size_t> index_new(rank());
			for (size_t i = 0; i < nnz(); i++) {
				auto ptr = index(i);
				for (size_t j = 0; j < rank(); j++)
					index_new[j] = ptr[perm[j]];
				std::copy(index_new.begin(), index_new.end(), ptr);
			}
		}

		sparse_tensor_t operator+(const sparse_tensor_t& l) {
			return tensor_sum(*this, l);
		}

		void operator+=(const sparse_tensor_t& l) {
			if (l.nnz() == 0)
				return;
			if (nnz() == 0) {
				*this = l;
				return;
			}

			*this = tensor_sum(*this, l);
		}

		// for the i-th column, return the indices
		index_p index(size_t i) const {
			return data.colptr + i * rank();
		}

		T& val(size_t i) const {
			return data.valptr[i];
		}

		index_t index_vector(size_t i) const {
			index_t result(rank());
			for (size_t j = 0; j < rank(); j++)
				result[j] = index(i)[j];
			return result;
		}

		std::unordered_map<index_type, index_t> chop_list(size_t pos) const {
			std::unordered_map<index_type, index_t> result;
			for (size_t i = 0; i < nnz(); i++) {
				result[index(i)[pos]].push_back(i);
			}
			return result;
		}

		sparse_tensor_t<T, SPARSE_COO> chop(slong pos, slong aa) const {
			std::vector<size_t> dims_new = dims();
			dims_new.erase(dims_new.begin() + pos);
			sparse_tensor_t<T, SPARSE_COO> result(dims_new);
			index_t index_new;
			index_new.reserve(rank() - 1);
			for (size_t i = 0; i < nnz(); i++) {
				if (index(i)[pos] != aa)
					continue;
				for (size_t j = 0; j < rank(); j++) {
					if (j != pos)
						index_new.push_back(index(i)[j]);
				}
				result.push_back(index_new, val(i));
				index_new.clear();
			}
			return result;
		}

		void convert_from_CSR(const sparse_tensor_t<T, SPARSE_CSR>& l) {
			data.init(prepend_num(l.data.dims, 1), l.data.rowptr[l.data.dims[0]]);
			std::vector<size_t> index(l.data.rank + 1);
			index[0] = 0;
			for (size_t i = 0; i < l.data.dims[0]; i++) {
				index[1] = i;
				for (size_t j = l.data.rowptr[i]; j < l.data.rowptr[i + 1]; j++) {
					for (size_t k = 0; k < l.data.rank - 1; k++)
						index[k + 2] = l.data.colptr[j * (l.data.rank - 1) + k];
					data.push_back(index, l.data.valptr[j]);
				}
			}
		}

		// constructor from CSR
		sparse_tensor_t(const sparse_tensor_t<T, SPARSE_CSR>& l) {
			convert_from_CSR(l);
		}
		sparse_tensor_t& operator=(const sparse_tensor_t<T, SPARSE_CSR>& l) {
			data.clear();
			convert_from_CSR(l);
			return *this;
		}

		void print_test() {
			for (size_t j = 0; j < data.rowptr[1]; j++) {
				for (size_t k = 0; k < data.rank - 1; k++)
					std::cout << data.colptr[j * (data.rank - 1) + k] << " ";
				std::cout << " : " << data.valptr[j] << std::endl;
			}
		}
	};

	// if A, B are sorted, then C is also sorted
	template <typename T> 
	sparse_tensor_t<T, SPARSE_COO> tensor_product(
		const sparse_tensor_t<T, SPARSE_COO>& A,
		const sparse_tensor_t<T, SPARSE_COO>& B, const field_t F) {

		std::vector<size_t> dimsB = B.dims();
		std::vector<size_t> dimsC = A.dims();
		dimsC.insert(dimsC.end(), dimsB.begin(), dimsB.end());

		sparse_tensor_t<T, SPARSE_COO> C(dimsC);

		if (A.nnz() == 0 || B.nnz() == 0) {
			return C;
		}

		C.reserve(A.nnz() * B.nnz());
		index_t indexC;
		for (size_t i = 0; i < A.nnz(); i++) {
			indexC = A.index_vector(i);
			for (size_t j = 0; j < B.nnz(); j++) {
				indexC.insert(indexC.end(), B.index(j), B.index(j) + B.rank());
				C.push_back(indexC, scalar_mul(A.val(i), B.val(j), F));
				indexC.resize(A.rank());
			}
		}

		return C;
	}

	// we assume that A, B are sorted, then C is also sorted
	template <typename T>
	sparse_tensor_t<T, SPARSE_COO> tensor_sum(
		const sparse_tensor_t<T, SPARSE_COO>& A,
		const sparse_tensor_t<T, SPARSE_COO>& B, const field_t F) {

		std::vector<size_t> dimsC = A.dims();

		if (dimsC != B.dims()) {
			std::cerr << "Error: The dimensions of the two tensors do not match." << std::endl;
			exit(1);
		}

		if (A.nnz() == 0)
			return B;
		if (B.nnz() == 0)
			return A;

		sparse_tensor_t<T, SPARSE_COO> C(dimsC, A.nnz() + B.nnz());

		// double pointer
		size_t i = 0, j = 0;
		while (i < A.nnz() && j < B.nnz()) {
			auto indexA = A.index_vector(i);
			auto indexB = B.index_vector(j);
			int cmp = lexico_compare(indexA, indexB);
			if (cmp < 0) {
				C.push_back(indexA, A.val(i));
				i++;
			}
			else if (cmp > 0) {
				C.push_back(indexB, B.val(j));
				j++;
			}
			else {
				C.push_back(indexA, scalar_add(A.val(i), B.val(j), F));
				i++; j++;
			}
		}
		while (i < A.nnz()) {
			C.push_back(A.index_vector(i), A.val(i));
			i++;
		}
		while (j < B.nnz()) {
			C.push_back(B.index_vector(j), B.val(j));
			j++;
		}

		return C;
	}

	template <typename T> 
	sparse_tensor_t<T, SPARSE_COO> tensor_contract(
		const sparse_tensor_t<T, SPARSE_COO>& A,
		const sparse_tensor_t<T, SPARSE_COO>& B,
		const slong i, const slong j, const field_t F) {

		std::vector<size_t> dimsA = A.dims();
		std::vector<size_t> dimsB = B.dims();

		if (dimsA[i] != dimsB[j]) {
			std::cerr << "Error: The dimensions of the two tensors do not match." << std::endl;
			exit(1);
		}

		std::vector<size_t> dimsC;
		for (size_t k = 0; k < dimsA.size(); k++) {
			if (k != i)
				dimsC.push_back(dimsA[k]);
		}
		for (size_t k = 0; k < dimsB.size(); k++) {
			if (k != j)
				dimsC.push_back(dimsB[k]);
		}

		sparse_tensor_t<T, SPARSE_COO> C(dimsC);
		sparse_tensor_t<T, SPARSE_COO> tmpC(dimsC);

		// search for the same indices
		auto choplist_1 = A.chop_list(i);
		auto choplist_2 = B.chop_list(j);

		for (size_t k = 0; k < dimsA[i]; k++) {
			tmpC.zero();

			if (choplist_1.contains(k) && choplist_2.contains(k)) {
				index_t list_1 = choplist_1[k];
				index_t list_2 = choplist_2[k];
				for (size_t m = 0; m < list_1.size(); m++) {
					for (size_t n = 0; n < list_2.size(); n++) {
						auto k1 = list_1[m];
						auto k2 = list_2[n];
						index_t indexC;
						for (size_t l = 0; l < A.rank(); l++) {
							if (l != i)
								indexC.push_back(A.index(k1)[l]);
						}
						for (size_t l = 0; l < B.rank(); l++) {
							if (l != j)
								indexC.push_back(B.index(k2)[l]);
						}

						tmpC.push_back(indexC, scalar_mul(A.val(k1), B.val(k2), F));
					}
				}
				C.sort_indices();
				tmpC.sort_indices();
				C = tensor_sum(C, tmpC, F);
			}
		}

		return C;
	}

	template <typename T>
	sparse_tensor_t<T, SPARSE_COO> tensor_contract(
		const sparse_tensor_t<T, SPARSE_COO>& A,
		const slong i, const slong j, const field_t F) {

		std::vector<size_t> dimsA = A.dims();

		std::vector<size_t> dimsC;
		for (size_t k = 0; k < dimsA.size(); k++) {
			if (k != i && k != j)
				dimsC.push_back(dimsA[k]);
		}

		std::vector<index_t> chop_list;

		// search for the same indices
		for (size_t k = 0; k < A.nnz(); k++) {
			if (A.index(k)[i] == A.index(k)[j]) {
				chop_list[A.index(k)[i]].push_back(k);
			}
		}

		sparse_tensor_t<T, SPARSE_COO> C(dimsC);
		sparse_tensor_t<T, SPARSE_COO> tmpC(dimsC);

		for (auto& list : chop_list) {
			tmpC.zero();
			for (size_t m = 0; m < list.size(); m++) {
				index_t indexC;
				for (size_t l = 0; l < A.rank(); l++) {
					if (l != i && l != j)
						indexC.push_back(A.index(list[m])[l]);
				}
				tmpC.push_back(indexC, A.val(list[m]));
			}
			C.sort_indices();
			tmpC.sort_indices();
			C = tensor_sum(C, tmpC, F);
		}

		return C;
	}

	template <typename T>
	sparse_tensor_t<T, SPARSE_COO> tensor_dot(
		const sparse_tensor_t<T, SPARSE_COO>& A,
		const sparse_tensor_t<T, SPARSE_COO>& B, const field_t F) {
		return tensor_contract(A, B, A.rank() - 1, 0, F);
	}

	// usually B is a matrix, and A is a tensor, we want to contract all the dimensions of A with B
	// e.g. change a basis of a tensor
	template <typename T>
	sparse_tensor_t<T, SPARSE_COO> tensor_transform(
		const sparse_tensor_t<T, SPARSE_COO>& A, const sparse_tensor_t<T, SPARSE_COO>& B, 
		const size_t start_index, const field_t F) {

		auto C = A;
		auto rank = A.rank();
		for (size_t i = start_index; i < rank; i++) {
			C = tensor_contract(C, B, start_index, 0, F);
		}

		return C;
	}

	// IO
	template <typename T> sparse_tensor_t<rat_t> COO_tensor_read(T& st) {
		if (!st.is_open())
			return sparse_tensor_t<rat_t>();
		std::string strLine;

		bool is_size = true;

		std::vector<size_t> dims;
		sparse_tensor_t<rat_t> tensor;

		while (getline(st, strLine)) {
			if (strLine[0] == '%')
				continue;

			auto tokens = sparse_rref::SplitString(strLine, " ");
			if (is_size) {
				for (size_t i = 0; i < tokens.size() - 1; i++)
					dims.push_back(std::stoull(tokens[i]));
				size_t nnz = std::stoull(tokens.back());
				tensor = sparse_tensor_t<rat_t, SPARSE_COO>(dims, nnz);
				is_size = false;
			}
			else {
				if (tokens.size() != dims.size() + 1) {
					std::cerr << "Error: wrong format in the matrix file" << std::endl;
					std::exit(-1);
				}
				index_t index;
				for (size_t i = 0; i < tokens.size() - 1; i++)
					index.push_back(std::stoull(tokens[i]) - 1);
				sparse_rref::DeleteSpaces(tokens.back());
				rat_t val(tokens.back());
				tensor.push_back(index, val);
			}
		}

		return tensor;
	}

} // namespace sparse_rref



#endif