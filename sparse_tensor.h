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

	// CSR format for sparse tensor
	template <typename T> struct sparse_tensor_struct {
		ulong rank;
		ulong alloc;
		ulong* colptr;
		T* valptr;
		std::vector<ulong> dims;
		std::vector<ulong> rowptr;

		//empty constructor
		sparse_tensor_struct() {
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
			for (ulong i = 0; i < alloc; i++)
				new (valptr + i) T();
		}

		sparse_tensor_struct(std::vector<ulong> l, ulong aoc = 8) {
			init(l, aoc);
		}

		// Copy constructor
		sparse_tensor_struct(const sparse_tensor_struct& l) {
			init(l.dims, l.alloc);
			std::copy(l.rowptr.begin(), l.rowptr.end(), rowptr.begin());
			std::copy(l.colptr, l.colptr + alloc * rank, colptr);
			for (ulong i = 0; i < alloc; i++)
				valptr[i] = l.valptr[i];
		}

		// Move constructor
		sparse_tensor_struct(sparse_tensor_struct&& l) noexcept {
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
			for (ulong i = 0; i < alloc; i++)
				valptr[i].~T();
			s_free(valptr);
			s_free(colptr);
			alloc = 0;
		}

		~sparse_tensor_struct() {
			clear();
		}

		void reserve(ulong size) {
			if (size == alloc)
				return;
			colptr = s_realloc<ulong>(colptr, size * rank);
			if (size > alloc) {
				valptr = s_realloc<T>(valptr, size);
				for (ulong i = alloc; i < size; i++)
					new (valptr + i) T();
			}
			else if (size < alloc) {
				for (ulong i = size; i < alloc; i++)
					valptr[i].~T();
				valptr = s_realloc<T>(valptr, size);
			}
			alloc = size;
		}

		void set_zero() {
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
				for (ulong i = 0; i < alloc; i++)
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
			for (ulong i = 0; i < alloc; i++)
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

		inline ulong nnz() {
			return rowptr[dims[0]];
		}

		std::vector<ulong> row_nums() {
			return sparse_rref::difference(rowptr);
		}

		ulong row_nnz(ulong i) {
			return rowptr[i + 1] - rowptr[i];
		}

		// remove zero entries
		void canonicalize() {
			ulong nnz = this->nnz();
			std::vector<ulong> rowptr_new(dims[0] + 1);
			ulong* colptr_new = s_malloc<ulong>(nnz * (rank - 1));
			T* valptr_new = s_malloc<T>(nnz);
			for (ulong i = 0; i < nnz; i++)
				new (valptr_new + i) T();
			ulong index = 0;
			rowptr_new[0] = 0;
			for (ulong i = 0; i < dims[0]; i++) {
				for (ulong j = rowptr[i]; j < rowptr[i + 1]; j++) {
					if (valptr[j] != 0) {
						for (ulong k = 0; k < rank - 1; k++)
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

		std::pair<ulong*, T*> row(ulong i) {
			return std::make_pair(colptr + rowptr[i] * (rank - 1), valptr + rowptr[i]);
		}

		ulong* entry_lower_bound(ulong* l) {
			auto begin = row(l[0]).first;
			auto end = row(l[0] + 1).first;
			if (begin == end)
				return end;
			return sparse_rref::lower_bound(begin, end, rank - 1, l + 1);
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

		// unordered, push back on the end of the row
		void push_back(std::vector<ulong> l, const T& val) {
			ulong row = l[0];
			ulong nnz = this->nnz();
			if (nnz + 1 > alloc)
				reserve((alloc + 1) * 2);
			size_t index = rowptr[row + 1];
			for (size_t i = nnz; i > index; i--) {
				for (size_t j = 0; j < rank - 1; j++)
					colptr[i * (rank - 1) + j] = colptr[(i - 1) * (rank - 1) + j];
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
		void insert(std::vector<ulong> l, const T& val, bool mode = true) {
			ulong trow = l[0];
			ulong nnz = this->nnz();
			if (nnz + 1 > alloc)
				reserve((alloc + 1) * 2);
			ulong* ptr = entry_lower_bound(l);
			size_t index = (ptr - colptr) / (rank - 1);
			bool exist = (ptr != row(trow + 1).first && std::equal(ptr, ptr + rank - 1, l.data() + 1));
			if (!exist || !mode) {
				for (size_t i = nnz; i > index; i--) {
					for (size_t j = 0; j < rank - 1; j++)
						colptr[i * (rank - 1) + j] = colptr[(i - 1) * (rank - 1) + j];
					valptr[i] = valptr[i - 1];
				}
				for (size_t i = 0; i < rank - 1; i++)
					colptr[index * (rank - 1) + i] = l[i + 1];
				valptr[index] = val;
				for (size_t i = trow + 1; i <= dims[0]; i++)
					rowptr[i]++;
				return;
			}
			valptr[index] = val;
		}

		// ordered add one value
		void insert_add(std::vector<ulong> l, const T& val) {
			ulong trow = l[0];
			ulong nnz = this->nnz();
			if (nnz + 1 > alloc)
				reserve((alloc + 1) * 2);
			ulong* ptr = entry_lower_bound(l);
			size_t index = (ptr - colptr) / (rank - 1);
			bool exist = (ptr != row(trow + 1).first && std::equal(ptr, ptr + rank - 1, l.data() + 1));
			if (!exist) {
				for (size_t i = nnz; i > index; i--) {
					for (size_t j = 0; j < rank - 1; j++)
						colptr[i * (rank - 1) + j] = colptr[(i - 1) * (rank - 1) + j];
					valptr[i] = valptr[i - 1];
				}
				for (size_t i = 0; i < rank - 1; i++)
					colptr[index * (rank - 1) + i] = l[i + 1];
				valptr[index] = val;
				for (size_t i = trow + 1; i <= dims[0]; i++)
					rowptr[i]++;
				return;
			}
			valptr[index] += val;
		}

		sparse_tensor_struct<T> transpose(const std::vector<ulong>& perm) {
			std::vector<ulong> l(rank);
			std::vector<ulong> lperm(rank);
			for (ulong i = 0; i < rank; i++)
				lperm[i] = dims[perm[i]];
			sparse_tensor_struct<T> B(lperm, nnz());
			for (ulong i = 0; i < dims[0]; i++) {
				for (ulong j = rowptr[i]; j < rowptr[i + 1]; j++) {
					l[0] = i;
					for (ulong k = 1; k < rank; k++)
						l[k] = colptr[j * (rank - 1) + k - 1];
					for (ulong k = 0; k < rank; k++)
						lperm[k] = l[perm[k]];
					B.insert(lperm, valptr[j], false);
				}
			}
			return B;
		}

		void sort_indices() {
			for (ulong i = 0; i < dims[0]; i++) {
				ulong rownnz = rowptr[i + 1] - rowptr[i];
				std::vector<ulong> perm(rownnz);
				for (ulong j = 0; j < rownnz; j++)
					perm[j] = j;
				std::sort(perm.begin(), perm.end(), [&](ulong a, ulong b) {
					return std::lexicographical_compare(
						colptr + (rowptr[i] + a) * (rank - 1), colptr + (rowptr[i] + a + 1) * (rank - 1),
						colptr + (rowptr[i] + b) * (rank - 1), colptr + (rowptr[i] + b + 1) * (rank - 1));
					});
				std::vector<ulong> colptr_new(rownnz * (rank - 1));
				std::vector<T> valptr_new(rownnz);
				for (ulong j = 0; j < rownnz; j++) {
					for (ulong k = 0; k < rank - 1; k++)
						colptr_new[j * (rank - 1) + k] = colptr[(rowptr[i] + perm[j]) * (rank - 1) + k];
					valptr_new[j] = valptr[rowptr[i] + perm[j]];
				}
				for (ulong j = 0; j < rownnz; j++) {
					for (ulong k = 0; k < rank - 1; k++)
						colptr[(rowptr[i] + j) * (rank - 1) + k] = colptr_new[j * (rank - 1) + k];
					valptr[rowptr[i] + j] = valptr_new[j];
				}
			}
		}
	};

	template <typename T, SPARSE_TYPE Type = SPARSE_CSR> struct sparse_tensor_t;

	template <typename T> struct sparse_tensor_t<T, SPARSE_CSR> {
		sparse_tensor_struct<T> data;

		sparse_tensor_t() {}
		~sparse_tensor_t() {}
		sparse_tensor_t(std::vector<ulong> l, ulong aoc = 8) : data(l, aoc) {}
		sparse_tensor_t(const sparse_tensor_t& l) : data(l.data) {}
		sparse_tensor_t(sparse_tensor_t&& l) noexcept : data(std::move(l.data)) {}
		sparse_tensor_t& operator=(const sparse_tensor_t& l) { data = l.data; return *this; }
		sparse_tensor_t& operator=(sparse_tensor_t&& l) noexcept { data = std::move(l.data); return *this; }

		inline ulong nnz() { return data.rowptr[data.dims[0]]; }
		inline ulong rank() { return data.rank; }
		inline std::vector<ulong> dims() { return data.dims; }
		inline void set_zero() { data.set_zero(); }
		inline void insert(std::vector<ulong> l, const T& val, bool mode = true) { data.insert(l, val, mode); }
		inline void push_back(std::vector<ulong> l, const T& val) { data.push_back(l, val); }
		inline void canonicalize() { data.canonicalize(); }
		inline void sort_indices() { data.sort_indices(); }
		inline sparse_tensor_t transpose(const std::vector<ulong>& perm) {
			sparse_tensor_t B;
			B.data = data.transpose(perm);
			return B;
		}

		void convert_from_COO(const sparse_tensor_t<T, SPARSE_COO>& l) {
			std::vector<ulong> dims(l.data.dims.begin() + 1, l.data.dims.end()); // remove the first dimension
			ulong nnz = l.data.rowptr[1];
			ulong rank = dims.size();
			data.init(dims, nnz);
			std::vector<ulong> index(rank);
			for (ulong i = 0; i < nnz; i++) {
				for (ulong j = 0; j < rank; j++)
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
			for (ulong i = 0; i < data.dims[0]; i++) {
				for (ulong j = data.rowptr[i]; j < data.rowptr[i + 1]; j++) {
					std::cout << i << " ";
					for (ulong k = 0; k < data.rank - 1; k++)
						std::cout << data.colptr[j * (data.rank - 1) + k] << " ";
					std::cout << " : " << data.valptr[j] << std::endl;
				}
			}
		}
	};

	template <typename T> struct sparse_tensor_t<T, SPARSE_COO> {
		sparse_tensor_struct<T> data;

		std::vector<ulong> prepend_num(const std::vector<ulong>& l, ulong num = 0) {
			std::vector<ulong> lp(l);
			lp.insert(lp.begin(), num);
			return lp;
		}

		sparse_tensor_t() {}
		~sparse_tensor_t() {}
		sparse_tensor_t(std::vector<ulong> l, ulong aoc = 8) : data(prepend_num(l, 1), aoc) {}
		sparse_tensor_t(const sparse_tensor_t& l) : data(l.data) {}
		sparse_tensor_t(sparse_tensor_t&& l) noexcept : data(std::move(l.data)) {}
		sparse_tensor_t& operator=(const sparse_tensor_t& l) { data = l.data; return *this; }
		sparse_tensor_t& operator=(sparse_tensor_t&& l) noexcept { data = std::move(l.data); return *this; }

		inline ulong nnz() const { return data.rowptr[1]; }
		inline ulong rank() const { return data.rank - 1; }
		inline std::vector<ulong> dims() const {
			std::vector<ulong> result(data.dims.begin() + 1, data.dims.end());
			return result;
		}
		inline void set_zero() { data.set_zero(); }
		inline void reserve(ulong size) { data.reserve(size); }
		inline void insert(std::vector<ulong> l, const T& val, bool mode = true) { data.insert(prepend_num(l), val, mode); }
		inline void insert_add(std::vector<ulong> l, const T& val) { data.insert_add(prepend_num(l), val); }
		inline void push_back(std::vector<ulong> l, const T& val) { data.push_back(prepend_num(l), val); }
		inline void canonicalize() { data.canonicalize(); }
		inline void sort_indices() { data.sort_indices(); }
		inline sparse_tensor_t transpose(const std::vector<ulong>& perm) {
			std::vector<ulong> perm_new(perm);
			for (auto& a : perm_new) { a++; }
			perm_new = prepend_num(perm_new, 0);
			sparse_tensor_t B;
			B.data = data.transpose(perm_new);
			return B;
		}

		void operator+=(const sparse_tensor_t& l) {
			if (l.nnz() == 0)
				return;
			if (nnz() == 0) {
				*this = l;
				return;
			}

			for (ulong i = 0; i < l.nnz(); i++)
				insert_add(l.index_vector(i), l.data.valptr[i]);
		}

		// for the i-th column, return the indices
		ulong* index(ulong i) const {
			return data.colptr + i * rank();
		}

		std::vector<ulong> index_vector(ulong i) const {
			std::vector<ulong> result(rank());
			for (ulong j = 0; j < rank(); j++)
				result[j] = *(index(i) + j);
			return result;
		}

		sparse_tensor_t<T, SPARSE_COO> chop(slong pos, slong aa) const {
			std::vector<ulong> dims_new = dims();
			dims_new.erase(dims_new.begin() + pos);
			sparse_tensor_t<T, SPARSE_COO> result(dims_new);
			std::vector<ulong> index_new;
			index_new.reserve(rank() - 1);
			for (ulong i = 0; i < nnz(); i++) {
				if (*(index(i) + pos) != aa)
					continue;
				for (ulong j = 0; j < rank(); j++) {
					if (j != pos)
						index_new.push_back(*(index(i) + j));
				}
				result.push_back(index_new, data.valptr[i]);
				index_new.clear();
			}
			return result;
		}

		void convert_from_CSR(const sparse_tensor_t<T, SPARSE_CSR>& l) {
			data.init(prepend_num(l.data.dims, 1), l.data.rowptr[l.data.dims[0]]);
			std::vector<ulong> index(l.data.rank + 1);
			index[0] = 0;
			for (ulong i = 0; i < l.data.dims[0]; i++) {
				index[1] = i;
				for (ulong j = l.data.rowptr[i]; j < l.data.rowptr[i + 1]; j++) {
					for (ulong k = 0; k < l.data.rank - 1; k++)
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
			for (ulong j = 0; j < data.rowptr[1]; j++) {
				for (ulong k = 0; k < data.rank - 1; k++)
					std::cout << data.colptr[j * (data.rank - 1) + k] << " ";
				std::cout << " : " << data.valptr[j] << std::endl;
			}
		}
	};

	template <typename T> struct sparse_tensor_t<T, SPARSE_LR> {
		std::vector<sparse_tensor_t<T, SPARSE_COO>> data;

		sparse_tensor_t() {}
		~sparse_tensor_t() {}
		sparse_tensor_t(std::vector<ulong> l, ulong aoc = 1) {
			data.resize(l[0]);
			for (ulong i = 0; i < l[0]; i++)
				data[i] = sparse_tensor_t<T, SPARSE_COO>(std::vector<ulong>(l.begin() + 1, l.end()), aoc);
		}
		sparse_tensor_t(const sparse_tensor_t& l) : data(l.data) {}
		sparse_tensor_t(sparse_tensor_t&& l) noexcept : data(std::move(l.data)) {}
		sparse_tensor_t& operator=(const sparse_tensor_t& l) { data = l.data; return *this; }
		sparse_tensor_t& operator=(sparse_tensor_t&& l) noexcept { data = std::move(l.data); return *this; }

		inline ulong nnz() {
			ulong res = 0;
			for (auto& a : data)
				res += a.nnz();
			return res;
		}
		inline ulong rank() { return data[0].rank() + 1; }
		inline std::vector<ulong> dims() {
			std::vector<ulong> result(rank());
			result[0] = data.size();
			auto otherdims = data[0].dims();
			for (ulong i = 1; i < rank(); i++)
				result[i] = otherdims[i - 1];
			return result;
		}

		inline void set_zero() {
			for (auto& a : data)
				a.set_zero();
		}
		inline void insert(std::vector<ulong> l, const T& val, bool mode = true) {
			data[l[0]].insert(std::vector<ulong>(l.begin() + 1, l.end()), val, mode);
		}
		inline void push_back(std::vector<ulong> l, const T& val) {
			data[l[0]].push_back(std::vector<ulong>(l.begin() + 1, l.end()), val);
		}
		inline void canonicalize() {
			for (auto& a : data)
				a.canonicalize();
		}

		void print_test() {
			for (ulong i = 0; i < data.size(); i++) {
				std::cout << "Row " << i << std::endl;
				data[i].print_test();
			}
		}
	};

	// if A, B are sorted, then C is also sorted
	template <typename T> 
	sparse_tensor_t<T, SPARSE_COO> tensor_product(
		const sparse_tensor_t<T, SPARSE_COO>& A,
		const sparse_tensor_t<T, SPARSE_COO>& B) {

		std::vector<ulong> dimsB = B.dims();
		std::vector<ulong> dimsC = A.dims();
		dimsC.insert(dimsC.end(), dimsB.begin(), dimsB.end());

		sparse_tensor_t<T, SPARSE_COO> C(dimsC);

		if (A.nnz() == 0 || B.nnz() == 0) {
			return C;
		}

		C.reserve(A.nnz() * B.nnz());
		std::vector<ulong> indexC;
		for (ulong i = 0; i < A.nnz(); i++) {
			indexC = A.index_vector(i);
			for (ulong j = 0; j < B.nnz(); j++) {
				indexC.insert(indexC.end(), B.index(j), B.index(j) + B.rank());
				C.push_back(indexC, A.data.valptr[i] * B.data.valptr[j]);
				indexC.resize(A.rank());
			}
		}

		return C;
	}

	// if A, B are sorted, then C is also sorted
	template <typename T>
	sparse_tensor_t<T, SPARSE_COO> tensor_sum(
		const sparse_tensor_t<T, SPARSE_COO>& A,
		const sparse_tensor_t<T, SPARSE_COO>& B) {

		std::vector<ulong> dimsC = A.dims();

		if (dimsC != B.dims()) {
			std::cerr << "Error: The dimensions of the two tensors do not match." << std::endl;
			exit(1);
		}

		sparse_tensor_t<T, SPARSE_COO> C;

		if (A.nnz() == 0) {
			C = B;
			return C;
		}
		if (B.nnz() == 0) {
			C = A;
			return C;
		}

		if (A.nnz() < B.nnz())
			return tensor_sum(B, A);

		C = A;
		C.reserve(A.nnz() + B.nnz());
		for (ulong j = 0; j < B.nnz(); j++) {
			C.insert_add(B.index_vector(j), B.data.valptr[j]);
		}

		return C;
	}

	template <typename T> 
	sparse_tensor_t<T, SPARSE_COO> tensor_contract(
		const sparse_tensor_t<T, SPARSE_COO>& A, 
		const sparse_tensor_t<T, SPARSE_COO>& B,
		const slong i, const slong j) {

		std::vector<ulong> dimsA = A.dims();
		std::vector<ulong> dimsB = B.dims();

		if (dimsA[i] != dimsB[j]) {
			std::cerr << "Error: The dimensions of the two tensors do not match." << std::endl;
			exit(1);
		}

		std::vector<ulong> dimsC;
		for (ulong k = 0; k < dimsA.size(); k++) {
			if (k != i)
				dimsC.push_back(dimsA[k]);
		}
		for (ulong k = 0; k < dimsB.size(); k++) {
			if (k != j)
				dimsC.push_back(dimsB[k]);
		}

		// do not alloc too much memory: A.nnz() * B.nnz()
		// they are upper bound
		sparse_tensor_t<T, SPARSE_COO> C(dimsC, A.nnz() + B.nnz());

		for (ulong k = 0; k < dimsA[i]; k++) {
			C += tensor_product(A.chop(i, k), B.chop(j, k));
		}

		return C;
	}

} // namespace sparse_rref

#endif