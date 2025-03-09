/*
	Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

	This file is part of Sparse_rref. The Sparse_rref is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/


#ifndef SPARSE_TENSOR_H
#define SPARSE_TENSOR_H

#include <execution> 
#include "sparse_rref.h"
#include "scalar.h"

// TODO: sparse tensor

enum SPARSE_TYPE {
	SPARSE_CSR, // Compressed sparse row
	SPARSE_COO, // List of lists
	SPARSE_LR  // List of rows
};

namespace sparse_rref {

	using index_type = uint8_t;
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
		// we require that rank >= 2
		void init(const std::vector<size_t>& l, size_t aoc = 8) {
			dims = l;
			rank = l.size();
			rowptr.resize(l[0] + 1);
			alloc = aoc;
			colptr = s_malloc<index_type>((rank - 1) * alloc);
			valptr = s_malloc<T>(alloc);
			for (size_t i = 0; i < alloc; i++)
				new (valptr + i) T();
		}

		sparse_tensor_struct(const std::vector<size_t>& l, size_t aoc = 8) {
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
				std::copy(l.colptr, l.colptr + alloc * (rank - 1), colptr);
				std::copy(l.valptr, l.valptr + alloc, valptr);
				return *this;
			}
			dims = l.dims;
			rank = l.rank;
			rowptr.resize(dims[0] + 1);
			if (alloc < l.alloc)
				reserve(l.alloc);
			std::copy(l.rowptr.begin(), l.rowptr.end(), rowptr.begin());
			std::copy(l.colptr, l.colptr + alloc * (rank - 1), colptr);
			std::copy(l.valptr, l.valptr + alloc, valptr);
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

		// remove zero entries, double pointer
		void canonicalize() {
			size_t nnz_now = nnz();
			size_t index = 0;
			std::vector<size_t> newrowptr(dims[0] + 1);
			newrowptr[0] = 0;
			for (size_t i = 0; i < dims[0]; i++) {
				for (size_t j = rowptr[i]; j < rowptr[i + 1]; j++) {
					if (valptr[j] != 0) {
						s_copy(colptr + index * (rank - 1), colptr + j * (rank - 1), rank - 1);
						valptr[index] = valptr[j];
						index++;
					}
				}
				newrowptr[i + 1] = index;
			}
			rowptr = newrowptr;
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
				std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
					auto ptra = colptr + (rowptr[i] + a) * (rank - 1);
					auto ptrb = colptr + (rowptr[i] + b) * (rank - 1);
					return lexico_compare(ptra, ptrb, rank - 1) < 0;
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

		inline size_t alloc() { return data.alloc; }
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

		// for the i-th column, return the indices
		index_p index(size_t i) const { return data.colptr + i * rank(); }
		T& val(size_t i) const { return data.valptr[i]; }

		index_t index_vector(size_t i) const {
			index_t result(rank());
			for (size_t j = 0; j < rank(); j++)
				result[j] = index(i)[j];
			return result;
		}

		inline size_t alloc() const { return data.alloc; }
		inline size_t nnz() const { return data.rowptr[1]; }
		inline size_t rank() const { return data.rank - 1; }
		inline std::vector<size_t> dims() const {
			std::vector<size_t> result(data.dims.begin() + 1, data.dims.end());
			return result;
		}
		inline void zero() { data.zero(); }
		inline void reserve(size_t size) { data.reserve(size); }
		inline void resize(size_t new_nnz) {
			if (new_nnz > alloc())
				reserve(new_nnz);
			data.rowptr[1] = new_nnz;
		}

		// change the dimensions of the tensor
		// it is dangerous, only for internal use
		inline void change_dims(const std::vector<size_t>& new_dims) {
			auto dims = prepend_num(new_dims, 1ULL);
			data.dims = dims;
			data.colptr = s_realloc<index_type>(data.colptr, new_dims.size() * alloc());
		}

		inline void insert(const index_t& l, const T& val, bool mode = true) { data.insert(prepend_num(l), val, mode); }
		inline void insert_add(const index_t& l, const T& val) { data.insert_add(prepend_num(l), val); }
		void push_back(const index_p l, const T& new_val) { 
			auto n_nnz = nnz();
			if (n_nnz + 1 > data.alloc)
				reserve((data.alloc + 1) * 2);
			std::copy(l, l + rank(), index(n_nnz));
			val(n_nnz) = new_val;
			data.rowptr[1]++; // increase the nnz
		}
		void push_back(const index_t& l, const T& new_val) { 
			auto n_nnz = nnz();
			if (n_nnz + 1 > data.alloc)
				reserve((data.alloc + 1) * 2);
			std::copy(l.begin(), l.end(), index(n_nnz));
			val(n_nnz) = new_val;
			data.rowptr[1]++; // increase the nnz
		}
		inline void canonicalize() { data.canonicalize(); }
		inline void sort_indices() { data.sort_indices(); }
		inline sparse_tensor_t transpose(const std::vector<size_t>& perm) {
			std::vector<size_t> perm_new(perm);
			for (auto& a : perm_new) { a++; }
			perm_new = prepend_num(perm_new, 0);
			sparse_tensor_t B;
			B.data = data.transpose(perm_new);
			B.sort_indices();
			return B;
		}

		std::vector<size_t> gen_perm() const {
			std::vector<size_t> perm = perm_init(nnz());

			auto r = rank();
			std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
				return lexico_compare(index(a), index(b), r) < 0;
				});
			return perm;
		}

		// if head_or_tail on, first sort pos-th index, then the other
		// if head_or_tail off, first sort the other, then pos-th index
		std::vector<size_t> gen_perm(size_t pos, const bool head_or_tail) const {
			auto r = rank();
			if ((head_or_tail && pos == 0) || (!head_or_tail && pos == r - 1))
				return gen_perm();

			std::vector<size_t> perm = perm_init(nnz());

			if (head_or_tail) {
				std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
					auto index_a = index(a);
					auto index_b = index(b);
					return index_a[pos] < index_b[pos] || (index_a[pos] == index_b[pos] && lexico_compare(index_a, index_b, r) < 0);
					});
			}
			else {
				std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
					auto index_a = index(a);
					auto index_b = index(b);

					// first compare the index before pos, len = pos not pos - 1!!!
					int t1 = (pos == 0 ? 0 : lexico_compare(index_a, index_b, pos));
					if (t1 < 0)
						return true;
					// then after compare the index after pos, len = r - pos - 1
					if (t1 == 0) {
						int t2 = lexico_compare(index_a + pos + 1, index_b + pos + 1, r - pos - 1);
						if (t2 < 0)
							return true;
						// finally compare the pos-th index
						if (t2 == 0)
							return index_a[pos] < index_b[pos];
					}
					return false;
					});
			}

			return perm;
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

		// true for sorted tensor
		std::unordered_map<index_type, std::vector<size_t>> chop_list(size_t pos, const bool mode = true) const {
			if (mode) {
				std::unordered_map<index_type, std::vector<size_t>> result;
				for (size_t i = 0; i < nnz(); i++) {
					result[index(i)[pos]].push_back(i);
				}
				return result;
			}
			else {
				auto perm = perm_init(nnz());
				// first sort pos-th index, then the other
				std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
					auto index_a = index(a);
					auto index_b = index(b);
					return index_a[pos] < index_b[pos]|| (index_a[pos] == index_b[pos] && 
						lexico_compare(index_a, index_b, rank()) < 0);
					});

				std::unordered_map<index_type, std::vector<size_t>> result;
				for (auto i : perm) {
					result[index(i)[pos]].push_back(i);
				}
				return result;
			}
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
					std::cout << (size_t)(data.colptr[j * (data.rank - 1) + k]) << " ";
				std::cout << " : " << data.valptr[j] << std::endl;
			}
		}
	};

	// we assume that A, B are sorted, then C is also sorted
	template <typename T> 
	sparse_tensor_t<T, SPARSE_COO> tensor_product(
		const sparse_tensor_t<T, SPARSE_COO>& A,
		const sparse_tensor_t<T, SPARSE_COO>& B, const field_t F, const bool mode = true) {

		std::vector<size_t> dimsB = B.dims();
		std::vector<size_t> dimsC = A.dims();
		dimsC.insert(dimsC.end(), dimsB.begin(), dimsB.end());

		sparse_tensor_t<T, SPARSE_COO> C(dimsC);

		if (A.nnz() == 0 || B.nnz() == 0) {
			return C;
		}

		C.reserve(A.nnz() * B.nnz());
		index_t indexC;

		if (mode) {
			for (size_t i = 0; i < A.nnz(); i++) {
				indexC = A.index_vector(i);
				for (size_t j = 0; j < B.nnz(); j++) {
					indexC.insert(indexC.end(), B.index(j), B.index(j) + B.rank());
					C.push_back(indexC, scalar_mul(A.val(i), B.val(j), F));
					indexC.resize(A.rank());
				}
			}
		}
		else {
			auto permA = A.gen_perm();
			auto permB = B.gen_perm();
			for (auto i : permA) {
				indexC = A.index_vector(i);
				for (auto j : permB) {
					indexC.insert(indexC.end(), B.index(j), B.index(j) + B.rank());
					C.push_back(indexC, scalar_mul(A.val(i), B.val(j), F));
					indexC.resize(A.rank());
				}
			}
		}

		return C;
	}

	// returned tensor is sorted
	template <typename T>
	sparse_tensor_t<T, SPARSE_COO> tensor_sum(
		const sparse_tensor_t<T, SPARSE_COO>& A, const sparse_tensor_t<T, SPARSE_COO>& B, 
		const field_t F, const bool mode = true) {

		// if one of the tensors is empty, it is ok that dims of A or B are not defined
		if (A.alloc() == 0)
			return B;
		if (B.alloc() == 0)
			return A;

		std::vector<size_t> dimsC = A.dims();
		auto rank = A.rank();

		if (dimsC != B.dims()) {
			std::cerr << "Error: The dimensions of the two tensors do not match." << std::endl;
			exit(1);
		}

		// if one of the tensors is zero
		if (A.nnz() == 0)
			return B;
		if (B.nnz() == 0)
			return A;

		sparse_tensor_t<T, SPARSE_COO> C(dimsC, A.nnz() + B.nnz());

		if (!mode) {
			auto Aperm = A.gen_perm();
			auto Bperm = B.gen_perm();

			// double pointer
			size_t i = 0, j = 0;
			while (i < A.nnz() && j < B.nnz()) {
				auto posA = Aperm[i];
				auto posB = Bperm[j];
				auto indexA = A.index(posA);
				auto indexB = B.index(posA);
				int cmp = lexico_compare(indexA, indexB, rank);
				if (cmp < 0) {
					C.push_back(indexA, A.val(posA));
					i++;
				}
				else if (cmp > 0) {
					C.push_back(indexB, B.val(posB));
					j++;
				}
				else {
					auto val = scalar_add(A.val(posA), B.val(posB), F);
					if (val != 0)
						C.push_back(indexA, val);
					i++; j++;
				}
			}
			while (i < A.nnz()) {
				auto posA = Aperm[i];
				C.push_back(A.index_vector(posA), A.val(posA));
				i++;
			}
			while (j < B.nnz()) {
				auto posB = Bperm[j];
				C.push_back(B.index_vector(posB), B.val(posB));
				j++;
			}
		}
		else {
			// double pointer
			size_t i = 0, j = 0;
			while (i < A.nnz() && j < B.nnz()) {
				auto indexA = A.index(i);
				auto indexB = B.index(i);
				int cmp = lexico_compare(indexA, indexB, rank);
				if (cmp < 0) {
					C.push_back(indexA, A.val(i));
					i++;
				}
				else if (cmp > 0) {
					C.push_back(indexB, B.val(j));
					j++;
				}
				else {
					auto val = scalar_add(A.val(i), B.val(j), F);
					if (val != 0)
						C.push_back(indexA, val);
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
		}

		return C;
	}

	// A += B, we assume that A and B are sorted
	template <typename T>
	void tensor_sum_replace( 
		sparse_tensor_t<T, SPARSE_COO>& A,
		const sparse_tensor_t<T, SPARSE_COO>& B, const field_t F) {

		// if one of the tensors is empty, it is ok that dims of A or B are not defined
		if (A.alloc() == 0) {
			A = B;
			return;
		}
		if (B.alloc() == 0)
			return;

		std::vector<size_t> dimsC = A.dims();
		auto rank = A.rank();

		if (dimsC != B.dims()) {
			std::cerr << "Error: The dimensions of the two tensors do not match." << std::endl;
			exit(1);
		}

		// if one of the tensors is zero
		if (A.nnz() == 0) {
			A = B;
			return;
		}
		if (B.nnz() == 0)
			return;

		if (&A == &B) {
			for (size_t i = 0; i < A.nnz(); i++) {
				A.val(i) = scalar_add(A.val(i), A.val(i), F);
			}
			return;
		}

		// double pointer, from the end to the beginning
		size_t ptr1 = A.nnz(), ptr2 = B.nnz();
		size_t ptr = A.nnz() + B.nnz();

		A.resize(ptr);

		while (ptr1 > 0 && ptr2 > 0) {
			int order = lexico_compare(A.index(ptr1 - 1), B.index(ptr2 - 1), rank);

			if (order == 0) {
				auto entry = scalar_add(A.val(ptr1 - 1), B.val(ptr2 - 1), F);
				if (entry != 0) {
					s_copy(A.index(ptr - 1), A.index(ptr1 - 1), rank);
					A.val(ptr - 1) = entry;
					ptr--;
				}
				ptr1--;
				ptr2--;
			}
			else if (order < 0) {
				s_copy(A.index(ptr - 1), B.index(ptr2 - 1), rank);
				A.val(ptr - 1) = B.val(ptr2 - 1);
				ptr2--;
				ptr--;
			}
			else {
				s_copy(A.index(ptr - 1), A.index(ptr1 - 1), rank);
				A.val(ptr - 1) = A.val(ptr1 - 1);
				ptr1--;
				ptr--;
			}
		}
		while (ptr2 > 0) {
			s_copy(A.index(ptr - 1), B.index(ptr2 - 1), rank);
			A.val(ptr - 1) = B.val(ptr2 - 1);
			ptr2--;
			ptr--;
		}

		// if ptr1 > 0, and ptr > 0
		for (size_t i = ptr1; i < ptr; i++) {
			A.val(i) = 0;
		}

		// // then remove the zero entries
		// A.canonicalize();
	}

	//// the result is sorted
	//template <typename T> 
	//sparse_tensor_t<T, SPARSE_COO> tensor_contract(
	//	const sparse_tensor_t<T, SPARSE_COO>& A,
	//	const sparse_tensor_t<T, SPARSE_COO>& B,
	//	const size_t i, const size_t j, const field_t F, const bool mode = true) {

	//	std::vector<size_t> dimsA = A.dims();
	//	std::vector<size_t> dimsB = B.dims();

	//	if (dimsA[i] != dimsB[j]) {
	//		std::cerr << "Error: The dimensions of the two tensors do not match." << std::endl;
	//		exit(1);
	//	}

	//	std::vector<size_t> dimsC;
	//	for (size_t k = 0; k < dimsA.size(); k++) {
	//		if (k != i)
	//			dimsC.push_back(dimsA[k]);
	//	}
	//	for (size_t k = 0; k < dimsB.size(); k++) {
	//		if (k != j)
	//			dimsC.push_back(dimsB[k]);
	//	}

	//	sparse_tensor_t<T, SPARSE_COO> C(dimsC);
	//	sparse_tensor_t<T, SPARSE_COO> tmpC(dimsC);
	//	std::vector<sparse_tensor_t<T, SPARSE_COO>> tmpCs(dimsA[i]);

	//	// search for the same indices
	//	// it is ok for unsorted tensors, chop_list will sort the indices
	//	auto choplist_1 = A.chop_list(i, mode);
	//	auto choplist_2 = B.chop_list(j, mode);

	//	index_t indexC(C.rank());
	//	for (index_type k = 0; k < dimsA[i]; k++) {
	//		if (choplist_1.contains(k) && choplist_2.contains(k)) {
	//			tmpC.zero();
	//			auto list_1 = choplist_1[k];
	//			auto list_2 = choplist_2[k];
	//			for (size_t m = 0; m < list_1.size(); m++) {
	//				for (size_t n = 0; n < list_2.size(); n++) {
	//					auto k1 = list_1[m];
	//					auto k2 = list_2[n];
	//					
	//					indexC.clear();
	//					for (size_t l = 0; l < A.rank(); l++) {
	//						if (l != i)
	//							indexC.push_back(A.index(k1)[l]);
	//					}
	//					for (size_t l = 0; l < B.rank(); l++) {
	//						if (l != j)
	//							indexC.push_back(B.index(k2)[l]);
	//					}

	//					tmpC.push_back(indexC, scalar_mul(A.val(k1), B.val(k2), F));
	//				}
	//			}
	//			tensor_sum_replace(C, tmpC, F);
	//		}
	//	}

	//	C.canonicalize();

	//	return C;
	//}

	template<typename T>
	void print_p(T* a, size_t t) {
		for (size_t i = 0; i < t; i++)
			std::cout << (ulong)(a[i]) << " ";
	}

	// the result is sorted
	template <typename T>
	sparse_tensor_t<T, SPARSE_COO> tensor_contract(
		const sparse_tensor_t<T, SPARSE_COO>& A,
		const sparse_tensor_t<T, SPARSE_COO>& B,
		const size_t i, const size_t j, const field_t F, thread_pool& pool) {

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

		auto permA = A.gen_perm(i, false);
		auto permB = B.gen_perm(j, false);
		// this means we move the i-th, j-th index to the last position

		std::vector<size_t> rowptrA;
		std::vector<size_t> rowptrB;
		
		auto equalexcept = [](const index_p a, const index_p b, const size_t pos, const size_t rank) {
			// do not compare the pos-th index
			for (size_t i = 0; i < rank; i++) 
				if (i != pos && a[i] != b[i])
					return false;
			return true;
			};

		auto r = A.rank();
		rowptrA.push_back(0);
		for (size_t k = 1; k < A.nnz(); k++) {
			if (A.index(permA[k])[i] <= A.index(permA[rowptrA.back()])[i])
				rowptrA.push_back(k);
			else if (!equalexcept(A.index(permA[rowptrA.back()]), A.index(permA[k]), i, r))
				rowptrA.push_back(k);
		}
		rowptrA.push_back(A.nnz());

		r = B.rank();
		rowptrB.push_back(0);
		for (size_t k = 1; k < B.nnz(); k++) {
			if (B.index(permB[k])[j] <= B.index(permB[rowptrB.back()])[j])
				rowptrB.push_back(k);
			else if (!equalexcept(B.index(permB[rowptrB.back()]), B.index(permB[k]), j, r))
				rowptrB.push_back(k);
		}
		rowptrB.push_back(B.nnz());

		sparse_tensor_t<T, SPARSE_COO> C(dimsC);

		//for (size_t k = 0; k < rowptrA.size() - 1; k++) {
		//	// from rowptrA[k] to rowptrA[k + 1] are the same
		//	auto startA = rowptrA[k];
		//	auto endA = rowptrA[k + 1];

		//	index_t indexC;
		//	for (size_t l = 0; l < A.rank(); l++) 
		//		if (l != i)
		//			indexC.push_back(A.index(permA[startA])[l]);

		//	for (size_t l = 0; l < rowptrB.size() - 1; l++) {
		//		auto startB = rowptrB[l];
		//		auto endB = rowptrB[l + 1];

		//		// if the maximum index of A is less than the minimum index of B, then continue
		//		if (A.index(permA[endA - 1])[i] < B.index(permB[startB])[j])
		//			continue;

		//		// double pointer to calculate the inner product
		//		size_t ptrA = startA, ptrB = startB;
		//		T entry = 0;
		//		while (ptrA < endA && ptrB < endB) {
		//			auto indA = A.index(permA[ptrA])[i];
		//			auto indB = B.index(permB[ptrB])[j];
		//			if (indA < indB) 
		//				ptrA++;
		//			else if (indB < indA)
		//				ptrB++;
		//			else {
		//				entry = scalar_add(entry, scalar_mul(A.val(permA[ptrA]), B.val(permB[ptrB]), F), F);
		//				ptrA++;
		//				ptrB++;
		//			}
		//		}

		//		if (entry != 0) {
		//			for (size_t m = 0; m < B.rank(); m++)
		//				if (m != j)
		//					indexC.push_back(B.index(permB[startB])[m]);

		//			C.push_back(indexC, entry);
		//			indexC.resize(A.rank() - 1);
		//		}
		//	}
		//}
		
		// parallel version
		auto nthread = pool.get_thread_count();
		std::vector<sparse_tensor_t<T, SPARSE_COO>> Cs(nthread, C);
		pool.detach_loop(0, rowptrA.size() - 1, [&](size_t k) {
			// from rowptrA[k] to rowptrA[k + 1] are the same
			auto startA = rowptrA[k];
			auto endA = rowptrA[k + 1];

			auto id = thread_id();

			index_t indexC;
			for (size_t l = 0; l < A.rank(); l++)
				if (l != i)
					indexC.push_back(A.index(permA[startA])[l]);

			for (size_t l = 0; l < rowptrB.size() - 1; l++) {
				auto startB = rowptrB[l];
				auto endB = rowptrB[l + 1];

				// if the maximum index of A is less than the minimum index of B, then continue
				if (A.index(permA[endA - 1])[i] < B.index(permB[startB])[j])
					continue;

				// double pointer to calculate the inner product
				size_t ptrA = startA, ptrB = startB;
				T entry = 0;
				while (ptrA < endA && ptrB < endB) {
					auto indA = A.index(permA[ptrA])[i];
					auto indB = B.index(permB[ptrB])[j];
					if (indA < indB)
						ptrA++;
					else if (indB < indA)
						ptrB++;
					else {
						entry = scalar_add(entry, scalar_mul(A.val(permA[ptrA]), B.val(permB[ptrB]), F), F);
						ptrA++;
						ptrB++;
					}
				}

				if (entry != 0) {
					for (size_t m = 0; m < B.rank(); m++)
						if (m != j)
							indexC.push_back(B.index(permB[startB])[m]);

					Cs[id].push_back(indexC, entry);
					indexC.resize(A.rank() - 1);
				}
			}
			}, 
			nthread); // num_block = num_thread

		pool.wait();

		// merge the results
		size_t allnnz = 0;
		size_t nownnz = 0;
		for (size_t i = 0; i < nthread; i++) {
			allnnz += Cs[i].nnz();
		}

		C.reserve(allnnz);
		C.resize(allnnz);
		for (size_t i = 0; i < nthread; i++) {
			// it is ordered, so we can directly push them back
			auto tmpnnz = Cs[i].nnz();
			T* valptr = C.data.valptr + nownnz;
			index_p colptr = C.data.colptr + nownnz * C.rank();
			s_copy(valptr, Cs[i].data.valptr, tmpnnz);
			s_copy(colptr, Cs[i].data.colptr, tmpnnz * C.rank());
			nownnz += tmpnnz;
		}

		return C;
	}

	// the result is sorted
	template <typename T>
	void tensor_contract_r(
		sparse_tensor_t<T, SPARSE_COO>& A, const sparse_tensor_t<T, SPARSE_COO>& B,
		const size_t i, const size_t j, const field_t F, thread_pool& pool) {

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

		auto permA = A.gen_perm(i, false);
		auto permB = B.gen_perm(j, false);
		// this means we move the i-th, j-th index to the last position

		std::vector<size_t> rowptrA;
		std::vector<size_t> rowptrB;

		auto equalexcept = [](const index_p a, const index_p b, const size_t pos, const size_t rank) {
			// do not compare the pos-th index
			for (size_t i = 0; i < rank; i++)
				if (i != pos && a[i] != b[i])
					return false;
			return true;
			};

		auto r = A.rank();
		rowptrA.push_back(0);
		for (size_t k = 1; k < A.nnz(); k++) {
			if (A.index(permA[k])[i] <= A.index(permA[rowptrA.back()])[i])
				rowptrA.push_back(k);
			else if (!equalexcept(A.index(permA[rowptrA.back()]), A.index(permA[k]), i, r))
				rowptrA.push_back(k);
		}
		rowptrA.push_back(A.nnz());

		r = B.rank();
		rowptrB.push_back(0);
		for (size_t k = 1; k < B.nnz(); k++) {
			if (B.index(permB[k])[j] <= B.index(permB[rowptrB.back()])[j])
				rowptrB.push_back(k);
			else if (!equalexcept(B.index(permB[rowptrB.back()]), B.index(permB[k]), j, r))
				rowptrB.push_back(k);
		}
		rowptrB.push_back(B.nnz());

		sparse_tensor_t<T, SPARSE_COO> C(dimsC);
		// parallel version
		auto nthread = pool.get_thread_count();
		std::vector<sparse_tensor_t<T, SPARSE_COO>> Cs(nthread, C);
		pool.detach_loop(0, rowptrA.size() - 1, [&](size_t k) {
			// from rowptrA[k] to rowptrA[k + 1] are the same
			auto startA = rowptrA[k];
			auto endA = rowptrA[k + 1];

			auto id = thread_id();

			index_t indexC;
			for (size_t l = 0; l < A.rank(); l++)
				if (l != i)
					indexC.push_back(A.index(permA[startA])[l]);

			for (size_t l = 0; l < rowptrB.size() - 1; l++) {
				auto startB = rowptrB[l];
				auto endB = rowptrB[l + 1];

				// if the maximum index of A is less than the minimum index of B, then continue
				if (A.index(permA[endA - 1])[i] < B.index(permB[startB])[j])
					continue;

				// double pointer to calculate the inner product
				size_t ptrA = startA, ptrB = startB;
				T entry = 0;
				while (ptrA < endA && ptrB < endB) {
					auto indA = A.index(permA[ptrA])[i];
					auto indB = B.index(permB[ptrB])[j];
					if (indA < indB)
						ptrA++;
					else if (indB < indA)
						ptrB++;
					else {
						entry = scalar_add(entry, scalar_mul(A.val(permA[ptrA]), B.val(permB[ptrB]), F), F);
						ptrA++;
						ptrB++;
					}
				}

				if (entry != 0) {
					for (size_t m = 0; m < B.rank(); m++)
						if (m != j)
							indexC.push_back(B.index(permB[startB])[m]);

					Cs[id].push_back(indexC, entry);
					indexC.resize(A.rank() - 1);
				}
			}
			},
			nthread); // num_block = num_thread

		pool.wait();

		// merge the results
		size_t allnnz = 0;
		size_t nownnz = 0;
		for (size_t i = 0; i < nthread; i++) {
			allnnz += Cs[i].nnz();
		}

		A.change_dims(dimsC);
		if (A.alloc() < allnnz) 
			A.reserve(allnnz);
		A.resize(allnnz);
		for (size_t i = 0; i < nthread; i++) {
			// it is ordered, so we can directly push them back
			auto tmpnnz = Cs[i].nnz();
			T* valptr = A.data.valptr + nownnz;
			index_p colptr = A.data.colptr + nownnz * A.rank();
			s_copy(valptr, Cs[i].data.valptr, tmpnnz);
			s_copy(colptr, Cs[i].data.colptr, tmpnnz * A.rank());
			nownnz += tmpnnz;
		}
	}

	// the result is not sorted
	// mode is only for tensor_contract inside
	template <typename T>
	sparse_tensor_t<T, SPARSE_COO> tensor_contract_2(
		const sparse_tensor_t<T, SPARSE_COO>& A,
		const sparse_tensor_t<T, SPARSE_COO>& B,
		const slong a, const field_t F, thread_pool& pool) {

		auto C = tensor_contract(A, B, a, 0, F, pool);
		std::vector<size_t> perm;
		for (size_t k = 0; k < A.rank() + B.rank() - 1; k++) {
			perm.push_back(k);
		}
		perm.erase(perm.begin() + A.rank() - 1);
		perm.insert(perm.begin() + a, A.rank() - 1);
		C.transpose_replace(perm);

		return C;
	}

	// the result is not sorted
	// mode is only for tensor_contract inside
	template <typename T>
	void tensor_contract_2_r(
		sparse_tensor_t<T, SPARSE_COO>& A,
		const sparse_tensor_t<T, SPARSE_COO>& B,
		const slong a, const field_t F, thread_pool& pool) {

		tensor_contract_r(A, B, a, 0, F, pool);
		std::vector<size_t> perm;
		for (size_t k = 0; k < A.rank() + B.rank() - 1; k++) {
			perm.push_back(k);
		}
		perm.erase(perm.begin() + A.rank() - 1);
		perm.insert(perm.begin() + a, A.rank() - 1);
		A.transpose_replace(perm);
	}

	template <typename T>
	sparse_tensor_t<T, SPARSE_COO> tensor_contract(
		const sparse_tensor_t<T, SPARSE_COO>& A,
		const size_t i, const size_t j, const field_t F, thread_pool& pool) {
		
		if (i > j)
			return tensor_contract(A, j, i, F, pool);

		if (i == j)
			return A; // do nothing
		
		// then i < j

		std::vector<size_t> dimsA = A.dims();
		auto rank = A.rank();

		std::vector<size_t> dimsC;
		for (size_t k = 0; k < dimsA.size(); k++) {
			if (k != i && k != j)
				dimsC.push_back(dimsA[k]);
		}

		std::vector<size_t> equal_ind_list;

		// search for the same indices
		for (size_t k = 0; k < A.nnz(); k++) {
			if (A.index(k)[i] == A.index(k)[j]) {
				equal_ind_list.push_back(k);
			}
		}

		// sort the indices
		auto perm = perm_init(equal_ind_list.size());
		std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
			// do not compare the i-th and j-th index
			auto ptr1 = A.index(equal_ind_list[a]);
			auto ptr2 = A.index(equal_ind_list[b]);

			// first compare the index before i, len = i, 1...i-1
			int t1 = lexico_compare(ptr1, ptr2, i);
			if (t1 != 0)
				return t1 < 0;
			// then compare the index before j, len = j - i - 1, i+1...j-1
			t1 = lexico_compare(ptr1 + i + 1, ptr2 + i + 1, j - i - 1);
			if (t1 != 0)
				return t1 < 0;
			// then compare the index after j, len = rank - j - 1, j+1...rank-1
			t1 = lexico_compare(ptr1 + j + 1, ptr2 + j + 1, A.rank() - j - 1);
			if (t1 != 0)
				return t1 < 0;
			// finally compare the i-th index
			return ptr1[i] < ptr2[i];
			});

		std::vector<size_t> rowptr;
		rowptr.push_back(0);
		
		auto equal_except_ij = [&](const index_p a, const index_p b) {
			// do not compare the i-th and j-th index
			for (size_t k = 0; k < rank; k++)
				if (k != i && k != j && a[k] != b[k])
					return false;
			return true;
			};

		for (size_t k = 1; k < equal_ind_list.size(); k++) {
			if (A.index(equal_ind_list[perm[k]])[i] >= A.index(equal_ind_list[perm[rowptr.back()]])[i] 
				|| !equal_except_ij(A.index(equal_ind_list[perm[k]]), A.index(equal_ind_list[perm[rowptr.back()]])))
				rowptr.push_back(k);
		}
		rowptr.push_back(equal_ind_list.size());

		auto nthread = pool.get_thread_count();
		sparse_tensor_t<T, SPARSE_COO> C(dimsC);
		std::vector<sparse_tensor_t<T, SPARSE_COO>> Cs(nthread, C);

		pool.detach_loop(0, rowptr.size() - 1, [&](size_t k) {
			// from rowptr[k] to rowptr[k + 1] are the same
			auto start = rowptr[k];
			auto end = rowptr[k + 1];
			T entry = 0;
			auto id = thread_id();
			for (size_t m = start; m < end; m++) {
				entry = scalar_add(entry, A.val(equal_ind_list[perm[m]]), F);
			}
			if (entry != 0) {
				index_t indexC;
				for (size_t l = 0; l < A.rank(); l++)
					if (l != i && l != j)
						indexC.push_back(A.index(equal_ind_list[perm[start]])[l]);
				Cs[id].push_back(indexC, entry);
			}
			}, nthread);

		pool.wait();

		// merge the results
		size_t allnnz = 0;
		size_t nownnz = 0;
		for (size_t i = 0; i < nthread; i++) {
			allnnz += Cs[i].nnz();
		}

		C.reserve(allnnz);
		C.resize(allnnz);
		for (size_t i = 0; i < nthread; i++) {
			// it is ordered, so we can directly push them back
			auto tmpnnz = Cs[i].nnz();
			T* valptr = C.data.valptr + nownnz;
			index_p colptr = C.data.colptr + nownnz * C.rank();
			s_copy(valptr, Cs[i].data.valptr, tmpnnz);
			s_copy(colptr, Cs[i].data.colptr, tmpnnz * C.rank());
			nownnz += tmpnnz;
		}

		return C;
	}

	template <typename T>
	void tensor_contract_r(
		sparse_tensor_t<T, SPARSE_COO>& A,
		const size_t i, const size_t j, const field_t F, thread_pool& pool) {

		if (i > j) {
			tensor_contract_r(A, j, i, F, pool);
		}

		if (i == j)
			return; // do nothing

		// then i < j

		std::vector<size_t> dimsA = A.dims();
		auto rank = A.rank();

		std::vector<size_t> dimsC;
		for (size_t k = 0; k < dimsA.size(); k++) {
			if (k != i && k != j)
				dimsC.push_back(dimsA[k]);
		}

		std::vector<size_t> equal_ind_list;

		// search for the same indices
		for (size_t k = 0; k < A.nnz(); k++) {
			if (A.index(k)[i] == A.index(k)[j]) {
				equal_ind_list.push_back(k);
			}
		}

		// sort the indices
		auto perm = perm_init(equal_ind_list.size());
		std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
			// do not compare the i-th and j-th index
			auto ptr1 = A.index(equal_ind_list[a]);
			auto ptr2 = A.index(equal_ind_list[b]);

			// first compare the index before i, len = i, 1...i-1
			int t1 = lexico_compare(ptr1, ptr2, i);
			if (t1 != 0)
				return t1 < 0;
			// then compare the index before j, len = j - i - 1, i+1...j-1
			t1 = lexico_compare(ptr1 + i + 1, ptr2 + i + 1, j - i - 1);
			if (t1 != 0)
				return t1 < 0;
			// then compare the index after j, len = rank - j - 1, j+1...rank-1
			t1 = lexico_compare(ptr1 + j + 1, ptr2 + j + 1, A.rank() - j - 1);
			if (t1 != 0)
				return t1 < 0;
			// finally compare the i-th index
			return ptr1[i] < ptr2[i];
			});

		std::vector<size_t> rowptr;
		rowptr.push_back(0);

		auto equal_except_ij = [&](const index_p a, const index_p b) {
			// do not compare the i-th and j-th index
			for (size_t k = 0; k < rank; k++)
				if (k != i && k != j && a[k] != b[k])
					return false;
			return true;
			};

		for (size_t k = 1; k < equal_ind_list.size(); k++) {
			if (A.index(equal_ind_list[perm[k]])[i] >= A.index(equal_ind_list[perm[rowptr.back()]])[i]
				|| !equal_except_ij(A.index(equal_ind_list[perm[k]]), A.index(equal_ind_list[perm[rowptr.back()]])))
				rowptr.push_back(k);
		}
		rowptr.push_back(equal_ind_list.size());

		auto nthread = pool.get_thread_count();
		sparse_tensor_t<T, SPARSE_COO> C(dimsC);
		std::vector<sparse_tensor_t<T, SPARSE_COO>> Cs(nthread, C);

		pool.detach_loop(0, rowptr.size() - 1, [&](size_t k) {
			// from rowptr[k] to rowptr[k + 1] are the same
			auto start = rowptr[k];
			auto end = rowptr[k + 1];
			T entry = 0;
			auto id = thread_id();
			for (size_t m = start; m < end; m++) {
				entry = scalar_add(entry, A.val(equal_ind_list[perm[m]]), F);
			}
			if (entry != 0) {
				index_t indexC;
				for (size_t l = 0; l < A.rank(); l++)
					if (l != i && l != j)
						indexC.push_back(A.index(equal_ind_list[perm[start]])[l]);
				Cs[id].push_back(indexC, entry);
			}
			}, nthread);

		pool.wait();

		// merge the results
		size_t allnnz = 0;
		size_t nownnz = 0;
		for (size_t i = 0; i < nthread; i++) {
			allnnz += Cs[i].nnz();
		}

		A.change_dims(dimsC);
		if (A.alloc() < allnnz)
			A.reserve(allnnz);
		A.resize(allnnz);
		for (size_t i = 0; i < nthread; i++) {
			// it is ordered, so we can directly push them back
			auto tmpnnz = Cs[i].nnz();
			T* valptr = A.data.valptr + nownnz;
			index_p colptr = A.data.colptr + nownnz * A.rank();
			s_copy(valptr, Cs[i].data.valptr, tmpnnz);
			s_copy(colptr, Cs[i].data.colptr, tmpnnz * A.rank());
			nownnz += tmpnnz;
		}
	}

	template <typename T>
	sparse_tensor_t<T, SPARSE_COO> tensor_dot(
		const sparse_tensor_t<T, SPARSE_COO>& A,
		const sparse_tensor_t<T, SPARSE_COO>& B, const field_t F, thread_pool& pool) {
		return tensor_contract(A, B, A.rank() - 1, 0, F, pool);
	}

	template <typename T>
	void tensor_dot_r(
		sparse_tensor_t<T, SPARSE_COO>& A,
		const sparse_tensor_t<T, SPARSE_COO>& B, const field_t F, thread_pool& pool) {
		tensor_contract_r(A, B, A.rank() - 1, 0, F, pool);
	}

	// usually B is a matrix, and A is a tensor, we want to contract all the dimensions of A with B
	// e.g. change a basis of a tensor
	// we always require that B is sorted
	template <typename T>
	sparse_tensor_t<T, SPARSE_COO> tensor_transform(
		const sparse_tensor_t<T, SPARSE_COO>& A, const sparse_tensor_t<T, SPARSE_COO>& B, 
		const size_t start_index, const field_t F, thread_pool& pool) {

		auto C = A;
		auto rank = A.rank();
		for (size_t i = start_index; i < rank; i++) {
			C = tensor_contract(C, B, start_index, 0, F, pool);
		}

		return C;
	}

	template <typename T>
	void tensor_transform_r(
		sparse_tensor_t<T, SPARSE_COO>& A, const sparse_tensor_t<T, SPARSE_COO>& B,
		const size_t start_index, const field_t F, thread_pool& pool) {

		auto rank = A.rank();
		for (size_t i = start_index; i < rank; i++) {
			tensor_contract_r(A, B, start_index, 0, F, pool);
		}
	}

	// IO

	template <typename T> sparse_tensor_t<size_t> COO_tensor_read(T& st, const field_t F) {
		if (!st.is_open())
			return sparse_tensor_t<size_t>();
		std::string strLine;

		bool is_size = true;

		std::vector<size_t> dims;
		sparse_tensor_t<size_t> tensor;
		index_t index;

		while (getline(st, strLine)) {
			if (strLine[0] == '%')
				continue;

			auto tokens = sparse_rref::SplitString(strLine, " ");
			if (is_size) {
				for (size_t i = 0; i < tokens.size() - 1; i++)
					dims.push_back(std::stoull(tokens[i]));
				size_t nnz = std::stoull(tokens.back());
				tensor = sparse_tensor_t<size_t, SPARSE_COO>(dims, nnz);
				index.reserve(dims.size());
				is_size = false;
			}
			else {
				if (tokens.size() != dims.size() + 1) {
					std::cerr << "Error: wrong format in the matrix file" << std::endl;
					std::exit(-1);
				}
				index.clear();
				for (size_t i = 0; i < tokens.size() - 1; i++)
					index.push_back(std::stoull(tokens[i]) - 1);
				sparse_rref::DeleteSpaces(tokens.back());
				rat_t val(tokens.back());
				ulong val2 = val % F->mod;
				tensor.push_back(index, val2);
			}
		}

		return tensor;
	}

} // namespace sparse_rref



#endif