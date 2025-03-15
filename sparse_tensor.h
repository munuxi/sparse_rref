/*
	Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

	This file is part of Sparse_rref. The Sparse_rref is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/


#ifndef SPARSE_TENSOR_H
#define SPARSE_TENSOR_H

#include <execution> 
#include <map>
#include "sparse_rref.h"
#include "scalar.h"

// TODO: sparse tensor

enum SPARSE_TYPE {
	SPARSE_CSR, // Compressed sparse row
	SPARSE_COO, // Coordinate list
	SPARSE_LR  // List of rows
};

namespace sparse_rref {

	// CSR format for sparse tensor
	template <typename index_type, typename T> struct sparse_tensor_struct {
		size_t rank;
		size_t alloc;
		index_type* colptr;
		T* valptr;
		std::vector<size_t> dims;
		std::vector<size_t> rowptr;

		using index_v = std::vector<index_type>;
		using index_p = index_type*;

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
			rowptr = std::vector<size_t>(l[0] + 1, 0);
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

		inline size_t nnz() {
			return rowptr[dims[0]];
		}

		// Copy assignment
		sparse_tensor_struct& operator=(const sparse_tensor_struct& l) {
			if (this == &l)
				return *this;
			auto nz = l.nnz();
			if (alloc == 0) {
				init(l.dims, nz);
				std::copy(l.rowptr.begin(), l.rowptr.end(), rowptr.begin());
				std::copy(l.colptr, l.colptr + nz * (rank - 1), colptr);
				std::copy(l.valptr, l.valptr + nz, valptr);
				return *this;
			}
			dims = l.dims;
			rank = l.rank;
			rowptr = l.rowptr;
			if (alloc < nz)
				reserve(nz);
			std::copy(l.colptr, l.colptr + nz * (rank - 1), colptr);
			std::copy(l.valptr, l.valptr + nz, valptr);
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

		index_p entry_lower_bound(const index_v& l) {
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

		index_p entry_ptr(const index_v& l) {
			return entry_ptr(l.data());
		}

		// unordered, push back on the end of the row
		void push_back(const index_v& l, const T& val) {
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
		void insert(const index_v& l, const T& val, bool mode = true) {
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
		void insert_add(const index_v& l, const T& val) {
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

		sparse_tensor_struct<index_type, T> transpose(const std::vector<size_t>& perm) {
			std::vector<size_t> l(rank);
			std::vector<size_t> lperm(rank);
			for (size_t i = 0; i < rank; i++)
				lperm[i] = dims[perm[i]];
			sparse_tensor_struct<index_type, T> B(lperm, nnz());
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
	template <typename index_type, typename T, SPARSE_TYPE Type = SPARSE_COO> struct sparse_tensor;

	template <typename index_type, typename T> struct sparse_tensor<index_type, T, SPARSE_CSR> {
		sparse_tensor_struct<index_type, T> data;

		using index_v = std::vector<index_type>;
		using index_p = index_type*;

		sparse_tensor() {}
		~sparse_tensor() {}
		sparse_tensor(std::vector<size_t> l, size_t aoc = 8) : data(l, aoc) {}
		sparse_tensor(const sparse_tensor& l) : data(l.data) {}
		sparse_tensor(sparse_tensor&& l) noexcept : data(std::move(l.data)) {}
		sparse_tensor& operator=(const sparse_tensor& l) { data = l.data; return *this; }
		sparse_tensor& operator=(sparse_tensor&& l) noexcept { data = std::move(l.data); return *this; }

		inline size_t alloc() const { return data.alloc; }
		inline size_t rank() const { return data.rank; }
		inline size_t nnz() const { return data.rowptr[data.dims[0]]; }
		inline std::vector<size_t> dims() const { return data.dims; }
		inline size_t dim(size_t i) const { return data.dims[i]; }
		inline void zero() { data.zero(); }
		inline void insert(const index_v& l, const T& val, bool mode = true) { data.insert(l, val, mode); }
		inline void push_back(const index_v& l, const T& val) { data.push_back(l, val); }
		inline void canonicalize() { data.canonicalize(); }
		inline void sort_indices() { data.sort_indices(); }
		inline sparse_tensor transpose(const std::vector<size_t>& perm) {
			sparse_tensor B;
			B.data = data.transpose(perm);
			return B;
		}

		void convert_from_COO(const sparse_tensor<index_type, T, SPARSE_COO>& l) {
			std::vector<size_t> dims(l.data.dims.begin() + 1, l.data.dims.end()); // remove the first dimension
			size_t nnz = l.data.rowptr[1];
			size_t rank = dims.size();
			data.init(dims, nnz);
			std::vector<index_type> index(rank);
			for (size_t i = 0; i < nnz; i++) {
				for (size_t j = 0; j < rank; j++)
					index[j] = l.data.colptr[i * rank + j];
				data.push_back(index, l.data.valptr[i]);
			}
		}

		// constructor from COO
		sparse_tensor(const sparse_tensor<index_type, T, SPARSE_COO>& l) { convert_from_COO(l); }
		sparse_tensor& operator=(const sparse_tensor<index_type, T, SPARSE_COO>& l) {
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
						std::cout << (size_t)data.colptr[j * (data.rank - 1) + k] << " ";
					std::cout << " : " << data.valptr[j] << std::endl;
				}
			}
		}
	};

	template <typename index_type, typename T> struct sparse_tensor<index_type, T, SPARSE_COO> {
		sparse_tensor_struct<index_type, T> data;

		using index_v = std::vector<index_type>;
		using index_p = index_type*;

		template <typename S>
		std::vector<S> prepend_num(const std::vector<S>& l, S num = 0) {
			std::vector<S> lp;
			lp.reserve(l.size() + 1);
			lp.push_back(num);
			lp.insert(lp.end(), l.begin(), l.end());
			return lp;
		}

		void init(const std::vector<size_t>& l, size_t aoc = 8) {
			data.init(prepend_num(l, 1ULL), aoc);
		}

		sparse_tensor() {}
		~sparse_tensor() {}
		sparse_tensor(const std::vector<size_t>& l, size_t aoc = 8) : data(prepend_num(l, 1ULL), aoc) {}
		sparse_tensor(const sparse_tensor& l) : data(l.data) {}
		sparse_tensor(sparse_tensor&& l) noexcept : data(std::move(l.data)) {}
		sparse_tensor& operator=(const sparse_tensor& l) { data = l.data; return *this; }
		sparse_tensor& operator=(sparse_tensor&& l) noexcept { data = std::move(l.data); return *this; }

		// for the i-th column, return the indices
		index_p index(size_t i) const { return data.colptr + i * rank(); }
		T& val(size_t i) const { return data.valptr[i]; }

		index_v index_vector(size_t i) const {
			index_v result(rank());
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
		inline size_t dim(size_t i) const { return data.dims[i + 1]; }
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

		inline void insert(const index_v& l, const T& val, bool mode = true) { data.insert(prepend_num(l), val, mode); }
		inline void insert_add(const index_v& l, const T& val) { data.insert_add(prepend_num(l), val); }
		void push_back(const index_p l, const T& new_val) { 
			 auto n_nnz = nnz();
			 if (n_nnz + 1 > data.alloc)
			 	reserve((data.alloc + 1) * 2);
			 s_copy(index(n_nnz), l, rank());
			 val(n_nnz) = new_val;
			 data.rowptr[1]++; // increase the nnz
		}
		void push_back(const index_v& l, const T& new_val) { 
			auto n_nnz = nnz();
			if (n_nnz + 1 > data.alloc)
				reserve((data.alloc + 1) * 2);
			std::copy(l.begin(), l.end(), index(n_nnz));
			val(n_nnz) = new_val;
			data.rowptr[1]++; // increase the nnz
		}
		inline void canonicalize() { data.canonicalize(); }
		inline void sort_indices() { data.sort_indices(); }
		inline sparse_tensor transpose(const std::vector<size_t>& perm) {
			std::vector<size_t> perm_new(perm);
			for (auto& a : perm_new) { a++; }
			perm_new = prepend_num(perm_new, 0);
			sparse_tensor B;
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

		std::vector<size_t> gen_perm(const std::vector<size_t>& index_perm) const {
			if (index_perm.size() != rank()) {
				std::cerr << "Error: gen_perm: index_perm size is not equal to rank" << std::endl;
				exit(1);
			}

			std::vector<size_t> perm = perm_init(nnz());
			std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
				return lexico_compare(index(a), index(b), index_perm) < 0;
				});

			return perm;
		}

		void transpose_replace(const std::vector<size_t>& perm, thread_pool* pool = nullptr) {
			std::vector<size_t> new_dims(rank() + 1);
			new_dims[0] = data.dims[0];

			for (size_t i = 0; i < rank(); i++)
				new_dims[i + 1] = data.dims[perm[i] + 1];
			data.dims = new_dims;

			if (pool == nullptr) {
				std::vector<size_t> index_new(rank());
				for (size_t i = 0; i < nnz(); i++) {
					auto ptr = index(i);
					for (size_t j = 0; j < rank(); j++)
						index_new[j] = ptr[perm[j]];
					std::copy(index_new.begin(), index_new.end(), ptr);
				}
			}
			else {
				pool->detach_blocks(0, nnz(), [&](size_t ss, size_t ee) {
					std::vector<size_t> index_new(rank());
					for (size_t i = ss; i < ee; i++) {
						auto ptr = index(i);
						for (size_t j = 0; j < rank(); j++)
							index_new[j] = ptr[perm[j]];
						std::copy(index_new.begin(), index_new.end(), ptr);
					}
					});
				pool->wait();
			}
		}

		sparse_tensor<index_type, T, SPARSE_COO> chop(slong pos, slong aa) const {
			std::vector<size_t> dims_new = dims();
			dims_new.erase(dims_new.begin() + pos);
			sparse_tensor<index_type, T, SPARSE_COO> result(dims_new);
			index_v index_new;
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

		// constructor from CSR
		sparse_tensor(const sparse_tensor<index_type, T, SPARSE_CSR>& l) {
			data.init(prepend_num(l.dims(), 1ULL), l.nnz());
			resize(l.nnz());

			auto r = rank();
			auto n_row = dim(0);

			// first copy the data
			s_copy(data.valptr, l.data.valptr, l.nnz());

			// then deal with the indices
			for (size_t i = 0; i < n_row; i++) {
				for (size_t j = l.data.rowptr[i]; j < l.data.rowptr[i + 1]; j++) {
					auto tmp_index = index(j);
					tmp_index[0] = i;
					for (size_t k = 0; k < r - 1; k++)
						tmp_index[k + 1] = l.data.colptr[j * (r - 1) + k];
				}
			}
		}

		sparse_tensor& operator=(const sparse_tensor<index_type, T, SPARSE_CSR>& l) {
			if (alloc() == 0) {
				init(l.dims(), l.nnz());
			}
			else {
				change_dims(l.dims());
				reserve(l.nnz());
			}

			auto r = rank();
			auto n_row = dim(0);

			// first copy the data
			s_copy(data.valptr, l.data.valptr, l.nnz());

			// then deal with the indices
			for (size_t i = 0; i < n_row; i++) {
				for (size_t j = l.data.rowptr[i]; j < l.data.rowptr[i + 1]; j++) {
					auto tmp_index = index(j);
					tmp_index[0] = i;
					for (size_t k = 0; k < r - 1; k++)
						tmp_index[k + 1] = l.data.colptr[j * (r - 1) + k];
				}
			}

			return *this;
		}

		sparse_tensor& operator=(sparse_tensor<index_type, T, SPARSE_CSR>&& l) {
			data.rank = l.rank() + 1;
			data.alloc = l.data.alloc;
			data.colptr = s_realloc(data.colptr, l.data.alloc * l.rank());
			std::swap(data.valptr, l.data.valptr); // no need to copy
			data.dims = prepend_num(l.dims(), 1ULL);
			data.rowptr = { 0, l.nnz() };

			auto r = l.rank();
			auto n_row = dim(0);

			// then deal with the indices
			for (size_t i = 0; i < n_row; i++) {
				for (size_t j = l.data.rowptr[i]; j < l.data.rowptr[i + 1]; j++) {
					auto tmp_index = index(j);
					tmp_index[0] = i;
					for (size_t k = 0; k < r - 1; k++)
						tmp_index[k + 1] = l.data.colptr[j * (r - 1) + k];
				}
			}

			return *this;
		}

		sparse_tensor(sparse_tensor<index_type, T, SPARSE_CSR>&& l) noexcept {
			data.rank = l.rank() + 1;
			data.alloc = l.data.alloc;
			data.colptr = s_malloc<index_type>(l.data.alloc * l.rank());
			data.valptr = l.data.valptr;
			l.data.valptr = NULL;
			data.dims = prepend_num(l.dims(), 1ULL);
			data.rowptr = { 0, l.nnz() };

			auto r = l.rank();
			auto n_row = dim(0);

			// then deal with the indices
			for (size_t i = 0; i < n_row; i++) {
				for (size_t j = l.data.rowptr[i]; j < l.data.rowptr[i + 1]; j++) {
					auto tmp_index = index(j);
					tmp_index[0] = i;
					for (size_t k = 0; k < r - 1; k++)
						tmp_index[k + 1] = l.data.colptr[j * (r - 1) + k];
				}
			}
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
	template <typename index_type, typename T> 
	sparse_tensor<index_type, T, SPARSE_COO> tensor_product(
		const sparse_tensor<index_type, T, SPARSE_COO>& A,
		const sparse_tensor<index_type, T, SPARSE_COO>& B, const field_t F) {

		std::vector<size_t> dimsB = B.dims();
		std::vector<size_t> dimsC = A.dims();
		dimsC.insert(dimsC.end(), dimsB.begin(), dimsB.end());

		sparse_tensor<index_type, T, SPARSE_COO> C(dimsC);

		if (A.nnz() == 0 || B.nnz() == 0) {
			return C;
		}

		C.reserve(A.nnz() * B.nnz());
		std::vector<index_type> indexC;

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

		return C;
	}

	// only for debug
	template<typename T>
	void print_p(T* a, size_t t) {
		for (size_t i = 0; i < t; i++)
			std::cout << (ulong)(a[i]) << " ";
	}

	// returned tensor is sorted
	template <typename index_type, typename T>
	sparse_tensor<index_type, T, SPARSE_COO> tensor_add(
		const sparse_tensor<index_type, T, SPARSE_COO>& A, 
		const sparse_tensor<index_type, T, SPARSE_COO>& B,
		const field_t F) {

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

		sparse_tensor<index_type, T, SPARSE_COO> C(dimsC, A.nnz() + B.nnz());

		auto Aperm = A.gen_perm();
		auto Bperm = B.gen_perm();

		// double pointer
		size_t i = 0, j = 0;
		// C.zero();
		while (i < A.nnz() && j < B.nnz()) {
			auto posA = Aperm[i];
			auto posB = Bperm[j];
			auto indexA = A.index(posA);
			auto indexB = B.index(posB);
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
			C.push_back(A.index(posA), A.val(posA));
			i++;
		}
		while (j < B.nnz()) {
			auto posB = Bperm[j];
			C.push_back(B.index(posB), B.val(posB));
			j++;
		}
		
		return C;
	}

	// A += B, we assume that A and B are sorted
	template <typename index_type, typename T>
	void tensor_sum_replace( 
		sparse_tensor<index_type, T, SPARSE_COO>& A,
		const sparse_tensor<index_type, T, SPARSE_COO>& B, const field_t F) {

		// if one of the tensors is empty, it is ok that dims of A or B are not defined
		if (A.alloc() == 0) {
			A = B;
			return;
		}
		if (B.alloc() == 0)
			return;

		auto dimsC = A.dims();
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

	// the result is sorted
	template <typename index_type, typename T>
	sparse_tensor<index_type, T, SPARSE_COO> tensor_contract(
		const sparse_tensor<index_type, T, SPARSE_COO>& A, 
		const sparse_tensor<index_type, T, SPARSE_COO>& B,
		const std::vector<size_t>& i1, const std::vector<size_t>& i2, 
		const field_t F, thread_pool* pool = nullptr) {

		using index_v = std::vector<index_type>;
		using index_p = index_type*;

		if (i1.size() != i2.size()) {
			std::cerr << "Error: tensor_contract: The size of the two contract sets do not match." << std::endl;
			exit(1);
		}

		if (i1.size() == 0) {
			return tensor_product(A, B, F);
		}

		auto dimsA = A.dims();
		auto dimsB = B.dims();

		for (size_t k = 0; k < i1.size(); k++) {
			if (dimsA[i1[k]] != dimsB[i2[k]]) {
				std::cerr << "Error: The dimensions of the two tensors do not match." << std::endl;
				exit(1);
			}
		}

		// the dimensions of the result
		std::vector<size_t> dimsC, index_perm_A, index_perm_B;
		for (size_t k = 0; k < dimsA.size(); k++) {
			// if k is not in i1, we add it to dimsC and index_perm_A
			if (std::find(i1.begin(), i1.end(), k) == i1.end()) {
				dimsC.push_back(dimsA[k]);
				index_perm_A.push_back(k);
			}
		}
		index_perm_A.insert(index_perm_A.end(), i1.begin(), i1.end());
		for (size_t k = 0; k < dimsB.size(); k++) {
			// if k is not in i2, we add it to dimsC and index_perm_B
			if (std::find(i2.begin(), i2.end(), k) == i2.end()) {
				dimsC.push_back(dimsB[k]);
				index_perm_B.push_back(k);
			}
		}
		index_perm_B.insert(index_perm_B.end(), i2.begin(), i2.end());

		auto permA = A.gen_perm(index_perm_A);
		auto permB = B.gen_perm(index_perm_B);

		std::vector<size_t> rowptrA;
		std::vector<size_t> rowptrB;

		auto equal_except = [](const index_p a, const index_p b, const std::vector<size_t>& perm, const size_t len) {
			for (size_t i = 0; i < len; i++) {
				if (a[perm[i]] != b[perm[i]])
					return false;
			}
			return true;
			};

		auto i1i2_size = i1.size();
		auto left_size_A = A.rank() - i1i2_size;
		auto left_size_B = B.rank() - i1i2_size;

		rowptrA.push_back(0);
		for (size_t k = 1; k < A.nnz(); k++) {
			if (!equal_except(A.index(permA[rowptrA.back()]), A.index(permA[k]), index_perm_A, left_size_A))
				rowptrA.push_back(k);
		}
		rowptrA.push_back(A.nnz());

		rowptrB.push_back(0);
		for (size_t k = 1; k < B.nnz(); k++) {
			if (!equal_except(B.index(permB[rowptrB.back()]), B.index(permB[k]), index_perm_B, left_size_B))
				rowptrB.push_back(k);
		}
		rowptrB.push_back(B.nnz());

		sparse_tensor<index_type, T, SPARSE_COO> C(dimsC);
		// parallel version
		size_t nthread;
		if (pool == nullptr)
			nthread = 1;
		else
			nthread = pool->get_thread_count();
		std::vector<sparse_tensor<index_type, T, SPARSE_COO>> Cs(nthread, C);

		auto temp_lexico_compare = [&](size_t a, size_t b) {
			auto ptrA = A.index(permA[a]);
			auto ptrB = B.index(permB[b]);
			for (size_t l = 0; l < i1i2_size; l++) {
				if (ptrA[i1[l]] < ptrB[i2[l]])
					return -1;
				if (ptrA[i1[l]] > ptrB[i2[l]])
					return 1;
			}
			return 0;
			};

		auto method = [&](size_t ss, size_t ee) {
			size_t id = 0;
			if (pool != nullptr)
				id = thread_id();

			index_v indexC(dimsC.size());

			for (size_t k = ss; k < ee; k++) {
				// from rowptrA[k] to rowptrA[k + 1] are the same
				auto startA = rowptrA[k];
				auto endA = rowptrA[k + 1];

				for (size_t l = 0; l < left_size_A; l++)
					indexC[l] = A.index(permA[startA])[index_perm_A[l]];

				for (size_t l = 0; l < rowptrB.size() - 1; l++) {
					auto startB = rowptrB[l];
					auto endB = rowptrB[l + 1];

					// if the maximum index of A is less than the minimum index of B, then continue
					if (temp_lexico_compare(endA - 1, startB) < 0)
						continue;

					// double pointer to calculate the inner product
					size_t ptrA = startA, ptrB = startB;
					T entry = 0;
					while (ptrA < endA && ptrB < endB) {
						auto t1 = temp_lexico_compare(ptrA, ptrB);
						if (t1 < 0)
							ptrA++;
						else if (t1 > 0)
							ptrB++;
						else {
							entry = scalar_add(entry, scalar_mul(A.val(permA[ptrA]), B.val(permB[ptrB]), F), F);
							ptrA++;
							ptrB++;
						}
					}

					if (entry != 0) {
						for (size_t l = 0; l < left_size_B; l++)
							indexC[left_size_A + l] = B.index(permB[startB])[index_perm_B[l]];

						Cs[id].push_back(indexC, entry);
					}
				}
			}
			};
	
		// parallel version

		if (pool != nullptr) {
			pool->detach_blocks(0, rowptrA.size() - 1, method, nthread); // num_block = num_thread
			pool->wait();

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
		else {
			method(0, rowptrA.size() - 1);
			return Cs[0];
		}
	}

	template <typename index_type, typename T>
	sparse_tensor<index_type, T, SPARSE_COO> tensor_contract(
		const sparse_tensor<index_type, T, SPARSE_COO>& A, 
		const sparse_tensor<index_type, T, SPARSE_COO>& B,
		const size_t i, const size_t j, const field_t F, thread_pool* pool = nullptr) {

		return tensor_contract(A, B, std::vector<size_t>{ i }, std::vector<size_t>{ j }, F, pool);
	}

	// the result is not sorted
	template <typename index_type, typename T>
	sparse_tensor<index_type, T, SPARSE_COO> tensor_contract_2(
		const sparse_tensor<index_type, T, SPARSE_COO>& A,
		const sparse_tensor<index_type, T, SPARSE_COO>& B,
		const slong a, const field_t F, thread_pool* pool = nullptr) {

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

	// self contraction
	template <typename index_type, typename T>
	sparse_tensor<index_type, T, SPARSE_COO> tensor_contract(
		const sparse_tensor<index_type, T, SPARSE_COO>& A,
		const size_t i, const size_t j, const field_t F, thread_pool* pool = nullptr) {
		
		using index_v = std::vector<index_type>;
		using index_p = index_type*;

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

		std::vector<size_t> index_perm;
		for (size_t k = 0; k < rank; k++) {
			if (k != i && k != j)
				index_perm.push_back(k);
		}
		index_perm.push_back(i);
		index_perm.push_back(j);

		auto perm = perm_init(equal_ind_list.size());
		std::sort(std::execution::par, perm.begin(), perm.end(), [&](size_t a, size_t b) {
			return lexico_compare(A.index(equal_ind_list[a]), A.index(equal_ind_list[b]), index_perm) < 0;
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
			if (!equal_except_ij(A.index(equal_ind_list[perm[k]]), A.index(equal_ind_list[perm[rowptr.back()]])))
				rowptr.push_back(k);
		}
		rowptr.push_back(equal_ind_list.size());

		sparse_tensor<index_type, T, SPARSE_COO> C(dimsC);

		if (pool != nullptr) {
			auto nthread = pool->get_thread_count();
			std::vector<sparse_tensor<index_type, T, SPARSE_COO>> Cs(nthread, C);

			pool->detach_blocks(0, rowptr.size() - 1, [&](size_t ss, size_t ee) {
				index_v indexC;
				indexC.reserve(rank - 2);
				for (size_t k = ss; k < ee; k++) {
					// from rowptr[k] to rowptr[k + 1] are the same
					auto start = rowptr[k];
					auto end = rowptr[k + 1];
					T entry = 0;
					auto id = thread_id();
					for (size_t m = start; m < end; m++) {
						entry = scalar_add(entry, A.val(equal_ind_list[perm[m]]), F);
					}
					if (entry != 0) {
						indexC.clear();
						for (size_t l = 0; l < A.rank(); l++)
							if (l != i && l != j)
								indexC.push_back(A.index(equal_ind_list[perm[start]])[l]);
						Cs[id].push_back(indexC, entry);
					}
				}}, nthread);

			pool->wait();

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
		}
		else {
			index_v indexC;
			indexC.reserve(rank - 2);
			for (size_t k = 0; k < rowptr.size() - 1; k++) {
				// from rowptr[k] to rowptr[k + 1] are the same
				auto start = rowptr[k];
				auto end = rowptr[k + 1];
				T entry = 0;
				for (size_t m = start; m < end; m++) {
					entry = scalar_add(entry, A.val(equal_ind_list[perm[m]]), F);
				}
				if (entry != 0) {
					indexC.clear();
					for (size_t l = 0; l < A.rank(); l++)
						if (l != i && l != j)
							indexC.push_back(A.index(equal_ind_list[perm[start]])[l]);
					C.push_back(indexC, entry);
				}
			}
		}

		return C;
	}

	template <typename index_type, typename T>
	sparse_tensor<index_type, T, SPARSE_COO> tensor_dot(
		const sparse_tensor<index_type, T, SPARSE_COO>& A,
		const sparse_tensor<index_type, T, SPARSE_COO>& B, 
		const field_t F, thread_pool* pool = nullptr) {
		return tensor_contract(A, B, A.rank() - 1, 0, F, pool);
	}

	// usually B is a matrix, and A is a tensor, we want to contract all the dimensions of A with B
	// e.g. change a basis of a tensor
	// we always require that B is sorted
	template <typename index_type, typename T>
	sparse_tensor<index_type, T, SPARSE_COO> tensor_transform(
		const sparse_tensor<index_type, T, SPARSE_COO>& A, 
		const sparse_tensor<index_type, T, SPARSE_COO>& B,
		const size_t start_index, const field_t F, thread_pool* pool = nullptr) {

		auto rank = A.rank();
		auto C = tensor_contract(A, B, start_index, 0, F, pool);
		for (size_t i = start_index + 1; i < rank; i++) {
			C = tensor_contract(C, B, start_index, 0, F, pool);
		}

		return C;
	}

	template <typename index_type, typename T>
	void tensor_transform_replace(
		sparse_tensor<index_type, T, SPARSE_COO>& A,
		const sparse_tensor<index_type, T, SPARSE_COO>& B,
		const size_t start_index, const field_t F, thread_pool* pool = nullptr) {

		auto rank = A.rank();
		for (size_t i = start_index; i < rank; i++) {
			A = tensor_contract(A, B, start_index, 0, F, pool);
		}
	}

	// tensors {A,B,...}
	// index_sets {{i1,j1,...}, {i2,j2,...}, ...}
	// |{i1,j1,...}| = A.rank(), |{i2,j2,...}| = B.rank(), ...
	// index with the same number will be contracted
	// and the other indices will be sorted

	// e.g. D = einstein_sum({ A,B,C }, { {0,1,4}, {2,1}, {2,3} })
	// D_{i0,i3,i4} = sum_{i1,i2} A_{i0,i1,i4} B_{i2,i1} C_{i2,i3}

	// TODO: it works, but the performance is not good
	template <typename index_type, typename T> 
	sparse_tensor<index_type, T, SPARSE_COO> einstein_sum(
		const std::vector<sparse_tensor<index_type, T, SPARSE_COO>*> tensors,
		const std::vector<std::vector<size_t>> index_sets,
		const field_t F, thread_pool* pool = nullptr) {

		auto nt = tensors.size();
		if (nt != index_sets.size()) {
			std::cerr << "Error: The number of tensors does not match the number of index sets." << std::endl;
			exit(1);
		}
		for (size_t i = 1; i < nt; i++) {
			if (tensors[i]->rank() != index_sets[i].size()) {
				std::cerr << "Error: The rank of the tensor does not match the index set." << std::endl;
				exit(1);
			}
		}

		// now is the valid case

		// first case is zero
		for (size_t i = 0; i < nt; i++) {
			if (tensors[i]->nnz() == 0)
				return sparse_tensor<index_type, T, SPARSE_COO>();
		}

		// compute the summed index
		std::map<size_t, std::vector<std::pair<size_t, size_t>>> index_map;
		for (size_t i = 0; i < nt; i++) {
			for (size_t j = 0; j < tensors[i]->rank(); j++) {
				index_map[index_sets[i][j]].push_back(std::make_pair(i, j));
			}
		}

		std::vector<std::vector<std::pair<size_t, size_t>>> contract_index;
		std::vector<std::pair<size_t, size_t>> free_index;
		for (auto& it : index_map) {
			if (it.second.size() > 1)
				contract_index.push_back(it.second);
			else
				free_index.push_back((it.second)[0]);
		}
		std::stable_sort(free_index.begin(), free_index.end(), 
			[&index_sets](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
				return index_sets[a.first][a.second] < index_sets[b.first][b.second];
			});
		
		std::vector<std::vector<size_t>> each_free_index(nt);
		std::vector<std::vector<size_t>> each_perm(nt);
		for (auto [p, q] : free_index) {
			each_free_index[p].push_back(q);
			each_perm[p].push_back(q);
		}
		for (auto& a : contract_index) {
			for (auto [p, q] : a) {
				each_perm[p].push_back(q);
			}
		}

		// restore the perm of the tensor
		std::vector<std::pair<sparse_tensor<index_type, T, SPARSE_COO>*, std::vector<size_t>>> tensor_perm_map;
		std::vector<std::vector<size_t>> pindx;
		std::vector<size_t> tindx(nt);
		for (size_t i = 0; i < nt; i++) {
			auto checkexist = [&](const std::pair<sparse_tensor<index_type, T, SPARSE_COO>*, std::vector<size_t>>& a) {
				if (a.first != tensors[i])
					return false;
				if (a.second.size() != each_perm[i].size())
					return false;
				for (size_t j = 0; j < a.second.size(); j++) {
					if (a.second[j] != each_perm[i][j])
						return false;
				}
				return true;
				};
			bool is_exist = false;
			for (size_t j = 0; j < tensor_perm_map.size(); j++) {
				if (checkexist(tensor_perm_map[j])) {
					is_exist = true;
					tindx[i] = j;
				}
			}
			if (!is_exist) {
				tensor_perm_map.push_back(std::make_pair(tensors[i], each_perm[i]));
				pindx.push_back(tensors[i]->gen_perm(each_perm[i]));
				tindx[i] = pindx.size() - 1;
			}
		}
	

		std::vector<std::vector<size_t>> each_rowptr(nt);
		for (size_t i = 0; i < nt; i++) {
			each_rowptr[i].push_back(0);
			for (size_t j = 1; j < tensors[i]->nnz(); j++) {
				bool is_same = true;
				for (auto aa : each_free_index[i]) {
					if (tensors[i]->index(pindx[tindx[i]][j])[aa] != tensors[i]->index(pindx[tindx[i]][j - 1])[aa]) {
						is_same = false;
						break;
					}
				}
				if (!is_same)
					each_rowptr[i].push_back(j);
			}
			each_rowptr[i].push_back(tensors[i]->nnz());
		}

		// first compute the dims of the result
		std::vector<size_t> dimsC;
		for (auto& aa : free_index)
			dimsC.push_back(tensors[aa.first]->dim(aa.second));

		sparse_tensor<index_type, T, SPARSE_COO> C(dimsC);

		int nthread = 1;
		if (pool != nullptr) {
			nthread = pool->get_thread_count();
		}

		std::vector<sparse_tensor<index_type, T, SPARSE_COO>> Cs(nthread, C);

		auto method = [&](size_t ss, size_t ee) {
			int id = 0;
			if (pool != nullptr)
				id = thread_id();

			std::vector<size_t> ptrs(nt, 0);
			ptrs[0] = ss;
			std::vector<size_t> internel_ptrs(nt);

			std::vector<index_type> index(free_index.size());

			// the outer loop 
			// 0 <= ptrs[i] < each_rowptr[i].size() - 1
			while (true) {
				// the internel loop 
				// each_rowptr[i][ptrs[i]] <= internel_ptrs[i] <  to each_rowptr[i][ptrs[i] + 1]
				
				std::vector<size_t> start_ptrs(nt);
				std::vector<size_t> end_ptrs(nt);
				for (size_t i = 0; i < nt; i++) {
					start_ptrs[i] = each_rowptr[i][ptrs[i]];
					end_ptrs[i] = each_rowptr[i][ptrs[i] + 1];
				}

				T entry = 0;

				multi_for(start_ptrs, end_ptrs, [&](const std::vector<size_t>& internel_ptrs) {
					bool is_zero = false;
					for (auto& a : contract_index) {
						auto num = tensors[a[0].first]->index(pindx[tindx[a[0].first]][internel_ptrs[a[0].first]])[a[0].second];
						for (size_t j = 1; j < a.size(); j++) {
							if (num != tensors[a[j].first]->index(pindx[tindx[a[j].first]][internel_ptrs[a[j].first]])[a[j].second]) {
								is_zero = true;
								break;
							}
						}
					}

					if (!is_zero) {
						T tmp = 1;
						for (auto j = 0; j < nt; j++)
							tmp = scalar_mul(tmp, tensors[j]->val(pindx[tindx[j]][internel_ptrs[j]]), F);
						entry = scalar_add(entry, tmp, F);
					}
					});

				if (entry != 0) {
					// compute the index
					for (size_t j = 0; j < free_index.size(); j++) {
						index[j] = tensors[free_index[j].first]->index(ptrs[free_index[j].first])[free_index[j].second];
					}

					Cs[id].push_back(index, entry);
				}

				bool is_end = false;
				for (int i = nt - 1; i > -2; i--) {
					if (i == -1) {
						is_end = true;
						break;
					}
					ptrs[i]++;
					if (i == 0) {
						if (ptrs[i] < ee)
							break;
						ptrs[i] = ss;
					}
					else {
						if (ptrs[i] < each_rowptr[i].size() - 1)
							break;
						ptrs[i] = 0;
					}
				}

				if (is_end)
					break;
			}
			};

		if (pool == nullptr) {
			method(0, each_rowptr[0].size() - 1);
			return Cs[0];
		}

		pool->detach_blocks(0, each_rowptr[0].size() - 1, method, nthread);
		pool->wait();

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
			index_type* colptr = C.data.colptr + nownnz * C.rank();
			s_copy(valptr, Cs[i].data.valptr, tmpnnz);
			s_copy(colptr, Cs[i].data.colptr, tmpnnz * C.rank());
			nownnz += tmpnnz;
		}

		return C;
	}

	template <typename index_type, typename T>
	sparse_tensor<index_type, T, SPARSE_COO> einstein_sum_2(
		const std::vector<sparse_tensor<index_type, T, SPARSE_COO>*> tensors,
		const std::vector<std::vector<size_t>> index_sets,
		const field_t F, thread_pool* pool = nullptr) {

		auto nt = tensors.size();
		if (nt != index_sets.size()) {
			std::cerr << "Error: The number of tensors does not match the number of index sets." << std::endl;
			exit(1);
		}
		for (size_t i = 1; i < nt; i++) {
			if (tensors[i]->rank() != index_sets[i].size()) {
				std::cerr << "Error: The rank of the tensor does not match the index set." << std::endl;
				exit(1);
			}
		}

		// now is the valid case

		// first case is zero
		for (size_t i = 0; i < nt; i++) {
			if (tensors[i]->nnz() == 0)
				return sparse_tensor<index_type, T, SPARSE_COO>();
		}

		// compute the summed index
		std::map<size_t, std::vector<std::pair<size_t, size_t>>> index_map;
		for (size_t i = 0; i < nt; i++) {
			for (size_t j = 0; j < tensors[i]->rank(); j++) {
				index_map[index_sets[i][j]].push_back(std::make_pair(i, j));
			}
		}

		std::vector<std::vector<std::pair<size_t, size_t>>> contract_index;
		std::vector<std::pair<size_t, size_t>> free_index;
		for (auto& it : index_map) {
			if (it.second.size() > 1)
				contract_index.push_back(it.second);
			else
				free_index.push_back((it.second)[0]);
		}
		std::stable_sort(free_index.begin(), free_index.end(),
			[&index_sets](const std::pair<size_t, size_t>& a, const std::pair<size_t, size_t>& b) {
				return index_sets[a.first][a.second] < index_sets[b.first][b.second];
			});

		std::vector<size_t> ptrs(nt, 0);
		std::vector<size_t> nnzs(nt);
		for (size_t i = 0; i < nt; i++)
			nnzs[i] = tensors[i]->nnz();

		// first find the first nonzero index
		std::vector<std::vector<size_t>> nonzero_index;
		while (true) {
			bool is_zero = false;
			for (auto& aa : contract_index) {
				// index in aa should be the same, otherwise zero
				auto num = tensors[aa[0].first]->index(ptrs[aa[0].first])[aa[0].second];
				for (size_t i = 1; i < aa.size(); i++) {
					auto num2 = tensors[aa[i].first]->index(ptrs[aa[i].first])[aa[i].second];
					if (num != num2) {
						is_zero = true;
						break;
					}
				}
				if (is_zero)
					break;
			}
			if (!is_zero)
				nonzero_index.push_back(ptrs);

			bool is_end = false;
			for (auto i = 0; i < nt + 1; i++) {
				if (i == nt) {
					is_end = true;
					break;
				}
				ptrs[i]++;
				if (ptrs[i] < nnzs[i])
					break;
				ptrs[i] = 0;
			}

			if (is_end)
				break;
		}

		auto compare_index = [&](size_t a, size_t b) {
			auto& ptr1 = nonzero_index[a];
			auto& ptr2 = nonzero_index[b];
			for (auto& aa : free_index) {
				if (tensors[aa.first]->index(ptr1[aa.first])[aa.second]
					< tensors[aa.first]->index(ptr2[aa.first])[aa.second])
					return -1;
				if (tensors[aa.first]->index(ptr1[aa.first])[aa.second]
					> tensors[aa.first]->index(ptr2[aa.first])[aa.second])
					return 1;
			}
			return 0;
			};

		// TODO: first permute each tensor, then the sort is not necessary
		// ...
		// avoid copy the index, just permute the index
		auto perm = perm_init(nonzero_index.size());
		std::sort(std::execution::par, perm.begin(), perm.end(),
			[&](size_t a, size_t b) {
				return compare_index(a, b) < 0;
			});

		// compute the rowptr
		std::vector<size_t> rowptr;
		rowptr.push_back(0);
		for (size_t i = 1; i < nonzero_index.size(); i++) {
			if (compare_index(perm[i], perm[rowptr.back()]) != 0)
				rowptr.push_back(i);
		}
		rowptr.push_back(nonzero_index.size());

		// now compute the result

		// first compute the dims of the result
		std::vector<size_t> dimsC;
		for (auto& aa : free_index)
			dimsC.push_back(tensors[aa.first]->dim(aa.second));

		sparse_tensor<index_type, T, SPARSE_COO> C(dimsC, rowptr.size());

		std::vector<index_type> index(free_index.size());
		for (size_t i = 0; i < rowptr.size() - 1; i++) {
			auto& ptr = nonzero_index[perm[rowptr[i]]];
			// first compute the index
			for (size_t j = 0; j < free_index.size(); j++) {
				index[j] = tensors[free_index[j].first]->index(ptr[free_index[j].first])[free_index[j].second];
			}

			T entry = 0;
			T tmp = 1;
			for (size_t k = rowptr[i]; k < rowptr[i + 1]; k++) {
				auto& ptr = nonzero_index[perm[k]];
				tmp = 1;
				for (size_t l = 0; l < nt; l++)
					tmp = scalar_mul(tmp, tensors[l]->val(ptr[l]), F);
				entry = scalar_add(entry, tmp, F);
			}

			if (entry != 0)
				C.push_back(index, entry);
		}

		return C;
	}

	// IO

	template <typename index_type, typename T> 
	sparse_tensor<index_type, size_t> COO_tensor_read(T& st, const field_t F) {
		if (!st.is_open())
			return sparse_tensor<index_type, size_t>();
		std::string strLine;

		bool is_size = true;

		using index_v = std::vector<index_type>;
		using index_p = index_type*;

		std::vector<size_t> dims;
		sparse_tensor<index_type, size_t> tensor;
		index_v index;

		while (getline(st, strLine)) {
			if (strLine[0] == '%')
				continue;

			auto tokens = sparse_rref::SplitString(strLine, " ");
			if (is_size) {
				for (size_t i = 0; i < tokens.size() - 1; i++)
					dims.push_back(std::stoull(tokens[i]));
				size_t nnz = std::stoull(tokens.back());
				tensor = sparse_tensor<index_type, size_t, SPARSE_COO>(dims, nnz);
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

	template <typename index_type, typename T> 
	sparse_tensor<index_type, size_t> COO_tensor_read_asyn(T& st, const field_t F) {
		if (!st.is_open())
			return sparse_tensor<index_type, size_t>();
		std::string strLine;

		bool is_size = true;

		using index_v = std::vector<index_type>;
		using index_p = index_type*;

		std::vector<size_t> dims;
		sparse_tensor<index_type, size_t> tensor;
		index_v index;

		LockFreeQueue<std::string> queue;
		std::atomic<ulong> nnz = 0;
		std::atomic<bool> wait_start = true;
		std::atomic<bool> is_done = false;

		std::thread t1([&]() {
			while (getline(st, strLine)) {
				if (strLine[0] == '%')
					continue;

				if (is_size) {
					auto tokens = sparse_rref::SplitString(strLine, " ");

					for (size_t i = 0; i < tokens.size() - 1; i++)
						dims.push_back(std::stoull(tokens[i]));
					size_t nnz = std::stoull(tokens.back());
					tensor = sparse_tensor<index_type, size_t, SPARSE_COO>(dims, nnz);
					index.reserve(dims.size());
					is_size = false;
				}
				else {
					queue.enqueue(strLine);
					nnz++;
					wait_start = false;
				}
			}
			is_done = true;
			}
		);

		std::thread t2([&]() {
			while (wait_start || nnz > 0 || !is_done) {
				if (nnz == 0)
					continue;

				auto strline = queue.dequeue();
				auto tokens = sparse_rref::SplitString(*strline, " ");

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

				nnz--;
			}
			}
		);

		t1.join();
		t2.join();

		return tensor;
	}
} // namespace sparse_rref

#endif