/*
	Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

	This file is part of Sparse_rref. The Sparse_rref is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

#ifndef SPARSE_VEC_H
#define SPARSE_VEC_H

#include "flint/nmod_vec.h"

#include "scalar.h"

namespace sparse_rref {

	template <typename T> struct sparse_vec_struct {
		ulong nnz = 0;
		ulong alloc = 0;
		slong* indices = NULL;
		T* entries = NULL;
	};

	// entries is useless for bool
	template <>
	struct sparse_vec_struct<bool> {
		ulong nnz = 0;
		ulong alloc = 0;
		slong* indices = NULL;
	};

	template <typename T> using sparse_vec_t = struct sparse_vec_struct<T>[1];

	typedef sparse_vec_t<ulong> snmod_vec_t;
	typedef sparse_vec_t<rat_t> sfmpq_vec_t;

	// sparse_vec

	// memory management
	template <typename T>
	void sparse_vec_realloc(sparse_vec_t<T> vec, ulong alloc) {
		if (alloc == vec->alloc)
			return;
		// so sparse_vec_realloc(vec,vec->alloc) is useless
		ulong old_alloc = vec->alloc;
		vec->alloc = alloc;
		if (vec->alloc > old_alloc) {
			// enlarge: init later
			vec->indices = s_realloc(vec->indices, vec->alloc);
			if constexpr (!std::is_same_v<T, bool>) {
				vec->entries = s_realloc(vec->entries, vec->alloc);
			}
			if constexpr (Flint::IsOneOf<T, int_t, rat_t>) {
				for (ulong i = old_alloc; i < vec->alloc; i++) {
					vec->entries[i].init();
					vec->entries[i] = 0;
				}
			}
		}
		else {
			// shrink: clear first
			if constexpr (Flint::IsOneOf<T, int_t, rat_t>) {
				for (ulong i = vec->alloc; i < old_alloc; i++)
					vec->entries[i].clear();
			}
			vec->indices = s_realloc(vec->indices, vec->alloc);
			if constexpr (!std::is_same_v<T, bool>) {
				vec->entries = s_realloc(vec->entries, vec->alloc);
			}
		}
	}


	// #define sparse_vec_entry_pointer(vec, index) ((vec)->entries + (index))

	template <typename T>
	inline T* sparse_vec_entry_pointer(const sparse_vec_t<T> vec, const slong index) {
		return vec->entries + index;
	}

	// alloc at least 1 to make sure that indices and entries are not NULL
	template <typename T>
	inline void sparse_vec_init(sparse_vec_t<T> vec, ulong alloc = 1) {
		vec->nnz = 0;
		vec->alloc = alloc;
		vec->indices = s_malloc<slong>(vec->alloc);
		if constexpr (std::is_same_v<T, bool>) {
			return;
		}
		if constexpr (!std::is_same_v<T, bool>) {
			vec->entries = s_malloc<T>(alloc);
		}
		if constexpr (std::is_same_v<T, fmpq>) {
			for (ulong i = 0; i < alloc; i++)
				fmpq_init(vec->entries + i);
		}
	}

	// just set vec to zero vector
#define sparse_vec_zero(__vec) ((__vec)->nnz = 0)

// set zero and clear memory
	template <typename T> inline void sparse_vec_clear(sparse_vec_t<T> vec) {
		vec->nnz = 0;
		s_free(vec->indices);
		vec->indices = NULL;
		if constexpr (std::is_same_v<T, fmpq>) {
			for (auto i = 0; i < vec->alloc; i++)
				fmpq_clear(vec->entries + i);
		}
		if constexpr (!std::is_same_v<T, bool>) {
			s_free(vec->entries);
			vec->entries = NULL;
		}
		vec->alloc = 0;
	}

	template <typename T>
	inline T* sparse_vec_entry(sparse_vec_t<T> vec, slong index, const bool isbinary = true) {
		if (vec->nnz == 0 || index < vec->indices[0] || index > vec->indices[vec->nnz - 1])
			return NULL;
		slong* ptr;
		if (isbinary)
			ptr = sparse_rref::binarysearch(vec->indices, vec->indices + vec->nnz, index);
		else
			ptr = std::find(vec->indices, vec->indices + vec->nnz, index);
		if (ptr == vec->indices + vec->nnz)
			return NULL;
		return sparse_vec_entry_pointer(vec, ptr - vec->indices);
	}

	// constructors
	template <typename T>
	inline void sparse_vec_set(sparse_vec_t<T> vec, const sparse_vec_t<T> src) {
		vec->nnz = src->nnz;
		if (vec->alloc < src->nnz)
			sparse_vec_realloc(vec, src->nnz);

		for (auto i = 0; i < src->nnz; i++) {
			vec->indices[i] = src->indices[i];
			if constexpr (!std::is_same_v<T, bool>) {
				*sparse_vec_entry_pointer(vec, i) = *sparse_vec_entry_pointer(src, i);
			}
		}
	}

	template <typename T>
	inline void sparse_vec_swap(sparse_vec_t<T> vec, sparse_vec_t<T> src) {
		std::swap(src->indices, vec->indices);
		std::swap(src->entries, vec->entries);
		std::swap(src->nnz, vec->nnz);
		std::swap(src->alloc, vec->alloc);
	}

	// this raw version assumes that the vec[index] = 0
	// equivalent to push_back
	template <typename T, uint8_t scale = 2>
	void _sparse_vec_set_entry(sparse_vec_t<T> vec, slong index, const T val) {
		if (vec->nnz == vec->alloc)
			sparse_vec_realloc(vec, scale * vec->alloc);
		vec->indices[vec->nnz] = index;
		if constexpr (!std::is_same_v<T, bool>) {
			*sparse_vec_entry_pointer(vec, vec->nnz) = val;
		}
		vec->nnz++;
	}

	template <uint8_t scale = 2>
	void _sparse_vec_set_entry(sparse_vec_t<bool> vec, slong index) {
		if (vec->nnz == vec->alloc)
			sparse_vec_realloc(vec, scale * vec->alloc);
		vec->indices[vec->nnz] = index;
		vec->nnz++;
	}

	template <typename T, typename S>
	inline void sparse_vec_set_entry(sparse_vec_t<T> vec, slong index, const T* val,
		bool isbinary = false) {
		// if val = 0, here we only set it as zero, but not remove it
		if constexpr (std::is_same_v<T, bool>) {
			return;
		}
		else {
			T* entry = sparse_vec_entry(vec, index, isbinary);
			if (entry != NULL)
				scalar_set(entry, val);
			_sparse_vec_set_entry(vec, index, val);
		}
	}

	// TODO: Implement a better sorting algorithm (sort only once)
	template <typename T>
	void sparse_vec_sort_indices(sparse_vec_t<T> vec) {
		if (vec->nnz <= 1)
			return;

		if constexpr (std::is_same_v<T, bool>) {
			std::sort(vec->indices, vec->indices + vec->nnz);
			return;
		}
		else {
			std::vector<slong> perm(vec->nnz);
			for (size_t i = 0; i < vec->nnz; i++)
				perm[i] = i;

			std::sort(perm.begin(), perm.end(), [&vec](slong a, slong b) {
				return vec->indices[a] < vec->indices[b];
				});

			bool is_sorted = true;
			for (size_t i = 0; i < vec->nnz; i++) {
				if (perm[i] != i) {
					is_sorted = false;
					break;
				}
			}
			if (is_sorted)
				return;

			T* entries = s_malloc<T>(vec->nnz);
			if constexpr (std::is_same_v<T, fmpq>) {
				for (size_t i = 0; i < vec->nnz; i++)
					scalar_init(entries + i);
			}

			// apply permutation
			for (size_t i = 0; i < vec->nnz; i++) {
				entries[i] = *sparse_vec_entry_pointer(vec, perm[i]);
			}
			for (size_t i = 0; i < vec->nnz; i++) {
				vec->entries[i] = entries[i];
			}

			if constexpr (std::is_same_v<T, fmpq>) {
				for (size_t i = 0; i < vec->nnz; i++)
					scalar_clear(entries + i);
			}
			s_free(entries);
			std::sort(vec->indices, vec->indices + vec->nnz);
		}
	}

	template <typename T>
	void sparse_vec_canonicalize(sparse_vec_t<T> vec) {
		if constexpr (std::is_same_v<T, bool>) { return; }

		ulong new_nnz = 0;
		ulong i = 0;
		for (; i < vec->nnz; i++) {
			if (*sparse_vec_entry_pointer(vec, i) != 0)
				break;
		}
		for (; i < vec->nnz; i++) {
			if (*sparse_vec_entry_pointer(vec, i) == 0)
				continue;
			vec->indices[new_nnz] = vec->indices[i];
			*sparse_vec_entry_pointer(vec, new_nnz) = *sparse_vec_entry_pointer(vec, i);
			new_nnz++;
		}
		vec->nnz = new_nnz;
	}

	template <typename T>
	inline void sparse_vec_compress(sparse_vec_t<T> vec) {
		sparse_vec_canonicalize(vec);
		sparse_vec_realloc(vec, vec->nnz);
	}

	// arithmetic operations

	// p should less than 2^(FLINT_BITS-1) (2^63(2^31) on 64(32)-bit machine)
	// scalar and all vec->entries[i] should less than p
	static inline void sparse_vec_rescale(snmod_vec_t vec, const ulong scalar, const field_t F) {
		_nmod_vec_scalar_mul_nmod_shoup(vec->entries, vec->entries, vec->nnz,
			scalar, F->mod);
	}

	static inline void sparse_vec_rescale(sfmpq_vec_t vec, const rat_t scalar, const field_t F = NULL) {
		for (ulong i = 0; i < vec->nnz; i++)
			vec->entries[i] *= scalar;
	}

	// we assume that vec and src are sorted, and the result is also sorted
	static int snmod_vec_add_mul(snmod_vec_t vec, const snmod_vec_t src,
		const ulong a, field_t F) {
		if (src->nnz == 0)
			return 0;

		auto p = F->mod;

		if (vec->nnz == 0) {
			sparse_vec_set(vec, src);
			sparse_vec_rescale(vec, a, F);
		}

		ulong na = a;
		ulong na_pr = n_mulmod_precomp_shoup(na, p.n);

		if (vec->nnz + src->nnz > vec->alloc)
			sparse_vec_realloc(vec, vec->nnz + src->nnz);

		ulong ptr1 = vec->nnz;
		ulong ptr2 = src->nnz;
		ulong ptr = vec->nnz + src->nnz;
		while (ptr1 > 0 && ptr2 > 0) {
			if (vec->indices[ptr1 - 1] == src->indices[ptr2 - 1]) {
				ulong entry =
					_nmod_add(vec->entries[ptr1 - 1],
						n_mulmod_shoup(na, src->entries[ptr2 - 1], na_pr, p.n), p);
				if (entry != 0) {
					vec->indices[ptr - 1] = vec->indices[ptr1 - 1];
					vec->entries[ptr - 1] = entry;
					ptr--;
				}
				ptr1--;
				ptr2--;
			}
			else if (vec->indices[ptr1 - 1] < src->indices[ptr2 - 1]) {
				vec->indices[ptr - 1] = src->indices[ptr2 - 1];
				vec->entries[ptr - 1] = n_mulmod_shoup(na, src->entries[ptr2 - 1], na_pr, p.n);
				ptr2--;
				ptr--;
			}
			else {
				vec->indices[ptr - 1] = vec->indices[ptr1 - 1];
				vec->entries[ptr - 1] = vec->entries[ptr1 - 1];
				ptr1--;
				ptr--;
			}
		}
		while (ptr2 > 0) {
			vec->indices[ptr - 1] = src->indices[ptr2 - 1];
			vec->entries[ptr - 1] = n_mulmod_shoup(na, src->entries[ptr2 - 1], na_pr, p.n);
			ptr2--;
			ptr--;
		}

		// if ptr1 > 0, and ptr > 0
		for (size_t i = ptr1; i < ptr; i++) {
			vec->entries[i] = 0;
		}

		vec->nnz += src->nnz;
		sparse_vec_canonicalize(vec);
		if (vec->alloc > 4 * vec->nnz)
			sparse_vec_realloc(vec, 2 * vec->nnz);

		return 0;
	}

	template <bool dir>
	int sfmpq_vec_addsub_mul(sfmpq_vec_t vec, const sfmpq_vec_t src, const rat_t a) {
		if (src->nnz == 0)
			return 0;

		if (vec->nnz == 0) {
			sparse_vec_set(vec, src);
			sparse_vec_rescale(vec, a);
		}

		rat_t na, entry;
		if constexpr (dir) {
			na = a;
		}
		else {
			na = -a;
		}

		if (vec->nnz + src->nnz > vec->alloc)
			sparse_vec_realloc(vec, vec->nnz + src->nnz);

		ulong ptr1 = vec->nnz;
		ulong ptr2 = src->nnz;
		ulong ptr = vec->nnz + src->nnz;
		while (ptr1 > 0 && ptr2 > 0) {
			if (vec->indices[ptr1 - 1] == src->indices[ptr2 - 1]) {
				entry = na * src->entries[ptr2 - 1];
				entry += vec->entries[ptr1 - 1];
				if (entry != 0) {
					vec->indices[ptr - 1] = vec->indices[ptr1 - 1];
					vec->entries[ptr - 1] = entry;
					ptr--;
				}
				ptr1--;
				ptr2--;
			}
			else if (vec->indices[ptr1 - 1] < src->indices[ptr2 - 1]) {
				entry = na * src->entries[ptr2 - 1];
				vec->indices[ptr - 1] = src->indices[ptr2 - 1];
				vec->entries[ptr - 1] = entry;
				ptr2--;
				ptr--;
			}
			else {
				vec->indices[ptr - 1] = vec->indices[ptr1 - 1];
				vec->entries[ptr - 1] = vec->entries[ptr1 - 1];
				ptr1--;
				ptr--;
			}
		}
		while (ptr2 > 0) {
			entry = na * src->entries[ptr2 - 1];
			vec->indices[ptr - 1] = src->indices[ptr2 - 1];
			vec->entries[ptr - 1] = entry;
			ptr2--;
			ptr--;
		}

		// if ptr1 > 0, and ptr > 0
		for (size_t i = ptr1; i < ptr; i++) {
			vec->entries[i] = 0;
		}

		vec->nnz += src->nnz;
		sparse_vec_canonicalize(vec);
		if (vec->alloc > 4 * vec->nnz)
			sparse_vec_realloc(vec, 2 * vec->nnz);

		return 0;
	}

	static inline int sfmpq_vec_add_mul(sfmpq_vec_t vec, const sfmpq_vec_t src, const fmpq_t a) {
		return sfmpq_vec_addsub_mul<true>(vec, src, a);
	}

	static inline int sfmpq_vec_sub_mul(sfmpq_vec_t vec, const sfmpq_vec_t src, const fmpq_t a) {
		return sfmpq_vec_addsub_mul<false>(vec, src, a);
	}

	static inline int snmod_vec_sub_mul(snmod_vec_t vec, const snmod_vec_t src, const ulong a, field_t F) {
		return snmod_vec_add_mul(vec, src, F->mod.n - a, F);
	}

	static inline int sparse_vec_add(snmod_vec_t vec, const snmod_vec_t src, field_t F) {
		return snmod_vec_add_mul(vec, src, (ulong)1, F);
	}

	static inline int sparse_vec_sub(snmod_vec_t vec, const snmod_vec_t src, field_t F) {
		return snmod_vec_add_mul(vec, src, F->mod.n - 1, F);
	}

	static inline int sparse_vec_sub_mul(snmod_vec_t vec, const snmod_vec_t src, const ulong* a, field_t F) {
		return snmod_vec_sub_mul(vec, src, *a, F);
	}

	static inline int sparse_vec_sub_mul(sfmpq_vec_t vec, const sfmpq_vec_t src, const fmpq_t a, field_t F = NULL) {
		return sfmpq_vec_sub_mul(vec, src, a);
	}

	static void snmod_vec_from_sfmpq(snmod_vec_t vec, const sfmpq_vec_t src, nmod_t p) {
		sparse_vec_realloc(vec, src->nnz);
		vec->alloc = src->nnz;
		vec->nnz = 0;
		for (size_t i = 0; i < src->nnz; i++) {
			ulong num = src->entries[i].num() % p;
			ulong den = src->entries[i].den() % p;
			ulong val = nmod_div(num, den, p);
			_sparse_vec_set_entry(vec, src->indices[i], val);
		}
	}

	// dot product
	// return true if the result is zero
	template <typename T>
	bool sparse_vec_dot(T& result, const sparse_vec_t<T> v1, const sparse_vec_t<T> v2, field_t F) {
		if (v1->nnz == 0 || v2->nnz == 0) {
			scalar_zero(result);
			return 0;
		}
		slong ptr1 = 0, ptr2 = 0;
		T tmp;
		while (ptr1 < v1->nnz && ptr2 < v2->nnz) {
			if (v1->indices[ptr1] == v2->indices[ptr2]) {
				scalar_mul(tmp, v1->entries + ptr1, v2->entries + ptr2, F);
				scalar_add(result, result, tmp, F);
				ptr1++;
				ptr2++;
			}
			else if (v1->indices[ptr1] < v2->indices[ptr2])
				ptr1++;
			else
				ptr2++;
		}
		return scalar_is_zero(result);
	}

	static std::pair<size_t, char*> snmod_vec_to_binary(sparse_vec_t<ulong> vec) {
		auto ratio = sizeof(ulong) / sizeof(char);
		char* buffer = s_malloc<char>((1 + 2 * vec->nnz) * ratio);
		std::memcpy(buffer, &(vec->nnz), sizeof(ulong));
		std::memcpy(buffer + ratio, vec->indices, vec->nnz * sizeof(ulong));
		std::memcpy(buffer + (1 + vec->nnz) * ratio, vec->entries, vec->nnz * sizeof(ulong));
		return std::make_pair((1 + 2 * vec->nnz) * ratio, buffer);
	}

	static void snmod_vec_from_binary(sparse_vec_t<ulong> vec, const char* buffer) {
		auto ratio = sizeof(ulong) / sizeof(char);
		std::memcpy(&(vec->nnz), buffer, sizeof(ulong));
		sparse_vec_realloc(vec, vec->nnz);
		std::memcpy(vec->indices, buffer + ratio, vec->nnz * sizeof(ulong));
		std::memcpy(vec->entries, buffer + (1 + vec->nnz) * ratio, vec->nnz * sizeof(ulong));
	}

	// debug only, not used to the large vector
	template <typename T> void print_vec_info(const sparse_vec_t<T> vec) {
		std::cout << "-------------------" << std::endl;
		std::cout << "nnz: " << vec->nnz << std::endl;
		std::cout << "alloc: " << vec->alloc << std::endl;
		std::cout << "indices: ";
		for (size_t i = 0; i < vec->nnz; i++)
			std::cout << vec->indices[i] << " ";
		std::cout << "\nentries: ";
		for (size_t i = 0; i < vec->nnz; i++)
			std::cout << scalar_to_str(vec->entries + i) << " ";
		std::cout << std::endl;
	}

	// sparse vector
	template <typename T> struct sparse_vec {
		std::vector<slong> indices;
		std::vector<T> entries; // entries is useless for bool

		// using PE_pair = std::pair<slong&, T&>;

		sparse_vec() {};
		~sparse_vec() {}

		void clear() {
			indices.clear();
			entries.clear();
		}

		void reserve(ulong n) {
			indices.reserve(n);
			// entries is useless for bool
			if constexpr (!std::is_same_v<T, bool>) {
				entries.reserve(n);
			}
		}

		sparse_vec(ulong n) { reserve(n); }

		inline void copy(const sparse_vec& src) {
			indices = src.indices;
			entries = src.entries;
		}

		sparse_vec(const sparse_vec& l) {
			copy(l);
		}

		size_t nnz() const {
			return indices.size();
		}

		sparse_vec(sparse_vec&& l) noexcept {
			indices = l.indices;
			entries = l.entries;
		}

		sparse_vec& operator=(const sparse_vec& l) {
			if (this == &l)
				return *this;

			copy(l);
			return *this;
		}

		sparse_vec& operator=(sparse_vec&& l) noexcept {
			if (this == &l)
				return *this;

			indices = l.indices;
			entries = l.entries;
		}

		void push_back(const ulong index, const T val) {
			indices.push_back(index);
			if constexpr (!std::is_same_v<T, bool>) {
				entries.push_back(val);
			}
		}

		void push_back(const std::pair<ulong, T> a) {
			indices.push_back(a.first);
			if constexpr (!std::is_same_v<T, bool>) {
				entries.push_back(a.second);
			}
		}

		slong& operator()(const ulong pos) {
			return indices[pos];
		}

		const T& operator()(const ulong pos) const {
			return entries[pos];
		}

		T& operator[](const ulong pos) {
			return entries[pos];
		}

		const T& operator[](const ulong pos) const {
			return entries[pos];
		}

		std::pair<slong&, T&> at(ulong pos) {
			return std::make_pair(indices[pos], entries[pos]);
		}

		const std::pair<slong&, T&> at(ulong pos) const {
			return std::make_pair(indices[pos], entries[pos]);
		}

		void canonicalize() {
			if constexpr (std::is_same_v<T, bool>) { return; }
			ulong new_nnz = 0;
			ulong nnz = indices.size();
			ulong i = 0;
			for (; i < nnz; i++) {
				if (entries[i] != 0)
					break;
			}
			for (; i < nnz; i++) {
				if (entries[i] == 0)
					continue;
				indices[new_nnz] = indices[i];
				entries[new_nnz] = entries[i];
				new_nnz++;
			}
			indices.resize(new_nnz);
			entries.resize(new_nnz);
		}

		void sort_indices() {
			ulong nnz = indices.size();
			if (nnz <= 1)
				return;

			if constexpr (std::is_same_v<T, bool>) {
				std::sort(indices.begin(), indices.end());
				return;
			}
			else {
				std::vector<slong> perm(nnz);
				for (size_t i = 0; i < nnz; i++)
					perm[i] = i;

				std::sort(perm.begin(), perm.end(), [&](slong a, slong b) {
					return indices[a] < indices[b];
					});

				bool is_sorted = true;
				for (size_t i = 0; i < nnz; i++) {
					if (perm[i] != i) {
						is_sorted = false;
						break;
					}
				}
				if (is_sorted)
					return;

				std::vector<T> tentries(nnz);

				// apply permutation
				for (size_t i = 0; i < nnz; i++) {
					tentries[i] = entries[perm[i]];
				}
				
				entries = tentries;

				std::sort(indices.begin(), indices.end());
			}
		}

		void compress() {
			canonicalize();
			sort_indices();
		}
	};

	using snmod_vec = sparse_vec<ulong>;
	using sfmpq_vec = sparse_vec<fmpq>;

	template <typename T>
	inline void _sparse_vec_rescale(sparse_vec<T>& vec, const T scalar, const field_t F) {
		if constexpr (std::is_same_v<T, ulong>) {
			_nmod_vec_scalar_mul_nmod_shoup(vec.entries, vec.entries, vec.nnz,
				scalar, F->mod);
		}
		else if constexpr (Flint::IsOneOf<T, int_t, rat_t>) {
			for (ulong i = 0; i < vec.nnz(); i++)
				vec.entries[i] *= scalar;
		}
	}
}
#endif
