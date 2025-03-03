/*
	Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

	This file is part of Sparse_rref. The Sparse_rref is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

#ifndef SPARSE_VEC_H
#define SPARSE_VEC_H

#include "flint/nmod_vec.h"

#include "sparse_rref.h"
#include "scalar.h"

namespace sparse_rref {
	// sparse vector
	template <typename T> struct sparse_vec {
		slong* indices;
		T* entries;
		ulong _nnz;
		ulong _alloc;

		// using PE_pair = std::pair<slong&, T&>;

		sparse_vec() {
			indices = NULL;
			entries = NULL;
			_nnz = 0;
			_alloc = 0;
		}

		void clear() {
			if (indices)
				s_free(indices);
			if (entries) {
				for (ulong i = 0; i < _alloc; i++)
					entries[i].~T();
				s_free(entries);
			}
			_alloc = 0;
			_nnz = 0;
		}

		~sparse_vec() {
			clear();
		}

		void reserve(ulong n) {
			if (n == _alloc || n == 0)
				return;

			if (_alloc == 0) {
				indices = s_malloc<slong>(n);
				entries = s_malloc<T>(n);
				for (ulong i = 0; i < n; i++)
					new (entries + i) T();
				_alloc = n;
				return;
			}

			indices = s_realloc(indices, n);

			if (n < _alloc) {
				for (ulong i = n; i < _alloc; i++)
					entries[i].~T();
				entries = s_realloc<T>(entries, n);
			}
			else {
				entries = s_realloc<T>(entries, n);
				for (ulong i = _alloc; i < n; i++)
					new (entries + i) T();
			}

			_alloc = n;
		}

		void resize(ulong n) {
			_nnz = n;
		}

		inline void copy(const sparse_vec& l) {
			if (this == &l)
				return;
			if (_alloc < l._nnz)
				reserve(l._nnz);
			for (ulong i = 0; i < l._nnz; i++) {
				indices[i] = l.indices[i];
				entries[i] = l.entries[i];
			}
			_nnz = l._nnz;
		}

		sparse_vec(const sparse_vec& l) {
			copy(l);
		}

		size_t nnz() const {
			return _nnz;
		}

		sparse_vec(sparse_vec&& l) noexcept {
			indices = l.indices;
			entries = l.entries;
			_nnz = l._nnz;
			_alloc = l._alloc;
			l.indices = NULL;
			l.entries = NULL;
			l._nnz = 0;
			l._alloc = 0;
		}

		template <typename U = T, typename std::enable_if<Flint::IsOneOf<U, ulong, int_t>,int>::type = 0>
		operator sparse_vec<rat_t>() {
			sparse_vec<rat_t> result;
			result.reserve(_nnz);
			result.zero();
			for (size_t i = 0; i < _nnz; i++) {
				result.push_back(indices[i], rat_t(entries[i]));
			}
			return result;
		}

		template <typename U = T, typename std::enable_if<Flint::IsOneOf<U, ulong>, int>::type = 0>
		operator sparse_vec<int_t>() {
			sparse_vec<int_t> result;
			result.reserve(_nnz);
			result.zero();
			for (size_t i = 0; i < _nnz; i++) {
				result.push_back(indices[i], int_t(entries[i]));
			}
			return result;
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

			clear();
			indices = l.indices;
			entries = l.entries;
			_nnz = l._nnz;
			_alloc = l._alloc;
			l.indices = NULL;
			l.entries = NULL;
			l._nnz = 0;
			l._alloc = 0;
			return *this;
		}

		void push_back(const slong index, const T& val) {
			if (_nnz + 1 > _alloc)
				reserve((1 + _alloc) * 2); // +1 to avoid _alloc = 0
			indices[_nnz] = index;
			entries[_nnz] = val;
			_nnz++;
		}

		void push_back(const std::pair<slong, T>& a) {
			push_back(a.first, a.second);
		}

		slong& operator()(const ulong pos) {
			return indices[pos];
		}

		const slong& operator()(const ulong pos) const {
			return indices[pos];
		}

		void zero() {
			_nnz = 0;
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
			ulong new_nnz = 0;
			ulong i = 0;
			for (; i < _nnz; i++) {
				if (entries[i] != 0)
					break;
			}
			for (; i < _nnz; i++) {
				if (entries[i] == 0)
					continue;
				indices[new_nnz] = indices[i];
				entries[new_nnz] = entries[i];
				new_nnz++;
			}
			_nnz = new_nnz;
		}

		void sort_indices() {
			if (_nnz <= 1)
				return;

			std::vector<slong> perm(_nnz);
			for (size_t i = 0; i < _nnz; i++)
				perm[i] = i;

			std::sort(perm.begin(), perm.end(), [&](slong a, slong b) {
				return indices[a] < indices[b];
				});

			bool is_sorted = true;
			for (size_t i = 0; i < _nnz; i++) {
				if (perm[i] != i) {
					is_sorted = false;
					break;
				}
			}
			if (is_sorted)
				return;

			std::vector<T> tentries(_nnz);

			// apply permutation
			for (size_t i = 0; i < _nnz; i++) {
				tentries[i] = entries[perm[i]];
			}
			for (size_t i = 0; i < _nnz; i++) {
				entries[i] = tentries[i];
			}

			std::sort(indices, indices + _nnz);
		}

		void compress() {
			canonicalize();
			sort_indices();
		}
	};

	template <> struct sparse_vec<bool> {
		slong* indices;
		ulong _nnz;
		ulong _alloc;

		sparse_vec() {
			indices = NULL;
			_nnz = 0;
			_alloc = 0;
		}

		void clear() {
			if (indices)
				s_free(indices);
			_alloc = 0;
			_nnz = 0;
		}

		~sparse_vec() { clear(); }

		void reserve(ulong n) {
			if (n == _alloc)
				return;

			if (_alloc == 0) {
				indices = s_malloc<slong>(n);
				_alloc = n;
				return;
			}

			indices = s_realloc(indices, n);
			_alloc = n;
		}

		void resize(ulong n) { _nnz = n; }

		inline void copy(const sparse_vec& l) {
			if (this == &l)
				return;
			if (_alloc < l._nnz)
				reserve(l._nnz);
			for (ulong i = 0; i < l._nnz; i++) {
				indices[i] = l.indices[i];
			}
			_nnz = l._nnz;
		}

		sparse_vec(const sparse_vec& l) { copy(l); }
		size_t nnz() const { return _nnz; }

		sparse_vec(sparse_vec&& l) noexcept {
			indices = l.indices;
			_nnz = l._nnz;
			_alloc = l._alloc;
			l.indices = NULL;
			l._nnz = 0;
			l._alloc = 0;
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

			clear();
			indices = l.indices;
			_nnz = l._nnz;
			_alloc = l._alloc;
			l.indices = NULL;
			l._nnz = 0;
			l._alloc = 0;
			return *this;
		}

		void push_back(const slong index, const bool val = true) {
			if (_nnz + 1 > _alloc)
				reserve((1 + _alloc) * 2); // +1 to avoid _alloc = 0
			indices[_nnz] = index;
			_nnz++;
		}

		slong& operator()(const ulong pos) { return indices[pos]; }
		const slong& operator()(const ulong pos) const { return indices[pos]; }
		void zero() { _nnz = 0; }
		void sort_indices() { std::sort(indices, indices + _nnz); }
		void compress() { sort_indices(); }
	};

	template <typename T>
	inline T* sparse_vec_entry(sparse_vec<T>& vec, const slong index, const bool isbinary = true) {
		if (vec.nnz() == 0)
			return NULL;
		slong* ptr;
		if (isbinary)
			ptr = sparse_rref::binarysearch(vec.indices, vec.indices + vec.nnz(), index);
		else
			ptr = std::find(vec.indices, vec.indices + vec.nnz(), index);
		if (ptr == vec.indices + vec.nnz())
			return NULL;
		return vec.entries + (ptr - vec.indices);
	}

	using snmod_vec = sparse_vec<ulong>;
	using sfmpq_vec = sparse_vec<rat_t>;

	template <typename T>
	inline void sparse_vec_rescale(sparse_vec<T>& vec, const T scalar, const field_t F) {
		if constexpr (std::is_same_v<T, ulong>) {
			_nmod_vec_scalar_mul_nmod_shoup(vec.entries, vec.entries, vec.nnz(), scalar, F->mod);
		}
		else if constexpr (Flint::IsOneOf<T, int_t, rat_t>) {
			for (ulong i = 0; i < vec.nnz(); i++)
				vec.entries[i] *= scalar;
		}
	}

	static void snmod_vec_from_sfmpq(snmod_vec& vec, const sfmpq_vec& src, nmod_t p) {
		vec.reserve(src.nnz());
		vec.zero();
		for (size_t i = 0; i < src.nnz(); i++) {
			ulong num = src[i].num() % p;
			ulong den = src[i].den() % p;
			ulong val = nmod_div(num, den, p);
			vec.push_back(src(i), val);
		}
	}

	// we assume that vec and src are sorted, and the result is also sorted
	static int snmod_vec_add_mul(snmod_vec& vec, const snmod_vec& src,
		const ulong a, field_t F) {
		if (src.nnz() == 0)
			return 0;

		auto p = F->mod;

		if (vec.nnz() == 0) {
			vec = src;
			sparse_vec_rescale(vec, a, F);
		}

		ulong na = a;
		ulong na_pr = n_mulmod_precomp_shoup(na, p.n);

		ulong ptr1 = vec.nnz();
		ulong ptr2 = src.nnz();
		ulong ptr = vec.nnz() + src.nnz();

		if (vec._alloc < ptr)
			vec.reserve(ptr);
		vec.resize(ptr);

		while (ptr1 > 0 && ptr2 > 0) {
			if (vec(ptr1 - 1) == src(ptr2 - 1)) {
				ulong entry =
					_nmod_add(vec[ptr1 - 1],
						n_mulmod_shoup(na, src[ptr2 - 1], na_pr, p.n), p);
				if (entry != 0) {
					vec(ptr - 1) = vec(ptr1 - 1);
					vec[ptr - 1] = entry;
					ptr--;
				}
				ptr1--;
				ptr2--;
			}
			else if (vec(ptr1 - 1) < src(ptr2 - 1)) {
				vec(ptr - 1) = src(ptr2 - 1);
				vec[ptr - 1] = n_mulmod_shoup(na, src[ptr2 - 1], na_pr, p.n);
				ptr2--;
				ptr--;
			}
			else {
				vec(ptr - 1) = vec(ptr1 - 1);
				vec[ptr - 1] = vec[ptr1 - 1];
				ptr1--;
				ptr--;
			}
		}
		while (ptr2 > 0) {
			vec(ptr - 1) = src(ptr2 - 1);
			vec[ptr - 1] = n_mulmod_shoup(na, src[ptr2 - 1], na_pr, p.n);
			ptr2--;
			ptr--;
		}

		// if ptr1 > 0, and ptr > 0
		for (size_t i = ptr1; i < ptr; i++) {
			vec[i] = 0;
		}

		vec.canonicalize();
		if (vec._alloc > 4 * vec.nnz())
			vec.reserve(2 * vec.nnz());

		return 0;
	}

	template <bool dir>
	int sfmpq_vec_addsub_mul(sfmpq_vec& vec, const sfmpq_vec& src, const rat_t& a) {
		if (src.nnz() == 0)
			return 0;

		if (vec.nnz() == 0) {
			vec = src;
			sparse_vec_rescale(vec, a, field_t{});
		}

		rat_t na, entry;
		if constexpr (dir) {
			na = a;
		}
		else {
			na = -a;
		}

		ulong ptr1 = vec.nnz();
		ulong ptr2 = src.nnz();
		ulong ptr = vec.nnz() + src.nnz();

		if (vec._alloc < ptr)
			vec.reserve(ptr);
		vec.resize(ptr);

		while (ptr1 > 0 && ptr2 > 0) {
			if (vec(ptr1 - 1) == src(ptr2 - 1)) {
				entry = na * src[ptr2 - 1];
				entry += vec[ptr1 - 1];
				if (entry != 0) {
					vec(ptr - 1) = vec(ptr1 - 1);
					vec[ptr - 1] = entry;
					ptr--;
				}
				ptr1--;
				ptr2--;
			}
			else if (vec(ptr1 - 1) < src(ptr2 - 1)) {
				entry = na * src[ptr2 - 1];
				vec(ptr - 1) = src(ptr2 - 1);
				vec[ptr - 1] = entry;
				ptr2--;
				ptr--;
			}
			else {
				vec(ptr - 1) = vec(ptr1 - 1);
				vec[ptr - 1] = vec[ptr1 - 1];
				ptr1--;
				ptr--;
			}
		}
		while (ptr2 > 0) {
			entry = na * src[ptr2 - 1];
			vec(ptr - 1) = src(ptr2 - 1);
			vec[ptr - 1] = entry;
			ptr2--;
			ptr--;
		}

		// if ptr1 > 0, and ptr > 0
		for (size_t i = ptr1; i < ptr; i++) {
			vec[i] = 0;
		}

		vec.canonicalize();
		if (vec._alloc > 4 * vec.nnz())
			vec.reserve(2 * vec.nnz());

		return 0;
	}

	static inline int sfmpq_vec_add_mul(sfmpq_vec& vec, const sfmpq_vec& src, const rat_t& a) {
		return sfmpq_vec_addsub_mul<true>(vec, src, a);
	}

	static inline int sfmpq_vec_sub_mul(sfmpq_vec& vec, const sfmpq_vec& src, const rat_t& a) {
		return sfmpq_vec_addsub_mul<false>(vec, src, a);
	}

	static inline int snmod_vec_sub_mul(snmod_vec& vec, const snmod_vec& src, const ulong a, field_t F) {
		return snmod_vec_add_mul(vec, src, F->mod.n - a, F);
	}

	static inline int sparse_vec_add(snmod_vec& vec, const snmod_vec& src, field_t F) {
		return snmod_vec_add_mul(vec, src, 1, F);
	}

	static inline int sparse_vec_sub(snmod_vec& vec, const snmod_vec& src, field_t F) {
		return snmod_vec_add_mul(vec, src, F->mod.n - 1, F);
	}

	static inline int sparse_vec_sub_mul(snmod_vec& vec, const snmod_vec& src, const ulong a, field_t F) {
		return snmod_vec_sub_mul(vec, src, a, F);
	}


	// dot product
	// return true if the result is zero
	template <typename T>
	T sparse_vec_dot(const sparse_vec<T> v1, const sparse_vec<T> v2, field_t F) {
		if (v1->nnz == 0 || v2->nnz == 0) {
			return T(0);
		}
		slong ptr1 = 0, ptr2 = 0;
		T result = 0;
		while (ptr1 < v1.nnz() && ptr2 < v2.nnz()) {
			if (v1(ptr1) == v2(ptr2)) {
				result = scalar_add(result, scalar_mul(v1[ptr1], v2[ptr2], F), F);
				ptr1++;
				ptr2++;
			}
			else if (v1(ptr1) < v2(ptr2))
				ptr1++;
			else
				ptr2++;
		}
		return result;
	}

	static std::pair<size_t, char*> snmod_vec_to_binary(const sparse_vec<ulong>& vec) {
		constexpr auto ratio = sizeof(ulong) / sizeof(char);
		auto nnz = vec.nnz();
		char* buffer = s_malloc<char>((1 + 2 * nnz) * ratio);
		std::memcpy(buffer, &nnz, sizeof(ulong));
		std::memcpy(buffer + ratio, vec.indices, nnz * sizeof(ulong));
		std::memcpy(buffer + (1 + nnz) * ratio, vec.entries, nnz * sizeof(ulong));
		return std::make_pair((1 + 2 * nnz) * ratio, buffer);
	}

	static void snmod_vec_from_binary(sparse_vec<ulong>& vec, const char* buffer) {
		constexpr auto ratio = sizeof(ulong) / sizeof(char);
		ulong nnz;
		std::memcpy(&nnz, buffer, sizeof(ulong));
		vec.reserve(nnz);
		vec.resize(nnz);
		std::memcpy(vec.indices, buffer + ratio, nnz * sizeof(ulong));
		std::memcpy(vec.entries, buffer + (1 + nnz) * ratio, nnz * sizeof(ulong));
	}

	// debug only, not used to the large vector
	template <typename T> void print_vec_info(const sparse_vec<T>& vec) {
		std::cout << "-------------------" << std::endl;
		std::cout << "nnz: " << vec.nnz() << std::endl;
		std::cout << "indices: ";
		for (size_t i = 0; i < vec.nnz(); i++)
			std::cout << vec(i) << " ";
		std::cout << "\nentries: ";
		for (size_t i = 0; i < vec.nnz(); i++)
			std::cout << scalar_to_str(vec[i]) << " ";
		std::cout << std::endl;
	}

}
#endif
