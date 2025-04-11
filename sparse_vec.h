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
	template <typename index_type, typename T> struct sparse_vec {
		index_type* indices = NULL;
		T* entries = NULL;
		size_t _nnz = 0;
		size_t _alloc = 0;

		sparse_vec() {
			indices = NULL;
			entries = NULL;
			_nnz = 0;
			_alloc = 0;
		}

		void clear() {
			if (_alloc == 0)
				return;
			s_free(indices);
			indices = NULL;
			for (size_t i = 0; i < _alloc; i++)
				entries[i].~T();
			s_free(entries);
			entries = NULL;
			_alloc = 0;
			_nnz = 0;
		}

		~sparse_vec() {
			clear();
		}

		void reserve(size_t n) {
			if (n == _alloc || n == 0)
				return;

			if (_alloc == 0) {
				indices = s_malloc<index_type>(n);
				entries = s_malloc<T>(n);
				for (size_t i = 0; i < n; i++)
					new (entries + i) T();
				_alloc = n;
				return;
			}

			indices = s_realloc(indices, n);

			if (n < _alloc) {
				for (size_t i = n; i < _alloc; i++)
					entries[i].~T();
				entries = s_realloc<T>(entries, n);
			}
			else {
				entries = s_realloc<T>(entries, n);
				for (size_t i = _alloc; i < n; i++)
					new (entries + i) T();
			}

			_alloc = n;
		}

		void resize(size_t n) {
			_nnz = n;
		}

		inline void copy(const sparse_vec& l) {
			if (this == &l)
				return;
			if (_alloc < l._nnz)
				reserve(l._nnz);
			for (size_t i = 0; i < l._nnz; i++) {
				indices[i] = l.indices[i];
				entries[i] = l.entries[i];
			}
			_nnz = l._nnz;
		}

		sparse_vec(const sparse_vec& l) {
			copy(l);
		}

		size_t nnz() const { return _nnz; }
		size_t alloc() const { return _alloc; }

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

		void push_back(const index_type index, const T& val) {
			if (_nnz + 1 > _alloc)
				reserve((1 + _alloc) * 2); // +1 to avoid _alloc = 0
			indices[_nnz] = index;
			entries[_nnz] = val;
			_nnz++;
		}

		inline void zero() { _nnz = 0; }
		inline void push_back(const std::pair<index_type, T>& a) { push_back(a.first, a.second); }
		index_type& operator()(const size_t pos) { return indices[pos]; }
		const index_type& operator()(const size_t pos) const { return indices[pos]; }
		T& operator[](const size_t pos) { return entries[pos]; }
		const T& operator[](const size_t pos) const { return entries[pos]; }

		std::pair<index_type&, T&> at(index_type pos) { return std::make_pair(indices[pos], entries[pos]); }
		const std::pair<index_type&, T&> at(index_type pos) const { return std::make_pair(indices[pos], entries[pos]); }

		template <typename U = T> requires Flint::IsOneOf<U, unsigned long, int_t>
		operator sparse_vec<index_type, rat_t>() {
			sparse_vec<index_type, rat_t> result;
			result.reserve(_nnz);
			result.zero();
			for (size_t i = 0; i < _nnz; i++) {
				result.push_back(indices[i], rat_t(entries[i]));
			}
			return result;
		}

		template <typename U = T> requires Flint::IsOneOf<U, size_t>
		operator sparse_vec<index_type, int_t>() {
			sparse_vec<index_type, int_t> result;
			result.reserve(_nnz);
			result.zero();
			for (size_t i = 0; i < _nnz; i++) {
				result.push_back(indices[i], int_t(entries[i]));
			}
			return result;
		}

		void canonicalize() {
			size_t new_nnz = 0;
			for (size_t i = 0; i < _nnz && new_nnz < _nnz; i++) {
				if (entries[i] != 0) {
					if (new_nnz != i) {
						indices[new_nnz] = indices[i];
						entries[new_nnz] = entries[i];
					}
					new_nnz++;
				}
			}
			_nnz = new_nnz;
		}

		void sort_indices() {
			if (_nnz <= 1 || std::is_sorted(indices, indices + _nnz))
				return;

			auto perm = perm_init(_nnz);
			std::sort(perm.begin(), perm.end(), [&](index_type a, index_type b) {
				return indices[a] < indices[b];
				});

			// apply permutation in-place using cycle swapping
			auto apply_permutation = [&](auto& data) {
				std::vector<bool> visited(_nnz, false);
				for (size_t i = 0; i < _nnz; ++i) {
					if (visited[i] || perm[i] == i) continue;
					size_t j = i;
					auto tmp = std::move(data[i]);
					while (!visited[j]) {
						visited[j] = true;
						size_t k = perm[j];
						if (k == i) {
							data[j] = std::move(tmp);
							break;
						}
						data[j] = std::move(data[k]);
						j = k;
					}
				}
				};

			apply_permutation(indices);
			apply_permutation(entries);
		}

		void compress() {
			canonicalize();
			sort_indices();
			reserve(_nnz);
		}
	};

	template <typename index_type> struct sparse_vec<index_type, bool> {
		index_type* indices = NULL;
		size_t _nnz = 0;
		size_t _alloc = 0;

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

		void reserve(size_t n) {
			if (n == _alloc)
				return;

			if (_alloc == 0) {
				indices = s_malloc<index_type>(n);
				_alloc = n;
				return;
			}

			indices = s_realloc(indices, n);
			_alloc = n;
		}

		void resize(size_t n) { _nnz = n; }

		inline void copy(const sparse_vec& l) {
			if (this == &l)
				return;
			if (_alloc < l._nnz)
				reserve(l._nnz);
			for (size_t i = 0; i < l._nnz; i++) {
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

		void push_back(const index_type index, const bool val = true) {
			if (_nnz + 1 > _alloc)
				reserve((1 + _alloc) * 2); // +1 to avoid _alloc = 0
			indices[_nnz] = index;
			_nnz++;
		}

		index_type& operator()(const size_t pos) { return indices[pos]; }
		const index_type& operator()(const size_t pos) const { return indices[pos]; }
		void zero() { _nnz = 0; }
		void sort_indices() { std::sort(indices, indices + _nnz); }
		void canonicalize() {}
		void compress() { sort_indices(); }
	};

	template <typename index_type, typename T>
	inline T* sparse_vec_entry(const sparse_vec<index_type, T>& vec, const index_type index, const bool isbinary = true) {
		if (vec.nnz() == 0)
			return NULL;
		index_type* ptr;
		if (isbinary)
			ptr = sparse_rref::binarysearch(vec.indices, vec.indices + vec.nnz(), index);
		else
			ptr = std::find(vec.indices, vec.indices + vec.nnz(), index);
		if (ptr == vec.indices + vec.nnz())
			return NULL;
		return vec.entries + (ptr - vec.indices);
	}

	template <typename index_type> using snmod_vec = sparse_vec<index_type, size_t>;
	template <typename index_type> using sfmpq_vec = sparse_vec<index_type, rat_t>;

	template <typename index_type, typename T>
	inline void sparse_vec_rescale(sparse_vec<index_type, T>& vec, const T scalar, const field_t F) {
		if constexpr (std::is_same_v<T, size_t>) {
			_nmod_vec_scalar_mul_nmod_shoup(vec.entries, vec.entries, vec.nnz(), scalar, F->mod);
		}
		else if constexpr (Flint::IsOneOf<T, int_t, rat_t>) {
			for (size_t i = 0; i < vec.nnz(); i++)
				vec.entries[i] *= scalar;
		}
	}

	template <typename index_type>
	static void snmod_vec_from_sfmpq(snmod_vec<index_type>& vec, const sfmpq_vec<index_type>& src, nmod_t p) {
		if (vec.alloc() < src.nnz())
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
	template <typename index_type>
	static int snmod_vec_add_mul(
		snmod_vec<index_type>& vec, const snmod_vec<index_type>& src,
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

		size_t ptr1 = vec.nnz();
		size_t ptr2 = src.nnz();
		size_t ptr = vec.nnz() + src.nnz();

		if (vec.alloc() < ptr)
			vec.reserve(ptr);
		vec.resize(ptr);

		while (ptr1 > 0 && ptr2 > 0) {
			if (vec(ptr1 - 1) == src(ptr2 - 1)) {
				auto entry =
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
		if (vec.alloc() > 4 * vec.nnz())
			vec.reserve(2 * vec.nnz());

		return 0;
	}

	template <typename index_type, bool dir>
	int sfmpq_vec_addsub_mul(sfmpq_vec<index_type>& vec, const sfmpq_vec<index_type>& src, const rat_t& a) {
		if (src.nnz() == 0)
			return 0;

		if (vec.nnz() == 0) {
			vec = src;
			sparse_vec_rescale(vec, a, nullptr);
		}

		rat_t na, entry;
		if constexpr (dir) {
			na = a;
		}
		else {
			na = -a;
		}

		size_t ptr1 = vec.nnz();
		size_t ptr2 = src.nnz();
		size_t ptr = vec.nnz() + src.nnz();

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

	template <typename index_type>
	static inline int sfmpq_vec_add_mul(sfmpq_vec<index_type>& vec, const sfmpq_vec<index_type>& src, const rat_t& a) {
		return sfmpq_vec_addsub_mul<index_type, true>(vec, src, a);
	}

	template <typename index_type>
	static inline int sfmpq_vec_sub_mul(sfmpq_vec<index_type>& vec, const sfmpq_vec<index_type>& src, const rat_t& a) {
		return sfmpq_vec_addsub_mul<index_type, false>(vec, src, a);
	}

	template <typename index_type>
	static inline int snmod_vec_sub_mul(snmod_vec<index_type>& vec, const snmod_vec<index_type>& src, const ulong a, field_t F) {
		return snmod_vec_add_mul(vec, src, F->mod.n - a, F);
	}

	template <typename index_type>
	static inline int sparse_vec_add(snmod_vec<index_type>& vec, const snmod_vec<index_type>& src, field_t F) {
		return snmod_vec_add_mul(vec, src, 1, F);
	}

	template <typename index_type>
	static inline int sparse_vec_sub(snmod_vec<index_type>& vec, const snmod_vec<index_type>& src, field_t F) {
		return snmod_vec_add_mul(vec, src, F->mod.n - 1, F);
	}

	template <typename index_type>
	static inline int sparse_vec_sub_mul(snmod_vec<index_type>& vec, const snmod_vec<index_type>& src, const ulong a, field_t F) {
		return snmod_vec_sub_mul(vec, src, a, F);
	}

	template <typename index_type>
	static inline int sparse_vec_sub_mul(sfmpq_vec<index_type>& vec, const sfmpq_vec<index_type>& src, const rat_t& a, field_t F) {
		return sfmpq_vec_sub_mul(vec, src, a);
	}

	// dot product
	// return true if the result is zero
	template <typename index_type, typename T>
	T sparse_vec_dot(const sparse_vec<index_type, T> v1, const sparse_vec<index_type, T> v2, field_t F) {
		if (v1.nnz() == 0 || v2.nnz() == 0) {
			return T(0);
		}
		size_t ptr1 = 0, ptr2 = 0;
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

	template <typename index_type>
	std::pair<char*, char*> snmod_vec_to_binary(const sparse_vec<index_type, ulong>& vec, char* buffer = NULL) {
		constexpr auto ratio_i = sizeof(index_type) / sizeof(char);
		constexpr auto ratio_e = sizeof(ulong) / sizeof(char);
		auto nnz = vec.nnz();
		if (buffer == NULL)
			buffer = s_malloc<char>(ratio_e + nnz * (ratio_i + ratio_e));
		auto ptr = buffer;
		std::memcpy(ptr, &nnz, sizeof(ulong));
		ptr += ratio_e;
		std::memcpy(ptr, vec.indices, nnz * sizeof(index_type));
		ptr += nnz * ratio_i;
		std::memcpy(ptr, vec.entries, nnz * sizeof(ulong));
		ptr += nnz * ratio_e;
		return std::make_pair(buffer, ptr);
	}

	template <typename index_type>
	char* snmod_vec_from_binary(sparse_vec<index_type, ulong>& vec, const char* buffer) {
		constexpr auto ratio_i = sizeof(index_type) / sizeof(char);
		constexpr auto ratio_e = sizeof(ulong) / sizeof(char);
		ulong nnz;
		std::memcpy(&nnz, buffer, sizeof(ulong));
		vec.reserve(nnz);
		vec.resize(nnz);
		std::memcpy(vec.indices, buffer + ratio_e, nnz * sizeof(index_type));
		std::memcpy(vec.entries, buffer + ratio_e + nnz * ratio_i, nnz * sizeof(ulong));
		char* ptr = (char*)(buffer + ratio_e + nnz * (ratio_i + ratio_e));
		return ptr;
	}

	// debug only, not used to the large vector
	template <typename index_type, typename T> void print_vec_info(const sparse_vec<index_type, T>& vec) {
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
