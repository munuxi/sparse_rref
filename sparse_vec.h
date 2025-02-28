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

		void resize(ulong n) {
			indices.resize(n);
			// entries is useless for bool
			if constexpr (!std::is_same_v<T, bool>) {
				entries.resize(n);
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

		void push_back(const slong index, const T val) {
			indices.push_back(index);
			if constexpr (!std::is_same_v<T, bool>) {
				entries.push_back(val);
			}
		}

		void push_back(const std::pair<slong, T> a) {
			indices.push_back(a.first);
			if constexpr (!std::is_same_v<T, bool>) {
				entries.push_back(a.second);
			}
		}

		slong& operator()(const ulong pos) {
			return indices[pos];
		}

		const slong& operator()(const ulong pos) const {
			return indices[pos];
		}

		void zero() {
			indices.clear();
			entries.clear();
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

	template <typename T>
	inline std::vector<T>::iterator sparse_vec_entry(sparse_vec<T>& vec, const slong index, const bool isbinary = true) {
		if (vec.nnz() == 0)
			return vec.entries.end();
		std::vector<slong>::iterator ptr;
		if (isbinary)
			ptr = sparse_rref::binarysearch(vec.indices, index);
		else
			ptr = std::find(vec.indices.begin(), vec.indices.end(), index);
		if (ptr == vec.indices.end())
			return vec.entries.end();
		return vec.entries.begin() + (ptr - vec.indices.begin());
	}

	using snmod_vec = sparse_vec<ulong>;
	using sfmpq_vec = sparse_vec<rat_t>;

	template <typename T>
	inline void sparse_vec_rescale(sparse_vec<T>& vec, const T scalar, const field_t F) {
		if constexpr (std::is_same_v<T, ulong>) {
			_nmod_vec_scalar_mul_nmod_shoup(vec.entries.data(), vec.entries.data(), vec.nnz(),
				scalar, F->mod);
		}
		else if constexpr (Flint::IsOneOf<T, int_t, rat_t>) {
			for (ulong i = 0; i < vec.nnz(); i++)
				vec.entries[i] *= scalar;
		}
	}

	static void snmod_vec_from_sfmpq(snmod_vec& vec, const sfmpq_vec& src, nmod_t p) {
		vec.reserve(src.nnz());
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

		vec.reserve(vec.nnz() + src.nnz());
		vec.resize(vec.nnz() + src.nnz());

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
		if (vec.indices.capacity() > 4 * vec.nnz())
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

		vec.reserve(vec.nnz() + src.nnz());
		vec.resize(vec.nnz() + src.nnz());

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
		if (vec.indices.capacity() > 4 * vec.nnz())
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
		return snmod_vec_add_mul(vec, src, (ulong)1, F);
	}

	static inline int sparse_vec_sub(snmod_vec& vec, const snmod_vec& src, field_t F) {
		return snmod_vec_add_mul(vec, src, F->mod.n - 1, F);
	}

	static inline int sparse_vec_sub_mul(snmod_vec& vec, const snmod_vec& src, const ulong a, field_t F) {
		return snmod_vec_sub_mul(vec, src, a, F);
	}

	static inline int sparse_vec_sub_mul(sfmpq_vec& vec, const sfmpq_vec& src, const rat_t& a, field_t F = NULL) {
		return sfmpq_vec_sub_mul(vec, src, a);
	}


	// dot product
	// return true if the result is zero
	template <typename T>
	T sparse_vec_dot(const sparse_vec<T> v1, const sparse_vec<T> v2, field_t F) {
		if (v1->nnz == 0 || v2->nnz == 0) {
			return T(0);
		}
		slong ptr1 = 0, ptr2 = 0;
		T result, tmp;
		result = 0;
		while (ptr1 < v1.nnz() && ptr2 < v2.nnz()) {
			if (v1(ptr1) == v2(ptr2)) {
				scalar_mul(tmp, v1[ptr1], v2[ptr2], F);
				scalar_add(result, result, tmp, F);
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
		std::memcpy(buffer + ratio, vec.indices.data(), nnz * sizeof(ulong));
		std::memcpy(buffer + (1 + nnz) * ratio, vec.entries.data(), nnz * sizeof(ulong));
		return std::make_pair((1 + 2 * nnz) * ratio, buffer);
	}

	static void snmod_vec_from_binary(sparse_vec<ulong>& vec, const char* buffer) {
		constexpr auto ratio = sizeof(ulong) / sizeof(char);
		ulong nnz;
		std::memcpy(&nnz, buffer, sizeof(ulong));
		vec.reserve(nnz);
		vec.resize(nnz);
		std::memcpy(vec.indices.data(), buffer + ratio, nnz * sizeof(ulong));
		std::memcpy(vec.entries.data(), buffer + (1 + nnz) * ratio, nnz * sizeof(ulong));
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
