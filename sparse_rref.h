/*
	Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

	This file is part of Sparse_rref. The Sparse_rref is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

#ifndef SPARSE_RREF_H
#define SPARSE_RREF_H

#include "thread_pool.hpp"
#include <algorithm>
#include <bitset>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <list>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

// flint 
#include "flint/nmod.h"
#include "flint/ulong_extras.h"

#ifdef NULL
#undef NULL
#endif
#define NULL nullptr

namespace sparse_rref {
	// Memory management

	template <typename T>
	inline T* s_malloc(const size_t size) {
		return (T*)std::malloc(size * sizeof(T));
	}

	template <typename T>
	inline void s_free(T* s) {
		std::free(s);
	}

	template <typename T>
	inline T* s_realloc(T* s, const size_t size) {
		return (T*)std::realloc(s, size * sizeof(T));
	}

	template <typename T>
	void s_memset(T* s, const T val, const T size) {
		std::fill(s, s + size, val);
	}

	// field

	enum RING {
		FIELD_QQ,    // fmpq
		FIELD_Fp,    // ulong
		RING_MulitFp // not implemented now
	};

	struct field_struct {
		enum RING ring;
		ulong rank = 1; // the rank of the product ring
		nmod_t* pvec = NULL;
	};
	typedef struct field_struct field_t[1];

	// rref_option

	struct rref_option {
		bool verbose = false;
		bool is_back_sub = true;
		bool pivot_dir = true; // true: row, false: col
		uint8_t method = 0;
		int print_step = 100;
		int search_depth = INT_MAX;
	};
	typedef struct rref_option rref_option_t[1];


	// version
	constexpr static const char version[] = "v0.2.4";

	// thread
	using thread_pool = BS::thread_pool<>; // thread pool
	inline size_t thread_id() {
		return BS::this_thread::get_index().value();
	}

	// if c++20, use std::countr_zero
	// if c++17, use flint_ctz (__builtin_ctzll or _tzcnt_u64)
#if __cplusplus >= 202002L
#include <bit>
	inline size_t ctz(ulong x) {
		return std::countr_zero(x);
	}
	inline size_t clz(ulong x) {
		return std::countl_zero(x);
	}
	inline size_t popcount(ulong x) {
		return std::popcount(x);
	}
#else
	inline size_t ctz(ulong x) {
		return flint_ctz(x);
	}
	inline size_t clz(ulong x) {
		return flint_clz(x);
	}
	inline size_t popcount(ulong x) {
		return FLINT_BIT_COUNT(x);
	}
#endif

	// string
	inline void DeleteSpaces(std::string& str) {
		str.erase(std::remove_if(str.begin(), str.end(),
			[](unsigned char x) { return std::isspace(x); }),
			str.end());
	}

	inline std::vector<std::string> SplitString(const std::string& s, const std::string delim) {
		auto start = 0ULL;
		auto end = s.find(delim);
		std::vector<std::string> result;
		while (end != std::string::npos) {
			result.push_back(s.substr(start, end - start));
			start = end + delim.length();
			end = s.find(delim, start);
		}
		result.push_back(s.substr(start, end));
		return result;
	}

	// time
	inline std::chrono::system_clock::time_point clocknow() {
		return std::chrono::system_clock::now();
	}

	inline double usedtime(std::chrono::system_clock::time_point start,
		std::chrono::system_clock::time_point end) {
		auto duration =
			std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		return ((double)duration.count() * std::chrono::microseconds::period::num /
			std::chrono::microseconds::period::den);
	}

	template <typename T> std::vector<T> difference(std::vector<T> l) {
		std::vector<T> result;
		for (size_t i = 1; i < l.size(); i++) {
			result.push_back(l[i] - l[i - 1]);
		}
		return result;
	}

	struct uset {
		constexpr static size_t bitset_size = std::numeric_limits<unsigned long long>::digits; // 64 or 32
		std::vector<std::bitset<bitset_size>> data;

		uset() {}

		void resize(size_t alllen) {
			auto len = alllen / bitset_size + 1;
			data.resize(len);
		}

		uset(size_t alllen) {
			resize(alllen);
		}

		~uset() {
			data.clear();
		}

		void insert(size_t val) {
			auto idx = val / bitset_size;
			auto pos = val % bitset_size;
			data[idx].set(pos);
		}

		bool count(size_t val) {
			auto idx = val / bitset_size;
			auto pos = val % bitset_size;
			return data[idx].test(pos);
		}

		void erase(size_t val) {
			auto idx = val / bitset_size;
			auto pos = val % bitset_size;
			data[idx].reset(pos);
		}

		void clear() {
			for (auto& d : data)
				d.reset();
		}

		std::bitset<bitset_size>& operator[](size_t idx) {
			return data[idx];
		}

		size_t size() {
			return data.size();
		}

		size_t length() {
			return data.size() * bitset_size;
		}

		std::vector<size_t> nonzero() {
			std::vector<ulong> result;
			std::vector<ulong> tmp;
			tmp.reserve(bitset_size);
			for (size_t i = 0; i < data.size(); i++) {
				if (data[i].any()) {
					// naive version
					// for (size_t j = 0; j < bitset_size; j++) {
					// 	if (data[i].test(j))
					// 		result.push_back(i * bitset_size + j);
					// }

					tmp.clear();
					ulong c = data[i].to_ullong();

					// only ctz version
					// while (c) {
					// 	auto ctzpos = ctz(c);
					// 	result.push_back(i * bitset_size + ctzpos);
					// 	c &= c - 1;
					// }

					while (c) {
						auto ctzpos = ctz(c);
						auto clzpos = bitset_size - 1 - clz(c);
						result.push_back(i * bitset_size + ctzpos);
						if (ctzpos == clzpos)
							break;
						tmp.push_back(i * bitset_size + clzpos);
						c = c ^ (1ULL << clzpos) ^ (1ULL << ctzpos);
					}
					result.insert(result.end(), tmp.rbegin(), tmp.rend());
				}
			}
			return result;
		}
	};

	template <typename T> inline T* binarysearch(T* begin, T* end, T val) {
		auto ptr = std::lower_bound(begin, end, val);
		if (ptr == end || *ptr == val)
			return ptr;
		else
			return end;
	}

	template <typename T> inline T* lower_bound(T* begin, T* end, ulong rank, T* val) {
		auto len = (end - begin) / rank;
		T** vec = s_malloc<T*>(len);
		for (size_t i = 0; i < len; i++)
			vec[i] = begin + rank * i;
		T** ptr_s = std::lower_bound(vec, vec + len, val,
			[&rank](const T* a, const T* b) {
				// lex order
				for (ulong i = 0; i < rank; i++) {
					if (a[i] < b[i])
						return true;
					else if (a[i] > b[i])
						return false;
				}
				return false;
			}
		);
		if (ptr_s == vec + len) {
			s_free(vec);
			return end;
		}
		T* ptr = *ptr_s;
		s_free(vec);
		return ptr;
	}

	template <typename T> inline T* binarysearch(T* begin, T* end, uint16_t rank, T* val) {
		auto ptr = sparse_rref::lower_bound(begin, end, rank, val);
		if (ptr == end || std::equal(ptr, ptr + rank, val))
			return ptr;
		else
			return end;
	}

	// IO
	using DataTuple = std::vector<std::tuple<slong, slong, std::string>>;

	std::tuple<ulong, ulong, std::string> read_lines(const std::string& str) {
		std::istringstream iss(str);
		char c;
		std::string token;
		slong t1 = -1;
		slong t2 = -1;
		std::string t3("");
		int count = 0;
		bool skip_space = true;
		while (iss.get(c)) {
			bool is_space = std::isspace(c);
			if (skip_space) {
				if (is_space)
					continue;
				else
					skip_space = false;
			}
			if (!is_space) {
				token.push_back(c);
			}
			else {
				if (count == 0)
					t1 = std::stoull(token);
				else if (count == 1)
					t2 = std::stoull(token);
				else
					t3 = token;
				count++;
				token.clear();
				skip_space = true;
			}
			if (count == 3)
				break;
		}
		return std::make_tuple(t1, t2, t3);
	}

	std::string read_file_buffer(std::string filename) {
		std::ifstream file(filename);
		if (!file.is_open()) {
			std::cerr << "Failed to open file: " << filename << std::endl;
			return "";
		}
		std::stringstream buffer;
		buffer << file.rdbuf();
		file.close();
		return buffer.str();
	}

	// sparse vector
	template <typename T> struct sparse_vec {
		ulong nnz;
		ulong alloc;
		ulong* indices;
		T* entries;

		void init(ulong n) {
			nnz = 0;
			alloc = n;
			indices = s_malloc<ulong>(n);
			entries = NULL;
			if constexpr (!std::is_same_v<T, bool>) {
				entries = s_malloc<T>(n);
			}
			if constexpr (std::is_same_v<T, fmpq>) {
				for (ulong i = 0; i < alloc; i++)
					fmpq_init(entries + i);
			}
		}

		inline bool is_alloced() {
			return alloc != 0;
		}

		sparse_vec(ulong n = 1) { init(n); }

		sparse_vec(const sparse_vec& l) {
			init(l.alloc);
			copy(l);
		}

		sparse_vec(sparse_vec&& l) noexcept {
			nnz = l.nnz;
			alloc = l.alloc;
			indices = l.indices;
			entries = l.entries;
			l.indices = NULL;
			l.entries = NULL;
			l.alloc = 0;
		}

		void clear() {
			s_free(indices);
			if constexpr (std::is_same_v<T, fmpq>) {
				for (auto i = 0; i < alloc; i++)
					fmpq_clear(entries + i);
			}
			if constexpr (!std::is_same_v<T, bool>) {
				s_free(entries);
			}
			alloc = 0;
			indices = NULL;
			entries = NULL;
		}

		~sparse_vec() {
			clear();
		}

		void realloc(ulong n) {
			if (alloc == n)
				return;
			indices = s_realloc(indices, n);
			if (n > alloc) {
				// enlarge: init later
				if constexpr (!std::is_same_v<T, bool>) {
					entries = s_realloc(entries, n);
				}
				if constexpr (std::is_same_v<T, fmpq>) {
					for (ulong i = alloc; i < n; i++)
						fmpq_init((fmpq*)(entries)+i);
				}
			}
			else {
				// shrink: clear first
				if constexpr (std::is_same_v<T, fmpq>) {
					for (ulong i = n; i < alloc; i++)
						fmpq_clear((fmpq*)(entries)+i);
				}
				if constexpr (!std::is_same_v<T, bool>) {
					entries = s_realloc(entries, n);
				}
			}
			alloc = n;
		}

		void copy(const sparse_vec& src) {
			nnz = src.nnz;
			if (alloc < src.nnz)
				realloc(src.nnz);
			for (auto i = 0; i < src.nnz; i++) {
				indices[i] = src.indices[i];
				if constexpr (!std::is_same_v<T, bool>) {
					if constexpr (std::is_same_v<T, fmpq>) {
						scalar_set(entries + i, src.entries + i);
					}
					else {
						entries[i] = src.entries[i];
					}
				}
			}
		}

		sparse_vec& operator=(const sparse_vec& l) {
			if (this == &l)
				return *this;
			if (alloc == 0)
				init(l.alloc);
			else if (alloc < l.nnz)
				realloc(l.nnz);

			copy(l);
			return *this;
		}

		sparse_vec& operator=(sparse_vec&& l) noexcept {
			if (this == &l)
				return *this;
			clear();
			nnz = l.nnz;
			alloc = l.alloc;
			indices = l.indices;
			entries = l.entries;
			l.indices = NULL;
			l.entries = NULL;
			l.alloc = 0;
			return *this;
		}

		void set_zero() {
			nnz = 0;
		}

		void push_back(ulong index, const T* val) {
			if (nnz == alloc)
				realloc(2 * alloc);
			indices[nnz] = index;
			if constexpr (!std::is_same_v<T, bool>) {
				if constexpr (std::is_same_v<T, fmpq>) {
					scalar_set(entries + nnz, val);
				}
				else {
					entries[nnz] = *val;
				}
			}
			nnz++;
		}

		T* operator[](ulong pos) {
			return entries + pos;
		}

		std::pair<slong, T*> at(ulong pos) {
			return std::make_pair(indices[pos], entries + pos);
		}

		void canonicalize() {
			if constexpr (std::is_same_v<T, bool>) { return; }
			ulong new_nnz = 0;
			ulong i = 0;
			for (; i < nnz; i++) {
				if (!scalar_is_zero(entries + i))
					break;
			}
			for (; i < nnz; i++) {
				if (scalar_is_zero(entries + i))
					continue;
				indices[new_nnz] = indices[i];
				scalar_set(entries + new_nnz, entries + i);
				new_nnz++;
			}
			nnz = new_nnz;
		}

		void sort_indices() {
			if (nnz <= 1)
				return;

			if constexpr (std::is_same_v<T, bool>) {
				std::sort(indices, indices + nnz);
				return;
			}
			else {
				std::vector<slong> perm(nnz);
				for (size_t i = 0; i < nnz; i++)
					perm[i] = i;

				std::sort(perm.begin(), perm.end(), [&indices](slong a, slong b) {
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

				T* tentries = s_malloc<T>(nnz);
				if constexpr (std::is_same_v<T, fmpq>) {
					for (size_t i = 0; i < nnz; i++)
						scalar_init(tentries + i);
				}

				// apply permutation
				for (size_t i = 0; i < nnz; i++) {
					scalar_set(tentries + i, entries + perm[i]);
				}
				scalar_set(entries, tentries, nnz);

				if constexpr (std::is_same_v<T, fmpq>) {
					for (size_t i = 0; i < nnz; i++)
						scalar_clear(tentries + i);
				}
				s_free(tentries);
				std::sort(indices, indices + nnz);
			}
		}

		void compress() {
			canonicalize();
			sort_indices();
		}
	};
}

#endif