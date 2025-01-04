/*
	Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

	This file is part of Sparse_rref. The Sparse_rref is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

#ifndef SPARSE_BASE_H
#define SPARSE_BASE_H

#include "thread_pool.hpp"
#include <algorithm>
#include <bitset>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <list>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>

#ifdef NULL
#undef NULL
#endif
#define NULL nullptr

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

namespace sparse_base {
	// version
	constexpr static const char version[] = "v0.2.4";

	// thread
	using thread_pool = BS::thread_pool<>; // thread pool
	inline size_t thread_id() {
		return BS::this_thread::get_index().value();
	}

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
		constexpr static size_t bitset_size = 0x100; // 256 for avx2, morden cpu should support it?
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
		auto ptr = sparse_base::lower_bound(begin, end, rank, val);
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
}

#endif