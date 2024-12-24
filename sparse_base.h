#ifndef SPARSE_BASE_H
#define SPARSE_BASE_H

#include "thread_pool.hpp"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <list>
#include <queue>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

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
	// string
	inline void DeleteSpaces(std::string& str) {
		str.erase(std::remove_if(str.begin(), str.end(),
			[](unsigned char x) { return std::isspace(x); }),
			str.end());
	}

	template <typename T> inline T* binarysearch(T* begin, T* end, T val) {
		auto ptr = std::lower_bound(begin, end, val);
		if (ptr == end || *ptr == val)
			return ptr;
		else
			return end;
	}

	template <typename T> inline T* lower_bound(T* begin, T* end, uint16_t rank, T* val) {
		auto len = (end - begin) / rank;
		T** vec = s_malloc<T*>(len);
		for (size_t i = 0; i < len; i++)
			vec[i] = begin + rank * i;
		T** ptr_s = std::lower_bound(vec, vec + len, val,
			[&rank](const T* a, const T* b) {
				// lex order
				for (uint16_t i = 0; i < rank; i++) {
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
}

#endif