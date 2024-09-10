#ifndef UTIL_H
#define UTIL_H

#include "thread_pool.hpp"
#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <iostream>
#include <list>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>

// scalar array
template <typename T> struct scalar_s {
	size_t rank = 1;
	T* data = NULL;
};

template <typename T> struct is_scalar_s : std::false_type {};
template <typename T> struct is_scalar_s<scalar_s<T>> : std::true_type {};
template <typename T> struct is_scalar_s<const scalar_s<T>> : std::true_type {};

template <typename T> struct scalar_s_decay { using type = T; };
template <typename T> struct scalar_s_decay<scalar_s<T>> { using type = T; };
template <typename T> struct scalar_s_decay<const scalar_s<T>> { using type = T; };


// Memory management

template <typename T>
T* s_malloc(const size_t size) {
	return (T*)std::malloc(size * sizeof(T));
}

template <typename T>
scalar_s<T>* s_malloc(const size_t size, const size_t rank) {
	scalar_s<T>* s = (scalar_s<T>*)std::malloc(size * sizeof(scalar_s<T>));
	s->data = (T*)std::malloc(rank * size * sizeof(T));
	for (size_t i = 0; i < size; i++) {
		s[i].rank = rank;
		s[i].data = s->data + i * rank;
	}
	return s;
}

template <typename T>
void s_free(T* s) {
	if constexpr (is_scalar_s<T>::value) {
		std::free(s->data);
		s->data = NULL;
	}
	std::free(s);
}

template <typename T>
T* s_realloc(T* s, const size_t size) {
	return (T*)std::realloc(s, size * sizeof(T));
}

template <typename T>
scalar_s<T>* s_realloc(scalar_s<T>* s, const size_t size, const size_t rank) {
	auto ptr = (T*)std::realloc(s->data, rank * size * sizeof(T));
	scalar_s<T>* new_s = (scalar_s<T>*)std::realloc(s, size * sizeof(scalar_s<T>));
	for (size_t i = 0; i < size; i++) {
		new_s[i].rank = rank;
		new_s[i].data = ptr + i * rank;
	}
	return new_s;
}

template <typename T>
void s_memset(T* s, const T val, const T size) {
	std::fill(s, s + size, val);
}

// field

enum RING {
	FIELD_F2,    // bool
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
	bool pivot_dir = true; // true: row, false: col
	int print_step = 100;
	int sort_step = 0;
	int search_min = 200;
	int search_depth = INT_MAX;
};
typedef struct rref_option rref_option_t[1];

// string
inline void DeleteSpaces(std::string& str) {
	str.erase(std::remove_if(str.begin(), str.end(),
		[](unsigned char x) { return std::isspace(x); }),
		str.end());
}

inline std::vector<std::string> SplitString(const std::string& s, std::string delim) {
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

#endif