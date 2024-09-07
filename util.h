#ifndef UTIL_H
#define UTIL_H

#include "flint/fmpq.h"
#include "flint/nmod.h"
#include "thread_pool.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <execution>
#include <iostream>
#include <list>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef NULL
#undef NULL
#endif
#define NULL nullptr

// get the bit at position bit
#define GET_BIT(x, bit) (((x) >> (bit)) & 1ULL)
// set the bit at position bit
#define SET_BIT_ONE(x, bit) ((x) |= (1ULL << (bit)))
#define SET_BIT_NIL(x, bit) ((x) &= ~(1ULL << (bit)))

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

struct rref_option {
	bool verbose = false;
	bool pivot_dir = true; // true: row, false: col
	int print_step = 100;
	int sort_step = 0;
	int search_min = 200;
	ulong search_depth = ULLONG_MAX;
};
typedef struct rref_option rref_option_t[1];

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

// Memory management

template <typename T>
T* s_malloc(const size_t size) {
	return (T*)malloc(size * sizeof(T));
}

template <typename T>
scalar_s<T>* s_malloc(const size_t size, const size_t rank) {
	scalar_s<T>* s = (scalar_s<T>*)malloc(size * sizeof(scalar_s<T>));
	s->data = (T*)malloc(rank * size * sizeof(T));
	for (size_t i = 0; i < size; i++) {
		s[i].rank = rank;
		s[i].data = s->data + i * rank;
	}
	return s;
}

template <typename T>
void s_free(T* s) {
	if constexpr (is_scalar_s<T>::value) {
		free(s->data);
		s->data = NULL;
	}
	free(s);
}

template <typename T>
T* s_realloc(T* s, const size_t size) {
	return (T*)realloc(s, size * sizeof(T));
}

template <typename T>
scalar_s<T>* s_realloc(scalar_s<T>* s, const size_t size, const size_t rank) {
	auto ptr = (T*)realloc(s->data, rank * size * sizeof(T));
	scalar_s<T>* new_s = (scalar_s<T>*)realloc(s, size * sizeof(scalar_s<T>));
	for (size_t i = 0; i < size; i++) {
		new_s[i].rank = rank;
		new_s[i].data = ptr + i * rank;
	}
	return new_s;
}

// field

inline void field_init(field_t field, enum RING ring, ulong rank, const ulong* pvec) {
	field->ring = ring;
	field->rank = rank;
	if (field->ring == FIELD_Fp || field->ring == RING_MulitFp) {
		field->pvec = s_malloc<nmod_t>(rank);
		for (ulong i = 0; i < rank; i++)
			nmod_init(field->pvec + i, pvec[i]);
	}
}

inline void field_init(field_t field, enum RING ring, const std::vector<ulong>& pvec) {
	field->ring = ring;
	field->rank = pvec.size();
	if (field->ring == FIELD_Fp || field->ring == RING_MulitFp) {
		field->pvec = s_malloc<nmod_t>(field->rank);
		for (ulong i = 0; i < field->rank; i++)
			nmod_init(field->pvec + i, pvec[i]);
	}
}

inline void field_set(field_t field, const field_t ff) {
	field->ring = ff->ring;
	field->rank = ff->rank;
	if (field->ring == FIELD_Fp || field->ring == RING_MulitFp) {
		field->pvec = s_realloc(field->pvec, field->rank);
		for (ulong i = 0; i < field->rank; i++)
			nmod_init(field->pvec + i, ff->pvec[i].n);
	}
}

template <typename T> inline T* binarysearch(T* begin, T* end, T val) {
	auto ptr = std::lower_bound(begin, end, val);
	if (ptr == end || *ptr == val)
		return ptr;
	else
		return end;
}

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

// vector
template <typename T>
void remove_indices(std::vector<T>& vec, std::vector<slong>& indices) {
	std::sort(indices.begin(), indices.end());
	auto it = indices.rbegin();
	for (; it != indices.rend(); ++it) {
		if (*it >= 0 && *it < vec.size()) {
			vec.erase(vec.begin() + *it);
		}
		else {
			std::cerr << "Index out of range: " << *it << std::endl;
		}
	}
}

#endif