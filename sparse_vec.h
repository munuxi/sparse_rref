#ifndef SPARSE_VEC_H
#define SPARSE_VEC_H

#include "flint/fmpq.h"
#include "flint/fmpq_vec.h"
#include "flint/nmod.h"
#include "flint/nmod_vec.h"
#include "flint/perm.h"
#include "flint/ulong_extras.h"

#include "scalar.h"
#include "util.h"

template <typename T> struct sparse_vec_struct {
	ulong len = 0;
	ulong nnz = 0;
	ulong alloc = 0;
	slong* indices = NULL;
	T* entries = NULL;
};

template <typename T> using sparse_vec_t = struct sparse_vec_struct<T>[1];

typedef sparse_vec_t<ulong> snmod_vec_t;
typedef sparse_vec_t<fmpq> sfmpq_vec_t;

// sparse_vec

// memory management
// realloc is ok to apply to NULL pointer
template <typename T>
inline void sparse_vec_realloc(sparse_vec_t<T> vec, ulong alloc) {
	if (alloc == vec->alloc)
		return;
	// so sparse_vec_realloc(vec,vec->alloc) is useless
	ulong old_alloc = vec->alloc;
	vec->alloc = std::min(alloc, vec->len);
	if (vec->alloc > old_alloc) {
		// enlarge: init later
		vec->indices =
			(slong*)realloc(vec->indices, vec->alloc * sizeof(slong));
		if constexpr (!std::is_same_v<T, bool>) {
			vec->entries = (T*)realloc(vec->entries, vec->alloc * sizeof(T));
		}
		if constexpr (std::is_same_v<T, fmpq>) {
			for (ulong i = old_alloc; i < vec->alloc; i++)
				fmpq_init((fmpq*)(vec->entries) + i);
		}
	}
	else {
		// shrink: clear first
		if constexpr (std::is_same_v<T, fmpq>) {
			for (ulong i = vec->alloc; i < old_alloc; i++)
				fmpq_clear((fmpq*)(vec->entries) + i);
		}
		vec->indices =
			(slong*)realloc(vec->indices, vec->alloc * sizeof(slong));
		if constexpr (!std::is_same_v<T, bool>) {
			vec->entries = (T*)realloc(vec->entries, vec->alloc * sizeof(T));
		}
	}
}


template <typename T>
inline void _sparse_vec_init(sparse_vec_t<T> vec, ulong len,
	ulong alloc) {
	vec->len = len;
	vec->nnz = 0;
	vec->alloc = alloc;
	vec->indices = (slong*)malloc(vec->alloc * sizeof(slong));
	if constexpr (std::is_same_v<T, bool>) {
		return;
	}
	vec->entries = (T*)malloc(vec->alloc * sizeof(T));
	if constexpr (std::is_same_v<T, fmpq>) {
		for (ulong i = 0; i < vec->alloc; i++)
			fmpq_init(vec->entries + i);
	}
}

// alloc at least 1 to make sure that indices and entries are not NULL
template <typename T>
inline void sparse_vec_init(sparse_vec_t<T> vec, ulong len) {
	_sparse_vec_init(vec, len, 1ULL);
}

// just set vec to zero vector
template <typename T> inline void sparse_vec_zero(sparse_vec_t<T> vec) {
	vec->nnz = 0;
}

// set zero and clear memory
template <typename T> inline void sparse_vec_clear(sparse_vec_t<T> vec) {
	free(vec->indices);
	if constexpr (std::is_same_v<T, fmpq>) {
		for (auto i = 0; i < vec->alloc; i++)
			fmpq_clear(vec->entries + i);
	}
	free(vec->entries);
	vec->nnz = 0;
	vec->alloc = 0;
	vec->indices = NULL;
	vec->entries = NULL;
}

template <typename T>
inline T* sparse_vec_entry(sparse_vec_t<T> vec, slong index,
	bool isbinary = false) {
	if (vec->nnz == 0 || index < 0 || (ulong)index >= vec->len)
		return NULL;
	slong* ptr;
	if (isbinary)
		ptr = binarysearch(vec->indices, vec->indices + vec->nnz, index);
	else
		ptr = std::find(vec->indices, vec->indices + vec->nnz, index);
	if (ptr == vec->indices + vec->nnz)
		return NULL;
	return vec->entries + (ptr - vec->indices);
}

template <typename T>
inline bool sparse_vec_is_same(const sparse_vec_t<T> vec,
	const sparse_vec_t<T> src) {
	if (src->len != vec->len || src->nnz != vec->nnz)
		return false;
	for (size_t i = 0; i < src->nnz; i++) {
		if (vec->indices[i] != src->indices[i])
			return false;
		if constexpr (!std::is_same_v<T, bool>) {
			if (!scalar_equal(vec->entries + i, src->entries + i))
				return false;
		}
	}
	return true;
}

// constructors
template <typename T>
inline void sparse_vec_set(sparse_vec_t<T> vec,
	const sparse_vec_t<T> src) {
	if (vec->alloc < src->nnz)
		sparse_vec_realloc(vec, src->nnz);

	vec->len = src->len;
	vec->nnz = src->nnz;
	for (ulong i = 0; i < src->nnz; i++) {
		vec->indices[i] = src->indices[i];
		if constexpr (!std::is_same_v<T, bool>) {
			scalar_set(vec->entries + i, src->entries + i);
		}
	}
}

// be careful to use it
template <typename T>
inline void sparse_vec_move(sparse_vec_t<T> vec, const sparse_vec_t<T> src) {
	sparse_vec_clear(vec);
	vec->indices = src->indices;
	vec->entries = src->entries;
	vec->len = src->len;
	vec->nnz = src->nnz;
	vec->alloc = src->alloc;
}

template <typename T>
inline void sparse_vec_swap(sparse_vec_t<T> vec, sparse_vec_t<T> src) {
	std::swap(src->indices, vec->indices);
	std::swap(src->entries, vec->entries);
	std::swap(src->len, vec->len);
	std::swap(src->nnz, vec->nnz);
	std::swap(src->alloc, vec->alloc);
}

// this raw version assumes that the vec[index] = 0
template <typename T, typename S>
inline void _sparse_vec_set_entry(sparse_vec_t<T> vec, slong index,
	S val) {
	if (index < 0 || (ulong)index >= vec->len)
		return;

	if (vec->nnz == vec->alloc) {
		ulong new_alloc = std::min(vec->alloc + (vec->alloc + 1) / 2, vec->len);
		sparse_vec_realloc(vec, new_alloc);
	}
	vec->indices[vec->nnz] = index;
	if constexpr (!std::is_same_v<T, bool>) {
		if constexpr (std::is_same_v<T, fmpq>) {
			fmpq_set(vec->entries + vec->nnz, val);
		}
		else {
			vec->entries[vec->nnz] = (T)val;
		}
	}
	vec->nnz++;
}

// template <typename T, typename S>
// inline void sparse_vec_set_entry(sparse_vec_t<T> vec, slong index, S val,
//                                         bool isbinary = false) {
//     if (index < 0 || (ulong)index >= vec->len)
//         return;
// 
//     // if val = 0, here we only set it as zero, but not remove it
//     T *entry = sparse_vec_entry(vec, index, isbinary);
//     if (entry != NULL) {
//         if constexpr (std::is_same_v<T, fmpq>)
//             fmpq_set((fmpq *)entry, val);
//         else
//             *entry = (T)val;
//         return;
//     }
//     _sparse_vec_set_entry(vec, index, val);
// }

// TODO: Implement a better sorting algorithm (sort only once)
template <typename T> void sparse_vec_sort_indices(sparse_vec_t<T> vec) {
	if (vec->nnz <= 1)
		return;

	if constexpr (std::is_same_v<T, bool>) {
		std::sort(vec->indices, vec->indices + vec->nnz);
		return;
	}

	std::vector<slong> perm(vec->nnz);
	for (size_t i = 0; i < vec->nnz; i++)
		perm[i] = i;

	T* entries = (T*)malloc(vec->nnz * sizeof(T));
	if constexpr (std::is_same_v<T, fmpq>) {
		for (size_t i = 0; i < vec->nnz; i++)
			fmpq_init(entries + i);
	}

	std::sort(perm.begin(), perm.end(), [&vec](slong a, slong b) {
		return vec->indices[a] < vec->indices[b];
		});

	// apply permutation
	for (size_t i = 0; i < vec->nnz; i++)
		scalar_set(entries + i, vec->entries + perm[i]);
	for (size_t i = 0; i < vec->nnz; i++)
		scalar_set(vec->entries + i, entries + i);

	std::sort(vec->indices, vec->indices + vec->nnz);
	if constexpr (std::is_same_v<T, fmpq>) {
		for (size_t i = 0; i < vec->nnz; i++)
			fmpq_clear(entries + i);
	}
	free(entries);
}

template <typename T>
inline void sparse_vec_canonicalize(sparse_vec_t<T> vec) {
	if constexpr (std::is_same_v<T, bool>) {
		return;
	}

	// sparse_vec_sort_indices(vec);
	ulong new_nnz = 0;
	for (size_t i = 0; i < vec->nnz; i++) {
		if (scalar_is_zero(vec->entries + i))
			continue;
		vec->indices[new_nnz] = vec->indices[i];
		scalar_set(vec->entries + new_nnz, vec->entries + i);

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

void snmod_vec_rescale(snmod_vec_t vec, ulong scalar, nmod_t p);
void snmod_vec_neg(snmod_vec_t vec, nmod_t p);
int snmod_vec_add(snmod_vec_t vec, const snmod_vec_t src, nmod_t p);
int snmod_vec_sub(snmod_vec_t vec, const snmod_vec_t src, nmod_t p);
int snmod_vec_add_mul(snmod_vec_t vec, const snmod_vec_t src, const ulong a, nmod_t p);
int snmod_vec_sub_mul(snmod_vec_t vec, const snmod_vec_t src, const ulong a, nmod_t p);

void sfmpq_vec_rescale(sfmpq_vec_t vec, const fmpq_t scalar);
void sfmpq_vec_neg(sfmpq_vec_t vec);
int sfmpq_vec_add_mul(sfmpq_vec_t vec, const sfmpq_vec_t src, const fmpq_t a);
int sfmpq_vec_sub_mul(sfmpq_vec_t vec, const sfmpq_vec_t src, const fmpq_t a);

void snmod_vec_from_sfmpq(snmod_vec_t vec, const sfmpq_vec_t src, nmod_t p);

// debug only, not used to the large vector
template <typename T> void print_vec_info(const sparse_vec_t<T> vec) {
	std::cout << "-------------------" << std::endl;
	std::cout << "len: " << vec->len << std::endl;
	std::cout << "nnz: " << vec->nnz << std::endl;
	std::cout << "alloc: " << vec->alloc << std::endl;
	std::cout << "indices: ";
	for (size_t i = 0; i < vec->nnz; i++)
		std::cout << vec->indices[i] << " ";
	std::cout << "\nentries: ";
	if constexpr (std::is_same_v<fmpq, T>) {
		for (size_t i = 0; i < vec->nnz; i++)
			std::cout << fmpq_get_str(NULL, 10, vec->entries + i) << " ";
	}
	else {
		for (size_t i = 0; i < vec->nnz; i++)
			std::cout << vec->entries[i] << " ";
	}
	std::cout << std::endl;
}

#endif
