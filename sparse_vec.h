#ifndef SPARSE_VEC_H
#define SPARSE_VEC_H

#include "flint/nmod_vec.h"

#include "scalar.h"

template <typename T> struct sparse_vec_struct {
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
			if constexpr (is_scalar_s<T>::value) {
				vec->entries = s_realloc(vec->entries, vec->alloc, vec->entries->rank);
			}
			else {
				vec->entries = s_realloc(vec->entries, vec->alloc);
			}
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
		vec->indices = s_realloc(vec->indices, vec->alloc);
		if constexpr (!std::is_same_v<T, bool>) {
			if constexpr (is_scalar_s<T>::value) {
				vec->entries = s_realloc(vec->entries, vec->alloc, vec->entries->rank);
			}
			else {
				vec->entries = s_realloc(vec->entries, vec->alloc);
			}
		}
	}
}

template <typename T>
inline T* sparse_vec_entry_pointer(const sparse_vec_t<T> vec, const slong index) {
	return vec->entries + index;
}

// alloc at least 1 to make sure that indices and entries are not NULL
template <typename T>
inline void sparse_vec_init(sparse_vec_t<T> vec, ulong alloc = 1, ulong rank = 1) {
	vec->nnz = 0;
	vec->alloc = alloc;
	vec->indices = s_malloc<slong>(vec->alloc);
	if constexpr (std::is_same_v<T, bool>) {
		return;
	}
	if constexpr (is_scalar_s<T>::value) {
		using S = typename scalar_s_decay<T>::type;
		vec->entries = s_malloc<S>(alloc, rank);
		vec->entries->rank = rank;
	}
	else {
		vec->entries = s_malloc<T>(alloc);
		if constexpr (std::is_same_v<T, fmpq>) {
			for (ulong i = 0; i < alloc; i++)
				fmpq_init(vec->entries + i);
		}
	}
}

// just set vec to zero vector
template <typename T> inline void sparse_vec_zero(sparse_vec_t<T> vec) {
	vec->nnz = 0;
}

// set zero and clear memory
template <typename T> inline void sparse_vec_clear(sparse_vec_t<T> vec) {
	vec->nnz = 0;
	vec->alloc = 0;
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
}

template <typename T>
inline T* sparse_vec_entry(sparse_vec_t<T> vec, slong index,
	const bool isbinary = true) {
	if (vec->nnz == 0 || index < vec->indices[0] || index > vec->indices[vec->nnz - 1])
		return NULL;
	slong* ptr;
	if (isbinary)
		ptr = binarysearch(vec->indices, vec->indices + vec->nnz, index);
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
			scalar_set(
				sparse_vec_entry_pointer(vec, i),
				sparse_vec_entry_pointer(src, i));
		}
	}
}

// be careful to use it
template <typename T>
inline void sparse_vec_move(sparse_vec_t<T> vec, const sparse_vec_t<T> src) {
	sparse_vec_clear(vec);
	vec->indices = src->indices;
	vec->entries = src->entries;
	vec->nnz = src->nnz;
	vec->alloc = src->alloc;
}

template <typename T>
inline void sparse_vec_swap(sparse_vec_t<T> vec, sparse_vec_t<T> src) {
	std::swap(src->indices, vec->indices);
	std::swap(src->entries, vec->entries);
	std::swap(src->nnz, vec->nnz);
	std::swap(src->alloc, vec->alloc);
}

// this raw version assumes that the vec[index] = 0
template <typename T>
void _sparse_vec_set_entry(sparse_vec_t<T> vec, slong index, const T* val) {
	if (vec->nnz == vec->alloc) {
		ulong new_alloc = 2 * vec->alloc;
		sparse_vec_realloc(vec, new_alloc);
	}
	vec->indices[vec->nnz] = index;
	if constexpr (!std::is_same_v<T, bool>) {
		if constexpr (std::is_same_v<T, fmpq>) {
			scalar_set(sparse_vec_entry_pointer(vec, vec->nnz), val);
		}
		else if constexpr (is_scalar_s<T>::value) {
			//std::cout << sparse_vec_entry_pointer(vec, vec->nnz)->data << std::endl;
			scalar_set(sparse_vec_entry_pointer(vec, vec->nnz)->data, val->data, vec->entries->rank);
		}
		else {
			// use scalar_set ??
			*sparse_vec_entry_pointer(vec, vec->nnz) = *val;
		}
	}
	vec->nnz++;
}

template <typename T, typename S>
inline void sparse_vec_set_entry(sparse_vec_t<T> vec, slong index, const T* val,
	bool isbinary = false) {
	// if val = 0, here we only set it as zero, but not remove it
	T* entry = sparse_vec_entry(vec, index, isbinary);
	if (entry != NULL) {
		if constexpr (std::is_same_v<T, fmpq>)
			scalar_set(entry, val);
		else
			*entry = *val;
		return;
	}
	_sparse_vec_set_entry(vec, index, val);
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

		if constexpr (is_scalar_s<T>::value) {
			using S = typename scalar_s_decay<T>::type;
			auto entries = s_malloc<S>(vec->nnz, vec->entries->rank);

			// apply permutation
			for (size_t i = 0; i < vec->nnz; i++) {
				scalar_set(entries + i,
					sparse_vec_entry_pointer(vec, perm[i]));
			}
			scalar_set(vec->entries, entries, vec->nnz);

			s_free(entries);
		}
		else {
			T* entries = s_malloc<T>(vec->nnz);
			if constexpr (std::is_same_v<T, fmpq>) {
				for (size_t i = 0; i < vec->nnz; i++)
					fmpq_init(entries + i);
			}

			// apply permutation
			for (size_t i = 0; i < vec->nnz; i++) {
				scalar_set(entries + i,
					sparse_vec_entry_pointer(vec, perm[i]));
			}
			scalar_set(vec->entries, entries, vec->nnz);

			if constexpr (std::is_same_v<T, fmpq>) {
				for (size_t i = 0; i < vec->nnz; i++)
					fmpq_clear(entries + i);
			}
			s_free(entries);
		}
		std::sort(vec->indices, vec->indices + vec->nnz);
	}
}

template <typename T>
void sparse_vec_canonicalize(sparse_vec_t<T> vec) {
	if constexpr (std::is_same_v<T, bool>) {
		return;
	}

	ulong new_nnz = 0;
	bool is_changed = false;
	for (size_t i = 0; i < vec->nnz; i++) {
		if (scalar_is_zero(sparse_vec_entry_pointer(vec, i))) {
			is_changed = true;
			continue;
		}
		if (is_changed) { // avoid copy to itself
			vec->indices[new_nnz] = vec->indices[i];
			scalar_set(
				sparse_vec_entry_pointer(vec, new_nnz),
				sparse_vec_entry_pointer(vec, i));
		}
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
static inline void snmod_vec_rescale(snmod_vec_t vec, ulong scalar, nmod_t p) {
	_nmod_vec_scalar_mul_nmod_shoup(vec->entries, vec->entries, vec->nnz,
		scalar, p);
}

int snmod_vec_add_mul(snmod_vec_t vec, const snmod_vec_t src, const ulong a, nmod_t p);
static inline int snmod_vec_sub_mul(snmod_vec_t vec, const snmod_vec_t src, const ulong a, nmod_t p) {
	return snmod_vec_add_mul(vec, src, nmod_neg(a, p), p);
}
//static inline int sparse_vec_add(snmod_vec_t vec, const snmod_vec_t src, field_t F) {
//	return snmod_vec_add_mul(vec, src, (ulong)1, F->pvec[0]);
//}
//static inline int sparse_vec_sub(snmod_vec_t vec, const snmod_vec_t src, field_t F) {
//	return snmod_vec_sub_mul(vec, src, (ulong)1, F->pvec[0]);
//}

static inline void sfmpq_vec_rescale(sfmpq_vec_t vec, const fmpq_t scalar) {
	for (ulong i = 0; i < vec->nnz; i++)
		fmpq_mul(vec->entries + i, vec->entries + i, scalar);
}

int sfmpq_vec_add_mul(sfmpq_vec_t vec, const sfmpq_vec_t src, const fmpq_t a);
int sfmpq_vec_sub_mul(sfmpq_vec_t vec, const sfmpq_vec_t src, const fmpq_t a);

static void snmod_vec_from_sfmpq(snmod_vec_t vec, const sfmpq_vec_t src, nmod_t p) {
	sparse_vec_realloc(vec, src->nnz);
	vec->alloc = src->nnz;
	vec->nnz = 0;
	for (size_t i = 0; i < src->nnz; i++) {
		ulong num = fmpz_get_nmod(fmpq_numref(src->entries + i), p);
		ulong den = fmpz_get_nmod(fmpq_denref(src->entries + i), p);
		ulong val = nmod_div(num, den, p);
		_sparse_vec_set_entry(vec, src->indices[i], &val);
	}
}

// dot product
// return true if the result is zero
template <typename T>
bool sparse_vec_dot(T* result, const sparse_vec_t<T> v1, const sparse_vec_t<T> v2, field_t F) {
	if (v1->nnz == 0 || v2->nnz == 0) {
		scalar_zero(result);
		return 0;
	}
	slong ptr1 = 0, ptr2 = 0;
	T tmp[1];
	scalar_init(tmp);
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
	scalar_clear(tmp);
	return scalar_is_zero(result);
}

std::pair<size_t, char*> snmod_vec_to_binary(sparse_vec_t<ulong> vec);
void snmod_vec_from_binary(sparse_vec_t<ulong> vec, const char* buffer);

// debug only, not used to the large vector
template <typename T> void print_vec_info(const sparse_vec_t<T> vec) {
	std::cout << "-------------------" << std::endl;
	std::cout << "nnz: " << vec->nnz << std::endl;
	std::cout << "alloc: " << vec->alloc << std::endl;
	if constexpr (is_scalar_s<T>::value) {
		std::cout << "rank: " << vec->entries->rank << std::endl;
	}
	std::cout << "indices: ";
	for (size_t i = 0; i < vec->nnz; i++)
		std::cout << vec->indices[i] << " ";
	std::cout << "\nentries: ";
	if constexpr (is_scalar_s<T>::value) {
		for (size_t i = 0; i < vec->nnz; i++) {
			auto data = vec->entries[i].data;
			for (size_t j = 0; j < vec->entries->rank - 1; j++)
				std::cout << data[j] << ", ";
			std::cout << data[vec->entries->rank - 1] << ";";
			std::cout << std::endl;
		}
	}
	else {
		for (size_t i = 0; i < vec->nnz; i++)
			std::cout << scalar_to_str(vec->entries + i) << " ";
	}
	std::cout << std::endl;
}

#endif
