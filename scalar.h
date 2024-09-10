#ifndef SCALAR_H
#define SCALAR_H

#include <type_traits>

#include "flint/fmpq.h"
#include "flint/nmod.h"
#include "flint/ulong_extras.h"

#include "base.h"

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

// scalar

inline void scalar_init(fmpq_t a) { fmpq_init(a); }
inline void scalar_init(ulong* a) { return; } // do nothing
template <typename T>
inline void scalar_init(scalar_s<T>* a, const ulong rank) {
	a->rank = rank;
	a->data = s_malloc<T>(rank);
	for (size_t i = 0; i < rank; i++)
		scalar_init(a->data + i);
}

inline void scalar_clear(fmpq_t a) { fmpq_clear(a); }
inline void scalar_clear(ulong* a) { return; } // do nothing
template <typename T>
inline void scalar_clear(scalar_s<T>* a) {
	for (size_t i = 0; i < a->rank; i++)
		scalar_clear(a->data + i);
	s_free(a->data);
	a->data = NULL;
}

inline bool scalar_is_zero(const fmpq_t a) { return fmpq_is_zero(a); }
inline bool scalar_is_zero(const ulong* a) { return (*a) == 0; }
template <typename T>
inline bool scalar_is_zero(const scalar_s<T>* a) {
	for (size_t i = 0; i < a->rank; i++)
		if (!scalar_is_zero(a->data + i))
			return false;
	return true;
}

inline bool scalar_equal(const fmpq_t a, const fmpq_t b) { return fmpq_equal(a, b); }
inline bool scalar_equal(const ulong* a, const ulong* b) { return (*a) == (*b); }
template <typename T>
inline bool scalar_equal(const scalar_s<T>* a, const scalar_s<T>* b) {
	for (size_t i = 0; i < a->rank; i++)
		if (!scalar_equal(a->data + i, b->data + i))
			return false;
	return true;
}

inline void scalar_set(fmpq_t a, const fmpq_t b) { fmpq_set(a, b); }
inline void scalar_set(ulong* a, const ulong* b) { *a = *b; }
template <typename T>
inline void scalar_set(T* a, const T* b, const ulong rank) {
	for (ulong i = 0; i < rank; i++)
		scalar_set(a + i, b + i);
}

inline void scalar_set(scalar_s<ulong>* a, const scalar_s<ulong>* b) {
	scalar_set(a->data, b->data, b->rank);
}

inline void scalar_zero(fmpq_t a) { fmpq_zero(a); }
inline void scalar_zero(ulong* a) { *a = 0; }

template <typename T>
inline void scalar_zero(scalar_s<T>* a) {
	for (size_t i = 0; i < a->rank; i++)
		scalar_zero(a->data + i);
}

inline void scalar_one(fmpq_t a) { fmpq_one(a); }
inline void scalar_one(ulong* a) { *a = 1; }

template <typename T>
inline void scalar_one(scalar_s<T>* a) {
	for (size_t i = 0; i < a->rank; i++)
		scalar_one(a->data + i);
}

// arithmetic

inline void scalar_add(ulong* a, const ulong* b, const ulong* c, const field_t field) {
	*a = _nmod_add(*b, *c, field->pvec[0]);
}
inline void scalar_add(fmpq_t a, const fmpq_t b, const fmpq_t c) { fmpq_add(a, b, c); }

template <typename T>
inline void scalar_add(scalar_s<T>* a, const scalar_s<T>* b, const scalar_s<T>* c, const field_t field) {
	for (size_t i = 0; i < field->rank; i++)
		scalar_add(a->data + i, b->data + i, c->data + i, field);
}

inline void scalar_sub(ulong* a, const ulong* b, const ulong* c, const field_t field) {
	*a = _nmod_sub(*b, *c, field->pvec[0]);
}
inline void scalar_sub(fmpq_t a, const fmpq_t b, const fmpq_t c) { fmpq_sub(a, b, c); }

template <typename T>
inline void scalar_sub(scalar_s<T>* a, const scalar_s<T>* b, const scalar_s<T>* c, const field_t field) {
	for (size_t i = 0; i < field->rank; i++)
		scalar_sub(a->data + i, b->data + i, c->data + i, field);
}

inline void scalar_mul(ulong* a, const ulong* b, const ulong* c, const field_t field) {
	*a = nmod_mul(*b, *c, field->pvec[0]);
}
inline void scalar_mul(fmpq_t a, const fmpq_t b, const fmpq_t c) { fmpq_mul(a, b, c); }

template <typename T>
inline void scalar_mul(scalar_s<T>* a, const scalar_s<T>* b, const scalar_s<T>* c, const field_t field) {
	for (size_t i = 0; i < field->rank; i++)
		scalar_mul(a->data + i, b->data + i, c->data + i, field);
}

inline void scalar_div(ulong* a, const ulong* b, const ulong* c, const field_t field) {
	*a = nmod_div(*b, *c, field->pvec[0]);
}
inline void scalar_div(fmpq_t a, const fmpq_t b, const fmpq_t c) { fmpq_div(a, b, c); }

template <typename T>
inline void scalar_div(scalar_s<T>* a, const scalar_s<T>* b, const scalar_s<T>* c, const field_t field) {
	for (size_t i = 0; i < field->rank; i++)
		scalar_div(a->data + i, b->data + i, c->data + i, field);
}

#endif