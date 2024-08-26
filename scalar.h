#ifndef SCALAR_H
#define SCALAR_H

#include "util.h"

inline void scalar_init(fmpq_t a) { fmpq_init(a); }
inline void scalar_init(ulong* a) { return; } // do nothing

template <typename T>
inline void scalar_init(T* a, const ulong rank) {
	for (ulong i = 0; i < rank; i++)
		scalar_init(a + i);
}

inline bool scalar_is_zero(const fmpq_t a) { return fmpq_is_zero(a); }
inline bool scalar_is_zero(const ulong* a) { return (*a) == 0; }

template <typename T>
inline bool scalar_is_zero(const T* a, const ulong rank) {
	for (ulong i = 0; i < rank; i++)
		if (!scalar_is_zero(a + i))
			return false;
	return true;
}

inline bool scalar_equal(const fmpq_t a, const fmpq_t b) { return fmpq_equal(a, b); }
inline bool scalar_equal(const ulong* a, const ulong* b) { return (*a) == (*b); }

template <typename T>
inline bool scalar_equal(const T* a, const T* b, const ulong rank) {
	for (ulong i = 0; i < rank; i++)
		if (!scalar_equal(a + i, b + i))
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

inline void scalar_zero(fmpq_t a) { fmpq_zero(a); }
inline void scalar_zero(ulong* a) { *a = 0; }

template <typename T>
inline bool scalar_zero(T* a, const ulong rank) {
	for (ulong i = 0; i < rank; i++)
		scalar_zero(a + i);
}

inline void scalar_one(fmpq_t a) { fmpq_one(a); }
inline void scalar_one(ulong* a) { *a = 1; }

template <typename T>
inline bool scalar_one(T* a, const ulong rank) {
	for (ulong i = 0; i < rank; i++)
		scalar_one(a + i);
}

// arithmetic

template <typename T>
inline void scalar_add(T a, const T b, const T c, const field_t field) {
	switch (field->ring) {
		case FIELD_QQ:
			fmpq_add(a, b, c);
			break;
		case FIELD_Fp:
			*a = nmod_add(*b, *c, field->pvec[0]);
			break;
		case RING_MulitFp:
			// not implemented now
			break;
		case FIELD_F2:
			break;
	}
}

template <typename T>
inline void scalar_sub(T a, const T b, const T c, const field_t field) {
	switch (field->ring) {
		case FIELD_QQ:
			fmpq_sub(a, b, c);
			break;
		case FIELD_Fp:
			*a = nmod_sub(*b, *c, field->pvec[0]);
			break;
		case RING_MulitFp:
			// not implemented now
			break;
		case FIELD_F2:
			break;
	}
}

template <typename T>
inline void scalar_mul(T a, const T b, const T c, const field_t field) {
	switch (field->ring) {
		case FIELD_QQ:
			fmpq_mul(a, b, c);
			break;
		case FIELD_Fp:
			*a = nmod_mul(*b, *c, field->pvec[0]);
			break;
		case RING_MulitFp:
			// not implemented now
			break;
		case FIELD_F2:
			break;
	}
}

template <typename T>
inline void scalar_div(T a, const T b, const T c, const field_t field) {
	switch (field->ring) {
		case FIELD_QQ:
			fmpq_div(a, b, c);
			break;
		case FIELD_Fp:
			*a = nmod_div(*b, *c, field->pvec[0]);
			break;
		case RING_MulitFp:
			// not implemented now
			break;
		case FIELD_F2:
			break;
	}
}

#endif