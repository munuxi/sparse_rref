#ifndef SCALAR_H
#define SCALAR_H

#include "util.h"

inline bool scalar_is_zero(fmpq_t a) { return fmpq_is_zero(a); }
inline bool scalar_is_zero(ulong* a) { return (*a) == 0; }

inline bool scalar_equal(fmpq_t a, fmpq_t b) { return fmpq_equal(a, b); }
inline bool scalar_equal(ulong* a, ulong* b) { return (*a) == (*b); }

inline void scalar_set(fmpq_t a, const fmpq_t b) { fmpq_set(a, b); }
inline void scalar_set(ulong* a, const ulong* b) { *a = *b; }

inline void scalar_one(fmpq_t a) { fmpq_one(a); }
inline void scalar_one(ulong* a) { *a = 1; }

inline void scalar_zero(fmpq_t a) { fmpq_zero(a); }
inline void scalar_zero(ulong* a) { *a = 0; }

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