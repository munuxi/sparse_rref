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

inline void scalar_add(fmpq_t a, const fmpq_t b, const fmpq_t c) {
	fmpq_add(a, b, c);
}
inline void scalar_add(ulong* a, const ulong* b, const ulong* c) {
	*a = (*b) + (*c);
}

inline void scalar_sub(fmpq_t a, const fmpq_t b, const fmpq_t c) {
	fmpq_sub(a, b, c);
}
inline void scalar_sub(ulong* a, const ulong* b, const ulong* c) {
	*a = (*b) - (*c);
}

inline void scalar_mul(fmpq_t a, const fmpq_t b, const fmpq_t c) {
	fmpq_mul(a, b, c);
}
inline void scalar_mul(ulong* a, const ulong* b, const ulong* c) {
	*a = (*b) * (*c);
}

#endif