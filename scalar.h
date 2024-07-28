#ifndef SCALAR_H
#define SCALAR_H

#include "util.h"

inline bool scalar_is_zero(fmpq_t a) { return fmpq_is_zero(a); }

inline bool scalar_is_zero(ulong *a) { return (*a) == 0; }

inline void scalar_set(fmpq_t a, const fmpq_t b) { fmpq_set(a, b); }

inline void scalar_set(ulong *a, const ulong *b) { *a = *b; }

inline void scalar_add(fmpq_t a, const fmpq_t b, const fmpq_t c) {
    fmpq_add(a, b, c);
}

inline void scalar_add(ulong *a, const ulong *b, const ulong *c) {
    *a = (*b) + (*c);
}

inline void scalar_sub(fmpq_t a, const fmpq_t b, const fmpq_t c) {
    fmpq_sub(a, b, c);
}

inline void scalar_sub(ulong *a, const ulong *b, const ulong *c) {
    *a = (*b) - (*c);
}

inline void scalar_mul(fmpq_t a, const fmpq_t b, const fmpq_t c) {
    fmpq_mul(a, b, c);
}

inline void scalar_mul(ulong *a, const ulong *b, const ulong *c) {
    *a = (*b) * (*c);
}

#endif