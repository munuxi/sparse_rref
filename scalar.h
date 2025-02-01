/*
	Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

	This file is part of Sparse_rref. The Sparse_rref is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

#ifndef SCALAR_H
#define SCALAR_H

#include <type_traits>

#include "flint/fmpq.h"
#include "flint/nmod.h"
#include "flint/ulong_extras.h"

#include "sparse_rref.h"

using namespace sparse_rref;

// field

static inline void field_init(field_t field, const enum RING ring, const ulong rank = 1, const ulong* pvec = NULL) {
	field->ring = ring;
	field->rank = rank;
	if (field->ring == FIELD_Fp || field->ring == RING_MulitFp) {
		field->pvec = s_malloc<nmod_t>(rank);
		for (ulong i = 0; i < rank; i++)
			nmod_init(field->pvec + i, pvec[i]);
	}
}

static inline void field_init(field_t field, const enum RING ring, const std::vector<ulong>& pvec) {
	field->ring = ring;
	field->rank = pvec.size();
	if (field->ring == FIELD_Fp || field->ring == RING_MulitFp) {
		field->pvec = s_malloc<nmod_t>(field->rank);
		for (ulong i = 0; i < field->rank; i++)
			nmod_init(field->pvec + i, pvec[i]);
	}
}

static inline void field_clear(field_t field) {
	s_free(field->pvec);
	field->pvec = NULL;
}

static inline void field_set(field_t field, const field_t ff) {
	field->ring = ff->ring;
	field->rank = ff->rank;
	if (field->ring == FIELD_Fp || field->ring == RING_MulitFp) {
		field->pvec = s_realloc(field->pvec, field->rank);
		for (ulong i = 0; i < field->rank; i++)
			nmod_init(field->pvec + i, ff->pvec[i].n);
	}
}

// scalar

static inline void scalar_init(fmpq_t a) { fmpq_init(a); }
static inline void scalar_init(fmpz_t a) { fmpz_init(a); }
static inline void scalar_init(ulong* a) { return; } // do nothing

static inline void scalar_clear(fmpq_t a) { fmpq_clear(a); }
static inline void scalar_clear(fmpz_t a) { fmpz_clear(a); }
static inline void scalar_clear(ulong* a) { return; } // do nothing

// TODO: avoid copy
static inline std::string scalar_to_str(fmpq_t a) {
	char* cstr = s_malloc<char>(fmpz_sizeinbase(fmpq_numref(a), 10) + fmpz_sizeinbase(fmpq_denref(a), 10) + 3);
	fmpq_get_str(cstr, 10, a);
	std::string str(cstr);
	s_free(cstr);
	return str;
}
static inline std::string scalar_to_str(ulong* a) { return std::to_string(*a); }

static inline bool scalar_is_zero(const fmpq_t a) { return fmpq_is_zero(a); }
static inline bool scalar_is_zero(const fmpz_t a) { return fmpz_is_zero(a); }
static inline bool scalar_is_zero(const ulong* a) { return (*a) == 0; }

static inline bool scalar_equal(const fmpq_t a, const fmpq_t b) { return fmpq_equal(a, b); }
static inline bool scalar_equal(const fmpz_t a, const fmpz_t b) { return fmpz_equal(a, b); }
static inline bool scalar_equal(const ulong* a, const ulong* b) { return (*a) == (*b); }

static inline void scalar_set(fmpq_t a, const fmpq_t b) { fmpq_set(a, b); }
static inline void scalar_set(fmpz_t a, const fmpz_t b) { fmpz_set(a, b); }
static inline void scalar_set(ulong* a, const ulong* b) { *a = *b; }
template <typename T>
inline void scalar_set(T* a, const T* b, const ulong len) {
	for (ulong i = 0; i < len; i++)
		scalar_set(a + i, b + i);
}

static inline void scalar_zero(fmpq_t a) { fmpq_zero(a); }
static inline void scalar_zero(fmpz_t a) { fmpz_zero(a); }
static inline void scalar_zero(ulong* a) { *a = 0; }

static inline void scalar_one(fmpq_t a) { fmpq_one(a); }
static inline void scalar_one(fmpz_t a) { fmpz_one(a); }
static inline void scalar_one(ulong* a) { *a = 1; }

// arithmetic

static inline void scalar_neg(ulong* a, const ulong* b, const field_t field) {
	*a = field->pvec[0].n - *b;
}
static inline void scalar_neg(fmpq_t a, const fmpq_t b, const field_t field) { fmpq_neg(a, b); }

static inline void scalar_inv(ulong* a, const ulong* b, const field_t field) {
	*a = nmod_inv(*b, field->pvec[0]);
}
static inline void scalar_inv(fmpq_t a, const fmpq_t b, const field_t field) { fmpq_inv(a, b); }

static inline void scalar_add(ulong* a, const ulong* b, const ulong* c, const field_t field) {
	*a = _nmod_add(*b, *c, field->pvec[0]);
}
static inline void scalar_add(fmpq_t a, const fmpq_t b, const fmpq_t c, const field_t field) { fmpq_add(a, b, c); }

static inline void scalar_sub(ulong* a, const ulong* b, const ulong* c, const field_t field) {
	*a = _nmod_sub(*b, *c, field->pvec[0]);
}
static inline void scalar_sub(fmpq_t a, const fmpq_t b, const fmpq_t c, const field_t field) { fmpq_sub(a, b, c); }

static inline void scalar_mul(ulong* a, const ulong* b, const ulong* c, const field_t field) {
	*a = nmod_mul(*b, *c, field->pvec[0]);
}
static inline void scalar_mul(fmpq_t a, const fmpq_t b, const fmpq_t c, const field_t field) { fmpq_mul(a, b, c); }

static inline void scalar_div(ulong* a, const ulong* b, const ulong* c, const field_t field) {
	*a = nmod_div(*b, *c, field->pvec[0]);
}
static inline void scalar_div(fmpq_t a, const fmpq_t b, const fmpq_t c, const field_t field) { fmpq_div(a, b, c); }

#endif