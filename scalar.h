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

#include "flint_wrapper.h"

#include "sparse_rref.h"

namespace sparse_rref {

	// field

	static inline void field_init(field_t field, const enum RING ring, ulong p) {
		field->ring = ring;
		nmod_init(&(field->mod), p);
	}

	// scalar


	using rat_t = Flint::rat_t;
	using int_t = Flint::int_t;

	// TODO: avoid copy
	static inline std::string scalar_to_str(const rat_t& a) { return a.get_str(10, true);}
	static inline std::string scalar_to_str(const int_t& a) { return a.get_str(10, true); }
	static inline std::string scalar_to_str(const ulong& a) { return std::to_string(a); }

	// arithmetic

	static inline void scalar_neg(ulong& a, const ulong b, const field_t field) {
		a = field->mod.n - b;
	}
	static inline void scalar_neg(rat_t& a, const rat_t& b, const field_t field) { a = -b; }

	static inline void scalar_inv(ulong& a, const ulong& b, const field_t field) {
		a = nmod_inv(b, field->mod);
	}
	static inline void scalar_inv(rat_t& a, const rat_t& b, const field_t field) { a = b.inv(); }

	static inline void scalar_add(ulong& a, const ulong b, const ulong c, const field_t field) {
		a = _nmod_add(b, c, field->mod);
	}
	static inline void scalar_add(rat_t& a, const rat_t b, const rat_t c, const field_t field) { a = b + c; }

	static inline void scalar_sub(ulong& a, const ulong b, const ulong c, const field_t field) {
		a = _nmod_sub(b, c, field->mod);
	}
	static inline void scalar_sub(rat_t& a, const rat_t b, const rat_t c, const field_t field) { a = b - c; }

	static inline void scalar_mul(ulong& a, const ulong b, const ulong c, const field_t field) {
		a = nmod_mul(b, c, field->mod);
	}
	static inline void scalar_mul(rat_t& a, const rat_t b, const rat_t c, const field_t field) { a = b * c; }

	static inline void scalar_div(ulong& a, const ulong b, const ulong c, const field_t field) {
		a = nmod_div(b, c, field->mod);
	}
	static inline void scalar_div(rat_t& a, const rat_t b, const rat_t c, const field_t field) { a = b / c; }

} // namespace sparse_rref

#endif