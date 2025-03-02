/*
    Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

    This file is part of Sparse_rref. The Sparse_rref is free software:
    you can redistribute it and/or modify it under the terms of the MIT
    License.
*/

#ifndef SCALAR_H
#define SCALAR_H

#include <string>
#include "flint/nmod.h"
#include "flint/fmpz.h"
#include "flint/fmpq.h"

// a simple wrapper for flint's fmpz and fmpq
namespace Flint {
    // for to_string, get_str and output, caution: the buffer is shared
    // multi-threading should be careful
    constexpr size_t MAX_STR_LEN = 1024;
    static char _buf[MAX_STR_LEN];

    // some concepts
    template<typename T, typename... Ts>
    concept IsOneOf = (std::same_as<T, Ts> || ...);

    template<typename T>
    concept flint_bare_type = IsOneOf<T, fmpz, fmpq>;

    template<typename T>
    concept flint_pointer_type = IsOneOf<T, fmpz_t, fmpq_t>;

    // builtin number types
    template <typename T>
    concept builtin_number = std::is_arithmetic_v<T>;

    template <typename T>
    concept builtin_integral = std::is_integral_v<T>;

    template <typename T>
    concept signed_builtin_integral = builtin_integral<T> && std::is_signed_v<T>;

    template <typename T>
    concept unsigned_builtin_integral = builtin_integral<T> && std::is_unsigned_v<T>;

#define NUM_INIT(type1, type2, fh, fd)                            \
    type1(const type2 a) {                                        \
        fh##_init(_data);                                         \
        fh##_set##fd(_data, a);                                   \
    }

#define NUM_OP_0(type1, op, type2, func, tail)                    \
    type1 operator##op(const type2 other) const {                 \
        type1 result;                                             \
        func(result._data, _data, other##tail);                   \
        return result;                                            \
    }

#define NUM_OP_00(type1, type2, fh, ft, tail)                     \
    NUM_OP_0(type1, +, type2, fh##_add##ft, tail);                \
    NUM_OP_0(type1, -, type2, fh##_sub##ft, tail);                \
    NUM_OP_0(type1, *, type2, fh##_mul##ft, tail)                 \

#define NUM_OP_01(type1, type2, fh, ft, tail)                     \
    NUM_OP_00(type1, type2, fh, ft, tail);                        \
    NUM_OP_0(type1, /, type2, fh##_mul##ft, tail)                 \

#define NUM_OP_1(type1, op, type2, func, tail)                    \
    type1 operator##op(const type2 other) const {                 \
        type1 result;                                             \
        func(result._data, _data, other##tail, FLOAT_PARA::prec); \
        return result;                                            \
    }

#define NUM_OP_10(type1, type2, fh, ft, tail)                     \
    NUM_OP_1(type1, +, type2, fh##_add##ft, tail);                \
    NUM_OP_1(type1, -, type2, fh##_sub##ft, tail);                \
    NUM_OP_1(type1, *, type2, fh##_mul##ft, tail)                 \

#define NUM_OP_11(type1, type2, fh, ft, tail)                     \
    NUM_OP_10(type1, type2, fh, ft, tail);                        \
    NUM_OP_1(type1, /, type2, fh##_div##ft, tail)                 \

#define NUM_SELF_OP_0(type1, op, type2, func, tail)               \
    type1& operator##op(const type2 other) {                      \
        func(_data, _data, other##tail);                          \
        return *this;                                             \
    }

#define NUM_SELF_OP_00(type1, type2, fh, ft, tail)                \
    NUM_SELF_OP_0(type1, +=, type2, fh##_add##ft, tail);          \
    NUM_SELF_OP_0(type1, -=, type2, fh##_sub##ft, tail);          \
    NUM_SELF_OP_0(type1, *=, type2, fh##_mul##ft, tail)           \

#define NUM_SELF_OP_01(type1, type2, fh, ft, tail)                \
    NUM_SELF_OP_00(type1, type2, fh, ft, tail);                   \
    NUM_SELF_OP_0(type1, /=, type2, fh##_div##ft, tail)           \

#define NUM_CMP_0(type, op, num, func, tail)                      \
    bool operator##op(const type other) const {                   \
        return func(_data, other##tail) op num;                   \
    }

#define NUM_EQUAL_0(type, func, tail)                             \
    bool operator==(const type other) const {                     \
        return func(_data, other##tail);                          \
    }

#define NUM_EQUAL_1(type1, type2, fh, ft)                         \
    bool operator==(const type2 other) const {                    \
        if (other == 0) {                                         \
            return fh##_is_zero(_data);                           \
        }                                                         \
        if (other == 1) {                                         \
            return fh##_is_one(_data);                            \
        }                                                         \
        return fh##_equal##ft(_data, other);                      \
    }

#define NUM_CMP_01(type1, type2, fh, ft, tail)                    \
    NUM_CMP_0(type2, < , 0, fh##_cmp##ft, tail);                  \
    NUM_CMP_0(type2, > , 0, fh##_cmp##ft, tail);                  \
    NUM_CMP_0(type2, <= , 0, fh##_cmp##ft, tail);                 \
    NUM_CMP_0(type2, >= , 0, fh##_cmp##ft, tail)

#define NUM_SET_0(type1, type2, fh, ft, tail)                     \
    type1& operator=(const type2 a) {                             \
        fh##_set##ft(_data, a);                                   \
        return *this;                                             \
    }

#define NUM_FUNC_0(type1, type2, func)                            \
    type1 func() const {                                          \
        type1 result;                                             \
        type2##_##func(result._data, _data);                      \
        return result;                                            \
    }

#define NUM_MACRO_s(MACRO, type1, type2)                          \
    MACRO(type1, int, type2, _si);                                \
    MACRO(type1, long, type2, _si);                               \
    MACRO(type1, slong, type2, _si)    

#define NUM_MACRO_u(MACRO, type1, type2)                          \
    MACRO(type1, unsigned int, type2, _ui);                       \
    MACRO(type1, unsigned long, type2, _ui);                      \
    MACRO(type1, ulong, type2, _ui)

    struct int_t {
        fmpz_t _data;

        void init() { fmpz_init(_data); }
        int_t() { fmpz_init(_data); }
		void clear() { fmpz_clear(_data); }
        ~int_t() { fmpz_clear(_data); }
        int_t(const int_t& other) { fmpz_init(_data); fmpz_set(_data, other._data); }
        int_t(int_t&& other) noexcept { fmpz_init(_data); fmpz_swap(_data, other._data); }

        NUM_INIT(int_t, fmpz_t, fmpz);
        NUM_MACRO_s(NUM_INIT, int_t, fmpz);
        NUM_MACRO_u(NUM_INIT, int_t, fmpz);

        int_t(const std::string& str) { fmpz_init(_data); fmpz_set_str(_data, str.c_str(), 10); }
        void set_str(const std::string& str, int base = 10) { fmpz_set_str(_data, str.c_str(), base); }

        int_t& operator=(const int_t& other) { if (this != &other) fmpz_set(_data, other._data); return *this; }
        int_t& operator=(int_t&& other) noexcept { if (this != &other) fmpz_swap(_data, other._data); return *this; }
        NUM_MACRO_s(NUM_SET_0, int_t, fmpz);
        NUM_MACRO_u(NUM_SET_0, int_t, fmpz);

        NUM_EQUAL_0(int_t&, fmpz_equal, ._data);
        NUM_MACRO_s(NUM_EQUAL_1, , fmpz);
        NUM_MACRO_u(NUM_EQUAL_1, , fmpz);

        NUM_CMP_01(, int_t&, fmpz, , ._data);
        NUM_MACRO_s(NUM_CMP_01, , fmpz);
        NUM_MACRO_u(NUM_CMP_01, , fmpz);

        NUM_OP_00(int_t, int_t&, fmpz, , ._data);
        NUM_MACRO_s(NUM_OP_00, int_t, fmpz);
        NUM_MACRO_u(NUM_OP_00, int_t, fmpz);

        NUM_SELF_OP_00(int_t, int_t&, fmpz, , ._data);
        NUM_MACRO_s(NUM_SELF_OP_00, int_t, fmpz);
        NUM_MACRO_u(NUM_SELF_OP_00, int_t, fmpz);

        template <unsigned_builtin_integral T>
        int_t pow(const T n) const { int_t result; fmpz_pow_ui(result._data, _data, n); return result; }
        int_t pow(const int_t& n) const { int_t result; fmpz_pow_fmpz(result._data, _data, n._data); return result; }
        NUM_FUNC_0(int_t, fmpz, abs);
        NUM_FUNC_0(int_t, fmpz, neg);

		ulong operator%(const nmod_t other) const { return fmpz_get_nmod(_data, other); }
		int_t operator%(const int_t& other) const { int_t result; fmpz_mod(result._data, _data, other._data); return result; }

        int_t operator-() const { return neg(); }

        std::string get_str(int base = 10, bool thread_safe = false) const {
            auto len = fmpz_sizeinbase(_data, base) + 3;

            if (thread_safe || len > MAX_STR_LEN - 4) {
                char* str = (char*)malloc(len * sizeof(char));
                fmpz_get_str(str, base, _data);
                std::string result(str);
                free(str);
                return result;
            }
            else {
                fmpz_get_str(_buf, base, _data);
                std::string result(_buf);
                return result;
            }
        }
    };

    struct rat_t {
        fmpq_t _data;

		void init() { fmpq_init(_data); }
        rat_t() { fmpq_init(_data); }
        void clear() { fmpq_clear(_data); }
        ~rat_t() { fmpq_clear(_data); }

        rat_t(const rat_t& other) { fmpq_init(_data); fmpq_set(_data, other._data); }
        rat_t(rat_t&& other) noexcept { fmpq_init(_data); fmpq_swap(_data, other._data); }
        NUM_INIT(rat_t, fmpq_t, fmpq);

        template <signed_builtin_integral T> rat_t(const T a, const T b) { fmpq_init(_data); fmpq_set_si(_data, a, b); }
        template <signed_builtin_integral T> rat_t(const T a) { fmpq_init(_data); fmpq_set_si(_data, a, 1); }
        template <unsigned_builtin_integral T> rat_t(const T a, const T b) { fmpq_init(_data); fmpq_set_ui(_data, a, b); }
        template <unsigned_builtin_integral T> rat_t(const T a) { fmpq_init(_data); fmpq_set_ui(_data, a, 1); }

        rat_t(const std::string& str) { fmpq_init(_data); fmpq_set_str(_data, str.c_str(), 10); }
        void set_str(const std::string& str, int base = 10) { fmpq_set_str(_data, str.c_str(), base); }

        int_t num() const { return fmpq_numref(_data); }
        int_t den() const { return fmpq_denref(_data); }

        rat_t& operator=(const rat_t& other) { if (this != &other) fmpq_set(_data, other._data); return *this; }
        rat_t& operator=(rat_t&& other) noexcept { if (this != &other) fmpq_swap(_data, other._data); return *this; }

        template <signed_builtin_integral T>
        rat_t& operator=(const T a) { fmpq_set_si(_data, a, 1); return *this; }
        template <unsigned_builtin_integral T>
        rat_t& operator=(const T a) { fmpq_set_ui(_data, a, 1); return *this; }

        NUM_OP_01(rat_t, rat_t&, fmpq, , ._data);
        NUM_OP_01(rat_t, int_t&, fmpq, _fmpz, ._data);
        NUM_MACRO_s(NUM_OP_00, rat_t, fmpq);
        NUM_MACRO_u(NUM_OP_00, rat_t, fmpq);

        NUM_SELF_OP_01(rat_t, rat_t&, fmpq, , ._data);
        NUM_SELF_OP_01(rat_t, int_t&, fmpq, _fmpz, ._data);
        NUM_MACRO_s(NUM_SELF_OP_00, rat_t, fmpq);
        NUM_MACRO_u(NUM_SELF_OP_00, rat_t, fmpq);

        NUM_CMP_01(, rat_t&, fmpq, , ._data);
        NUM_CMP_01(, int_t&, fmpq, _fmpz, ._data);
        NUM_MACRO_s(NUM_CMP_01, , fmpq);
        NUM_MACRO_u(NUM_CMP_01, , fmpq);

        bool operator==(const rat_t& other) const { return fmpq_equal(_data, other._data); }
        bool operator==(const int_t& other) const { return fmpq_equal_fmpz((fmpq*)_data, (fmpz*)other._data); }
        template <builtin_integral T> bool operator==(const T other) const {
            if (other == 0) {
                return fmpq_is_zero(_data);
            } if (other == 1) {
                return fmpq_is_one(_data);
            } return fmpq_equal_si((fmpq*)_data, other);
        };
        template<typename T> bool operator!=(const T other) const { return !operator==(other); }

        rat_t pow(const int_t& n) const { rat_t result; fmpq_pow_fmpz(result._data, _data, n._data); return result; }
        template <signed_builtin_integral T>
        rat_t pow(const T n) const { rat_t result; fmpq_pow_si(result._data, _data, n); return result; }
        NUM_FUNC_0(rat_t, fmpq, abs);
        NUM_FUNC_0(rat_t, fmpq, neg);
        NUM_FUNC_0(rat_t, fmpq, inv);

        rat_t operator-() const { return neg(); }

        std::string get_str(int base = 10, bool thread_safe = false) const {
            auto len = fmpz_sizeinbase(fmpq_numref(_data), base) +
                fmpz_sizeinbase(fmpq_denref(_data), base) + 3;

            if (thread_safe || len > MAX_STR_LEN - 4) {
                char* str = (char*)malloc(len * sizeof(char));
                fmpq_get_str(nullptr, base, _data);
                std::string result(str);
                free(str);
                return result;
            }
            else {
                fmpq_get_str(_buf, base, _data);
                std::string result(_buf);
                return result;
            }
        }
    };

    // our number types
    template<typename T>
    concept Flint_type = IsOneOf<T, int_t, rat_t>;

    template <typename T>
    T& operator<< (T& os, const int_t& i) {
        auto len = fmpz_sizeinbase(i._data, 10);
        if (len > MAX_STR_LEN - 4) {
            char* str = fmpz_get_str(nullptr, 10, i._data);
            os << str;
            flint_free(str);
        }
        else {
            fmpz_get_str(_buf, 10, i._data);
            os << _buf;
        }
        return os;
    }

    template <typename T>
    T& operator<< (T& os, const rat_t& r) {
        auto len = fmpz_sizeinbase(fmpq_numref(r._data), 10)
            + fmpz_sizeinbase(fmpq_denref(r._data), 10) + 3;

        if (len > MAX_STR_LEN - 4) {
            char* str = fmpq_get_str(nullptr, 10, r._data);
            os << str;
            flint_free(str);
        }
        else {
            fmpq_get_str(_buf, 10, r._data);
            os << _buf;
        }

        return os;
    }

    template <typename T, Flint_type S> S operator+(const T r, const S& c) { return c + r; }
    template <typename T, Flint_type S> S operator-(const T r, const S& c) { return (-c) + r; }
    template <typename T, Flint_type S> S operator*(const T r, const S& c) { return c * r; }
    template <builtin_number T, Flint_type S> S operator/(const T r, const S& c) { return (r == 1 ? c.inv() : S(r) / c); }
    template <typename T, Flint_type S> S pow(const S& c, const T& r) { return c.pow(r); }

    rat_t operator/(const int_t& r, const int_t& c) {
        rat_t result;
        fmpq_set_fmpz_frac(result._data, r._data, c._data);
        return result;
    }

    // other functions

    int_t factorial(const ulong n) {
        int_t result;
        fmpz_fac_ui(result._data, n);
        return result;
    }

	int rational_reconstruct(rat_t& q, const int_t& a, const int_t& mod) {
        return  fmpq_reconstruct_fmpz(q._data, a._data, mod._data);
	}

    // CRT
	int_t CRT(const int_t& r1, const int_t& m1, ulong r2, ulong m2) {
		int_t result;
        fmpz_CRT_ui(result._data, r1._data, m1._data, r2, m2, 0);
		return result;
	}

    int_t CRT(const int_t& r1, const int_t& m1, const int_t& r2, const int_t& m2) {
        int_t result;
        fmpz_CRT(result._data, r1._data, m1._data, r2._data, m2._data, 0);
        return result;
    }

#undef NUM_INIT
#undef NUM_OP_0
#undef NUM_OP_00
#undef NUM_OP_01
#undef NUM_OP_1
#undef NUM_OP_10
#undef NUM_OP_11
#undef NUM_SELF_OP_0
#undef NUM_SELF_OP_00
#undef NUM_SELF_OP_01
#undef NUM_SELF_OP_1
#undef NUM_SELF_OP_10
#undef NUM_SELF_OP_11
#undef NUM_CMP_0
#undef NUM_EQUAL_0
#undef NUM_EQUAL_1
#undef NUM_EQUAL_1s
#undef NUM_EQUAL_1u
#undef NUM_CMP_01
#undef NUM_SET_0
#undef NUM_SET_01
#undef NUM_FUNC_0
#undef NUM_FUNC_1
#undef NUM_MACRO_s
#undef NUM_MACRO_u
#undef NUM_MACRO_d

}

namespace sparse_rref {

    // field
    enum RING {
        FIELD_QQ,    // fmpq
        FIELD_Fp,    // ulong
        OTHER_RING // not implemented now
    };

    struct field_struct {
        enum RING ring;
        nmod_t mod;
    };
    typedef struct field_struct field_t[1];

    static inline void field_init(field_t field, const enum RING ring, ulong p) {
        field->ring = ring;
        nmod_init(&(field->mod), p);
    }

    // scalar
    using rat_t = Flint::rat_t;
    using int_t = Flint::int_t;

    // TODO: avoid copy
    static inline std::string scalar_to_str(const rat_t& a) { return a.get_str(10, true); }
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
}

#endif