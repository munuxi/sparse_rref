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

    struct int_t {
        fmpz_t _data;

        void init() { fmpz_init(_data); }
        int_t() { fmpz_init(_data); }
		void clear() { fmpz_clear(_data); }
        ~int_t() { fmpz_clear(_data); }
        int_t(const int_t& other) { fmpz_init(_data); fmpz_set(_data, other._data); }
        int_t(int_t&& other) noexcept { fmpz_init(_data); fmpz_swap(_data, other._data); }
		int_t(const fmpz_t a) { fmpz_init(_data); fmpz_set(_data, a); }

		template <signed_builtin_integral T> int_t(const T a) { fmpz_init(_data); fmpz_set_si(_data, a); }
		template <unsigned_builtin_integral T> int_t(const T a) { fmpz_init(_data); fmpz_set_ui(_data, a); }

        void set_str(const std::string& str, int base = 10) { fmpz_set_str(_data, str.c_str(), base); }
		int_t(const std::string& str) { set_str(str); }

        int_t& operator=(const int_t& other) { if (this != &other) fmpz_set(_data, other._data); return *this; }
        int_t& operator=(int_t&& other) noexcept { if (this != &other) fmpz_swap(_data, other._data); return *this; }
		template <signed_builtin_integral T> int_t& operator=(const T a) { fmpz_set_si(_data, a); return *this; }
		template <unsigned_builtin_integral T> int_t& operator=(const T a) { fmpz_set_ui(_data, a); return *this; }

		bool operator==(const int_t other) const { return fmpz_equal(_data, other._data); }
		template <unsigned_builtin_integral T> bool operator==(const T other) const { return fmpz_equal_ui(_data, other); }
		template <signed_builtin_integral T> bool operator==(const T other) const { return fmpz_equal_si(_data, other); }
		bool operator!=(const int_t other) const { return !operator==(other); }

		auto operator<=>(const int_t other) const { return fmpz_cmp(_data, other._data) <=> 0; }
		template <unsigned_builtin_integral T> auto operator<=>(const T other) const { return fmpz_cmp_ui(_data, other) <=> 0; }
		template <signed_builtin_integral T> auto operator<=>(const T other) const { return fmpz_cmp_si(_data, other) <=> 0; }

		int_t operator+(const int_t& other) const { int_t result; fmpz_add(result._data, _data, other._data); return result; }
		template <unsigned_builtin_integral T> int_t operator+(const T other) const { int_t result; fmpz_add_ui(result._data, _data, other); return result; }
		template <signed_builtin_integral T> int_t operator+(const T other) const { int_t result; fmpz_add_si(result._data, _data, other); return result; }

		int_t operator-(const int_t& other) const { int_t result; fmpz_sub(result._data, _data, other._data); return result; }
		template <unsigned_builtin_integral T> int_t operator-(const T other) const { int_t result; fmpz_sub_ui(result._data, _data, other); return result; }
		template <signed_builtin_integral T> int_t operator-(const T other) const { int_t result; fmpz_sub_si(result._data, _data, other); return result; }

		int_t operator*(const int_t& other) const { int_t result; fmpz_mul(result._data, _data, other._data); return result; }
		template <unsigned_builtin_integral T> int_t operator*(const T other) const { int_t result; fmpz_mul_ui(result._data, _data, other); return result; }
		template <signed_builtin_integral T> int_t operator*(const T other) const { int_t result; fmpz_mul_si(result._data, _data, other); return result; }

		void operator+=(const int_t& other) { fmpz_add(_data, _data, other._data); }
		template <unsigned_builtin_integral T> void operator+=(const T other) { fmpz_add_ui(_data, _data, other); }
		template <signed_builtin_integral T> void operator+=(const T other) { fmpz_add_si(_data, _data, other); }

		void operator-=(const int_t& other) { fmpz_sub(_data, _data, other._data); }
		template <unsigned_builtin_integral T> void operator-=(const T other) { fmpz_sub_ui(_data, _data, other); }
		template <signed_builtin_integral T> void operator-=(const T other) { fmpz_sub_si(_data, _data, other); }

		void operator*=(const int_t& other) { fmpz_mul(_data, _data, other._data); }
		template <unsigned_builtin_integral T> void operator*=(const T other) { fmpz_mul_ui(_data, _data, other); }
		template <signed_builtin_integral T> void operator*=(const T other) { fmpz_mul_si(_data, _data, other); }

        template <unsigned_builtin_integral T>
        int_t pow(const T n) const { int_t result; fmpz_pow_ui(result._data, _data, n); return result; }
        int_t pow(const int_t& n) const { int_t result; fmpz_pow_fmpz(result._data, _data, n._data); return result; }
		int_t abs() const { int_t result; fmpz_abs(result._data, _data); return result; }
		int_t neg() const { int_t result; fmpz_neg(result._data, _data); return result; }

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
		rat_t(const fmpq_t a) { fmpq_init(_data); fmpq_set(_data, a); }

        template <signed_builtin_integral T> rat_t(const T a, const T b) { fmpq_init(_data); fmpq_set_si(_data, a, b); }
        template <signed_builtin_integral T> rat_t(const T a) { fmpq_init(_data); fmpq_set_si(_data, a, 1); }
        template <unsigned_builtin_integral T> rat_t(const T a, const T b) { fmpq_init(_data); fmpq_set_ui(_data, a, b); }
        template <unsigned_builtin_integral T> rat_t(const T a) { fmpq_init(_data); fmpq_set_ui(_data, a, 1); }

        void set_str(const std::string& str, int base = 10) { fmpq_set_str(_data, str.c_str(), base); }
        rat_t(const std::string& str) { set_str(str); }

        int_t num() const { return fmpq_numref(_data); }
        int_t den() const { return fmpq_denref(_data); }

        rat_t& operator=(const rat_t& other) { if (this != &other) fmpq_set(_data, other._data); return *this; }
        rat_t& operator=(rat_t&& other) noexcept { if (this != &other) fmpq_swap(_data, other._data); return *this; }

        template <signed_builtin_integral T> rat_t& operator=(const T a) { fmpq_set_si(_data, a, 1); return *this; }
        template <unsigned_builtin_integral T> rat_t& operator=(const T a) { fmpq_set_ui(_data, a, 1); return *this; }

        auto operator<=>(const rat_t& other) const { return fmpq_cmp(_data, other._data) <=> 0; }
        template <unsigned_builtin_integral T> auto operator<=>(const T other) const { return fmpq_cmp_ui(_data, other) <=> 0; }
        template <signed_builtin_integral T> auto operator<=>(const T other) const { return fmpq_cmp_si(_data, other) <=> 0; }
        auto operator<=>(const int_t& other) const { return fmpq_cmp_fmpz(_data, other._data) <=> 0; }

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

		rat_t operator+(const rat_t& other) const { rat_t result; fmpq_add(result._data, _data, other._data); return result; }
		template <unsigned_builtin_integral T> rat_t operator+(const T other) const { rat_t result; fmpq_add_ui(result._data, _data, other); return result; }
		template <signed_builtin_integral T> rat_t operator+(const T other) const { rat_t result; fmpq_add_si(result._data, _data, other); return result; }
        rat_t operator+(const int_t& other) const { rat_t result; fmpq_add_fmpz(result._data, _data, other._data); return result; }

		rat_t operator-(const rat_t& other) const { rat_t result; fmpq_sub(result._data, _data, other._data); return result; }
		template <unsigned_builtin_integral T> rat_t operator-(const T other) const { rat_t result; fmpq_sub_ui(result._data, _data, other); return result; }
		template <signed_builtin_integral T> rat_t operator-(const T other) const { rat_t result; fmpq_sub_si(result._data, _data, other); return result; }
		rat_t operator-(const int_t& other) const { rat_t result; fmpq_sub_fmpz(result._data, _data, other._data); return result; }

		rat_t operator*(const rat_t& other) const { rat_t result; fmpq_mul(result._data, _data, other._data); return result; }
		template <unsigned_builtin_integral T> rat_t operator*(const T other) const { rat_t result; fmpq_mul_ui(result._data, _data, other); return result; }
		template <signed_builtin_integral T> rat_t operator*(const T other) const { rat_t result; fmpq_mul_si(result._data, _data, other); return result; }
        rat_t operator*(const int_t& other) const { rat_t result; fmpq_mul_fmpz(result._data, _data, other._data); return result; }

		rat_t operator/(const rat_t& other) const { rat_t result; fmpq_div(result._data, _data, other._data); return result; }
		template <unsigned_builtin_integral T> rat_t operator/(const T other) const { rat_t result; fmpq_div_ui(result._data, _data, other); return result; }
		template <signed_builtin_integral T> rat_t operator/(const T other) const { rat_t result; fmpq_div_si(result._data, _data, other); return result; }
		rat_t operator/(const int_t& other) const { rat_t result; fmpq_div_fmpz(result._data, _data, other._data); return result; }

		void operator+=(const rat_t& other) { fmpq_add(_data, _data, other._data); }
		template <unsigned_builtin_integral T> void operator+=(const T other) { fmpq_add_ui(_data, _data, other); }
		template <signed_builtin_integral T> void operator+=(const T other) { fmpq_add_si(_data, _data, other); }
		void operator+=(const int_t& other) { fmpq_add_fmpz(_data, _data, other._data); }

		void operator-=(const rat_t& other) { fmpq_sub(_data, _data, other._data); }
		template <unsigned_builtin_integral T> void operator-=(const T other) { fmpq_sub_ui(_data, _data, other); }
		template <signed_builtin_integral T> void operator-=(const T other) { fmpq_sub_si(_data, _data, other); }
		void operator-=(const int_t& other) { fmpq_sub_fmpz(_data, _data, other._data); }

		void operator*=(const rat_t& other) { fmpq_mul(_data, _data, other._data); }
		template <unsigned_builtin_integral T> void operator*=(const T other) { fmpq_mul_ui(_data, _data, other); }
		template <signed_builtin_integral T> void operator*=(const T other) { fmpq_mul_si(_data, _data, other); }
		void operator*=(const int_t& other) { fmpq_mul_fmpz(_data, _data, other._data); }

		void operator/=(const rat_t& other) { fmpq_div(_data, _data, other._data); }
		template <unsigned_builtin_integral T> void operator/=(const T other) { fmpq_div_ui(_data, _data, other); }
		template <signed_builtin_integral T> void operator/=(const T other) { fmpq_div_si(_data, _data, other); }
		void operator/=(const int_t& other) { fmpq_div_fmpz(_data, _data, other._data); }

        ulong operator%(const nmod_t other) const { 
			auto nummod = num() % other;
			auto denmod = den() % other;
            return nmod_div(nummod, denmod, other);
        }

        rat_t pow(const int_t& n) const { rat_t result; fmpq_pow_fmpz(result._data, _data, n._data); return result; }
        template <signed_builtin_integral T>
        rat_t pow(const T n) const { rat_t result; fmpq_pow_si(result._data, _data, n); return result; }
		rat_t inv() const { rat_t result; fmpq_inv(result._data, _data); return result; }
		rat_t abs() const { rat_t result; fmpq_abs(result._data, _data); return result; }
		rat_t neg() const { rat_t result; fmpq_neg(result._data, _data); return result; }

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

    static inline ulong scalar_neg(const ulong b, const field_t field) {
        return field->mod.n - b;
    }
    static inline rat_t scalar_neg(const rat_t& b, const field_t field) { return -b; }

    static inline ulong scalar_inv(const ulong b, const field_t field) {
        return nmod_inv(b, field->mod);
    }
    static inline rat_t scalar_inv(const rat_t& b, const field_t field) { return b.inv(); }

    static inline ulong scalar_add(const ulong b, const ulong c, const field_t field) {
        return _nmod_add(b, c, field->mod);
    }
    static inline rat_t scalar_add(const rat_t& b, const rat_t& c, const field_t field) { return b + c; }

    static inline ulong scalar_sub(const ulong b, const ulong c, const field_t field) {
        return _nmod_sub(b, c, field->mod);
    }
    static inline rat_t scalar_sub(const rat_t& b, const rat_t& c, const field_t field) { return b - c; }

    static inline ulong scalar_mul(const ulong b, const ulong c, const field_t field) {
        return nmod_mul(b, c, field->mod);
    }
    static inline rat_t scalar_mul(const rat_t& b, const rat_t& c, const field_t field) { return b * c; }

    static inline ulong scalar_div(const ulong b, const ulong c, const field_t field) {
        return nmod_div(b, c, field->mod);
    }
    static inline rat_t scalar_div(const rat_t& b, const rat_t& c, const field_t field) { return b / c; }
}

#endif