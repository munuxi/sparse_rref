#pragma once

#include <string>
#include "flint/fmpz.h"
#include "flint/fmpq.h"
#include "flint/arb.h"
#include "flint/acb.h"

namespace Flint {

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

#define NUM_SELF_OP_1(type1, op, type2, func, tail)               \
    type1& operator##op(const type2 other) {                      \
        func(_data, _data, other##tail, FLOAT_PARA::prec);        \
        return *this;                                             \
    }

#define NUM_SELF_OP_10(type1, type2, fh, ft, tail)                \
    NUM_SELF_OP_1(type1, +=, type2, fh##_add##ft, tail);          \
    NUM_SELF_OP_1(type1, -=, type2, fh##_sub##ft, tail);          \
    NUM_SELF_OP_1(type1, *=, type2, fh##_mul##ft, tail)           \

#define NUM_SELF_OP_11(type1, type2, fh, ft, tail)                \
    NUM_SELF_OP_10(type1, type2, fh, ft, tail);                   \
    NUM_SELF_OP_1(type1, /=, type2, fh##_div##ft, tail)           \

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

#define NUM_FUNC_1(type1, type2, func)                            \
    type1 func() const {                                          \
        type1 result;                                             \
        type2##_##func(result._data, _data, FLOAT_PARA::prec);    \
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

#define NUM_MACRO_d(MACRO, type1, type2)                          \
    MACRO(type1, float, type2, _d);                               \
    MACRO(type1, double, type2, _d)

    static char _buf[1024];

    struct integer {
        fmpz_t _data;

        integer() { fmpz_init(_data); }
        ~integer() { fmpz_clear(_data); }
        integer(const integer& other) { fmpz_init(_data); fmpz_set(_data, other._data); }
        integer(integer&& other) noexcept { fmpz_init(_data); fmpz_swap(_data, other._data); }

        NUM_INIT(integer, fmpz_t, fmpz);
        NUM_MACRO_s(NUM_INIT, integer, fmpz);
        NUM_MACRO_u(NUM_INIT, integer, fmpz);

        integer(const std::string& str) { fmpz_init(_data); fmpz_set_str(_data, str.c_str(), 10); }

        integer& operator=(const integer& other) { fmpz_set(_data, other._data); return *this; }
        integer& operator=(integer&& other) noexcept { fmpz_swap(_data, other._data); return *this; }
        NUM_MACRO_s(NUM_SET_0, integer, fmpz);
        NUM_MACRO_u(NUM_SET_0, integer, fmpz);

        NUM_EQUAL_0(integer&, fmpz_equal, ._data);
        NUM_MACRO_s(NUM_EQUAL_1, , fmpz);
        NUM_MACRO_u(NUM_EQUAL_1, , fmpz);

        NUM_CMP_01(, integer&, fmpz, , ._data);
        NUM_MACRO_s(NUM_CMP_01, , fmpz);
        NUM_MACRO_u(NUM_CMP_01, , fmpz);

        NUM_OP_00(integer, integer&, fmpz, , ._data);
        NUM_MACRO_s(NUM_OP_00, integer, fmpz);
        NUM_MACRO_u(NUM_OP_00, integer, fmpz);

        NUM_SELF_OP_00(integer, integer&, fmpz, , ._data);
        NUM_MACRO_s(NUM_SELF_OP_00, integer, fmpz);
        NUM_MACRO_u(NUM_SELF_OP_00, integer, fmpz);

        integer pow(const ulong n) const { integer result; fmpz_pow_ui(result._data, _data, n); return result; }
        NUM_FUNC_0(integer, fmpz, abs);
        NUM_FUNC_0(integer, fmpz, neg);

        integer operator-() const { return neg(); }
    };

    struct rational {
        fmpq_t _data;

        rational() { fmpq_init(_data); }
        ~rational() { fmpq_clear(_data); }

        rational(const rational& other) { fmpq_init(_data); fmpq_set(_data, other._data); }
        rational(rational&& other) noexcept { fmpq_init(_data); fmpq_swap(_data, other._data); }
        NUM_INIT(rational, fmpq_t, fmpq);

        rational(const slong a, const slong b) { fmpq_init(_data); fmpq_set_si(_data, a, b); }
        rational(const int a, const int b) { fmpq_init(_data); fmpq_set_si(_data, a, b); }
        rational(const long a, const long b) { fmpq_init(_data); fmpq_set_si(_data, a, b); }
        rational(const ulong a, const ulong b) { fmpq_init(_data); fmpq_set_ui(_data, a, b); }
        rational(const unsigned int a, const unsigned int b) { fmpq_init(_data); fmpq_set_ui(_data, a, b); }
        rational(const unsigned long a, const unsigned long b) { fmpq_init(_data); fmpq_set_ui(_data, a, b); }

        rational& operator=(const rational& other) { fmpq_set(_data, other._data); return *this; }
        rational& operator=(rational&& other) noexcept { fmpq_swap(_data, other._data); return *this; }
        rational& operator=(const int a) { fmpq_set_si(_data, a, 1); return *this; }
        rational& operator=(const long a) { fmpq_set_si(_data, a, 1); return *this; }
        rational& operator=(const slong a) { fmpq_set_si(_data, a, 1); return *this; }
        rational& operator=(const ulong a) { fmpq_set_ui(_data, a, 1); return *this; }
        rational& operator=(const unsigned int a) { fmpq_set_ui(_data, a, 1); return *this; }
        rational& operator=(const unsigned long a) { fmpq_set_ui(_data, a, 1); return *this; }

        NUM_OP_01(rational, rational&, fmpq, , ._data);
        NUM_OP_01(rational, integer&, fmpq, _fmpz, ._data);
        NUM_MACRO_s(NUM_OP_00, rational, fmpq);
        NUM_MACRO_u(NUM_OP_00, rational, fmpq);

        NUM_SELF_OP_01(rational, rational&, fmpq, , ._data);
        NUM_SELF_OP_01(rational, integer&, fmpq, _fmpz, ._data);
        NUM_MACRO_s(NUM_SELF_OP_00, rational, fmpq);
        NUM_MACRO_u(NUM_SELF_OP_00, rational, fmpq);

        NUM_CMP_01(, rational&, fmpq, , ._data);
        NUM_CMP_01(, integer&, fmpq, _fmpz, ._data);
        NUM_MACRO_s(NUM_CMP_01, , fmpq);
        NUM_MACRO_u(NUM_CMP_01, , fmpq);

        bool operator==(const rational& other) const { return fmpq_equal(_data, other._data); }
        bool operator==(const integer& other) const { return fmpq_equal_fmpz((fmpq*)_data, (fmpz*)other._data); }
        template <typename T> bool operator==(const T other) const {
            if (other == 0) {
                return fmpq_is_zero(_data);
            } if (other == 1) {
                return fmpq_is_one(_data);
            } return fmpq_equal_si((fmpq*)_data, other);
        };
        template<typename T> bool operator!=(const T other) const { return !operator==(other); }

        rational pow(const integer& n) const { rational result; fmpq_pow_fmpz(result._data, _data, n._data); return result; }
        rational pow(const slong n) const { rational result; fmpq_pow_si(result._data, _data, n); return result; }
        NUM_FUNC_0(rational, fmpq, abs);
        NUM_FUNC_0(rational, fmpq, neg);

        rational operator-() const { return neg(); }

        std::string to_string() {
            auto len = fmpz_sizeinbase(fmpq_numref(_data), 10) +
                fmpz_sizeinbase(fmpq_denref(_data), 10) + 3;

            if (len > 1020) {
                char* str = fmpq_get_str(nullptr, 10, _data);
                std::string result(str);
                flint_free(str);
                return result;
            }
            else {
                fmpq_get_str(_buf, 10, _data);
                std::string result(_buf);
                return result;
            }
        }
    };

    // use FLOAT_PARA::prec to set the precision of real and complex
    namespace FLOAT_PARA {
        static ulong prec = 52;
    }

    void set_prec(ulong prec) { FLOAT_PARA::prec = prec; }
    ulong get_prec() { return FLOAT_PARA::prec; }

    struct real {
        arb_t _data;

        real() { arb_init(_data); }
        ~real() { arb_clear(_data); }
        real(const real& other) { arb_init(_data); arb_set(_data, other._data); }
        real(real&& other) noexcept { arb_init(_data); arb_swap(_data, other._data); }

        NUM_INIT(real, arb_t, arb);
        NUM_MACRO_s(NUM_INIT, real, arb);
        NUM_MACRO_u(NUM_INIT, real, arb);
        NUM_MACRO_d(NUM_INIT, real, arb);

        real(std::string str) { arb_init(_data); arb_set_str(_data, str.c_str(), FLOAT_PARA::prec); }
        real(const integer& a) { arb_init(_data); arb_set_fmpz(_data, a._data); }
        real(const rational& a) { arb_init(_data); arb_set_fmpq(_data, a._data, FLOAT_PARA::prec); }

        real& operator=(const real& other) { arb_set(_data, other._data); return *this; }
        real& operator=(real&& other) noexcept { arb_swap(_data, other._data); return *this; }
        NUM_MACRO_s(NUM_SET_0, real, arb);
        NUM_MACRO_u(NUM_SET_0, real, arb);

        NUM_EQUAL_0(real&, arb_equal, ._data);
        NUM_MACRO_s(NUM_EQUAL_1, , arb);

        template<typename T> bool operator!=(const T other) const { return !operator==(other); }

        bool operator<(const real& other) const { return arb_lt(_data, other._data); }
        bool operator>(const real& other) const { return arb_gt(_data, other._data); }
        bool operator<=(const real& other) const { return arb_le(_data, other._data); }
        bool operator>=(const real& other) const { return arb_ge(_data, other._data); }

        real operator-() const { return neg(); }

        NUM_OP_11(real, real&, arb, , ._data);
        NUM_OP_11(real, integer&, arb, _fmpz, ._data);
        NUM_MACRO_u(NUM_OP_11, real, arb);
        NUM_MACRO_s(NUM_OP_11, real, arb);

        NUM_SELF_OP_11(real, real&, arb, , ._data);
        NUM_SELF_OP_11(real, integer&, arb, _fmpz, ._data);
        NUM_MACRO_u(NUM_SELF_OP_11, real, arb);
        NUM_MACRO_s(NUM_SELF_OP_11, real, arb);

        void set_str(std::string str) {
            arb_set_str(_data, str.c_str(), FLOAT_PARA::prec);
        }

        std::string get_str() const {
            char* str = arb_get_str(_data, 10, ARB_STR_NO_RADIUS);
            std::string result(str);
            flint_free(str);
            return result;
        }

        real abs() const { real result; arb_abs(result._data, _data); return result; }
        real neg() const { real result; arb_neg(result._data, _data); return result; }
        real pow(const real& n) const { real result; arb_pow(result._data, _data, n._data, FLOAT_PARA::prec); return result; }
        real pow(const ulong n) const { real result; arb_pow_ui(result._data, _data, n, FLOAT_PARA::prec); return result; }

        double get_d() const { return arf_get_d(arb_midref(_data), ARF_RND_NEAR); }
        slong get_si() const { return arf_get_si(arb_midref(_data), ARF_RND_NEAR); }

        // some constants
        real e() const { real result; arb_const_e(result._data, FLOAT_PARA::prec); return result; }
        real e(ulong prec) const { real result; arb_const_e(result._data, prec); return result; }
        real pi() const { real result; arb_const_pi(result._data, FLOAT_PARA::prec); return result; }
        real pi(ulong prec) const { real result; arb_const_pi(result._data, prec); return result; }
        real log2() const { real result; arb_const_log2(result._data, FLOAT_PARA::prec); return result; }
        real log2(ulong prec) const { real result; arb_const_log2(result._data, prec); return result; }
        real log10() const { real result; arb_const_log10(result._data, FLOAT_PARA::prec); return result; }
        real log10(ulong prec) const { real result; arb_const_log10(result._data, prec); return result; }
        real euler() const { real result; arb_const_euler(result._data, FLOAT_PARA::prec); return result; } // Euler's constant
        real euler(ulong prec) const { real result; arb_const_euler(result._data, prec); return result; }

        // some basic functions with one parameter
        NUM_FUNC_1(real, arb, inv);
        NUM_FUNC_1(real, arb, sin);
        NUM_FUNC_1(real, arb, sin_pi);
        NUM_FUNC_1(real, arb, cos);
        NUM_FUNC_1(real, arb, cos_pi);
        NUM_FUNC_1(real, arb, tan);
        NUM_FUNC_1(real, arb, tan_pi);
        NUM_FUNC_1(real, arb, cot);
        NUM_FUNC_1(real, arb, cot_pi);
        NUM_FUNC_1(real, arb, asin);
        NUM_FUNC_1(real, arb, acos);
        NUM_FUNC_1(real, arb, atan);
        NUM_FUNC_1(real, arb, sinh);
        NUM_FUNC_1(real, arb, cosh);
        NUM_FUNC_1(real, arb, tanh);
        NUM_FUNC_1(real, arb, asinh);
        NUM_FUNC_1(real, arb, acosh);
        NUM_FUNC_1(real, arb, atanh);
        NUM_FUNC_1(real, arb, exp);
        NUM_FUNC_1(real, arb, log);
        NUM_FUNC_1(real, arb, gamma);
        NUM_FUNC_1(real, arb, lgamma); // log(gamma)
        NUM_FUNC_1(real, arb, rgamma); // 1/gamma
        NUM_FUNC_1(real, arb, digamma); // psi = d(log(gamma))/dx
        NUM_FUNC_1(real, arb, zeta);
    };

    struct complex {
        acb_t _data;

        complex() { acb_init(_data); }
        ~complex() { acb_clear(_data); }
        complex(const complex& other) { acb_init(_data); acb_set(_data, other._data); }
        complex(complex&& other) noexcept { acb_init(_data); acb_swap(_data, other._data); }

        NUM_INIT(complex, acb_t, acb);
        NUM_MACRO_s(NUM_INIT, complex, acb);
        NUM_MACRO_u(NUM_INIT, complex, acb);
        NUM_MACRO_d(NUM_INIT, complex, acb);

        complex(const real& a) { acb_init(_data); acb_set_arb(_data, a._data); }
        complex(const integer& a) { acb_init(_data); acb_set_fmpz(_data, a._data); }
        complex(const rational& a) { acb_init(_data); acb_set_fmpq(_data, a._data, FLOAT_PARA::prec); }
        complex(const real& a, const real& b) { acb_init(_data); acb_set_arb_arb(_data, a._data, b._data); }
        complex(const slong a, const slong b) { acb_init(_data); acb_set_si_si(_data, a, b); }

        real re() const { return acb_realref(_data); }
        real im() const { return acb_imagref(_data); }

        complex& operator=(const complex& other) { acb_set(_data, other._data); return *this; }
        complex& operator=(complex&& other) noexcept { acb_swap(_data, other._data); return *this; }
        NUM_MACRO_s(NUM_SET_0, complex, acb);
        NUM_MACRO_u(NUM_SET_0, complex, acb);

        NUM_EQUAL_0(complex&, acb_equal, ._data);
        NUM_MACRO_s(NUM_EQUAL_1, , acb);

        template<typename T> bool operator!=(const T other) const { return !operator==(other); }

        bool is_zero() const { return acb_is_zero(_data); }

        complex operator-() const { return neg(); }

        NUM_OP_11(complex, complex&, acb, , ._data);
        NUM_OP_11(complex, real&, acb, _arb, ._data);
        NUM_OP_11(complex, integer&, acb, _fmpz, ._data);
        NUM_MACRO_u(NUM_OP_11, complex, acb);
        NUM_MACRO_s(NUM_OP_11, complex, acb);

        NUM_SELF_OP_11(complex, complex&, acb, , ._data);
        NUM_SELF_OP_11(complex, real&, acb, _arb, ._data);
        NUM_SELF_OP_11(complex, integer&, acb, _fmpz, ._data);
        NUM_MACRO_s(NUM_SELF_OP_11, complex, acb);
        NUM_MACRO_u(NUM_SELF_OP_11, complex, acb);

        real abs() const { real result; acb_abs(result._data, _data, FLOAT_PARA::prec); return result; }
        complex neg() const { complex result; acb_neg(result._data, _data); return result; }
        complex conj() const { complex result; acb_conj(result._data, _data); return result; }
        complex pow(const complex& n) const { complex result; acb_pow(result._data, _data, n._data, FLOAT_PARA::prec); return result; }
        complex pow(const slong n) const { complex result; acb_pow_si(result._data, _data, n, FLOAT_PARA::prec); return result; }
        complex pow(const int n) const { complex result; acb_pow_si(result._data, _data, n, FLOAT_PARA::prec); return result; }
        complex pow(const ulong n) const { complex result; acb_pow_ui(result._data, _data, n, FLOAT_PARA::prec); return result; }

        complex pi() const { complex result; acb_const_pi(result._data, FLOAT_PARA::prec); return result; }

        // some basic functions with one parameter
        NUM_FUNC_1(complex, acb, inv);
        NUM_FUNC_1(complex, acb, sin);
        NUM_FUNC_1(complex, acb, sin_pi);
        NUM_FUNC_1(complex, acb, cos);
        NUM_FUNC_1(complex, acb, cos_pi);
        NUM_FUNC_1(complex, acb, tan);
        NUM_FUNC_1(complex, acb, tan_pi);
        NUM_FUNC_1(complex, acb, cot);
        NUM_FUNC_1(complex, acb, cot_pi);
        NUM_FUNC_1(complex, acb, asin);
        NUM_FUNC_1(complex, acb, acos);
        NUM_FUNC_1(complex, acb, atan);
        NUM_FUNC_1(complex, acb, sinh);
        NUM_FUNC_1(complex, acb, cosh);
        NUM_FUNC_1(complex, acb, tanh);
        NUM_FUNC_1(complex, acb, asinh);
        NUM_FUNC_1(complex, acb, acosh);
        NUM_FUNC_1(complex, acb, atanh);
        NUM_FUNC_1(complex, acb, exp);
        NUM_FUNC_1(complex, acb, log);
        NUM_FUNC_1(complex, acb, gamma);
        NUM_FUNC_1(complex, acb, lgamma); // log(gamma)
        NUM_FUNC_1(complex, acb, rgamma); // 1/gamma
        NUM_FUNC_1(complex, acb, digamma); // psi = d(log(gamma))/dx
        NUM_FUNC_1(complex, acb, zeta);

    };

    template <typename T>
    T& operator<< (T& os, const integer& i) {
        auto len = fmpz_sizeinbase(i._data, 10);
        if (len > 1020) {
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
    T& operator<< (T& os, const rational& r) {
        auto len = fmpz_sizeinbase(fmpq_numref(r._data), 10)
            + fmpz_sizeinbase(fmpq_denref(r._data), 10) + 3;

        if (len > 1020) {
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

    template <typename T>
    T& operator<< (T& os, const real& r) {
        char* str = arb_get_str(r._data, FLOAT_PARA::prec, ARB_STR_NO_RADIUS);
        os << str;
        flint_free(str);
        return os;
    }

    template <typename T>
    T& operator<< (T& os, const complex& c) {
        os << c.re() << " + " << c.im() << " * I";
        return os;
    }

    template <typename T, typename S> S operator+(const T r, const S& c) { return c + r; }
    template <typename T, typename S> S operator-(const T r, const S& c) { return (-c) + r; }
    template <typename T, typename S> S operator*(const T r, const S& c) { return c * r; }
    template <typename T, typename S> S operator/(const T r, const S& c) { return (r == 1 ? c.inv() : S(r) / c); }
    template <typename T, typename S> S pow(const S& c, const T& r) { return c.pow(r); }

    rational operator/(const integer& r, const integer& c) {
        rational result;
        fmpq_set_fmpz_frac(result._data, r._data, c._data);
        return result;
    }

    // some basic functions with one parameter
    template <typename T> inline T log(const T& x) { return x.log(); }
    template <typename T> inline T exp(const T& x) { return x.exp(); }
    template <typename T> inline T sin(const T& x) { return x.sin(); }
    template <typename T> inline T sin_pi(const T& x) { return x.sin_pi(); }
    template <typename T> inline T cos(const T& x) { return x.cos(); }
    template <typename T> inline T cos_pi(const T& x) { return x.cos_pi(); }
    template <typename T> inline T tan(const T& x) { return x.tan(); }
    template <typename T> inline T tan_pi(const T& x) { return x.tan_pi(); }
    template <typename T> inline T cot(const T& x) { return x.cot(); }
    template <typename T> inline T cot_pi(const T& x) { return x.cot_pi(); }
    template <typename T> inline T asin(const T& x) { return x.asin(); }
    template <typename T> inline T acos(const T& x) { return x.acos(); }
    template <typename T> inline T atan(const T& x) { return x.atan(); }
    template <typename T> inline T sinh(const T& x) { return x.sinh(); }
    template <typename T> inline T cosh(const T& x) { return x.cosh(); }
    template <typename T> inline T tanh(const T& x) { return x.tanh(); }
    template <typename T> inline T asinh(const T& x) { return x.asinh(); }
    template <typename T> inline T acosh(const T& x) { return x.acosh(); }
    template <typename T> inline T atanh(const T& x) { return x.atanh(); }
    template <typename T> inline T gamma(const T& x) { return x.gamma(); }
    template <typename T> inline T lgamma(const T& x) { return x.lgamma(); }
    template <typename T> inline T digamma(const T& x) { return x.digamma(); }
    template <typename T> inline T zeta(const T& x) { return x.zeta(); }

    // some constants
    template <typename T> inline T const_e() { return T().e(); }
    template <typename T> inline T const_e(ulong prec) { return T().e(prec); }
    template <typename T> inline T pi() { return T().pi(); }
    template <typename T> inline T pi(ulong prec) { return T().pi(prec); }
    template <typename T> inline T log2() { return T().log2(); }
    template <typename T> inline T log2(ulong prec) { return T().log2(prec); }
    template <typename T> inline T log10() { return T().log10(); }
    template <typename T> inline T log10(ulong prec) { return T().log10(prec); }
    template <typename T> inline T euler() { return T().euler(); }
    template <typename T> inline T euler(ulong prec) { return T().euler(prec); }

    // some basic functions with two parameters
    complex polylog(const complex& s, const complex& z) {
        complex result;
        acb_polylog(result._data, s._data, z._data, FLOAT_PARA::prec);
        return result;
    }

    real polylog(const real& s, const real& z) {
        real result;
        arb_polylog(result._data, s._data, z._data, FLOAT_PARA::prec);
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