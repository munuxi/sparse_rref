#ifndef SFMPQ_VEC_H
#define SFMPQ_VEC_H

#include <algorithm>

#include "sparse_vec.h"
#include "snmod_vec.h"

// print stuff
void print_dense_vec(sfmpq_vec_t vec) ;

// arithmetic operations
void sfmpq_vec_rescale(sfmpq_vec_t vec, const fmpq_t scalar) ;
void sfmpq_vec_neg(sfmpq_vec_t vec) ;
// int sfmpq_vec_add(sfmpq_vec_t vec, const sfmpq_vec_t src) ;
int sfmpq_vec_sub_scalar_sorted(sfmpq_vec_t prevec, const sfmpq_vec_t src, const fmpq_t a);
int sfmpq_vec_sub_scalar_sorted_cached(sfmpq_vec_t prevec, const sfmpq_vec_t src, sfmpq_vec_t cache, const fmpq_t a);
int sfmpq_vec_add_sorted(sfmpq_vec_t vec, const sfmpq_vec_t src);
int sfmpq_vec_add_mul_sorted(sfmpq_vec_t vec, const sfmpq_vec_t src, const fmpq_t a);

void snmod_vec_from_sfmpq(snmod_vec_t vec, const sfmpq_vec_t src, nmod_t p) ;

#endif // SFMPQ_VEC_H