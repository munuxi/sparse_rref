#ifndef SNMOD_VEC_H
#define SNMOD_VEC_H

#include "sparse_vec.h"

// arithmetic operations
void snmod_vec_rescale(snmod_vec_t vec, ulong scalar, nmod_t p);
void snmod_vec_neg(snmod_vec_t vec, nmod_t p);
int snmod_vec_add(snmod_vec_t vec, const snmod_vec_t src, nmod_t p);
int snmod_vec_sub(snmod_vec_t vec, const snmod_vec_t src, nmod_t p);
int snmod_vec_sub_scalar(snmod_vec_t vec, const snmod_vec_t src, const ulong a,
                         nmod_t p);
int snmod_vec_sub_scalar_sorted(snmod_vec_t vec, const snmod_vec_t src,
                                const ulong a, nmod_t p);
int snmod_vec_sub_scalar_sorted_cached(snmod_vec_t vec, const snmod_vec_t src,
                                       snmod_vec_t cache, const ulong a,
                                       nmod_t p);
int snmod_vec_add_densed(snmod_vec_t vec, ulong *src, nmod_t p);
int snmod_vec_sub_densed(snmod_vec_t vec, ulong *src, nmod_t p);

#endif // snmod_VEC_H