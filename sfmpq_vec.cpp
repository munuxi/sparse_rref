#include "sparse_vec.h"
#include <iostream>

void sfmpq_vec_rescale(sfmpq_vec_t vec, const fmpq_t scalar) {
    for (ulong i = 0; i < vec->nnz; i++)
        scalar_mul(vec->entries + i, vec->entries + i, scalar);
}

void sfmpq_vec_neg(sfmpq_vec_t vec) {
    for (ulong i = 0; i < vec->nnz; i++)
        fmpq_neg(vec->entries + i, vec->entries + i);
}

int sfmpq_vec_add_sorted(sfmpq_vec_t vec, const sfmpq_vec_t src) {
    // -1 : Different lengths
    if (vec->len != src->len)
        return -1;

    if (src->nnz == 0)
        return 0;

    if (vec->nnz == 0) {
        sparse_vec_set(vec, src);
    }

    fmpq_t entry;
    fmpq_init(entry);

    sfmpq_vec_t tmpvec;
    _sparse_vec_init(tmpvec, vec->len, vec->nnz + src->nnz);

    ulong ptr1 = 0;
    ulong ptr2 = 0;
    while (ptr1 < vec->nnz && ptr2 < src->nnz) {
        if (vec->indices[ptr1] == src->indices[ptr2]) {
            scalar_add(entry, vec->entries + ptr1, src->entries + ptr2);
            if (entry != 0)
                _sparse_vec_set_entry(tmpvec, vec->indices[ptr1], entry);
            ptr1++;
            ptr2++;
        } else if (vec->indices[ptr1] < src->indices[ptr2]) {
            if (!scalar_is_zero(vec->entries + ptr1))
                _sparse_vec_set_entry(tmpvec, vec->indices[ptr1],
                                      vec->entries + ptr1);
            ptr1++;
        } else {
            if (!scalar_is_zero(src->entries + ptr2)) {
                _sparse_vec_set_entry(tmpvec, src->indices[ptr2],
                                      src->entries + ptr2);
            }
            ptr2++;
        }
    }
    while (ptr1 < vec->nnz) {
        if (!scalar_is_zero(vec->entries + ptr1))
            _sparse_vec_set_entry(tmpvec, vec->indices[ptr1],
                                  vec->entries + ptr1);
        ptr1++;
    }
    while (ptr2 < src->nnz) {
        if (!scalar_is_zero(src->entries + ptr2)) {
            _sparse_vec_set_entry(tmpvec, src->indices[ptr2],
                                  src->entries + ptr2);
        }
        ptr2++;
    }

    sparse_vec_set(vec, tmpvec);
    sparse_vec_clear(tmpvec);
    fmpq_clear(entry);
    return 0;
}

int sfmpq_vec_add_mul_sorted(sfmpq_vec_t vec, const sfmpq_vec_t src,
                             const fmpq_t a) {
    // -1 : Different lengths
    if (vec->len != src->len)
        return -1;

    if (src->nnz == 0)
        return 0;

    if (vec->nnz == 0) {
        sparse_vec_set(vec, src);
    }

    fmpq_t entry;
    fmpq_init(entry);

    sfmpq_vec_t tmpvec;
    _sparse_vec_init(tmpvec, vec->len, vec->nnz + src->nnz);

    ulong ptr1 = 0;
    ulong ptr2 = 0;
    while (ptr1 < vec->nnz && ptr2 < src->nnz) {
        if (vec->indices[ptr1] == src->indices[ptr2]) {
            scalar_mul(entry, a, src->entries + ptr2);
            scalar_add(entry, vec->entries + ptr1, entry);
            if (entry != 0)
                _sparse_vec_set_entry(tmpvec, vec->indices[ptr1], entry);
            ptr1++;
            ptr2++;
        } else if (vec->indices[ptr1] < src->indices[ptr2]) {
            if (!scalar_is_zero(vec->entries + ptr1))
                _sparse_vec_set_entry(tmpvec, vec->indices[ptr1],
                                      vec->entries + ptr1);
            ptr1++;
        } else {
            if (!scalar_is_zero(src->entries + ptr2)) {
                scalar_mul(entry, a, src->entries + ptr2);
                _sparse_vec_set_entry(tmpvec, src->indices[ptr2], entry);
            }
            ptr2++;
        }
    }
    while (ptr1 < vec->nnz) {
        if (!scalar_is_zero(vec->entries + ptr1))
            _sparse_vec_set_entry(tmpvec, vec->indices[ptr1],
                                  vec->entries + ptr1);
        ptr1++;
    }
    while (ptr2 < src->nnz) {
        if (!scalar_is_zero(src->entries + ptr2)) {
            scalar_mul(entry, a, src->entries + ptr2);
            _sparse_vec_set_entry(tmpvec, src->indices[ptr2], entry);
        }
        ptr2++;
    }

    sparse_vec_set(vec, tmpvec);
    sparse_vec_clear(tmpvec);
    fmpq_clear(entry);
    return 0;
}

// we assume that vec and src are sorted, and the result is also sorted
int sfmpq_vec_sub_scalar_sorted(sfmpq_vec_t vec, const sfmpq_vec_t src,
                                const fmpq_t a) {
    // -1 : Different lengths
    if (vec->len != src->len)
        return -1;

    if (src->nnz == 0)
        return 0;

    if (vec->nnz == 0) {
        sparse_vec_set(vec, src);
        sfmpq_vec_neg(vec);
        sfmpq_vec_rescale(vec, a);
    }

    fmpq_t na, entry;
    fmpq_init(na);
    fmpq_init(entry);
    fmpq_neg(na, a);

    sfmpq_vec_t tmpvec;
    _sparse_vec_init(tmpvec, vec->len, vec->nnz + src->nnz);

    ulong ptr1 = 0;
    ulong ptr2 = 0;
    while (ptr1 < vec->nnz && ptr2 < src->nnz) {
        if (vec->indices[ptr1] == src->indices[ptr2]) {
            scalar_mul(entry, na, src->entries + ptr2);
            scalar_add(entry, vec->entries + ptr1, entry);
            if (entry != 0)
                _sparse_vec_set_entry(tmpvec, vec->indices[ptr1], entry);
            ptr1++;
            ptr2++;
        } else if (vec->indices[ptr1] < src->indices[ptr2]) {
            _sparse_vec_set_entry(tmpvec, vec->indices[ptr1], vec->entries + ptr1);
            ptr1++;
        } else {
            scalar_mul(entry, na, src->entries + ptr2);
            _sparse_vec_set_entry(tmpvec, src->indices[ptr2], entry);
            ptr2++;
        }
    }
    while (ptr1 < vec->nnz) {
        _sparse_vec_set_entry(tmpvec, vec->indices[ptr1], vec->entries + ptr1);
        ptr1++;
    }
    while (ptr2 < src->nnz) {
        scalar_mul(entry, na, src->entries + ptr2);
        _sparse_vec_set_entry(tmpvec, src->indices[ptr2], entry);
        ptr2++;
    }

    sparse_vec_swap(vec, tmpvec);
    sparse_vec_clear(tmpvec);
    fmpq_clear(na);
    fmpq_clear(entry);
    return 0;
}

// we assume that vec and src are sorted, and the result is also sorted
// TODO: it will memory leak
int sfmpq_vec_sub_scalar_sorted_cached(sfmpq_vec_t vec, const sfmpq_vec_t src,
                                       sfmpq_vec_t cache, const fmpq_t a) {
    // -1 : Different lengths
    if (vec->len != src->len)
        return -1;

    if (src->nnz == 0)
        return 0;

    if (vec->nnz == 0) {
        sparse_vec_set(vec, src);
        sfmpq_vec_neg(vec);
        sfmpq_vec_rescale(vec, a);
    }

    fmpq_t na, entry;
    fmpq_init(na);
    fmpq_init(entry);
    fmpq_neg(na, a);

    cache->nnz = 0;
    cache->len = src->len;
    if (vec->nnz + src->nnz > cache->alloc)
        sparse_vec_realloc(cache, vec->nnz + src->nnz);

    ulong ptr1 = 0;
    ulong ptr2 = 0;
    while (ptr1 < vec->nnz && ptr2 < src->nnz) {
        if (vec->indices[ptr1] == src->indices[ptr2]) {
            scalar_mul(entry, na, src->entries + ptr2);
            scalar_add(entry, vec->entries + ptr1, entry);
            if (entry != 0)
                _sparse_vec_set_entry(cache, vec->indices[ptr1], entry);
            ptr1++;
            ptr2++;
        } else if (vec->indices[ptr1] < src->indices[ptr2]) {
            if (!scalar_is_zero(vec->entries + ptr1))
                _sparse_vec_set_entry(cache, vec->indices[ptr1],
                                      vec->entries + ptr1);
            ptr1++;
        } else {
            if (!scalar_is_zero(src->entries + ptr2)) {
                scalar_mul(entry, na, src->entries + ptr2);
                _sparse_vec_set_entry(cache, src->indices[ptr2], entry);
            }
            ptr2++;
        }
    }
    while (ptr1 < vec->nnz) {
        if (!scalar_is_zero(vec->entries + ptr1))
            _sparse_vec_set_entry(cache, vec->indices[ptr1],
                                  vec->entries + ptr1);
        ptr1++;
    }
    while (ptr2 < src->nnz) {
        if (!scalar_is_zero(src->entries + ptr2)) {
            scalar_mul(entry, na, src->entries + ptr2);
            _sparse_vec_set_entry(cache, src->indices[ptr2], entry);
        }
        ptr2++;
    }

    sparse_vec_swap(vec, cache);
    fmpq_clear(na);
    fmpq_clear(entry);
    return 0;
}

void print_dense_vec(sfmpq_vec_t vec) {
    for (size_t i = 0; i < vec->len; i++) {
        auto entry = sparse_vec_entry(vec, i);
        if (entry == NULL)
            printf("0");
        else
            fmpq_print(entry);
        printf(" ");
    }
    printf("\n");
}

void snmod_vec_from_sfmpq(snmod_vec_t vec, const sfmpq_vec_t src, nmod_t p) {
    sparse_vec_realloc(vec, src->nnz);
    vec->alloc = src->nnz;
    vec->nnz = 0;
    for (size_t i = 0; i < src->nnz; i++) {
        ulong num = fmpz_get_nmod(fmpq_numref(src->entries + i), p);
        ulong den = fmpz_get_nmod(fmpq_denref(src->entries + i), p);
        ulong val = nmod_div(num, den, p);
        _sparse_vec_set_entry(vec, src->indices[i], val);
    }
}