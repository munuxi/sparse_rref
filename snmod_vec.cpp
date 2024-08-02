#include "sparse_vec.h"
#include <iostream>

void snmod_vec_neg(snmod_vec_t vec, nmod_t p) {
    _nmod_vec_neg(vec->entries, vec->entries, vec->nnz, p);
}

// p should less than 2^(FLINT_BITS-1) (2^63(2^31) on 64(32)-bit machine)
// scalar and all vec->entries[i] should less than p
void snmod_vec_rescale(snmod_vec_t vec, ulong scalar, nmod_t p) {
    _nmod_vec_scalar_mul_nmod_shoup(vec->entries, vec->entries, vec->nnz,
                                    scalar, p);
}

int snmod_vec_add(snmod_vec_t vec, const snmod_vec_t src, nmod_t p) {
    // -1 : Different lengths
    if (vec->len != src->len)
        return -1;

    ulong new_nnz = std::max(vec->len, vec->nnz + src->nnz);
    if (new_nnz > vec->alloc)
        sparse_vec_realloc(vec, new_nnz);

    if (vec->nnz < 100) {
        for (size_t i = 0; i < src->nnz; i++) {
            ulong *entry = sparse_vec_entry(vec, src->indices[i]);
            if (entry != NULL) {
                *entry = nmod_add(*entry, src->entries[i], p);
            } else {
                vec->indices[vec->nnz] = src->indices[i];
                scalar_set(vec->entries + vec->nnz, src->entries + i);
                vec->nnz++;
            }
        }
    } else { // first sort it and binary search
        slong *perm = (slong *)malloc(vec->nnz * sizeof(slong));
        for (auto i = 0; i < vec->nnz; i++)
            perm[i] = i;
        slong *indices = (slong *)malloc(vec->nnz * sizeof(slong));
        std::sort(perm, perm + vec->nnz, [&vec](slong a, slong b) {
            return vec->indices[a] < vec->indices[b];
        });
        for (auto i = 0; i < vec->nnz; i++)
            indices[i] = vec->indices[perm[i]];

        auto oldnnz = vec->nnz;
        for (size_t i = 0; i < src->nnz; i++) {
            auto ptr = binarysearch(indices, indices + oldnnz, src->indices[i]);
            if (ptr == indices + oldnnz) { // not found
                vec->indices[vec->nnz] = src->indices[i];
                vec->entries[vec->nnz] = src->entries[i];
                vec->nnz++;
            } else {
                ulong *entry = vec->entries + perm[ptr - indices];
                *entry = nmod_add(*entry, src->entries[i], p);
            }
        }
        free(perm);
        free(indices);
    }

    // spsarse_vec_compress(vec);
    return 0;
}

// src is densed vector
int snmod_vec_add_densed(snmod_vec_t vec, ulong *src, nmod_t p) {
    for (size_t i = 0; i < vec->nnz; i++) {
        src[vec->indices[i]] =
            nmod_add(src[vec->indices[i]], vec->entries[i], p);
    }
    vec->nnz = 0;
    for (size_t i = 0; i < vec->len; i++) {
        if (src[i] != 0)
            _sparse_vec_set_entry(vec, i, src[i]);
    }
    return 0;
}

int snmod_vec_sub(snmod_vec_t vec, const snmod_vec_t src, nmod_t p) {
    // -1 : Different lengths
    if (vec->len != src->len)
        return -1;

    ulong new_nnz = std::max(vec->len, vec->nnz + src->nnz);
    if (new_nnz > vec->alloc)
        sparse_vec_realloc(vec, new_nnz);

    if (vec->nnz < 100) {
        for (size_t i = 0; i < src->nnz; i++) {
            ulong *entry = sparse_vec_entry(vec, src->indices[i]);
            if (entry != NULL) {
                *entry = nmod_sub(*entry, src->entries[i], p);
            } else {
                vec->indices[vec->nnz] = src->indices[i];
                vec->entries[vec->nnz] = nmod_neg(src->entries[i], p);
                vec->nnz++;
            }
        }
    } else { // first sort it and binary search
        slong *perm = (slong *)malloc(vec->nnz * sizeof(slong));
        for (size_t i = 0; i < vec->nnz; i++)
            perm[i] = i;
        slong *indices = (slong *)malloc(vec->nnz * sizeof(slong));
        std::sort(perm, perm + vec->nnz, [&vec](slong a, slong b) {
            return vec->indices[a] < vec->indices[b];
        });
        for (size_t i = 0; i < vec->nnz; i++)
            indices[i] = vec->indices[perm[i]];

        auto oldnnz = vec->nnz;
        for (size_t i = 0; i < src->nnz; i++) {
            auto ptr = binarysearch(indices, indices + oldnnz, src->indices[i]);
            if (ptr == indices + oldnnz) { // not found
                vec->indices[vec->nnz] = src->indices[i];
                vec->entries[vec->nnz] = nmod_neg(src->entries[i], p);
                vec->nnz++;

            } else {
                ulong *entry = vec->entries + perm[ptr - indices];
                *entry = nmod_sub(*entry, src->entries[i], p);
            }
        }
        free(perm);
        free(indices);
    }

    // spsarse_vec_compress(vec);
    return 0;
}

// c.f. the relization of _nmod_vec_scalar_mul_nmod_shoup from FLINT
// void _nmod_vec_scalar_mul_nmod_shoup(nn_ptr res, nn_srcptr vec,
// 	slong len, ulong c, nmod_t mod)
// {
// 	slong i;
// 	ulong w_pr;
// 	w_pr = n_mulmod_precomp_shoup(c, mod.n);
// 	for (i = 0; i < len; i++)
// 		res[i] = n_mulmod_shoup(c, vec[i], w_pr, mod.n);
// }

// we assume that src is sorted
int snmod_vec_sub_scalar(snmod_vec_t vec, const snmod_vec_t src, const ulong a,
                         nmod_t p) {
    // -1 : Different lengths
    if (vec->len != src->len)
        return -1;

    if (src->nnz == 0)
        return 0;

    if (vec->nnz == 0) {
        sparse_vec_set(vec, src);
        snmod_vec_neg(vec, p);
        snmod_vec_rescale(vec, a, p);
    }

    ulong new_nnz = std::max(vec->len, vec->nnz + src->nnz);
    if (new_nnz > vec->alloc)
        sparse_vec_realloc(vec, new_nnz);

    ulong a_pr = n_mulmod_precomp_shoup(a, p.n);

    if (vec->nnz < 100) {
        for (size_t i = 0; i < src->nnz; i++) {
            ulong *entry = sparse_vec_entry(vec, src->indices[i]);
            if (entry != NULL) {
                *entry = nmod_sub(
                    *entry, n_mulmod_shoup(a, src->entries[i], a_pr, p.n), p);
            } else {
                vec->indices[vec->nnz] = src->indices[i];
                vec->entries[vec->nnz] =
                    n_mulmod_shoup(a, nmod_neg(src->entries[i], p), a_pr, p.n);
                vec->nnz++;
                // SET_BIT_ONE(vec->bitlist, src->indices[i] / vec->lenstep);
            }
        }
    } else { // first sort it and binary search
        slong *perm = (slong *)malloc(vec->nnz * sizeof(slong));
        for (size_t i = 0; i < vec->nnz; i++)
            perm[i] = i;
        slong *indices = (slong *)malloc(vec->nnz * sizeof(slong));
        std::sort(perm, perm + vec->nnz, [&vec](slong a, slong b) {
            return vec->indices[a] < vec->indices[b];
        });
        for (size_t i = 0; i < vec->nnz; i++)
            indices[i] = vec->indices[perm[i]];

        slong oldvecnnz =
            vec->nnz; // vec->nnz will change in the following loop

        ulong ptr1 = 0;
        ulong ptr2 = 0;
        while (ptr1 < oldvecnnz && ptr2 < src->nnz) {
            if (indices[ptr1] == src->indices[ptr2]) {
                ulong *entry = vec->entries + perm[ptr1];
                *entry = nmod_sub(
                    *entry, n_mulmod_shoup(a, src->entries[ptr2], a_pr, p.n),
                    p);
                ptr1++;
                ptr2++;
            } else if (indices[ptr1] < src->indices[ptr2]) {
                ptr1++;
            } else {
                vec->indices[vec->nnz] = src->indices[ptr2];
                vec->entries[vec->nnz] = n_mulmod_shoup(
                    a, nmod_neg(src->entries[ptr2], p), a_pr, p.n);
                vec->nnz++;
                ptr2++;
                // SET_BIT_ONE(vec->bitlist, src->indices[ptr2] / vec->lenstep);
            }
        }
        // if ptr1 == oldvecnnz but still ptr2 < src->nnz
        while (ptr2 < src->nnz) {
            vec->indices[vec->nnz] = src->indices[ptr2];
            vec->entries[vec->nnz] =
                n_mulmod_shoup(a, nmod_neg(src->entries[ptr2], p), a_pr, p.n);
            vec->nnz++;
            ptr2++;
            // SET_BIT_ONE(vec->bitlist, src->indices[ptr2] / vec->lenstep);
        }

        free(perm);
        free(indices);
    }

    // sparse_vec_setbitlist(vec);
    // sparse_vec_canonicalize(vec);
    return 0;
}

// we assume that vec and src are sorted, and the result is also sorted
int snmod_vec_sub_scalar_sorted(snmod_vec_t vec, const snmod_vec_t src,
                                const ulong a, nmod_t p) {
    // -1 : Different lengths
    if (vec->len != src->len)
        return -1;

    if (src->nnz == 0)
        return 0;

    if (vec->nnz == 0) {
        sparse_vec_set(vec, src);
        snmod_vec_neg(vec, p);
        snmod_vec_rescale(vec, a, p);
    }

    ulong na = nmod_neg(a, p);
    ulong na_pr = n_mulmod_precomp_shoup(na, p.n);

    snmod_vec_t tmpvec;
    _sparse_vec_init(tmpvec, vec->len, vec->nnz + src->nnz);

    ulong ptr1 = 0;
    ulong ptr2 = 0;
    while (ptr1 < vec->nnz && ptr2 < src->nnz) {
        if (vec->indices[ptr1] == src->indices[ptr2]) {
            ulong entry =
                nmod_add(vec->entries[ptr1],
                         n_mulmod_shoup(na, src->entries[ptr2], na_pr, p.n), p);
            if (entry != 0)
                _sparse_vec_set_entry(tmpvec, vec->indices[ptr1], entry);

            ptr1++;
            ptr2++;
        } else if (vec->indices[ptr1] < src->indices[ptr2]) {
            _sparse_vec_set_entry(tmpvec, vec->indices[ptr1], vec->entries[ptr1]);
            ptr1++;
        } else {
            ulong entry = n_mulmod_shoup(na, src->entries[ptr2], na_pr, p.n);
            _sparse_vec_set_entry(tmpvec, src->indices[ptr2], entry);
            ptr2++;
        }
    }
    while (ptr2 < src->nnz) {
        ulong entry = n_mulmod_shoup(na, src->entries[ptr2], na_pr, p.n);
        _sparse_vec_set_entry(tmpvec, src->indices[ptr2], entry);
        ptr2++;
    }
    // while (ptr1 < vec->nnz) {
    //     _sparse_vec_set_entry(tmpvec, vec->indices[ptr1], vec->entries[ptr1]);
    //     ptr1++;
    // }
    memcpy(tmpvec->indices + tmpvec->nnz, vec->indices + ptr1, (vec->nnz - ptr1) * sizeof(slong));
    memcpy(tmpvec->entries + tmpvec->nnz, vec->entries + ptr1, (vec->nnz - ptr1) * sizeof(ulong));
    tmpvec->nnz += vec->nnz - ptr1;

    sparse_vec_swap(vec, tmpvec);
    sparse_vec_clear(tmpvec);

    return 0;
}

// we assume that vec and src are sorted, and the result is also sorted
int snmod_vec_sub_scalar_sorted_cached(snmod_vec_t vec, const snmod_vec_t src,
                                       snmod_vec_t cache, const ulong a,
                                       nmod_t p) {
    // -1 : Different lengths
    if (vec->len != src->len)
        return -1;

    if (src->nnz == 0)
        return 0;

    if (vec->nnz == 0) {
        sparse_vec_set(vec, src);
        snmod_vec_neg(vec, p);
        snmod_vec_rescale(vec, a, p);
    }

    cache->nnz = 0;
    cache->len = src->len;
    if (vec->nnz + src->nnz > cache->alloc)
        sparse_vec_realloc(cache, vec->nnz + src->nnz);

    ulong na = nmod_neg(a, p);
    ulong na_pr = n_mulmod_precomp_shoup(na, p.n);

    ulong ptr1 = 0;
    ulong ptr2 = 0;
    while (ptr1 < vec->nnz && ptr2 < src->nnz) {
        if (vec->indices[ptr1] == src->indices[ptr2]) {
            ulong entry =
                nmod_add(vec->entries[ptr1],
                         n_mulmod_shoup(na, src->entries[ptr2], na_pr, p.n), p);
            if (entry != 0)
                _sparse_vec_set_entry(cache, vec->indices[ptr1], entry);

            ptr1++;
            ptr2++;
        } else if (vec->indices[ptr1] < src->indices[ptr2]) {
            if (vec->entries[ptr1] != 0)
                _sparse_vec_set_entry(cache, vec->indices[ptr1],
                                      vec->entries[ptr1]);
            ptr1++;
        } else {
            if (src->entries[ptr2] != 0) {
                ulong entry =
                    n_mulmod_shoup(na, src->entries[ptr2], na_pr, p.n);
                _sparse_vec_set_entry(cache, src->indices[ptr2], entry);
            }
            ptr2++;
        }
    }
    while (ptr1 < vec->nnz) {
        if (vec->entries[ptr1] != 0)
            _sparse_vec_set_entry(cache, vec->indices[ptr1],
                                  vec->entries[ptr1]);
        ptr1++;
    }
    while (ptr2 < src->nnz) {
        if (src->entries[ptr2] != 0) {
            ulong entry = n_mulmod_shoup(na, src->entries[ptr2], na_pr, p.n);
            _sparse_vec_set_entry(cache, src->indices[ptr2], entry);
        }
        ptr2++;
    }

    sparse_vec_swap(vec, cache);

    return 0;
}

// src is densed vector
int snmod_vec_sub_densed(snmod_vec_t vec, ulong *src, nmod_t p) {
    _nmod_vec_neg(src, src, vec->len, p);
    for (size_t i = 0; i < vec->nnz; i++) {
        src[vec->indices[i]] =
            nmod_add(src[vec->indices[i]], vec->entries[i], p);
    }
    vec->nnz = 0;
    for (size_t i = 0; i < vec->len; i++) {
        if (src[i] != 0)
            _sparse_vec_set_entry(vec, i, src[i]);
    }
    return 0;
}

void print_dense_vec(snmod_vec_t vec) {
    for (size_t i = 0; i < vec->len; i++) {
        auto entry = sparse_vec_entry(vec, i);
        if (entry == NULL)
            printf("0");
        else
            std::cout << entry;
        printf(" ");
    }
    printf("\n");
}
