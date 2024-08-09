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

int sfmpq_vec_add_mul(sfmpq_vec_t vec, const sfmpq_vec_t src, const fmpq_t a) {
	// -1 : Different lengths
	if (vec->len != src->len)
		return -1;

	if (src->nnz == 0)
		return 0;

	if (vec->nnz == 0) {
		sparse_vec_set(vec, src);
		sfmpq_vec_rescale(vec, a);
	}

	fmpq_t na, entry;
	fmpq_init(na);
	fmpq_set(na, a);
	fmpq_init(entry);

	if (vec->nnz + src->nnz > vec->alloc)
		sparse_vec_realloc(vec, vec->nnz + src->nnz);

	ulong ptr1 = vec->nnz;
	ulong ptr2 = src->nnz;
	ulong ptr = vec->nnz + src->nnz;
	while (ptr1 > 0 && ptr2 > 0) {
		if (vec->indices[ptr1 - 1] == src->indices[ptr2 - 1]) {
			scalar_mul(entry, na, src->entries + ptr2 - 1);
			scalar_add(entry, vec->entries + ptr1 - 1, entry);
			if (!scalar_is_zero(entry)) {
				vec->indices[ptr - 1] = vec->indices[ptr1 - 1];
				fmpq_set(vec->entries + ptr - 1, entry);
				ptr--;
			}
			ptr1--;
			ptr2--;
		}
		else if (vec->indices[ptr1 - 1] < src->indices[ptr2 - 1]) {
			scalar_mul(entry, na, src->entries + ptr2 - 1);
			vec->indices[ptr - 1] = src->indices[ptr2 - 1];
			fmpq_set(vec->entries + ptr - 1, entry);
			ptr2--;
			ptr--;
		}
		else {
			vec->indices[ptr - 1] = vec->indices[ptr1 - 1];
			fmpq_set(vec->entries + ptr - 1, vec->entries + ptr1 - 1);
			ptr1--;
			ptr--;
		}
	}
	while (ptr2 > 0) {
		scalar_mul(entry, na, src->entries + ptr2 - 1);
		vec->indices[ptr - 1] = src->indices[ptr2 - 1];
		fmpq_set(vec->entries + ptr - 1, entry);
		ptr2--;
		ptr--;
	}

	// if ptr1 > 0, and ptr > 0
	for (size_t i = ptr1; i < ptr; i++) {
		fmpq_zero(vec->entries + i);
	}

	vec->nnz += src->nnz;
	sparse_vec_canonicalize(vec);
	if (vec->alloc > 4 * vec->nnz)
		sparse_vec_realloc(vec, 2 * vec->nnz);

	fmpq_clear(na);
	fmpq_clear(entry);
	return 0;
}

// we assume that vec and src are sorted, and the result is also sorted
int sfmpq_vec_sub_mul(sfmpq_vec_t vec, const sfmpq_vec_t src, const fmpq_t a) {
	// -1 : Different lengths
	if (vec->len != src->len)
		return -1;

	if (src->nnz == 0)
		return 0;

	if (vec->nnz == 0) {
		sparse_vec_set(vec, src);
		sfmpq_vec_rescale(vec, a);
		sfmpq_vec_neg(vec);
	}

	fmpq_t na, entry;
	fmpq_init(na);
	fmpq_neg(na, a);
	fmpq_init(entry);

	if (vec->nnz + src->nnz > vec->alloc)
		sparse_vec_realloc(vec, vec->nnz + src->nnz);

	ulong ptr1 = vec->nnz;
	ulong ptr2 = src->nnz;
	ulong ptr = vec->nnz + src->nnz;
	while (ptr1 > 0 && ptr2 > 0) {
		if (vec->indices[ptr1 - 1] == src->indices[ptr2 - 1]) {
			scalar_mul(entry, na, src->entries + ptr2 - 1);
			scalar_add(entry, vec->entries + ptr1 - 1, entry);
			if (!scalar_is_zero(entry)) {
				vec->indices[ptr - 1] = vec->indices[ptr1 - 1];
				fmpq_set(vec->entries + ptr - 1, entry);
				ptr--;
			}
			ptr1--;
			ptr2--;
		}
		else if (vec->indices[ptr1 - 1] < src->indices[ptr2 - 1]) {
			scalar_mul(entry, na, src->entries + ptr2 - 1);
			vec->indices[ptr - 1] = src->indices[ptr2 - 1];
			fmpq_set(vec->entries + ptr - 1, entry);
			ptr2--;
			ptr--;
		}
		else {
			vec->indices[ptr - 1] = vec->indices[ptr1 - 1];
			fmpq_set(vec->entries + ptr - 1, vec->entries + ptr1 - 1);
			ptr1--;
			ptr--;
		}
	}
	while (ptr2 > 0) {
		scalar_mul(entry, na, src->entries + ptr2 - 1);
		vec->indices[ptr - 1] = src->indices[ptr2 - 1];
		fmpq_set(vec->entries + ptr - 1, entry);
		ptr2--;
		ptr--;
	}

	// if ptr1 > 0, and ptr > 0
	for (size_t i = ptr1; i < ptr; i++) {
		fmpq_zero(vec->entries + i);
	}

	vec->nnz += src->nnz;
	sparse_vec_canonicalize(vec);
	if (vec->alloc > 4 * vec->nnz)
		sparse_vec_realloc(vec, 2 * vec->nnz);

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