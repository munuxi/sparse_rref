#include "sparse_vec.h"

int sfmpq_vec_add_mul(sfmpq_vec_t vec, const sfmpq_vec_t src, const fmpq_t a) {
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
			fmpq_mul(entry, na, src->entries + ptr2 - 1);
			fmpq_add(entry, vec->entries + ptr1 - 1, entry);
			if (!scalar_is_zero(entry)) {
				vec->indices[ptr - 1] = vec->indices[ptr1 - 1];
				fmpq_set(vec->entries + ptr - 1, entry);
				ptr--;
			}
			ptr1--;
			ptr2--;
		}
		else if (vec->indices[ptr1 - 1] < src->indices[ptr2 - 1]) {
			fmpq_mul(entry, na, src->entries + ptr2 - 1);
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
		fmpq_mul(entry, na, src->entries + ptr2 - 1);
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
	if (src->nnz == 0)
		return 0;

	if (vec->nnz == 0) {
		sparse_vec_set(vec, src);
		sfmpq_vec_rescale(vec, a);
		for (size_t i = 0; i < vec->nnz; i++) {
			fmpq_neg(vec->entries + i, vec->entries + i);
		}
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
			fmpq_mul(entry, na, src->entries + ptr2 - 1);
			fmpq_add(entry, vec->entries + ptr1 - 1, entry);
			if (!scalar_is_zero(entry)) {
				vec->indices[ptr - 1] = vec->indices[ptr1 - 1];
				fmpq_set(vec->entries + ptr - 1, entry);
				ptr--;
			}
			ptr1--;
			ptr2--;
		}
		else if (vec->indices[ptr1 - 1] < src->indices[ptr2 - 1]) {
			fmpq_mul(entry, na, src->entries + ptr2 - 1);
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
		fmpq_mul(entry, na, src->entries + ptr2 - 1);
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
