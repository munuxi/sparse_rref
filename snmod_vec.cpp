#include "sparse_vec.h"

void snmod_vec_neg(snmod_vec_t vec, nmod_t p) {
	_nmod_vec_neg(vec->entries, vec->entries, vec->nnz, p);
}

// p should less than 2^(FLINT_BITS-1) (2^63(2^31) on 64(32)-bit machine)
// scalar and all vec->entries[i] should less than p
void snmod_vec_rescale(snmod_vec_t vec, ulong scalar, nmod_t p) {
	_nmod_vec_scalar_mul_nmod_shoup(vec->entries, vec->entries, vec->nnz,
		scalar, p);
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

// we assume that vec and src are sorted, and the result is also sorted
int snmod_vec_add_mul(snmod_vec_t vec, const snmod_vec_t src,
	const ulong a, nmod_t p) {
	// -1 : Different lengths
	if (vec->len != src->len)
		return -1;

	if (src->nnz == 0)
		return 0;

	if (vec->nnz == 0) {
		sparse_vec_set(vec, src);
		snmod_vec_rescale(vec, a, p);
	}

	ulong na = a;
	ulong na_pr = n_mulmod_precomp_shoup(na, p.n);

	if (vec->nnz + src->nnz > vec->alloc)
		sparse_vec_realloc(vec, vec->nnz + src->nnz);

	ulong ptr1 = vec->nnz;
	ulong ptr2 = src->nnz;
	ulong ptr = vec->nnz + src->nnz;
	while (ptr1 > 0 && ptr2 > 0) {
		if (vec->indices[ptr1 - 1] == src->indices[ptr2 - 1]) {
			ulong entry =
				nmod_add(vec->entries[ptr1 - 1],
					n_mulmod_shoup(na, src->entries[ptr2 - 1], na_pr, p.n), p);
			if (entry != 0) {
				vec->indices[ptr - 1] = vec->indices[ptr1 - 1];
				vec->entries[ptr - 1] = entry;
				ptr--;
			}
			ptr1--;
			ptr2--;
		}
		else if (vec->indices[ptr1 - 1] < src->indices[ptr2 - 1]) {
			ulong entry = n_mulmod_shoup(na, src->entries[ptr2 - 1], na_pr, p.n);
			vec->indices[ptr - 1] = src->indices[ptr2 - 1];
			vec->entries[ptr - 1] = entry;
			ptr2--;
			ptr--;
		}
		else {
			vec->indices[ptr - 1] = vec->indices[ptr1 - 1];
			vec->entries[ptr - 1] = vec->entries[ptr1 - 1];
			ptr1--;
			ptr--;
		}
	}
	while (ptr2 > 0) {
		ulong entry = n_mulmod_shoup(na, src->entries[ptr2 - 1], na_pr, p.n);
		vec->indices[ptr - 1] = src->indices[ptr2 - 1];
		vec->entries[ptr - 1] = entry;
		ptr2--;
		ptr--;
	}

	// if ptr1 > 0, and ptr > 0
	for (size_t i = ptr1; i < ptr; i++) {
		vec->entries[i] = 0;
	}

	vec->nnz += src->nnz;
	sparse_vec_canonicalize(vec);
	if (vec->alloc > 4 * vec->nnz)
		sparse_vec_realloc(vec, 2 * vec->nnz);

	return 0;
}

int snmod_vec_sub_mul(snmod_vec_t vec, const snmod_vec_t src, const ulong a, nmod_t p) {
	return snmod_vec_add_mul(vec, src, nmod_neg(a, p), p);
}

int snmod_vec_add(snmod_vec_t vec, const snmod_vec_t src, nmod_t p) {
	return snmod_vec_add_mul(vec, src, (ulong)1, p);
}

int snmod_vec_sub(snmod_vec_t vec, const snmod_vec_t src, nmod_t p) {
	return snmod_vec_sub_mul(vec, src, (ulong)1, p);
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
