/*
    Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

    This file is part of Sparse_rref. The Sparse_rref is free software:
    you can redistribute it and/or modify it under the terms of the MIT
    License.
*/

/*
    To compile the library, WolframLibrary.h and WolframSparseLibrary.h are required,
    which are included in the Mathematica installation directory.

    The output of modrref is the join of the rref of A and its kernel.

    To load it in Mathematica, use the following code (as an example):

    ```mathematica

    rreflib = LibraryFunctionLoad[
        "rreflib.dll", "modrref",
        {{LibraryDataType[SparseArray],
          "Constant"}, {Integer}, {Integer}, {Integer}}, {LibraryDataType[SparseArray],
          "Shared"}
    ];
    
    (* the first matrix is the result of rref, and the second is its kernel *)
    modprref[mat_SparseArray, p_Integer, nthread_ : 1, method_ : 1] := 
        With[{joinedmat = rreflib[mat, p, nthread, method]},
         {joinedmat[[;; Length@mat]], joinedmat[[Length@mat + 1 ;;]]}];

    ```
*/

#include <string>
#include "sparse_mat.h"
#include "mma/WolframLibrary.h"
#include "mma/WolframSparseLibrary.h"

EXTERN_C DLLEXPORT int modrref(WolframLibraryData ld, mint Argc, MArgument *Args, MArgument Res) {
    auto mat = MArgument_getMSparseArray(Args[0]);
    auto p = MArgument_getInteger(Args[1]);
    auto nthreads = MArgument_getInteger(Args[2]);
    auto method = MArgument_getInteger(Args[3]);

    auto sf = ld->sparseLibraryFunctions;

    auto ranks = sf->MSparseArray_getRank(mat);
    if (ranks != 2 && sf->MSparseArray_getImplicitValue(mat) != 0)
        return LIBRARY_FUNCTION_ERROR;
    
    auto dims = sf->MSparseArray_getDimensions(mat);
    auto nrows = dims[0];
    auto ncols = dims[1];

    auto m_rowptr = sf->MSparseArray_getRowPointers(mat);
    auto m_colptr = sf->MSparseArray_getColumnIndices(mat);
    auto m_valptr = sf->MSparseArray_getExplicitValues(mat);

    // rowptr, valptr, colptr are managed by mathematica
    // do not free them
    mint* rowptr = ld->MTensor_getIntegerData(*m_rowptr);
    mint* valptr = ld->MTensor_getIntegerData(*m_valptr);
    mint* colptr = ld->MTensor_getIntegerData(*m_colptr);

    auto nnz = rowptr[nrows];
    BS::thread_pool pool((int)nthreads);
    rref_option_t opt;
    if (method == 0) 
        opt->pivot_dir = true;
    else 
        opt->pivot_dir = false;

    // init a sparse matrix
    nmod_t pp;
    fmpz_t tmp;
    fmpz_init(tmp);
    nmod_init(&pp, (ulong)p);
    sparse_mat_t<ulong> A, K;
    sparse_mat_init(A, nrows, ncols);
    for (auto i = 0; i < nrows; i++) {
        auto therow = A->rows + i;
        for (auto k = rowptr[i]; k < rowptr[i + 1]; k++) {
            fmpz_set_si(tmp, valptr[k]);
            fmpz_mod_ui(tmp, tmp, p);
            ulong entry = fmpz_get_ui(tmp);
            _sparse_vec_set_entry(therow, colptr[k] - 1, &entry);
        }
    }
    fmpz_clear(tmp);

    field_t F;
    field_init(F, FIELD_Fp, std::vector<ulong>{(ulong)p});

    auto pivots = sparse_mat_rref(A, F, pool, opt);
    auto len = sparse_mat_rref_kernel(K, A, pivots, F, pool);
    auto rank = pivots.size();
    
	if (len == 0) 
        nnz = sparse_mat_nnz(A);
    else 
        nnz = sparse_mat_nnz(A) + sparse_mat_nnz(K);

    MTensor pos, val, dim;
    mint dims_r2[] = {nnz, 2};
    ld->MTensor_new(MType_Integer, 2, dims_r2, &pos);
    mint dims_r1[] = {nnz};
    ld->MTensor_new(MType_Integer, 1, dims_r1, &val);
    dims_r1[0] = 2;
    ld->MTensor_new(MType_Integer, 1, dims_r1, &dim);

    mint* dimdata = ld->MTensor_getIntegerData(dim);
    dimdata[0] = A->nrow + K->nrow;
    dimdata[1] = ncols;
    mint* valdata = ld->MTensor_getIntegerData(val);
    mint* posdata = ld->MTensor_getIntegerData(pos);

    auto nownnz = 0;
    // output A
    for (auto i = 0; i < A->nrow; i++){
        auto therow = A->rows + i;
        for (auto j = 0; j < therow->nnz; j++){
            posdata[2 * nownnz] = i + 1;
            posdata[2 * nownnz + 1] = therow->indices[j] + 1;
            valdata[nownnz] = therow->entries[j];
            nownnz++;
        }
    }
    if (len > 0) {
        // output K
        for (auto i = 0; i < K->nrow; i++) {
            auto therow = K->rows + i;
            for (auto j = 0; j < therow->nnz; j++) {
                posdata[2 * nownnz] = A->nrow + i + 1;
                posdata[2 * nownnz + 1] = therow->indices[j] + 1;
                valdata[nownnz] = therow->entries[j];
                nownnz++;
            }
        }
        sparse_mat_clear(K);
    }
    sparse_mat_clear(A);

    MSparseArray result = 0;
    auto err = sf->MSparseArray_fromExplicitPositions(pos, val, dim, 0, &result);

    ld->MTensor_free(pos);
    ld->MTensor_free(val);
    ld->MTensor_free(dim);
    
    field_clear(F);
    
    if (err)
        return LIBRARY_FUNCTION_ERROR;

    MArgument_setMSparseArray(Res, result);
    // MArgument_setMTensor(Res, pos);

    return LIBRARY_NO_ERROR;
}
