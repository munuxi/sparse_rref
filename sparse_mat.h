#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include "sparse_vec.h"

template <typename T> struct sparse_mat_struct {
    ulong nrow;
    ulong ncol;
    sparse_vec_struct<T> *rows;
};

template <typename T> using sparse_mat_t = struct sparse_mat_struct<T>[1];

typedef sparse_mat_t<ulong> snmod_mat_t;
typedef sparse_mat_t<fmpq> sfmpq_mat_t;

template <typename T>
static inline void _sparse_mat_init(sparse_mat_t<T> mat, ulong nrow, ulong ncol,
                                    ulong alloc) {
    mat->nrow = nrow;
    mat->ncol = ncol;
    mat->rows =
        (sparse_vec_struct<T> *)malloc(nrow * sizeof(sparse_vec_struct<T>));
    for (size_t i = 0; i < nrow; i++)
        _sparse_vec_init(mat->rows + i, ncol, alloc);
}

template <typename T>
static inline void sparse_mat_init(sparse_mat_t<T> mat, ulong nrow,
                                   ulong ncol) {
    _sparse_mat_init(mat, nrow, ncol, 1ULL);
}

template <typename T> static inline void sparse_mat_clear(sparse_mat_t<T> mat) {
    for (size_t i = 0; i < mat->nrow; i++)
        sparse_vec_clear(mat->rows + i);
    free(mat->rows);
    mat->nrow = 0;
    mat->ncol = 0;
    mat->rows = NULL;
}

template <typename T> static inline ulong sparse_mat_nnz(sparse_mat_t<T> mat) {
    ulong nnz = 0;
    for (size_t i = 0; i < mat->nrow; i++)
        nnz += mat->rows[i].nnz;
    return nnz;
}

template <typename T>
static inline ulong sparse_mat_alloc(sparse_mat_t<T> mat) {
    ulong alloc = 0;
    for (size_t i = 0; i < mat->nrow; i++)
        alloc += mat->rows[i].alloc;
    return alloc;
}

template <typename T>
static inline void sparse_mat_compress(sparse_mat_t<T> mat) {
    for (size_t i = 0; i < mat->nrow; i++)
        sparse_vec_realloc(mat->rows + i, mat->rows[i].nnz);
}

template <typename T>
static inline T *sparse_mat_entry(sparse_mat_t<T> mat, slong row, slong col,
                                  bool isbinary = false) {
    if (row < 0 || col < 0 || (ulong)row >= mat->nrow ||
        (ulong)col >= mat->ncol)
        return NULL;
    return sparse_vec_entry(mat->rows + row, col, isbinary);
}

template <typename T, typename S>
static inline void _sparse_mat_set_entry(sparse_mat_t<T> mat, slong row,
                                         slong col, S val) {
    if (row < 0 || col < 0 || (ulong)row >= mat->nrow ||
        (ulong)col >= mat->ncol)
        return;
    _sparse_vec_set_entry(mat->rows + row, col, val);
}

template <typename T>
static inline void sparse_mat_clear_zero_row(sparse_mat_t<T> mat) {
    ulong newnrow = 0;
    for (size_t i = 0; i < mat->nrow; i++) {
        if (mat->rows[i].nnz != 0) {
            mat->rows[newnrow] = mat->rows[i];
            newnrow++;
        } else {
            sparse_vec_clear(mat->rows + i);
        }
    }
    mat->nrow = newnrow;
}

template <typename T>
static inline void sparse_mat_transpose_pointer(sparse_mat_t<T *> mat2,
                                                sparse_mat_t<T> mat) {
    for (size_t i = 0; i < mat2->nrow; i++)
        mat2->rows[i].nnz = 0;

    for (size_t i = 0; i < mat->nrow; i++) {
        auto therow = mat->rows + i;
        for (size_t j = 0; j < therow->nnz; j++) {
            // if (scalar_is_zero(therow->entries + j))
            // 	continue;
            auto col = therow->indices[j];
            _sparse_vec_set_entry(mat2->rows + col, i, therow->entries + j);
        }
    }
}

template <typename T>
static inline void sparse_mat_transpose(sparse_mat_t<T> mat2,
                                        sparse_mat_t<T> mat) {
    for (size_t i = 0; i < mat2->nrow; i++)
        mat2->rows[i].nnz = 0;

    for (size_t i = 0; i < mat->nrow; i++) {
        auto therow = mat->rows + i;
        for (size_t j = 0; j < therow->nnz; j++) {
            if (scalar_is_zero(therow->entries + j))
                continue;
            auto col = therow->indices[j];
            _sparse_vec_set_entry(mat2->rows + col, i, therow->entries + j);
        }
    }
}

slong *sfmpq_mat_rref(sfmpq_mat_t mat, BS::thread_pool &pool,
                      rref_option_t opt);

static inline void snmod_mat_from_sfmpq(snmod_mat_t mat, const sfmpq_mat_t src,
                                        nmod_t p) {
    for (size_t i = 0; i < src->nrow; i++) {
        auto row = src->rows + i;
        snmod_vec_from_sfmpq(mat->rows + i, row, p);
    }
}

// IO
template <typename T> void sfmpq_mat_read(sfmpq_mat_t mat, T &st) {
    if (!st.is_open())
        return;
    std::string strLine;

    bool is_size = true;
    fmpq_t val;
    fmpq_init(val);

    int totalprint = 0;

    while (getline(st, strLine)) {
        if (strLine[0] == '%')
            continue;

        auto tokens = SplitString(strLine, " ");
        if (is_size) {
            ulong nrow = std::stoul(tokens[0]);
            ulong ncol = std::stoul(tokens[1]);
            ulong nnz = std::stoul(tokens[2]);
            // here we alloc 1, or alloc nnz/ncol ?
            sparse_mat_init(mat, nrow, ncol);
            is_size = false;
        } else {
            ulong row = std::stoul(tokens[0]) - 1;
            ulong col = std::stoul(tokens[1]) - 1;
            DeleteSpaces(tokens[2]);
            fmpq_set_str(val, tokens[2].c_str(), 10);
            _sparse_vec_set_entry(mat->rows + row, col, val);
        }
    }
}

template <typename T> void sfmpq_mat_write(sfmpq_mat_t mat, T &st) {
    // sfmpq_mat_compress(mat);
    st << "%%MatrixMarket matrix coordinate rational general" << '\n';
    st << mat->nrow << " " << mat->ncol << " " << sparse_mat_nnz(mat) << '\n';
    for (size_t i = 0; i < mat->nrow; i++) {
        auto therow = mat->rows + i;
        for (size_t j = 0; j < therow->nnz; j++) {
            if (fmpq_is_zero(therow->entries + j))
                continue;
            auto thenum = fmpq_get_str(NULL, 10, therow->entries + j);
            st << i + 1 << " " << therow->indices[j] + 1 << " " << thenum
               << '\n';
        }
    }
}

template <typename T> void sfmpq_mat_dense_write(sfmpq_mat_t mat, T &st) {
    // sfmpq_mat_compress(mat);
    for (size_t i = 0; i < mat->nrow; i++) {
        for (size_t j = 0; j < mat->ncol; j++) {
            auto entry = sparse_mat_entry(mat, i, j);
            if (entry == NULL)
                st << "0 ";
            else
                st << fmpq_get_str(NULL, 10, entry) << " ";
        }
        st << '\n';
    }
}

slong *snmod_mat_rref(snmod_mat_t mat, nmod_t p, BS::thread_pool &pool,
                      rref_option_t opt);

// IO
template <typename T> void snmod_mat_read(snmod_mat_t mat, nmod_t p, T &st) {
    if (!st.is_open())
        return;
    std::string strLine;

    bool is_size = true;
    ulong val;

    int totalprint = 0;

    while (getline(st, strLine)) {
        if (strLine[0] == '%')
            continue;

        auto tokens = SplitString(strLine, " ");
        if (is_size) {
            ulong nrow = std::stoul(tokens[0]);
            ulong ncol = std::stoul(tokens[1]);
            ulong nnz = std::stoul(tokens[2]);
            // here we alloc 1, or alloc nnz/ncol ?
            sparse_mat_init(mat, nrow, ncol);
            is_size = false;
        } else {
            ulong row = std::stoul(tokens[0]) - 1;
            ulong col = std::stoul(tokens[1]) - 1;
            DeleteSpaces(tokens[2]);
            slong ss = std::stoll(tokens[2]);
            if (ss < 0) {
                val = -ss;
                NMOD_RED(val, val, p);
                val = nmod_neg(val, p);
            } else {
                val = ss;
                NMOD_RED(val, val, p);
            }
            _sparse_vec_set_entry(mat->rows + row, col, val);
        }
    }
}

template <typename T> void snmod_mat_write(snmod_mat_t mat, T &st) {
    // snmod_mat_compress(mat);
    st << "%%MatrixMarket matrix coordinate integer general" << '\n';
    st << mat->nrow << " " << mat->ncol << " " << sparse_mat_nnz(mat) << '\n';
    for (size_t i = 0; i < mat->nrow; i++) {
        auto therow = mat->rows + i;
        for (size_t j = 0; j < therow->nnz; j++) {
            if (scalar_is_zero(therow->entries + j))
                continue;
            ulong thenum = therow->entries[j];
            st << i + 1 << " " << therow->indices[j] + 1 << " " << thenum
               << '\n';
        }
    }
}

// never use it somehow
template <typename T> void snmod_mat_dense_write(snmod_mat_t mat, T &st) {
    // snmod_mat_compress(mat);
    slong shiftedentry = 0;
    for (size_t i = 0; i < mat->nrow; i++) {
        for (size_t j = 0; j < mat->ncol; j++) {
            auto entry = sparse_mat_entry(mat, i, j);
            if (entry == NULL)
                st << "0 ";
            else
                st << *entry << " ";
        }
        st << '\n';
    }
}

#endif
