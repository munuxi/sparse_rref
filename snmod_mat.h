#ifndef SNMOD_MAT_H
#define SNMOD_MAT_H

#include "snmod_vec.h"
#include "sparse_mat.h"
#include <fstream>
#include <iomanip>
#include <iostream>

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