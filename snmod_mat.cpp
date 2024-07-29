#include "sparse_mat.h"
#include <cstring>
#include "thread_pool.hpp"

// Row x - a*Row y
static inline void snmod_mat_xmay(snmod_mat_t mat, slong x, slong y, ulong a,
                                  nmod_t p) {
    snmod_vec_sub_scalar_sorted(mat->rows + x, mat->rows + y, a, p);
}

// Row x - a*Row y
static inline void snmod_mat_xmay_cached(snmod_mat_t mat, slong x, slong y,
                                         snmod_vec_t cache, ulong a, nmod_t p) {
    snmod_vec_sub_scalar_sorted_cached(mat->rows + x, mat->rows + y, cache, a,
                                       p);
}

static void snmod_mat_transpose_index(sindex_mat_t mat2, snmod_mat_t mat) {
    for (size_t i = 0; i < mat2->nrow; i++)
        mat2->rows[i].nnz = 0;

    for (size_t i = 0; i < mat->nrow; i++) {
        auto therow = mat->rows + i;
        for (size_t j = 0; j < therow->nnz; j++) {
            if (scalar_is_zero(therow->entries + j))
                continue;
            auto col = therow->indices[j];
            _sindex_vec_set_entry(mat2->rows + col, i);
        }
    }
}

// first look for rows with only one nonzero value and eliminate them
// we assume that mat is canonical, i.e. each index is sorted
// and the result is also canonical
static ulong eliminate_row_with_one_nnz(snmod_mat_t mat,
                                        sparse_mat_t<ulong *> tranmat,
                                        slong *donelist) {
    auto localcounter = 0;
    slong *pivlist = (slong *)malloc(mat->nrow * sizeof(slong));
    slong *collist = (slong *)malloc(mat->ncol * sizeof(slong));
    for (size_t i = 0; i < mat->nrow; i++)
        pivlist[i] = -1;
    for (size_t i = 0; i < mat->ncol; i++)
        collist[i] = -1;
    for (size_t i = 0; i < mat->nrow; i++) {
        if (donelist[i] != -1)
            continue;
        if (mat->rows[i].nnz == 1) {
            if (collist[mat->rows[i].indices[0]] == -1) {
                localcounter++;
                pivlist[i] = mat->rows[i].indices[0];
                collist[mat->rows[i].indices[0]] = i;
            }
        }
    }

    if (localcounter == 0) {
        free(pivlist);
        free(collist);
        return localcounter;
    }

    sparse_mat_transpose_pointer(tranmat, mat);
    for (size_t i = 0; i < mat->nrow; i++) {
        if (pivlist[i] == -1)
            continue;
        auto thecol = tranmat->rows + pivlist[i];
        ulong colone = 0;
        for (size_t j = 0; j < thecol->nnz; j++) {
            if (thecol->indices[j] == i) {
                colone = j;
                *(thecol->entries[j]) = 1;
            } else
                *(thecol->entries[j]) = 0;
        }
    }

    for (size_t i = 0; i < mat->nrow; i++) {
        sparse_vec_canonicalize(mat->rows + i);
    }

    for (size_t i = 0; i < mat->nrow; i++)
        if (pivlist[i] != -1)
            donelist[i] = pivlist[i];

    free(pivlist);
    free(collist);
    return localcounter;
}

static ulong eliminate_row_with_one_nnz_rec(snmod_mat_t mat,
                                            sparse_mat_t<ulong *> tranmat,
                                            slong *donelist,
                                            bool verbose = false) {
    slong depth = 0;
    ulong oldnnz = 0;
    ulong localcounter = 0;
    ulong count = 0;

    do {
        oldnnz = sparse_mat_nnz(mat);
        localcounter = eliminate_row_with_one_nnz(mat, tranmat, donelist);
        if (verbose) {
            std::cout << "\r-- eliminating rows with only one element, depth: "
                      << depth << ", eliminated row: " << count << std::flush;
        }
        count += localcounter;
        depth++;
    } while (localcounter > 0);
    return count;
}

static inline slong rowcolpart(std::vector<slong> conp, slong nrow) {
    slong i;
    for (i = 0; i < conp.size(); i++)
        if (conp[i] >= nrow)
            break;
    return i;
}

slong findrowpivot(snmod_mat_t mat, slong col, slong *rowpivs,
                   std::vector<slong> &colparts, std::vector<slong> &kkparts,
                   std::vector<std::vector<slong>> &rowparts, slong *dolist,
                   ulong **entrylist, slong &dolist_len,
                   BS::thread_pool &pool) {

    auto mm = colparts[col];
    auto rowlist = rowparts[mm];
    // parallelize this loop

    auto loop = [&](slong i) {
        if (rowlist[i] == -1)
            return;
        auto entry = sparse_mat_entry(mat, rowlist[i], col, true);
        if (entry == NULL || scalar_is_zero(entry))
            return;
        // add one to make it nonzero
        entrylist[i] = entry;
        dolist[i] = 1;
    };

    pool.detach_loop<slong>(0, kkparts[mm], loop);
    pool.wait();

    slong rowi = -1; // choose the row with the minimal nnz
    ulong mininnz = ULLONG_MAX;
    for (size_t i = 0; i < kkparts[mm]; i++) {
        if (dolist[i] == -1)
            continue;

        auto row = rowparts[mm][i];

        if (rowpivs[row] !=
            -1) // do not choose rows that have been used to eliminate
            continue;

        dolist[dolist_len] = row;
        entrylist[dolist_len] = entrylist[i];
        dolist_len++;

        if (mat->rows[row].nnz < mininnz) { // check the nnz of the row
            rowi = row;
            mininnz = mat->rows[row].nnz;
        }
    }
    return rowi;
}

auto findmanypivots(snmod_mat_t mat, sparse_mat_t<ulong *> tranmat,
                    slong *rowpivs, std::vector<slong> &colperm,
                    std::vector<slong>::iterator start,
                    size_t max_depth = ULLONG_MAX) {

    std::vector<std::pair<slong, std::vector<slong>::iterator>> pivots;
    std::unordered_set<slong> prows;
    prows.reserve(std::min((size_t)4096, max_depth));
    for (auto col = start; col < colperm.end(); col++) {
        if (col - start > max_depth)
            break;
        bool flag = true;
        auto thecol = tranmat->rows + *col;
        auto indices = thecol->indices;
        for (size_t i = 0; i < thecol->nnz; i++) {
            flag = flag && (prows.find(indices[i]) == prows.end());
            if (!flag)
                break;
        }
        if (!flag)
            continue;

        if (thecol->nnz == 0)
            continue;
        slong row;
        ulong mnnz = ULLONG_MAX;
        for (size_t i = 0; i < thecol->nnz; i++) {
            if (rowpivs[indices[i]] != -1)
                continue;
            if (mat->rows[indices[i]].nnz < mnnz) {
                row = indices[i];
                mnnz = mat->rows[row].nnz;
            }
        }
        if (mnnz != ULLONG_MAX) {
            pivots.push_back(std::make_pair(row, col));
            prows.insert(row);
        }
    }
    return pivots;
}

// we assume that the matrix is full rank in rows
slong *snmod_mat_rref(snmod_mat_t mat, nmod_t p, BS::thread_pool &pool,
                      rref_option_t opt) {
    // first canonicalize, sort and compress the matrix
    sparse_mat_compress(mat);
    for (size_t i = 0; i < mat->nrow; i++) {
        sparse_vec_sort_indices(mat->rows + i);
    }

    ulong rank = 0;

    ulong init_nnz = sparse_mat_nnz(mat);
    ulong now_nnz = init_nnz;

    ulong scalar;

    // store the pivots that have been used
    // -1 is not used
    slong *rowpivs = (slong *)malloc(mat->nrow * sizeof(slong));
    for (size_t i = 0; i < mat->nrow; i++)
        rowpivs[i] = -1;
    // perm the col
    std::vector<slong> colperm(mat->ncol);
    for (size_t i = 0; i < mat->ncol; i++) {
        colperm[i] = i;
    }

    // look for row with only one non-zero entry

    // compute the transpose of pointers of the matrix
    sparse_mat_t<ulong *> tranmat;
    sparse_mat_init(tranmat, mat->ncol, mat->nrow);

    bool verbose = opt->verbose;
    ulong count =
        eliminate_row_with_one_nnz_rec(mat, tranmat, rowpivs, verbose);
    rank += count;
    now_nnz = sparse_mat_nnz(mat);
    if (verbose) {
        std::cout << "\n** eliminated " << count
                  << " rows, and reduce nnz: " << init_nnz << " -> " << now_nnz
                  << std::endl;
    }

    sparse_mat_transpose_pointer(tranmat, mat);

    // sort pivots by nnz, it will be faster
    std::stable_sort(colperm.begin(), colperm.end(),
                     [&tranmat](slong a, slong b) {
                         return tranmat->rows[a].nnz < tranmat->rows[b].nnz;
                     });

    ulong **entrylist = (ulong **)malloc(mat->nrow * sizeof(ulong *));
    slong *dolist = (slong *)malloc(mat->nrow * sizeof(slong));

    // look for pivot cols with only one nonzero element
    slong countone = 0;
    for (size_t i = 0; i < mat->ncol; i++) {
        if (tranmat->rows[colperm[i]].nnz == 1) {
            auto row = tranmat->rows[colperm[i]].indices[0];
            if (rowpivs[row] != -1)
                continue;
            rowpivs[row] = colperm[i];
            dolist[countone] = row;
            entrylist[countone] =
                sparse_vec_entry(mat->rows + row, rowpivs[row], true);
            countone++;
            rank++;
        } else if (tranmat->rows[colperm[i]].nnz > 1)
            break; // since it's sorted
    }

    // eliminate the rows with only one nonzero element
    // we can do this in parallel

    pool.detach_loop<slong>(0, countone, [&](slong i) {
        ulong scalar;
        scalar = nmod_inv(*(entrylist[i]), p);
        snmod_vec_rescale(mat->rows + dolist[i], scalar, p);
    });
    pool.wait();

    if (verbose) {
        std::cout << ">> there are " << countone
                  << " columns that has already been eliminated" << std::endl;
    }

    // Matching
    // BipartiteMatcher matcher(mat->nrow, mat->ncol);
    //
    // for (size_t i = 0; i < mat->nrow; i++){
    // 	if (rowpivs[i] != -1)
    // 		continue;
    // 	auto therow = mat->rows + i;
    // 	for (size_t j = 0; j < therow->nnz; j++){
    // 			matcher.addEdge(i + 1, therow->indices[j] + 1 +
    // mat->nrow);
    // }
    //
    // slong maxMatching = matcher.maxMatching();
    // std::cout << "Maximum matching is " << maxMatching << std::endl;
    //
    // std::vector<std::pair<slong, slong>> pairs = matcher.getMatchingPairs();
    // for (const auto& pair : pairs) {
    //     std::cout << "Row " << pair.first << " is matched with Column " <<
    //     pair.second << std::endl;
    // }

    init_nnz = sparse_mat_nnz(mat);

    countone = 0;
    for (size_t i = 0; i < mat->ncol; i++) {
        if (tranmat->rows[colperm[i]].nnz <= 1)
            countone++;
        else
            break; // since it's sorted
    }

    // we cut the matrix as
    // (*********| A A A ... A A A )
    // (*********| A A A ... A A A )
    // (*********| A A A ... A A A )
    //  countone |   submatrix B
    // countone part is eliminated

    if (verbose)
        std::cout << "-- computing connected components" << std::endl;

    Graph adjgraph((mat->ncol) + (mat->nrow));

    for (size_t i = countone; i < mat->ncol; i++) {
        auto thecol = tranmat->rows + colperm[i];
        for (size_t j = 0; j < thecol->nnz; j++) {
            adjgraph.addEdge(thecol->indices[j], mat->nrow + colperm[i]);
        }
    }

    auto connected_components = adjgraph.findMaximalConnectedComponents();

    if (verbose) {
        std::cout << "** found " << connected_components.size() << " components"
                  << std::endl;
    }

    std::sort(connected_components.begin(), connected_components.end(),
              [](std::vector<slong> &a, std::vector<slong> &b) {
                  return a.size() < b.size();
              });

    for (auto &g : connected_components)
        std::sort(g.begin(), g.end());

    // use colparts to label components
    std::vector<slong> colparts(mat->ncol);
    std::vector<slong> kkparts(connected_components.size());
    std::vector<std::vector<slong>> rowparts(connected_components.size());
    for (size_t i = 0; i < mat->ncol; i++)
        colparts[i] = -1;
    for (auto mm = 0; mm < connected_components.size(); mm++) {
        auto g = connected_components[mm];
        kkparts[mm] = rowcolpart(g, mat->nrow);
        rowparts[mm] = std::vector<slong>(g.begin(), g.begin() + kkparts[mm]);
        for (size_t i = kkparts[mm]; i < g.size(); i++)
            colparts[g[i] - mat->nrow] = mm;
    }
    adjgraph.clear();

    std::stable_sort(
        colperm.begin() + countone, colperm.end(),
        [&colparts](slong a, slong b) { return colparts[a] < colparts[b]; });

    // int num_cache = pool.get_thread_count();
    // snmod_mat_t cachemat;
    // sparse_mat_init(cachemat, num_cache, mat->ncol);

    // upper triangle (with respect to row and col perm)

    sparse_mat_transpose_pointer(tranmat, mat);

    // upper triangle (with respect to row and col perm)
    while (countone < mat->ncol) {
        auto start = clocknow();

        if (colparts[colperm[countone]] == -1) {
            countone++;
            continue;
        }

        auto ps = findmanypivots(mat, tranmat, rowpivs, colperm,
                                 colperm.begin() + countone, opt->search_depth);

        for (auto kk = 0; kk < ps.size(); kk++) {
            auto start = clocknow();

            auto pp = ps[kk];
            auto row = pp.first;
            auto col = *pp.second;
            rowpivs[row] = col;
            rank++;
            scalar = nmod_inv(*sparse_mat_entry(mat, row, col, true), p);
            snmod_vec_rescale(mat->rows + row, scalar, p);
            auto thecol = tranmat->rows + col;
            pool.detach_loop<slong>(0, thecol->nnz, [&](slong i) {
                if (thecol->indices[i] == row)
                    return;
                auto entry =
                    sparse_mat_entry(mat, thecol->indices[i], col, true);
                // auto this_id = BS::this_thread::get_index().value();
                // snmod_mat_xmay_cached(mat, thecol->indices[i], row,
                // cachemat->rows + this_id, *entry, p);
                snmod_mat_xmay(mat, thecol->indices[i], row, *entry, p);
            });
            pool.wait();
            auto end = clocknow();

            if (verbose && (countone + kk + 1) % opt->printlen == 0) {
                now_nnz = sparse_mat_nnz(mat);
                std::cout << "\r-- Col: " << countone + kk + 1 << "/"
                          << mat->ncol << "  " << "row to eliminate: "
                          << tranmat->rows[*pp.second].nnz - 1 << "  "
                          << "rank: " << rank << "  " << "nnz: " << now_nnz
                          << "  " << "density: "
                          << (double)now_nnz / (mat->nrow * mat->ncol) << "  "
                          << "speed: " << 1 / usedtime(start, end) << " col/s"
                          << std::flush;
            }
        }

        std::vector<slong> tempcolperm(colperm.begin() + countone,
                                       colperm.end());
        std::vector<slong> diffs(ps.size());
        for (size_t i = 0; i < diffs.size(); i++)
            diffs[i] = ps[i].second - (colperm.begin() + countone);
        remove_indices(tempcolperm, diffs);

        for (auto i = 0; i < ps.size(); i++)
            colperm[countone + i] = *ps[i].second;
        for (auto i = 0; i < tempcolperm.size(); i++)
            colperm[countone + ps.size() + i] = tempcolperm[i];

        // for (size_t i = 0; i < ps.size(); i++){
        //	std::swap(colperm[countone + i], *ps[i].second);
        // }

        count = eliminate_row_with_one_nnz_rec(mat, tranmat, rowpivs, false);
        rank += count;
        now_nnz = sparse_mat_nnz(mat);
        // if (verbose) {
        //	std::cout << "\n** eliminated " << count << " rows, and nnz is "
        //		<< now_nnz << std::endl;
        // }

        countone += ps.size() + 1;

        sparse_mat_transpose_pointer(tranmat, mat);
        // sort pivots by nnz, it will be faster
        std::stable_sort(colperm.begin() + countone, colperm.end(),
                         [&tranmat, &colparts](slong a, slong b) {
                             if (colparts[a] < colparts[b]) {
                                 return true;
                             } else if (colparts[a] == colparts[b]) {
                                 return tranmat->rows[a].nnz <
                                        tranmat->rows[b].nnz;
                             } else
                                 return false;
                         });

        if (ps.size() < opt->search_min) {
            break;
        }
    }

    for (auto kk = countone; kk < mat->ncol; kk++) {
        auto start = clocknow();

        // first scan to get the list of rows to eliminate
        // and determine the the pivot with minimal nnz

        auto pivot = colperm[kk];

        if (colparts[pivot] == -1) {
            continue;
        }

        slong dolist_len = 0;
        for (auto i = 0; i < mat->nrow; i++)
            dolist[i] = -1;
        slong rowi =
            findrowpivot(mat, pivot, rowpivs, colparts, kkparts, rowparts,
                         dolist, entrylist, dolist_len, pool);

        if (rowi == -1)
            continue;

        rowpivs[rowi] = pivot;
        rank++;

        scalar = nmod_inv(*sparse_vec_entry(mat->rows + rowi, pivot, true), p);
        snmod_vec_rescale(mat->rows + rowi, scalar, p);

        pool.detach_loop<slong>(0, dolist_len, [&](slong i) {
            if (dolist[i] == rowi)
                return;
            snmod_mat_xmay(mat, dolist[i], rowi, *entrylist[i], p);
        });
        pool.wait();

        if (kk % opt->sort_step == 0 || kk == mat->ncol - 1) {
            // auto oldalloc = sparse_mat_alloc(mat);

            if (verbose)
                std::cout << std::endl;
            count =
                eliminate_row_with_one_nnz_rec(mat, tranmat, rowpivs, verbose);
            rank += count;
            now_nnz = sparse_mat_nnz(mat);
            if (verbose) {
                std::cout << "\n** eliminated " << count << " rows, and nnz is "
                          << now_nnz << std::endl;
            }

            // sort pivots by nnz, it will be faster
            std::stable_sort(colperm.begin() + kk + 1, colperm.end(),
                             [&tranmat, &colparts](slong a, slong b) {
                                 if (colparts[a] < colparts[b]) {
                                     return true;
                                 } else if (colparts[a] == colparts[b]) {
                                     return tranmat->rows[a].nnz <
                                            tranmat->rows[b].nnz;
                                 } else
                                     return false;
                             });

            // if (verbose) {
            //	std::cout << "-- cleaning up: " <<
            //		"alloc " << oldalloc << " shrink to 2*nnz " <<
            //std::endl;
            // }
        }

        auto end = clocknow();

        if (verbose) {
            now_nnz = sparse_mat_nnz(mat);
            if (kk % opt->printlen == 0 || kk == mat->ncol - 1) {
                std::cout << "\r-- Col: " << kk + 1 << "/" << mat->ncol << "  "
                          << "row to eliminate: " << dolist_len << "  "
                          << "rank: " << rank << "  " << "nnz: " << now_nnz
                          << "  " << "density: "
                          << (double)now_nnz / (mat->nrow * mat->ncol) << "  "
                          << "speed: " << 1 / usedtime(start, end) << " col/s"
                          << std::flush;
            }
        }
    }

    if (verbose) {
        std::cout << "\n** Rank: " << rank << " nnz: " << sparse_mat_nnz(mat)
                  << "  " << std::endl;
    }

    if (verbose) {
        std::cout << "\n>> Reverse solving: " << std::endl;
    }

    std::vector<slong> colpivs(mat->ncol, -1);
    for (size_t i = 0; i < mat->nrow; i++) {
        if (rowpivs[i] != -1)
            colpivs[rowpivs[i]] = i;
    }

    // the ordering of pivots is important
    // which makes the submatrix is upper triangular
    // i.e. mat(row(p[j]), col(p[i])) = 0 for i < j
    std::vector<std::pair<slong, slong>> pivots;
    for (slong i = 0; i < mat->ncol; i++) {
        auto col = colperm[i];
        auto row = colpivs[col];
        if (row != -1) {
            pivots.push_back(std::make_pair(row, col));
        }
    }

    // we only need to look for pivots once by transposing
    // since the matrix is upper triangular
    sparse_mat_transpose_pointer(tranmat, mat);

    for (size_t i = 0; i < pivots.size(); i++) {
        auto pp = pivots[pivots.size() - 1 - i]; // reverse order
        auto thecol = tranmat->rows + pp.second;
        auto start = clocknow();
        if (thecol->nnz != 1) {
            pool.detach_loop<slong>(0, thecol->nnz, [&](slong j) {
                if (thecol->indices[j] == pp.first)
                    return;
                auto entry =
                    sparse_mat_entry(mat, thecol->indices[j], pp.second, true);
                snmod_mat_xmay(mat, thecol->indices[j], pp.first, *entry, p);
            });
            pool.wait();
        }
        auto end = clocknow();
        if (verbose) {
            if (i % opt->printlen == 0 || i == pivots.size() - 1) {
                now_nnz = sparse_mat_nnz(mat);
                std::cout << "\r-- Row: " << (i + 1) << "/" << pivots.size()
                          << "  " << "row to eliminate: " << thecol->nnz - 1
                          << "  " << "nnz: " << now_nnz << "  " << "density: "
                          << (double)now_nnz / (mat->nrow * mat->ncol) << "  "
                          << "speed: " << 1 / usedtime(start, end) << " row/s"
                          << std::flush;
            }
        }
    }

    if (verbose) {
        std::cout << std::endl;
    }

    sparse_mat_clear(tranmat);

    free(entrylist);
    free(dolist);

    if (verbose) {
        std::cout << std::endl;
    }

    return rowpivs;
}