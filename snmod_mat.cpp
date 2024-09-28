#include <set>
#include "sparse_mat.h"

using iter = std::vector<slong>::iterator;

// Row x - a*Row y

inline void snmod_mat_xmay(snmod_mat_t mat, slong x, slong y, ulong a,
	nmod_t p) {
	snmod_vec_sub_mul(mat->rows + x, mat->rows + y, a, p);
}

// first write a stupid one
// TODO: Gilbert-Peierls algorithm for parallel computation 
// see https://hal.science/hal-01333670/document
void schur_complete(snmod_mat_t mat, slong row, std::vector<std::pair<slong, slong>>& pivots, 
	std::vector<slong>& leftcols, int ordering, nmod_t p, ulong* tmpvec) {
	if (ordering < 0) {
		std::vector<std::pair<slong, slong>> npivots(pivots.rbegin(), pivots.rend());
		schur_complete(mat, row, npivots, leftcols, -ordering, p, tmpvec);
	}

	auto therow = sparse_mat_row(mat, row);

	if (therow->nnz == 0)
		return;
	
	// if pivots size is small, we can use the sparse vector
	// to save to cost of converting between sparse and dense
	// vectors, otherwise we use dense vector
	if (pivots.size() < 100) {
		for (auto [r, c] : pivots) {
			auto entry = sparse_vec_entry(therow, c);
			if (entry == NULL)
				continue;
			auto row = mat->rows + r;
			snmod_vec_sub_mul(therow, row, *entry, p);
		}
		return;
	}

	std::set<slong> nonzero_c; // it's important that it is sorted

	s_memset(tmpvec, (ulong)0, mat->ncol);
	for (size_t i = 0; i < therow->nnz; i++) {
		nonzero_c.insert(therow->indices[i]);
		tmpvec[therow->indices[i]] = therow->entries[i];
	}

	for (auto [r, c] : pivots) {
		auto entry = tmpvec[c];
		if (entry == 0)
			continue;

		auto row = sparse_mat_row(mat, r);

		ulong e_pr = n_mulmod_precomp_shoup(entry, p.n);
		for (size_t i = 0; i < row->nnz; i++) {
			tmpvec[row->indices[i]] = _nmod_sub(tmpvec[row->indices[i]],
				n_mulmod_shoup(entry, row->entries[i], e_pr, p.n), p);
			nonzero_c.insert(row->indices[i]);
		}
	}
	therow->nnz = 0;
	for (auto i : nonzero_c) {
		if (tmpvec[i] != 0)
			_sparse_vec_set_entry(therow, i, tmpvec + i);
	}
}

// upper solver : ordering = -1
// lower solver : ordering = 1
void triangular_solver(snmod_mat_t mat, std::vector<std::pair<slong, slong>>& pivots,
	rref_option_t opt, int ordering, nmod_t p, BS::thread_pool& pool) {
	bool verbose = opt->verbose;
	auto printstep = opt->print_step;

	std::vector<std::vector<slong>> tranmat(mat->ncol);

	// we only need to compute the transpose of the submatrix involving pivots

	for (size_t i = 0; i < pivots.size(); i++) {
		auto therow = sparse_mat_row(mat, pivots[i].first);
		for (size_t j = 0; j < therow->nnz; j++) {
			if (scalar_is_zero(therow->entries + j))
				continue;
			auto col = therow->indices[j];
			tranmat[col].push_back(pivots[i].first);
		}
	}

	for (size_t i = 0; i < pivots.size(); i++) {
		size_t index = i;
		if (ordering < 0) 
			index = pivots.size() - 1 - i;
		auto pp = pivots[index];
		auto thecol = tranmat[pp.second];
		auto start = clocknow();
		if (thecol.size() > 1) {
			auto loop = [&](slong j) {
				auto r = thecol[j];
				if (r == pp.first)
					return;
				auto entry = sparse_mat_entry(mat, r, pp.second, true);
				snmod_mat_xmay(mat, r, pp.first, *entry, p);
				};
			pool.detach_loop<slong>(0, thecol.size(), loop);
			pool.wait();
		}
		auto end = clocknow();

		if ((i % printstep == 0 || i == pivots.size() - 1) && verbose && thecol.size() > 1) {
			auto now_nnz = sparse_mat_nnz(mat);
			std::cout << "\r-- Row: " << (i + 1) << "/" << pivots.size()
				<< "  " << "row to eliminate: " << thecol.size() - 1
				<< "  " << "nnz: " << now_nnz << "  " << "density: "
				<< (double)100 * now_nnz / (mat->nrow * mat->ncol)
				<< "%  " << "speed: " << 1 / usedtime(start, end)
				<< " row/s" << std::flush;
		}
	}
}

std::vector<std::pair<slong, slong>> snmod_mat_rref_c(snmod_mat_t mat, nmod_t p, BS::thread_pool& pool,
	rref_option_t opt) {
	// first canonicalize, sort and compress the matrix
	for (size_t i = 0; i < mat->nrow; i++) {
		sparse_vec_sort_indices(mat->rows + i);
		sparse_vec_canonicalize(mat->rows + i);
	}

	ulong init_nnz = sparse_mat_nnz(mat);
	ulong now_nnz = init_nnz;

	// store the pivots that have been used
	// -1 is not used
	std::vector<slong> rowpivs(mat->nrow, -1);
	std::vector<std::pair<slong, slong>> pivots;
	// perm the col
	std::vector<slong> colperm(mat->ncol);
	for (size_t i = 0; i < mat->ncol; i++) 
		colperm[i] = i;

	// look for row with only one non-zero entry

	// compute the transpose of pointers of the matrix
	sparse_mat_t<ulong*> tranmatp;
	sparse_mat_init(tranmatp, mat->ncol, mat->nrow);

	bool verbose = opt->verbose;
	ulong count =
		eliminate_row_with_one_nnz_rec(mat, tranmatp, rowpivs, verbose);
	now_nnz = sparse_mat_nnz(mat);
	if (verbose) {
		std::cout << "\n** eliminated " << count
			<< " rows, and reduce nnz: " << init_nnz << " -> " << now_nnz
			<< std::endl;
	}

	sparse_mat_transpose(tranmatp, mat);

	// sort pivots by nnz, it will be faster
	std::stable_sort(colperm.begin(), colperm.end(),
		[&tranmatp](slong a, slong b) {
			return tranmatp->rows[a].nnz < tranmatp->rows[b].nnz;
		});

	// look for pivot cols with only one nonzero element
	slong kk = 0;
	std::fill(rowpivs.begin(), rowpivs.end(), -1);
	for (; kk < mat->ncol; kk++) {
		auto nnz = tranmatp->rows[colperm[kk]].nnz;
		if (nnz == 0)
			continue;
		if (nnz == 1) {
			auto row = tranmatp->rows[colperm[kk]].indices[0];
			if (rowpivs[row] != -1)
				continue;
			rowpivs[row] = colperm[kk];
			auto e = sparse_mat_entry(mat, row, rowpivs[row], true);
			auto scalar = nmod_inv(*e, p);
			snmod_vec_rescale(mat->rows + row, scalar, p);
			pivots.push_back(std::make_pair(row, colperm[kk]));
		}
		else if (nnz > 1)
			break; // since it's sorted
	}
	sparse_mat_clear(tranmatp);

	ulong* cachedensedmat = s_malloc<ulong>(mat->ncol * pool.get_thread_count());

	sparse_mat_t<bool> tranmat;
	sparse_mat_init(tranmat, mat->ncol, mat->nrow);
	sparse_mat_transpose(tranmat, mat);

	std::vector<slong> leftrows;
	leftrows.reserve(mat->nrow);
	for (size_t i = 0; i < mat->nrow; i++) {
		if (rowpivs[i] != -1 || mat->rows[i].nnz == 0)
			continue;
		leftrows.push_back(i);
	}

	double oldpr = 0;
	// upper triangle (with respect to row and col perm)
	while (kk < mat->ncol) {
		auto start = clocknow();

		auto ps = findmanypivots_c(mat, tranmat, rowpivs, colperm,
			colperm.begin() + kk, opt->search_depth);
		if (ps.size() == 0)
			break;

		std::vector<std::pair<slong, slong>> n_pivots;
		for (auto i = ps.rbegin(); i != ps.rend(); i++) {
			auto [r, cp] = *i;
			rowpivs[r] = *cp;
			n_pivots.push_back(std::make_pair(r, *cp));
			pivots.push_back(std::make_pair(r, *cp));
			ulong scalar = nmod_inv(*sparse_mat_entry(mat, r, *cp), p);
			snmod_vec_rescale(mat->rows + r, scalar, p);
		}

		ulong n_leftrows = 0;
		for (size_t i = 0; i < leftrows.size(); i++) {
			auto row = leftrows[i];
			if (rowpivs[row] != -1 || mat->rows[row].nnz == 0)
				continue;
			leftrows[n_leftrows] = row;
			n_leftrows++;
		}
		leftrows.resize(n_leftrows);

		std::vector<slong> leftcols(colperm.begin() + kk, colperm.end());
		std::sort(leftcols.begin(), leftcols.end());
		
		std::vector<uint8_t> flags(leftrows.size(), 0);

		pool.detach_loop<slong>(0, leftrows.size(), [&](slong i) {
			auto id = BS::this_thread::get_index().value();
			schur_complete(mat, leftrows[i], n_pivots, 
				leftcols, 1, p, cachedensedmat + id * mat->ncol);
			flags[i] = 1;
			});

		// reorder the cols, move ps to the front
		std::unordered_set<slong> indices(ps.size());
		for (size_t i = 0; i < ps.size(); i++)
			indices.insert(ps[i].second - colperm.begin());
		std::vector<slong> result(colperm.begin(), colperm.begin() + kk);
		result.reserve(colperm.size());
		for (auto ind : ps) {
			result.push_back(*ind.second);
		}
		for (auto it = kk; it < mat->ncol; it++) {
			if (indices.find(it) == indices.end()) {
				result.push_back(colperm[it]);
			}
		}
		colperm = std::move(result);
		std::vector<slong> donelist(rowpivs);

		bool print_once = true; // print at least once
		// we need first clear the transpose matrix
		for (auto i = 0; i < tranmat->nrow; i++)
			tranmat->rows[i].nnz = 0;

		ulong localcount = 0;
		while (localcount < leftrows.size()) {
			for (size_t i = 0; i < leftrows.size(); i++) {
				if (flags[i]) {
					auto row = leftrows[i];
					auto therow = mat->rows + row;
					for (size_t j = 0; j < therow->nnz; j++) {
						auto col = therow->indices[j];
						_sparse_vec_set_entry(tranmat->rows + col, row, (bool*)NULL);
					}
					flags[i] = 0;
					localcount++;
				}
			}

			double pr = kk + (1.0 * ps.size() * localcount) / leftrows.size();
			if (verbose && (print_once || pr - oldpr > opt->print_step)) {
				auto end = clocknow();
				now_nnz = sparse_mat_nnz(mat);
				std::cout << "\r-- Col: " << (int)pr << "/"
					<< mat->ncol
					<< "  rank: " << pivots.size() << "  " << "nnz: " << now_nnz
					<< "  " << "density: "
					<< 100 * (double)now_nnz / (mat->nrow * mat->ncol)
					<< "%  " << "speed: " <<
					((pr - oldpr) / usedtime(start, end))
					<< " col/s" << std::flush;
				oldpr = pr;
				start = end;
				print_once = false;
			}
		}
		pool.wait();

		kk += ps.size();
	}

	if (verbose) {
		std::cout << "\n** Rank: " << pivots.size() << " nnz: " << sparse_mat_nnz(mat)
			<< "  " << std::endl;
		std::cout << "\n>> Reverse solving: " << std::endl;
	}

	// the matrix is upper triangular
	triangular_solver(mat, pivots, opt, -1, p, pool);

	if (verbose) {
		std::cout << '\n' << std::endl;
	}

	sparse_mat_clear(tranmat);

	free(cachedensedmat);

	return pivots;
}

std::vector<std::pair<slong, slong>> snmod_mat_rref_r(snmod_mat_t mat, nmod_t p, BS::thread_pool& pool,
	rref_option_t opt) {
	// first canonicalize, sort and compress the matrix

	for (size_t i = 0; i < mat->nrow; i++) {
		sparse_vec_sort_indices(mat->rows + i);
		sparse_vec_canonicalize(mat->rows + i);
	}
	// sparse_mat_compress(mat);

	std::vector<slong> rowperm(mat->nrow);
	for (size_t i = 0; i < mat->nrow; i++)
		rowperm[i] = i;

	auto printstep = opt->print_step;
	bool verbose = opt->verbose;

	ulong init_nnz = sparse_mat_nnz(mat);
	ulong now_nnz = init_nnz;

	// store the pivots that have been used
	// -1 is not used
	std::vector<slong> rowpivs(mat->nrow, -1);
	std::vector<slong> colpivs(mat->ncol, -1);

	sparse_mat_t<ulong*> tranmatp;
	sparse_mat_init(tranmatp, mat->ncol, mat->nrow);
	ulong count =
		eliminate_row_with_one_nnz_rec(mat, tranmatp, rowpivs, verbose);
	now_nnz = sparse_mat_nnz(mat);
	if (verbose) {
		std::cout << "\n** eliminated " << count
			<< " rows, and reduce nnz: " << init_nnz << " -> " << now_nnz
			<< std::endl;
	}
	init_nnz = now_nnz;
	sparse_mat_clear(tranmatp);

	// sort rows by nnz
	std::stable_sort(rowperm.begin(), rowperm.end(),
		[&mat](slong a, slong b) {
			if (mat->rows[a].nnz < mat->rows[b].nnz) {
				return true;
			}
			else if (mat->rows[a].nnz == mat->rows[b].nnz) {
				auto ri1 = mat->rows[a].indices;
				auto ri2 = mat->rows[b].indices;
				auto nnz = mat->rows[a].nnz;
				return std::lexicographical_compare(ri1, ri1 + nnz, ri2, ri2 + nnz);
			}
			else
				return false;
		});

	std::vector<std::pair<slong, slong>> pivots;

	sparse_mat_t<bool> tranmat;
	sparse_mat_init(tranmat, mat->ncol, mat->nrow);

	ulong* cachedensedmat = s_malloc<ulong>(mat->ncol * pool.get_thread_count());

	// skip the rows with only one/zero nonzero element
	slong kk;
	for (kk = 0; kk < mat->nrow; kk++) {
		auto row = rowperm[kk];
		auto therow = sparse_mat_row(mat, row);
		if (therow->nnz == 0)
			continue;
		else if (therow->nnz == 1) {
			auto col = therow->indices[0];
			pivots.push_back(std::make_pair(row, col));
			rowpivs[row] = col;
			colpivs[col] = row;
		}
		else
			break;
	}

	sparse_mat_transpose_part(tranmat, mat, rowperm);

	double oldstatus = 0;
	while (kk < mat->nrow) {
		auto start = clocknow();
		auto row = rowperm[kk];
		std::vector<std::pair<slong, slong>> n_pivots;

		if (mat->rows[row].nnz == 0) {
			kk++;
			continue;
		}

		pool.wait();
		std::vector<std::pair<slong, iter>> ps;
		ps = findmanypivots_r(mat, tranmat, colpivs,
			rowperm, rowperm.begin() + kk, opt->search_depth);

		for (auto& [c, rp] : ps) {
			pivots.push_back(std::make_pair(*rp, c));
			n_pivots.push_back(std::make_pair(*rp, c));
			colpivs[c] = *rp;
			rowpivs[*rp] = c;
			ulong scalar = nmod_inv(*sparse_mat_entry(mat, *rp, c), p);
			snmod_vec_rescale(mat->rows + *rp, scalar, p);
		}

		// reorder the rows, move ps to the front
		std::unordered_set<slong> indices(ps.size());
		for (size_t i = 0; i < ps.size(); i++)
			indices.insert(ps[i].second - rowperm.begin());
		std::vector<slong> result(rowperm.begin(), rowperm.begin() + kk);
		result.reserve(rowperm.size());
		for (auto ind : ps) {
			result.push_back(*ind.second);
		}
		for (ulong it = kk; it < mat->nrow; it++) {
			if (indices.find(it) == indices.end()) {
				result.push_back(rowperm[it]);
			}
		}
		rowperm = std::move(result);

		std::vector<slong> leftcols;
		for (size_t i = 0; i < mat->ncol; i++) {
			if (colpivs[i] == -1)
				leftcols.push_back(i);
		}

		kk += ps.size();
		slong newpiv = ps.size();
		
		ulong tran_count = 0;
		// flags[i] is true if the i-th row has been computed
		std::vector<uint8_t> flags(mat->nrow - kk, 0);
		// and then compute the elimination of the rows asynchronizely
		pool.detach_loop<slong>(kk, mat->nrow, [&](slong i) {
			if (rowpivs[rowperm[i]] != -1)
				return;
			auto id = BS::this_thread::get_index().value();
			schur_complete(mat, rowperm[i], n_pivots, leftcols, 1, p,
				cachedensedmat + id * mat->ncol);
			flags[i - kk] = 1;
			});
		std::vector<slong> leftrows(rowperm.begin() + kk, rowperm.end());
		for (size_t i = 0; i < tranmat->nrow; i++)
			tranmat->rows[i].nnz = 0;
		// compute the transpose of the submatrix and print the status asynchronizely
		bool print_once = true;
		while (tran_count < leftrows.size()) {
			for (size_t i = 0; i < leftrows.size(); i++) {
				if (flags[i]) {
					auto row = leftrows[i];
					auto therow = mat->rows + row;
					for (size_t j = 0; j < therow->nnz; j++) {
						auto col = therow->indices[j];
						_sparse_vec_set_entry(tranmat->rows + col, row, (bool*)NULL);
					}
					tran_count++;
					flags[i] = 0;
				}
			}
			auto status = (kk - newpiv + 1) + ((double)tran_count / (mat->nrow - kk)) * newpiv;
			if (verbose && (print_once || status - oldstatus > printstep)) {
				auto end = clocknow();
				now_nnz = sparse_mat_nnz(mat);
				std::cout << "\r-- Row: " << (int)std::floor(status) << "/" << mat->nrow
					<< "  rank: " << pivots.size()
					<< "  nnz: " << now_nnz << "  " << "density: "
					<< (double)100 * now_nnz / (mat->nrow * mat->ncol) << "%"
					<< "  speed: " << (status - oldstatus) / usedtime(start, end)
					<< " row/s" << std::flush;
				oldstatus = status;
				start = end;
				print_once = false;
			}
		}
	}

	free(cachedensedmat);

	if (verbose) {
		std::cout << "\n** Rank: " << pivots.size()
			<< " nnz: " << sparse_mat_nnz(mat) << std::endl
			<< "\n>> Reverse solving: " << std::endl;
	}

	// the matrix is upper triangular
	triangular_solver(mat, pivots, opt, -1, p, pool);

	if (verbose) {
		std::cout << std::endl;
	}

	sparse_mat_clear(tranmat);
	return pivots;
}


std::vector<std::pair<slong, slong>> snmod_mat_rref(snmod_mat_t mat, nmod_t p, BS::thread_pool& pool, rref_option_t opt) {
	if (opt->pivot_dir)
		return snmod_mat_rref_r(mat, p, pool, opt);
	else
		return snmod_mat_rref_c(mat, p, pool, opt);
}

ulong snmod_mat_rref_kernel(snmod_mat_t K, const snmod_mat_t M, const std::vector<std::pair<slong, slong>>& pivots, nmod_t p, BS::thread_pool& pool) {
	auto rank = pivots.size();
	if (rank == M->ncol)
		return 0; // full rank, no kernel

	ulong m1 = 1;

	if (rank == 0) {
		sparse_mat_init(K, M->ncol, M->ncol);
		for (size_t i = 0; i < M->ncol; i++)
			_sparse_vec_set_entry(sparse_mat_row(K, i), i, &m1);
		return M->ncol;
	}
	m1 = nmod_neg(m1, p);

	snmod_mat_t rows, trows;
	sparse_mat_init(rows, rank, M->ncol);
	sparse_mat_init(trows, M->ncol, rank);
	for (size_t i = 0; i < rank; i++) {
		sparse_vec_set(sparse_mat_row(rows, i), sparse_mat_row(M, pivots[i].first));
	}
	sparse_mat_transpose(trows, rows);
	sparse_mat_clear(rows);

	sparse_mat_init(K, M->ncol - rank, M->ncol);
	for (size_t i = 0; i < K->nrow; i++)
		sparse_mat_row(K, i)->nnz = 0;

	std::vector<slong> colpivs(M->ncol, -1);
	std::vector<slong> nonpivs;
	for (size_t i = 0; i < rank; i++)
		colpivs[pivots[i].second] = pivots[i].first;

	for (auto i = 0; i < M->ncol; i++)
		if (colpivs[i] == -1)
			nonpivs.push_back(i);

	pool.detach_loop<size_t>(0, nonpivs.size(), [&](size_t i) {
		auto thecol = sparse_mat_row(trows, nonpivs[i]);
		auto k_vec = sparse_mat_row(K, i);
		sparse_vec_realloc(k_vec, thecol->nnz + 1);
		for (size_t j = 0; j < thecol->nnz; j++) {
			_sparse_vec_set_entry(k_vec,
				pivots[thecol->indices[j]].second,
				thecol->entries+j);
		}
		_sparse_vec_set_entry(k_vec, nonpivs[i], &m1);
		sparse_vec_sort_indices(k_vec); // sort the indices
		});
	pool.wait();

	sparse_mat_clear(trows);
	return M->ncol - rank;
}