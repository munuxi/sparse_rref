#include "sparse_mat.h"

using iter = std::vector<slong>::iterator;

std::vector<std::pair<slong, slong>> sparse_mat_rref_c(sfmpq_mat_t mat, field_t F,
	BS::thread_pool& pool, rref_option_t opt) {
	// first canonicalize, sort and compress the matrix
	for (size_t i = 0; i < mat->nrow; i++) {
		sparse_vec_sort_indices(mat->rows + i);
		sparse_vec_canonicalize(mat->rows + i);
	}
	// sparse_mat_compress(mat);

	ulong init_nnz = sparse_mat_nnz(mat);
	ulong now_nnz = init_nnz;

	fmpq scalar[1];
	scalar_init(scalar);

	// store the pivots that have been used
	// -1 is not used
	std::vector<slong> rowpivs(mat->nrow, -1);
	std::vector<slong> colpivs(mat->ncol, -1);
	std::vector<std::pair<slong, slong>> pivots;
	// perm the col
	std::vector<slong> colperm(mat->ncol);
	for (size_t i = 0; i < mat->ncol; i++)
		colperm[i] = i;

	// look for row with only one non-zero entry

	// compute the transpose of pointers of the matrix
	sparse_mat_t<fmpq*> tranmatp;
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
			colpivs[colperm[kk]] = row;
			auto e = sparse_mat_entry(mat, row, rowpivs[row], true);
			scalar_inv(scalar, e, F);
			sparse_vec_rescale(mat->rows + row, scalar);
			pivots.push_back(std::make_pair(row, colperm[kk]));
		}
		else if (nnz > 1)
			break; // since it's sorted
	}
	sparse_mat_clear(tranmatp);

	fmpq* cachedensedmat = s_malloc<fmpq>(mat->ncol * pool.get_thread_count());
	for (size_t i = 0; i < mat->ncol * pool.get_thread_count(); i++) {
		scalar_init(cachedensedmat + i);
	}

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
			colpivs[*cp] = r;
			scalar_inv(scalar, sparse_mat_entry(mat, r, *cp), F);
			sparse_vec_rescale(mat->rows + r, scalar);
			n_pivots.push_back(std::make_pair(r, *cp));
			pivots.push_back(std::make_pair(r, *cp));
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

		std::vector<uint8_t> flags(leftrows.size(), 0);
		bool mode = ((double)100 * now_nnz / (mat->nrow * mat->ncol) < SPARSE_BOUND);
		pool.detach_loop<slong>(0, leftrows.size(), [&](slong i) {
			auto id = BS::this_thread::get_index().value();
			schur_complete(mat, leftrows[i], n_pivots, 1, F, cachedensedmat + id * mat->ncol, mode);
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
	triangular_solver(mat, pivots, F, opt, -1, pool);

	if (verbose) {
		std::cout << '\n' << std::endl;
	}

	sparse_mat_clear(tranmat);

	for (size_t i = 0; i < mat->ncol * pool.get_thread_count(); i++) {
		scalar_clear(cachedensedmat + i);
	}
	s_free(cachedensedmat);

	return pivots;
}

auto sparse_mat_rref_r(sfmpq_mat_t mat, field_t F, BS::thread_pool& pool, rref_option_t opt) {
	// first canonicalize, sort and compress the matrix
	for (size_t i = 0; i < mat->nrow; i++) {
		sparse_vec_sort_indices(mat->rows + i);
		sparse_vec_canonicalize(mat->rows + i);
	}

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

	sparse_mat_t<fmpq*> tranmatp;
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

	fmpq* cachedensedmat = s_malloc<fmpq>(mat->ncol * pool.get_thread_count());
	for (size_t i = 0; i < mat->ncol * pool.get_thread_count(); i++)
		scalar_init(cachedensedmat + i);

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

		if (mat->rows[row].nnz == 0) {
			kk++;
			continue;
		}

		pool.wait();
		auto ps = findmanypivots_r(mat, tranmat, colpivs,
			rowperm, rowperm.begin() + kk, opt->search_depth);

		if (ps.size() == 0)
			break;

		std::vector<std::pair<slong, slong>> n_pivots;

		fmpq_t scalar;
		scalar_init(scalar);

		for (auto& [c, rp] : ps) {
			pivots.push_back(std::make_pair(*rp, c));
			n_pivots.push_back(std::make_pair(*rp, c));
			colpivs[c] = *rp;
			rowpivs[*rp] = c;
			scalar_inv(scalar, sparse_mat_entry(mat, *rp, c, true), F);
			sparse_vec_rescale(mat->rows + *rp, scalar);
		}

		scalar_clear(scalar);

		// reorder the rows, move ps to the front
		std::unordered_set<slong> indices(ps.size());
		for (size_t i = 0; i < ps.size(); i++)
			indices.insert(ps[i].second - rowperm.begin());
		std::vector<slong> result(rowperm.begin(), rowperm.begin() + kk);
		result.reserve(rowperm.size());
		for (auto ind : ps) {
			result.push_back(*ind.second);
		}
		for (auto it = kk; it < mat->nrow; it++) {
			if (indices.find(it) == indices.end()) {
				result.push_back(rowperm[it]);
			}
		}
		rowperm = std::move(result);

		kk += ps.size();
		slong newpiv = ps.size();

		ulong tran_count = 0;
		// flags[i] is true if the i-th row has been computed
		std::vector<uint8_t> flags(mat->nrow - kk, 0);
		// and then compute the elimination of the rows asynchronizely
		bool mode = ((double)100 * now_nnz / (mat->nrow * mat->ncol) < SPARSE_BOUND);
		pool.detach_loop<slong>(kk, mat->nrow, [&](slong i) {
			if (rowpivs[rowperm[i]] != -1)
				return;
			auto id = BS::this_thread::get_index().value();
			schur_complete(mat, rowperm[i], n_pivots, 1, F, cachedensedmat + id * mat->ncol, mode);
			flags[i - kk] = 1;
			});
		std::vector<slong> leftrows(rowperm.begin() + kk, rowperm.end());
		for (size_t i = 0; i < tranmat->nrow; i++)
			tranmat->rows[i].nnz = 0;
		// compute the transpose of the submatrix and print the status asynchronizely
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
			if (verbose && status - oldstatus > printstep) {
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
			}
		}
	}

	for (size_t i = 0; i < mat->ncol * pool.get_thread_count(); i++)
		scalar_clear(cachedensedmat + i);
	s_free(cachedensedmat);

	if (verbose) {
		std::cout << "\n** Rank: " << pivots.size()
			<< " nnz: " << sparse_mat_nnz(mat) << std::endl
			<< "\n>> Reverse solving: " << std::endl;
	}

	// the matrix is upper triangular
	triangular_solver(mat, pivots, F, opt, -1, pool);

	if (verbose) {
		std::cout << std::endl;
	}

	sparse_mat_clear(tranmat);

	return pivots;
}

std::vector<std::pair<slong, slong>> sparse_mat_rref(sfmpq_mat_t mat, field_t F, BS::thread_pool& pool, rref_option_t opt) {
	if (opt->pivot_dir)
		return sparse_mat_rref_r(mat, F, pool, opt);
	else
		return sparse_mat_rref_c(mat, F, pool, opt);
}