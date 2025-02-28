/*
	Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

	This file is part of Sparse_rref. The Sparse_rref is free software: 
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/


#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include "sparse_vec.h"

namespace sparse_rref {
	typedef std::pair<slong, slong> pivot_t;

	// new sparse matrix
	template <typename T> struct sparse_mat {
		ulong nrow;
		ulong ncol;
		std::vector<sparse_vec<T>> rows;

		void init(ulong r, ulong c) {
			nrow = r;
			ncol = c;
			rows = std::vector<sparse_vec<T>>(r);
		}

		sparse_mat() { nrow = 0; ncol = 0; }
		~sparse_mat() {}
		sparse_mat(ulong r, ulong c) { init(r, c); }

		sparse_vec<T>& operator[](ulong i) { return rows[i]; }
		const sparse_vec<T>& operator[](ulong i) const { return rows[i]; }

		sparse_mat(const sparse_mat& l) {
			init(l.nrow, l.ncol);
			for (size_t i = 0; i < nrow; i++)
				rows[i] = l.rows[i];
		}

		sparse_mat(sparse_mat&& l) noexcept {
			nrow = l.nrow;
			ncol = l.ncol;
			rows = l.rows;
		}

		void realloc(ulong r) {
			rows.reverse(r);
		}

		sparse_mat& operator=(const sparse_mat& l) {
			if (this == &l)
				return *this;
			nrow = l.nrow;
			ncol = l.ncol;
			rows = l.rows;
			return *this;
		}

		sparse_mat& operator=(sparse_mat&& l) noexcept {
			if (this == &l)
				return *this;
			nrow = l.nrow;
			ncol = l.ncol;
			rows = l.rows;
			return *this;
		}

		void zero() {
			for (size_t i = 0; i < nrow; i++)
				rows[i].zero();
		}

		ulong nnz() {
			ulong n = 0;
			for (size_t i = 0; i < nrow; i++)
				n += rows[i].nnz();
			return n;
		}

		void compress() {
			for (size_t i = 0; i < nrow; i++) {
				rows[i].compress();
				rows[i].sort_indices();
				rows[i].reserve(rows[i].nnz());
			}
		}

		void clear_zero_row() {
			ulong new_nrow = 0;
			for (size_t i = 0; i < nrow; i++) {
				if (rows[i].nnz() != 0) {
					std::swap(rows[new_nrow], rows[i]);
					new_nrow++;
				}
			}
			nrow = new_nrow;
			rows.resize(nrow);
		}

		T* entry(ulong r, ulong c, bool isbinary = true) {
			return sparse_vec_entry(rows[r], c, isbinary);
		}

		sparse_mat<T> transpose() {
			sparse_mat<T> res(ncol, nrow);
			for (size_t i = 0; i < nrow; i++)
				res[i].zero();

			for (size_t i = 0; i < nrow; i++) {
				for (size_t j = 0; j < rows[i].nnz(); j++) {
					res[rows[i](j)].push_back(i, rows[i][j]);
				}
			}
			return std::move(res);
		}
	};

	typedef sparse_mat<ulong> snmod_mat;
	typedef sparse_mat<rat_t> sfmpq_mat;

	template <typename T>
	inline std::vector<T>::iterator sparse_mat_entry(sparse_mat<T>& mat, ulong r, ulong c, bool isbinary = true) {
		return sparse_vec_entry(mat[r], c, isbinary);
	}

	template <typename T>
	inline ulong sparse_mat_nnz(sparse_mat<T>& mat) {
		return mat.nnz();
	}

	template <typename T>
	inline void sparse_mat_compress(sparse_mat<T>& mat) {
		mat.compress();
	}

	template <typename T, typename S>
	void sparse_mat_transpose_part_replace(sparse_mat<S>& tranmat, const sparse_mat<T>& mat, const std::vector<slong>& rows) {
		tranmat.zero();

		for (size_t i = 0; i < rows.size(); i++) {
			for (size_t j = 0; j < mat[rows[i]].nnz(); j++) {
				auto col = mat[rows[i]](j);
				T ptr = mat[rows[i]][j];
				if constexpr (std::is_same_v<S, T>) {
					tranmat[col].push_back(rows[i], ptr);
				}
				else if constexpr (std::is_same_v<S, T*>) {
					tranmat[col].push_back(rows[i], &ptr);
				}
				else {
					tranmat[col].push_back(rows[i], true);
				}
			}
		}
	}


	template <typename T, typename S>
	void sparse_mat_transpose_replace(sparse_mat<S>& tranmat, const sparse_mat<T>& mat) {
		std::vector<slong> rows(mat.nrow);
		for (size_t i = 0; i < mat.nrow; i++)
			rows[i] = i;
		sparse_mat_transpose_part_replace(tranmat, mat, rows);
	}


	// rref staffs

	// first look for rows with only one nonzero value and eliminate them
	// we assume that mat is canonical, i.e. each index is sorted
	// and the result is also canonical
	template <typename T>
	ulong eliminate_row_with_one_nnz(sparse_mat<T>& mat,
		sparse_mat<T*>& tranmat, std::vector<slong>& donelist, bool is_tran = false) {
		auto localcounter = 0;
		std::vector<slong> pivlist(mat.nrow, -1);
		std::vector<slong> collist(mat.ncol, -1);
		for (size_t i = 0; i < mat.nrow; i++) {
			if (donelist[i] != -1)
				continue;
			if (mat[i].nnz() == 1) {
				if (collist[mat[i](0)] == -1) {
					localcounter++;
					pivlist[i] = mat[i](0);
					collist[mat[i](0)] = i;
				}
			}
		}

		if (localcounter == 0)
			return localcounter;

		if (!is_tran)
			sparse_mat_transpose_replace(tranmat, mat);
		for (size_t i = 0; i < mat.nrow; i++) {
			if (pivlist[i] == -1)
				continue;
			auto& thecol = tranmat[pivlist[i]];
			for (size_t j = 0; j < thecol.nnz(); j++) {
				if (thecol(j) == i) {
					*(thecol[j]) = 1;
				}
				else
					*(thecol[j]) = 0;
			}
		}

		for (size_t i = 0; i < mat.nrow; i++)
			mat[i].canonicalize();

		for (size_t i = 0; i < mat.nrow; i++)
			if (pivlist[i] != -1)
				donelist[i] = pivlist[i];

		return localcounter;
	}

	template <typename T>
	ulong eliminate_row_with_one_nnz_rec(sparse_mat<T>& mat,
		sparse_mat<T*>& tranmat,
		std::vector<slong>& donelist, rref_option_t opt,
		slong max_depth = INT_MAX) {
		slong depth = 0;
		ulong localcounter = 0;
		ulong count = 0;
		bool verbose = opt->verbose;
		bool dir = true;

		std::string dirstr = (dir) ? "Col" : "Row";
		ulong ndir = (dir) ? mat.ncol : mat.nrow;

		ulong oldnnz = mat.nnz();
		int bitlen_nnz = (int)std::floor(std::log(oldnnz) / std::log(10)) + 3;
		int bitlen_ndir = (int)std::floor(std::log(ndir) / std::log(10)) + 1;

		do {
			localcounter = eliminate_row_with_one_nnz(mat, tranmat, donelist);
			if (verbose) {
				oldnnz = mat.nnz();
				std::cout << "-- " << dirstr << ": " << std::setw(bitlen_ndir)
					<< localcounter << "/" << ndir
					<< "  rank: " << std::setw(bitlen_ndir) << count
					<< "  nnz: " << std::setw(bitlen_nnz) << oldnnz
					<< "  density: " << std::setprecision(6) << std::setw(8)
					<< 100 * (double)oldnnz / (mat.nrow * mat.ncol) << "%"
					<< "    \r" << std::flush;
			}
			count += localcounter;
			depth++;
		} while (localcounter > 0 && depth < max_depth);
		return count;
	}

	// TODO: add a DFS algorithm to find a maximal compatible set
	// //     |    a2 b2 |    |    a2 b2 |    |    a2 b2|    |    a2 b2 |   
	// //  11 | a1  *  * | 10 | a1  *  0 | 01 | a1  *  *| 00 | a1  *  0 |   
	// //     | b1  *  * |    | b1  *  * |    | b1  0  *|    | b1  0  * |   
	// //   3               2               1              0 
	// 
	// int compatible_degree(std::unordered_set<std::pair<slong, slong>>& adjmat,
	// 	std::pair<slong, slong>& a, std::pair<slong, slong>& b) {
	// 	auto [a1,a2] = a;
	// 	auto [b1,b2] = b;
	// 	if (a1 == b1 || a2 == b2) // same row or same col
	// 		return 3;
	// 	bool test1 = adjmat.find(std::make_pair(a1, b2)) == adjmat.end();
	// 	bool test2 = adjmat.find(std::make_pair(b1, a2)) == adjmat.end();
	// 	if (test1 && test2)
	// 		return 0;
	// 	if (test1)
	// 		return 2;
	// 	if (test2)
	// 		return 1;
	// 	return 3;
	// }

	using iter = std::vector<slong>::iterator;

	template <typename T, typename S>
	std::vector<std::pair<slong, iter>> findmanypivots(const sparse_mat<T>& mat, const sparse_mat<S>& tranmat,
		std::vector<slong>& rdivpivs, std::vector<slong>& dirperm, bool mat_dir, size_t max_depth = ULLONG_MAX) {

		if (!mat_dir)
			return findmanypivots(tranmat, mat, rdivpivs, dirperm, true, max_depth);

		auto start = dirperm.begin();
		auto end = dirperm.end();

		auto ndir = mat.nrow;
		auto nrdir = tranmat.nrow;

		std::list<std::pair<slong, iter>> pivots;
		std::unordered_set<slong> pdirs;
		pdirs.reserve(std::min((size_t)4096, max_depth));

		// rightlook first
		for (auto dir = start; dir < end; dir++) {
			if ((ulong)(dir - start) > max_depth)
				break;

			auto& thedir = mat[*dir];
			if (thedir.nnz() == 0)
				continue;

			slong rdiv;
			ulong mnnz = ULLONG_MAX;
			bool flag = true;

			for (size_t i = 0; i < thedir.nnz(); i++) {
				flag = (pdirs.count(thedir(i)) == 0);
				if (!flag)
					break;
				if (rdivpivs[thedir(i)] != -1)
					continue;
				ulong newnnz = tranmat[thedir(i)].nnz();
				if (newnnz < mnnz) {
					rdiv = thedir(i);
					mnnz = newnnz;
				}
				// make the result stable
				else if (newnnz == mnnz && thedir(i) < rdiv) {
					rdiv = thedir(i);
				}
			}
			if (!flag)
				continue;
			if (mnnz != ULLONG_MAX) {
				pivots.push_back(std::make_pair(rdiv, dir));
				pdirs.insert(rdiv);
			}
		}

		// leftlook then
		pdirs.clear();
		// make a table to help to look for dir pointers
		std::vector<iter> dirptrs(ndir, end);
		for (auto it = start; it != end; it++)
			dirptrs[*it] = it;

		for (auto p : pivots)
			pdirs.insert(*(p.second));

		for (size_t i = 0; i < nrdir; i++) {
			if (pivots.size() > max_depth)
				break;
			auto rdir = i;
			// auto rdir = nrdir - i - 1; // reverse ordering
			if (rdivpivs[rdir] != -1)
				continue;

			slong dir = 0;
			ulong mnnz = ULLONG_MAX;
			bool flag = true;

			auto& tc = tranmat[rdir];

			for (size_t j = 0; j < tc.nnz(); j++) {
				if (dirptrs[tc(j)] == end)
					continue;
				flag = (pdirs.count(tc(j)) == 0);
				if (!flag)
					break;
				if (mat[tc(j)].nnz() < mnnz) {
					mnnz = mat[tc(j)].nnz();
					dir = tc(j);
				}
				// make the result stable
				else if (mat[tc(j)].nnz() == mnnz && tc(j) < dir) {
					dir = tc(j);
				}
			}
			if (!flag)
				continue;
			if (mnnz != ULLONG_MAX) {
				pivots.push_front(std::make_pair(rdir, dirptrs[dir]));
				pdirs.insert(dir);
			}
		}

		std::vector<std::pair<slong, iter>> result(pivots.begin(), pivots.end());
		return result;
	}

	// upper solver : ordering = -1
	// lower solver : ordering = 1
	template <typename T>
	void triangular_solver(sparse_mat<T>& mat, std::vector<pivot_t>& pivots,
		field_t F, rref_option_t opt, int ordering, sparse_rref::thread_pool& pool) {
		bool verbose = opt->verbose;
		auto printstep = opt->print_step;

		std::vector<std::vector<slong>> tranmat(mat.ncol);

		// we only need to compute the transpose of the submatrix involving pivots

		for (size_t i = 0; i < pivots.size(); i++) {
			auto& therow = mat[pivots[i].first];
			for (size_t j = 0; j < therow.nnz(); j++) {
				if (therow[j] == 0)
					continue;
				auto col = therow(j);
				tranmat[col].push_back(pivots[i].first);
			}
		}

		size_t count = 0;
		size_t nthreads = pool.get_thread_count();
		for (size_t i = 0; i < pivots.size(); i++) {
			size_t index = i;
			if (ordering < 0)
				index = pivots.size() - 1 - i;
			auto pp = pivots[index];
			auto& thecol = tranmat[pp.second];
			auto start = sparse_rref::clocknow();
			if (thecol.size() > 1) {
				pool.detach_loop<slong>(0, thecol.size(), [&](slong j) {
					auto r = thecol[j];
					if (r == pp.first)
						return;
					auto entry = *sparse_mat_entry(mat, r, pp.second);
					sparse_vec_sub_mul(mat[r], mat[pp.first], entry, F);
					},
					((thecol.size() < 20 * nthreads) ? 0 : thecol.size() / 10));
			}
			pool.wait();

			if (verbose && (i % printstep == 0 || i == pivots.size() - 1) && thecol.size() > 1) {
				count++;
				auto end = sparse_rref::clocknow();
				auto now_nnz = mat.nnz();
				std::cout << "\r-- Row: " << (i + 1) << "/" << pivots.size()
					<< "  " << "row to eliminate: " << thecol.size() - 1
					<< "  " << "nnz: " << now_nnz << "  " << "density: "
					<< (double)100 * now_nnz / (mat.nrow * mat.ncol)
					<< "%  " << "speed: " << count / sparse_rref::usedtime(start, end)
					<< " row/s" << std::flush;
				start = sparse_rref::clocknow();
				count = 0;
			}
		}
		if (opt->verbose)
			std::cout << std::endl;
	}

	template <typename T>
	void triangular_solver(sparse_mat<T>& mat, std::vector<std::vector<pivot_t>>& pivots,
		field_t F, rref_option_t opt, int ordering, sparse_rref::thread_pool& pool) {
		std::vector<pivot_t> n_pivots;
		for (auto p : pivots)
			n_pivots.insert(n_pivots.end(), p.begin(), p.end());
		triangular_solver(mat, n_pivots, F, opt, ordering, pool);
	}

	// dot product
	template <typename T>
	inline int sparse_mat_dot_sparse_vec(sparse_vec<T> result, const sparse_mat<T>& mat, const sparse_vec<T> vec, field_t F) {
		result.zero();
		if (vec.nnz() == 0 || mat.nnz() == 0)
			return 0;

		for (size_t i = 0; i < mat.nrow; i++) {
			T tmp = sparse_vec_dot(mat[i], vec, F);
			if (tmp != 0)
				result.push_back(i, tmp);
		}
		return 1;
	}

	// A = B * C
	template <typename T>
	inline sparse_mat<T> sparse_mat_dot_sparse_mat(const sparse_mat<T>& B, const sparse_mat<T>& C, field_t F) {
		sparse_mat<T> A;

		sparse_mat<T> Ct(C.ncol, C.nrow);
		sparse_mat_transpose_replace(Ct, C);

		for (size_t i = 0; i < B.nrow; i++)
			sparse_mat_dot_sparse_vec(A[i], B, Ct[i], F);

		return 0;
	}

	template <typename T>
	size_t apart_pivots(sparse_mat<T>& mat, std::vector<pivot_t>& pivots, size_t index) {
		auto [sr, sc] = pivots[index];
		std::unordered_set<slong> colset;
		colset.reserve(mat.ncol);
		colset.insert(sc);
		size_t i = index + 1;
		for (; i < pivots.size(); i++) {
			auto [r, c] = pivots[i];
			bool flag = true;
			for (auto j = 0; flag && (j < mat[r].nnz()); j++) {
				flag = (colset.count(mat[r](j)) == 0);
			}
			if (!flag)
				break;
			colset.insert(c);
		}
		return i;
	}

	// SLOW!!!
	template <typename T>
	std::pair<std::vector<pivot_t>, std::vector<pivot_t>> apart_pivots_2(sparse_mat<T>& mat, std::vector<pivot_t>& pivots) {
		if (pivots.size() == 0)
			return std::make_pair(std::vector<pivot_t>(), std::vector<pivot_t>());
		auto [sr, sc] = pivots[0];
		std::unordered_set<slong> colset;
		colset.reserve(mat.ncol);
		colset.insert(sc);
		std::vector<pivot_t> n_pivots;
		std::vector<pivot_t> left_pivots;
		n_pivots.push_back(pivots[0]);
		size_t i = 1;
		for (; i < pivots.size(); i++) {
			auto [r, c] = pivots[i];
			bool flag = true;
			for (auto j = 0; flag && (j < mat[r].nnz()); j++) {
				flag = (colset.count(mat[r](j)) == 0);
			}
			if (!flag) {
				left_pivots.push_back(pivots[i]);
				continue;
			}
			colset.insert(c);
			n_pivots.push_back(pivots[i]);
		}
		return std::make_pair(n_pivots, left_pivots);
	}

	// first write a stupid one
	template <typename T>
	void schur_complete(sparse_mat<T>& mat, slong k, std::vector<pivot_t>& pivots,
		field_t F, T* tmpvec, sparse_rref::uset& nonzero_c) {

		if (mat[k].nnz() == 0)
			return;

		// sparse_rref::uset nonzero_c(mat.ncol);
		nonzero_c.clear();

		for (size_t i = 0; i < mat[k].nnz(); i++) {
			nonzero_c.insert(mat[k](i));
			tmpvec[mat[k](i)] = mat[k][i];
		}
		ulong e_pr;
		for (auto [r, c] : pivots) {
			if (!nonzero_c.count(c))
				continue;
			T entry = tmpvec[c];
			auto& row = mat[r];
			if constexpr (std::is_same_v<T, ulong>) {
				e_pr = n_mulmod_precomp_shoup(tmpvec[c], F->mod.n);
			}
			for (size_t i = 0; i < row.nnz(); i++) {
				if (!nonzero_c.count(row(i))) {
					nonzero_c.insert(row(i));
					tmpvec[row(i)] = 0;
				}
				if constexpr (std::is_same_v<T, ulong>) {
					tmpvec[row(i)] = _nmod_sub(tmpvec[row(i)],
						n_mulmod_shoup(entry, row[i], e_pr, F->mod.n), F->mod);
				}
				else if constexpr (std::is_same_v<T, rat_t>) {
					tmpvec[row(i)] -= entry * (row[i]);
				}
				if (tmpvec[row(i)] == 0)
					nonzero_c.erase(row(i));
			}
		}

		mat[k].zero();
		auto pos = nonzero_c.nonzero();
		for (auto p : pos) {
			mat[k].push_back(p, tmpvec[p]);
		}
	}

	// TODO: CHECK!!!
	// SLOW!!!
	template <typename T>
	void triangular_solver_2_rec(sparse_mat<T>& mat, std::vector<std::vector<slong>>& tranmat, std::vector<pivot_t>& pivots,
		field_t F, rref_option_t opt, sparse_rref::thread_pool& pool, T* cachedensedmat,
		std::vector<sparse_rref::uset>& nonzero_c, size_t n_split, size_t rank, size_t& process) {

		bool verbose = opt->verbose;
		opt->verbose = false;
		if (pivots.size() < n_split) {
			triangular_solver(mat, pivots, F, opt, -1, pool);
			opt->verbose = verbose;
			process += pivots.size();
			return;
		}

		std::vector<pivot_t> sub_pivots(pivots.end() - n_split, pivots.end());
		std::vector<pivot_t> left_pivots(pivots.begin(), pivots.end() - n_split);

		std::unordered_set<slong> pre_leftrows;
		for (auto [r, c] : sub_pivots)
			pre_leftrows.insert(tranmat[c].begin(), tranmat[c].end());
		for (auto [r, c] : sub_pivots)
			pre_leftrows.erase(r);
		std::vector<slong> leftrows(pre_leftrows.begin(), pre_leftrows.end());

		// for printing
		ulong now_nnz = mat.nnz();
		int bitlen_nnz = (int)std::floor(std::log(now_nnz) / std::log(10)) + 3;
		int bitlen_nrow = (int)std::floor(std::log(rank) / std::log(10)) + 1;

		auto clock_begin = sparse_rref::clocknow();
		std::atomic<size_t> cc = 0;
		pool.detach_blocks<ulong>(0, leftrows.size(), [&](const ulong s, const ulong e) {
			auto id = sparse_rref::thread_id();
			for (size_t i = s; i < e; i++) {
				schur_complete(mat, leftrows[i], sub_pivots, F, cachedensedmat + id * mat.ncol, nonzero_c[id]);
				cc++;
			}
			}, ((n_split < 20 * pool.get_thread_count()) ? 0 : leftrows.size() / 10));

		if (verbose) {
			ulong old_cc = cc;
			while (cc < leftrows.size()) {
				// stop for a while
				std::this_thread::sleep_for(std::chrono::microseconds(1000));
				now_nnz = mat.nnz();
				size_t status = (size_t)std::floor(1.0 * sub_pivots.size() * cc / leftrows.size());
				std::cout << "-- Row: " << std::setw(bitlen_nrow)
					<< process + status << "/" << rank
					<< "  nnz: " << std::setw(bitlen_nnz) << now_nnz
					<< "  density: " << std::setprecision(6) << std::setw(8)
					<< 100 * (double)now_nnz / (rank * mat.ncol) << "%"
					<< "  speed: " << std::setprecision(2) << std::setw(6)
					<< 1.0 * sub_pivots.size() * (cc - old_cc) / leftrows.size() / sparse_rref::usedtime(clock_begin, sparse_rref::clocknow())
					<< " row/s    \r" << std::flush;
				clock_begin = sparse_rref::clocknow();
				old_cc = cc;
			}
		}

		pool.wait();

		triangular_solver(mat, sub_pivots, F, opt, -1, pool);
		opt->verbose = verbose;
		process += sub_pivots.size();

		triangular_solver_2_rec(mat, tranmat, left_pivots, F, opt, pool, cachedensedmat, nonzero_c, n_split, rank, process);
	}

	template <typename T>
	void triangular_solver_2(sparse_mat<T>& mat, std::vector<pivot_t>& pivots,
		field_t F, rref_option_t opt, sparse_rref::thread_pool& pool) {

		// prepare the tmp array
		auto nthreads = pool.get_thread_count();
		std::vector<T> cachedensedmat(mat.ncol * nthreads);
		std::vector<sparse_rref::uset> nonzero_c(nthreads);
		for (size_t i = 0; i < nthreads; i++)
			nonzero_c[i].resize(mat.ncol);

		// we only need to compute the transpose of the submatrix involving pivots
		std::vector<std::vector<slong>> tranmat(mat.ncol);
		for (size_t i = 0; i < pivots.size(); i++) {
			auto& therow = mat[pivots[i].first];
			for (size_t j = 0; j < therow.nnz(); j++) {
				if (therow[j] == 0)
					continue;
				tranmat[therow(j)].push_back(pivots[i].first);
			}
		}

		size_t process = 0;
		// TODO: better split strategy
		size_t n_split = std::max(pivots.size() / 128ULL, 1024ULL);
		size_t rank = pivots.size();
		triangular_solver_2_rec(mat, tranmat, pivots, F, opt, pool, cachedensedmat.data(), nonzero_c, n_split, rank, process);

		if (opt->verbose)
			std::cout << std::endl;
	}

	template <typename T>
	void triangular_solver_2(sparse_mat<T>& mat, std::vector<std::vector<pivot_t>>& pivots,
		field_t F, rref_option_t opt, sparse_rref::thread_pool& pool) {

		std::vector<pivot_t> n_pivots;
		for (auto p : pivots)
			n_pivots.insert(n_pivots.end(), p.begin(), p.end());

		triangular_solver_2(mat, n_pivots, F, opt, pool);
	}

	// TODO: TEST!!! 
	// TODO: add ordering
	// if already know the pivots, we can directly do the rref
	template <typename T>
	void sparse_mat_direct_rref(sparse_mat<T>& mat,
		std::vector<std::vector<pivot_t>>& pivots,
		field_t F, sparse_rref::thread_pool& pool, rref_option_t opt) {
		T scalar;

		// first set rows not in pivots to zero
		std::vector<slong> rowset(mat.nrow, -1);
		for (auto p : pivots)
			for (auto [r, c] : p)
				rowset[r] = c;
		for (size_t i = 0; i < mat.nrow; i++)
			if (rowset[i] == -1)
				mat[i].zero();

		mat.compress();

		sparse_mat<T*> tranmatp(mat.ncol, mat.nrow);
		std::vector<slong> tmplist(mat.nrow, -1);
		eliminate_row_with_one_nnz_rec(mat, tranmatp, tmplist, opt);

		//auto n_pivots = pivots[0];
		//for (auto [r, c] : n_pivots) {
		//	auto therow = mat[r];
		//	therow.nnz() = 1;
		//	therow->indices[0] = c;
		//	scalar_one(therow->entries);
		//}

		// then do the elimination parallelly
		auto nthreads = pool.get_thread_count();
		std::vector<T> cachedensedmat(mat.ncol * nthreads);
		std::vector<sparse_rref::uset> nonzero_c(nthreads);
		for (size_t i = 0; i < nthreads; i++)
			nonzero_c[i].resize(mat.ncol);

		for (auto i = 0; i < pivots.size(); i++) {
			auto n_pivots = pivots[i];
			if (n_pivots.size() == 0)
				continue;

			// rescale the pivots
			for (auto [r, c] : n_pivots) {
				scalar_inv(scalar, *sparse_mat_entry(mat, r, c), F);
				sparse_vec_rescale(mat[r], scalar, F);
				rowset[r] = -1;
			}

			// the first is done by eliminate_row_with_one_nnz_rec
			if (i == 0)
				continue;

			std::vector<slong> leftrows;
			for (size_t j = 0; j < mat.nrow; j++) {
				if (rowset[j] != -1)
					leftrows.push_back(j);
			}

			// upper solver
			// TODO: check mode
			pool.detach_blocks<ulong>(0, leftrows.size(), [&](const ulong s, const ulong e) {
				auto id = sparse_rref::thread_id();
				for (ulong j = s; j < e; j++) {
					schur_complete(mat, leftrows[j], n_pivots, F, cachedensedmat.data() + id * mat.ncol, nonzero_c[id]);
				}
				}, ((leftrows.size() < 20 * nthreads) ? 0 : leftrows.size() / 10));
			pool.wait();
		}
	}

	template <typename T>
	std::vector<std::vector<pivot_t>> sparse_mat_rref_c(sparse_mat<T>& mat, field_t F,
		sparse_rref::thread_pool& pool, rref_option_t opt) {
		// first canonicalize, sort and compress the matrix
		mat.compress();

		T scalar;

		// perm the col
		std::vector<slong> leftcols(mat.ncol);
		for (size_t i = 0; i < mat.ncol; i++)
			leftcols[i] = i;

		auto printstep = opt->print_step;
		bool verbose = opt->verbose;

		ulong now_nnz = mat.nnz();

		// store the pivots that have been used
		// -1 is not used
		std::vector<slong> rowpivs(mat.nrow, -1);
		std::vector<std::vector<pivot_t>> pivots;

		// look for row with only one non-zero entry

		// compute the transpose of pointers of the matrix
		sparse_mat<T*> tranmatp(mat.ncol, mat.nrow);
		ulong count =
			eliminate_row_with_one_nnz_rec(mat, tranmatp, rowpivs, opt);
		now_nnz = mat.nnz();

		sparse_mat_transpose_replace(tranmatp, mat);

		// sort pivots by nnz, it will be faster
		std::stable_sort(leftcols.begin(), leftcols.end(),
			[&tranmatp](slong a, slong b) {
				return tranmatp[a].nnz() < tranmatp[b].nnz();
			});

		// look for pivot cols with only one nonzero element
		ulong kk = 0;
		std::fill(rowpivs.begin(), rowpivs.end(), -1);
		std::vector<pivot_t> n_pivots;
		for (; kk < mat.ncol; kk++) {
			auto nnz = tranmatp[leftcols[kk]].nnz();
			if (nnz == 0)
				continue;
			if (nnz == 1) {
				auto row = tranmatp[leftcols[kk]](0);
				if (rowpivs[row] != -1)
					continue;
				rowpivs[row] = leftcols[kk];
				scalar_inv(scalar, *sparse_mat_entry(mat, row, rowpivs[row]), F);
				sparse_vec_rescale(mat[row], scalar, F);
				n_pivots.push_back(std::make_pair(row, leftcols[kk]));
			}
			else if (nnz > 1)
				break; // since it's sorted
		}
		leftcols.erase(leftcols.begin(), leftcols.begin() + kk);
		pivots.push_back(std::move(n_pivots));
		auto rank = pivots[0].size();

		auto nthreads = pool.get_thread_count();
		std::vector<T> cachedensedmat(mat.ncol * nthreads);
		std::vector<sparse_rref::uset> nonzero_c(nthreads);
		for (size_t i = 0; i < nthreads; i++)
			nonzero_c[i].resize(mat.ncol);

		sparse_mat<bool> tranmat(mat.ncol, mat.nrow);
		sparse_mat_transpose_replace(tranmat, mat);

		std::vector<slong> leftrows;
		leftrows.reserve(mat.nrow);
		for (size_t i = 0; i < mat.nrow; i++) {
			if (rowpivs[i] != -1 || mat.rows[i].nnz() == 0)
				continue;
			leftrows.push_back(i);
		}

		// for printing
		double oldpr = 0;
		int bitlen_nnz = (int)std::floor(std::log(now_nnz) / std::log(10)) + 3;
		int bitlen_ncol = (int)std::floor(std::log(mat.ncol) / std::log(10)) + 1;

		while (kk < mat.ncol) {
			auto start = sparse_rref::clocknow();

			auto ps = findmanypivots(mat, tranmat, rowpivs, leftcols, false);
			if (ps.size() == 0)
				break;

			n_pivots.clear();
			for (auto i = ps.rbegin(); i != ps.rend(); i++) {
				auto [r, cp] = *i;
				rowpivs[r] = *cp;
				n_pivots.push_back(std::make_pair(r, *cp));
				scalar_inv(scalar, *sparse_mat_entry(mat, r, *cp), F);
				sparse_vec_rescale(mat[r], scalar, F);
			}
			pivots.push_back(n_pivots);
			rank += n_pivots.size();

			ulong n_leftrows = 0;
			for (size_t i = 0; i < leftrows.size(); i++) {
				auto row = leftrows[i];
				if (rowpivs[row] != -1 || mat.rows[row].nnz() == 0)
					continue;
				leftrows[n_leftrows] = row;
				n_leftrows++;
			}
			leftrows.resize(n_leftrows);

			std::vector<int> flags(leftrows.size(), 0);
			pool.detach_blocks<ulong>(0, leftrows.size(), [&](const ulong s, const ulong e) {
				auto id = sparse_rref::thread_id();
				for (ulong i = s; i < e; i++) {
					schur_complete(mat, leftrows[i], n_pivots, F, cachedensedmat.data() + id * mat.ncol, nonzero_c[id]);
					flags[i] = 1;
				}
				}, (leftrows.size() < 20 * nthreads ? 0 : leftrows.size() / 10));

			// reorder the cols, move ps to the front
			std::unordered_set<slong> indices(mat.ncol);
			for (auto [r, c] : ps)
				indices.insert(*c);
			std::vector<slong> result;
			result.reserve(leftcols.size());
			for (auto it : leftcols) {
				if (indices.count(it) == 0)
					result.push_back(it);
			}
			leftcols = std::move(result);
			std::vector<slong> donelist(rowpivs);

			bool print_once = true; // print at least once
			// we need first set the transpose matrix zero
			tranmat.zero();

			ulong localcount = 0;
			while (localcount < leftrows.size()) {
				for (size_t i = 0; i < leftrows.size(); i++) {
					if (flags[i]) {
						auto row = leftrows[i];
						for (size_t j = 0; j < mat[row].nnz(); j++) {
							tranmat[mat[row](j)].push_back(row, true);
						}
						flags[i] = 0;
						localcount++;
					}
				}

				double pr = kk + (1.0 * ps.size() * localcount) / leftrows.size();
				if (verbose && (print_once || pr - oldpr > printstep)) {
					auto end = sparse_rref::clocknow();
					now_nnz = mat.nnz();
					std::cout << "-- Col: " << std::setw(bitlen_ncol)
						<< (int)pr << "/" << mat.ncol
						<< "  rank: " << std::setw(bitlen_ncol) << rank
						<< "  nnz: " << std::setw(bitlen_nnz) << now_nnz
						<< "  density: " << std::setprecision(6) << std::setw(8)
						<< 100 * (double)now_nnz / (mat.nrow * mat.ncol) << "%"
						<< "  speed: " << std::setprecision(2) << std::setw(8) <<
						((pr - oldpr) / sparse_rref::usedtime(start, end))
						<< " col/s    \r" << std::flush;
					oldpr = pr;
					start = end;
					print_once = false;
				}
			}
			pool.wait();

			kk += ps.size();
		}

		if (verbose)
			std::cout << "\n** Rank: " << rank << " nnz: " << mat.nnz() << std::endl;

		return pivots;
	}

	// convert
	static inline void snmod_mat_from_sfmpq(sparse_mat<ulong>& mat, const sparse_mat<rat_t>& src, nmod_t p) {
		for (size_t i = 0; i < src.nrow; i++)
			snmod_vec_from_sfmpq(mat[i], src[i], p);
	}

	template <typename T>
	std::vector<std::vector<pivot_t>> sparse_mat_rref(sparse_mat<T>& mat, field_t F,
		sparse_rref::thread_pool& pool, rref_option_t opt) {
		std::vector<std::vector<pivot_t>> pivots = sparse_mat_rref_c(mat, F, pool, opt);

		if (opt->is_back_sub) {
			if (opt->verbose)
				std::cout << "\n>> Reverse solving: " << std::endl;
			// triangular_solver(mat, pivots, F, opt, -1, pool);
			triangular_solver_2(mat, pivots, F, opt, pool);
		}
		return pivots;
	}

	// TODO: check!!! 
	// parallel version !!!
	// output some information !!!
	std::vector<std::vector<pivot_t>> sparse_mat_rref_reconstruct(sparse_mat<rat_t>& mat,
		sparse_rref::thread_pool& pool, rref_option_t opt) {
		std::vector<std::vector<pivot_t>> pivots;

		ulong prime = n_nextprime(1ULL << 50, 0);
		field_t F;
		field_init(F, FIELD_Fp, prime);

		sparse_mat<ulong> matul(mat.nrow, mat.ncol);
		snmod_mat_from_sfmpq(matul, mat, F->mod);

		pivots = sparse_mat_rref_c(matul, F, pool, opt);

		if (opt->is_back_sub)
			triangular_solver_2(matul, pivots, F, opt, pool);

		int_t mod, mod1;

		mod = prime;

		int isok = 1;
		sparse_mat<rat_t> matq(mat.nrow, mat.ncol);

		// we use mod1 here as a temp variable
		for (auto i = 0; i < mat.nrow; i++) {
			matq[i].reserve(matul[i].nnz());
			auto& therow = matul[i];
			auto& therowq = matq[i];
			therowq.resize(therow.nnz());
			for (size_t j = 0; j < therow.nnz(); j++) {
				therowq(j) = therow(j);
				mod1 = therow[j];
				if (isok)
					isok = rational_reconstruct(therowq[j], mod1, mod);
			}
		}

		sparse_mat<int_t> matz(mat.nrow, mat.ncol);
		if (!isok) {
			for (auto i = 0; i < mat.nrow; i++) {
				auto& therow = matul[i];
				auto& therowz = matz[i];
				therowz.resize(therow.nnz());
				for (size_t j = 0; j < therow.nnz(); j++) {
					therowz(j) = therow(j);
					therowz[j] = therow[j];
				}
			}
		}

		while (!isok) {
			isok = 1;
			prime = n_nextprime(prime, 0);
			mod1 = mod * prime;
			field_init(F, FIELD_Fp, prime);
			snmod_mat_from_sfmpq(matul, mat, F->mod);
			sparse_mat_direct_rref(matul, pivots, F, pool, opt);
			if (opt->is_back_sub) {
				opt->verbose = false;
				triangular_solver_2(matul, pivots, F, opt, pool);
			}
			for (auto i = 0; i < mat.nrow; i++) {
				auto& therow = matul[i];
				auto& therowz = matz[i];
				auto& therowq = matq[i];
				for (size_t j = 0; j < therow.nnz(); j++) {
					therowz[j] = CRT(therowz[j], mod, therow[j], prime);
					if (isok)
						isok = rational_reconstruct(therowq[j], therowz[j], mod1);
				}
			}
			mod = mod1;
		}
		opt->verbose = true;

		std::swap(mat, matq);

		return pivots;
	}

	template <typename T>
	sparse_mat<T> sparse_mat_rref_kernel(const sparse_mat<T>& M,
		const std::vector<pivot_t>& pivots, field_t F, sparse_rref::thread_pool& pool) {

		sparse_mat<T> K;
		auto rank = pivots.size();
		if (rank == M.ncol)
			return K;

		T m1 = 1;

		if (rank == 0) {
			K.init(M.ncol, M.ncol);
			for (size_t i = 0; i < M.ncol; i++)
				K[i].push_back(i, m1);
			return K;
		}
		scalar_neg(m1, m1, F);

		sparse_mat<T> rows(rank, M.ncol);
		sparse_mat<T> trows(M.ncol, rank);
		for (size_t i = 0; i < rank; i++) {
			rows[i] = M[pivots[i].first];
		}
		sparse_mat_transpose_replace(trows, rows);

		K.init(M.ncol - rank, M.ncol);
		for (size_t i = 0; i < K.nrow; i++)
			K[i].zero();

		std::vector<slong> colpivs(M.ncol, -1);
		std::vector<slong> nonpivs;
		for (size_t i = 0; i < rank; i++)
			colpivs[pivots[i].second] = pivots[i].first;

		for (auto i = 0; i < M.ncol; i++)
			if (colpivs[i] == -1)
				nonpivs.push_back(i);

		pool.detach_loop<size_t>(0, nonpivs.size(), [&](size_t i) {
			auto thecol = trows[nonpivs[i]];
			auto k_vec = K[i];
			k_vec.reserve(thecol.nnz() + 1);
			for (size_t j = 0; j < thecol.nnz(); j++) {
				k_vec.push_back(pivots[thecol(j)].second, thecol[j]);
			}
			k_vec.push_back(nonpivs[i], m1);
			k_vec.sort_indices();
			});
		pool.wait();

		return K;
	}

	template <typename T>
	sparse_mat<T> sparse_mat_rref_kernel(const sparse_mat<T>& M,
		const std::vector<std::vector<pivot_t>>& pivots, field_t F, sparse_rref::thread_pool& pool) {
		std::vector<pivot_t> n_pivots;
		for (auto& p : pivots)
			n_pivots.insert(n_pivots.end(), p.begin(), p.end());
		return sparse_mat_rref_kernel(M, n_pivots, F, pool);
	}

	// IO
	template <typename T> void sfmpq_mat_read(sfmpq_mat& mat, T& st) {
		if (!st.is_open())
			return;
		std::string strLine;

		bool is_size = true;

		while (getline(st, strLine)) {
			if (strLine[0] == '%')
				continue;

			auto tokens = sparse_rref::SplitString(strLine, " ");
			if (is_size) {
				ulong nrow = std::stoul(tokens[0]);
				ulong ncol = std::stoul(tokens[1]);
				// ulong nnz = std::stoul(tokens[2]);
				// here we alloc 1, or alloc nnz/ncol ?
				mat.init(nrow, ncol);
				is_size = false;
			}
			else {
				if (tokens.size() != 3) {
					std::cerr << "Error: wrong format in the matrix file" << std::endl;
					std::exit(-1);
				}
				slong row = std::stoll(tokens[0]) - 1;
				slong col = std::stoll(tokens[1]) - 1;
				// SMS stop at 0 0 0
				if (row < 0 || col < 0)
					break;
				sparse_rref::DeleteSpaces(tokens[2]);
				rat_t val(tokens[2]);
				mat[row].push_back(col, val);
			}
		}
	}

	template <typename T, typename S> void sparse_mat_write(sparse_mat<T>& mat, S& st) {
		if constexpr (std::is_same_v<T, rat_t>) {
			st << "%%MatrixMarket matrix coordinate rational general" << '\n';
		}
		else {
			st << "%%MatrixMarket matrix coordinate integer general" << '\n';
		}
		st << mat.nrow << ' ' << mat.ncol << ' ' << mat.nnz() << '\n';
		for (size_t i = 0; i < mat.nrow; i++) {
			for (size_t j = 0; j < mat[i].nnz(); j++) {
				if (mat[i][j] == 0)
					continue;
				st << i + 1 << ' ' << mat[i][j] + 1 << ' ' << mat[i][j] << '\n';
			}
		}
	}

	static std::pair<size_t, char*> snmod_mat_to_binary(sparse_mat<ulong>& mat) {
		auto ratio = sizeof(ulong) / sizeof(char);
		auto nnz = mat.nnz();
		auto len = (3 + mat.nrow + 2 * nnz) * ratio;
		char* buffer = s_malloc<char>(len);
		char* ptr = buffer;
		std::memcpy(ptr, &(mat.nrow), sizeof(ulong)); ptr += ratio;
		std::memcpy(ptr, &(mat.ncol), sizeof(ulong)); ptr += ratio;
		std::memcpy(ptr, &nnz, sizeof(ulong)); ptr += ratio;
		for (size_t i = 0; i < mat.nrow; i++) {
			auto& therow = mat[i];
			auto nnz = therow.nnz();
			std::memcpy(ptr, &nnz, sizeof(ulong)); ptr += ratio;
			std::memcpy(ptr, therow.indices.data(), nnz * sizeof(ulong)); ptr += nnz * ratio;
			std::memcpy(ptr, therow.entries.data(), nnz * sizeof(ulong)); ptr += nnz * ratio;
		}
		return std::make_pair(len, buffer);
	}

	sparse_mat<ulong> snmod_mat_from_binary(char* buffer) {
		auto ratio = sizeof(ulong) / sizeof(char);
		char* ptr = buffer;
		ulong nnz, nrow, ncol;
		std::memcpy(&nrow, ptr, sizeof(ulong)); ptr += ratio;
		std::memcpy(&ncol, ptr, sizeof(ulong)); ptr += ratio;
		std::memcpy(&nnz, ptr, sizeof(ulong)); ptr += ratio;
		sparse_mat<ulong> mat(nrow, ncol);
		for (size_t i = 0; i < mat.nrow; i++) {
			auto& therow = mat[i];
			ulong nnz;
			std::memcpy(&nnz, ptr, sizeof(ulong)); ptr += ratio;
			therow.resize(nnz);
			std::memcpy(therow.indices.data(), ptr, nnz * sizeof(ulong)); ptr += nnz * ratio;
			std::memcpy(therow.entries.data(), ptr, nnz * sizeof(ulong)); ptr += nnz * ratio;
		}

		return mat;
	}

} // namespace sparse_rref

#endif
