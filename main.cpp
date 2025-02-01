/*
	Copyright (C) 2024 Zhenjie Li (Li, Zhenjie)

	This file is part of Sparse_rref. The Sparse_rref is free software:
	you can redistribute it and/or modify it under the terms of the MIT
	License.
*/

// // use mimalloc to replace the default malloc
// #include <cstdlib>
// #include "mimalloc-override.h"
// #include "mimalloc-new-delete.h"

#include <filesystem>
#include <fstream>
#include <iostream>

#include "argparse.hpp"
#include "sparse_mat.h"

#define printtime(str)                                                         \
    std::cout << (str) << " spent " << std::fixed << std::setprecision(6)      \
              << sparse_base::usedtime(start, end) << " seconds." << std::endl

#define printmatinfo(mat)                                                      \
    std::cout << "nnz: " << sparse_mat_nnz(mat) << " ";                        \
    std::cout << "nrow: " << (mat)->nrow << " ";                               \
    std::cout << "ncol: " << (mat)->ncol << std::endl

int main(int argc, char** argv) {
	argparse::ArgumentParser program("sparserref", sparse_base::version);
	program.set_usage_max_line_width(80);
	program.add_description("(exact) Sparse Reduced Row Echelon Form " + std::string(sparse_base::version));
	program.add_argument("input_file")
		.help("input file in matrix market format");
	program.add_argument("-o", "--output")
		.help("output file in matrix market format")
		.default_value("input_file.rref")
		.nargs(1);
	program.add_usage_newline();
	program.add_argument("-k", "--kernel")
		.default_value(false)
		.help("output the kernel")
		.implicit_value(true)
		.nargs(0);
	program.add_argument("--output-pivots")
		.help("output pivots")
		.default_value(false)
		.implicit_value(true)
		.nargs(0);
	program.add_usage_newline();
	program.add_argument("-F", "--field")
		.default_value("QQ")
		.help("QQ: rational field\nZp or Fp: Z/p for a prime p")
		.nargs(1);
	program.add_argument("-p", "--prime")
		.default_value("34534567")
		.help("a prime number, only vaild when field is Zp ")
		.nargs(1);
	program.add_argument("-t", "--threads")
		.help("the number of threads ")
		.default_value(1)
		.nargs(1)
		.scan<'i', int>();
	program.add_argument("-pd", "--pivot_direction")
		.help("the direction to select pivots")
		.default_value("col")
		.nargs(1);
	program.add_usage_newline();
	program.add_argument("-V", "--verbose")
		.default_value(false)
		.help("prints information of calculation")
		.implicit_value(true)
		.nargs(0);
	program.add_argument("-ps", "--print_step")
		.default_value(100)
		.help("print step when --verbose is enabled")
		.nargs(1)
		.scan<'i', int>();
	program.add_usage_newline();
	program.add_argument("--no-backward-substitution")
		.help("no backward substitution")
		.default_value(false)
		.implicit_value(true)
		.nargs(0);

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		std::exit(1);
	}

	ulong prime;
	nmod_t p;
	if (program.get<std::string>("--field") == "QQ") {
		prime = 0;
	}
	else if (program.get<std::string>("--field") == "Zp" || program.get<std::string>("--field") == "Fp") {
		if (program.get<std::string>("--prime") == "34534567") {
			prime = 34534567;
		}
		else {
			auto str = program.get<std::string>("--prime");
			fmpz_t prep;
			fmpz_init(prep);
			fmpz_set_str(prep, str.c_str(), 10);
			int is_max = fmpz_cmp_ui(prep, 1ULL << ((FLINT64) ? 63 : 31));
			if (is_max > 0) {
				std::cerr << "The prime number is too large: " << str
					<< std::endl;
				std::cerr << "It should be less than " << 2 << "^"
					<< ((FLINT64) ? 63 : 31) << std::endl;
				std::exit(1);
			}
			prime = fmpz_get_ui(prep);
			fmpz_clear(prep);
			if (!n_is_prime(prime)) {
				prime = n_nextprime(prime - 1, 0);
				std::cerr
					<< "The number is not a prime, use a near prime instead."
					<< std::endl;
			}
		}
		nmod_init(&p, prime);
		std::cout << "Using prime: " << prime << std::endl;
	}
	else {
		std::cerr << "The field is not valid: "
			<< program.get<std::string>("--field") << std::endl;
		std::exit(1);
	}

	int nthread = program.get<int>("--threads");
	sparse_base::thread_pool pool(nthread);
	std::cout << "using " << nthread << " threads" << std::endl;

	field_t F;
	if (prime == 0)
		field_init(F, FIELD_QQ, 1, NULL);
	else
		field_init(F, FIELD_Fp, std::vector<ulong>{prime});

	sparse_mat_t<fmpq> mat_Q;
	sparse_mat_t<ulong> mat_Zp;

	auto start = sparse_base::clocknow();
	auto input_file = program.get<std::string>("input_file");
	std::filesystem::path filePath = input_file;
	if (!std::filesystem::exists(filePath)) {
		std::cerr << "File does not exist: " << filePath << std::endl;
		return 1;
	}

	std::ifstream file(filePath);
	sfmpq_mat_read(mat_Q, file);
	if (prime != 0) {
		sparse_mat_init(mat_Zp, mat_Q->nrow, mat_Q->ncol);
		snmod_mat_from_sfmpq(mat_Zp, mat_Q, p);
		sparse_mat_clear(mat_Q);
	}
	file.close();

	auto end = sparse_base::clocknow();
	std::cout << "-------------------" << std::endl;
	printtime("read");

	if (prime == 0) {
		printmatinfo(mat_Q);
	}
	else {
		printmatinfo(mat_Zp);
	}

	std::cout << "-------------------" << std::endl;
	std::cout << "RREFing: " << std::endl;
	
	rref_option_t opt;
	opt->verbose = (program["--verbose"] == true);
	opt->is_back_sub = (program["--no-backward-substitution"] == false);
	opt->print_step = program.get<int>("--print_step");
	opt->pivot_dir = (program.get<std::string>("--pivot_direction") == "col");

	start = sparse_base::clocknow();
	std::vector<std::vector<pivot_t>> pivots;
	if (prime == 0) {
		// pivots = sparse_mat_rref(mat_Q, F, pool, opt);
		pivots = sparse_mat_rref_reconstruct(mat_Q, pool, opt);
	}
	else {
		pivots = sparse_mat_rref(mat_Zp, F, pool, opt);
	}

	end = sparse_base::clocknow();
	std::cout << "-------------------" << std::endl;
	printtime("RREF");

	size_t rank = 0;
	for (auto p : pivots) {
		rank += p.size();
	}
	std::cout << "rank: " << rank << " ";
	if (prime == 0) {
		printmatinfo(mat_Q);
	}
	else {
		printmatinfo(mat_Zp);
	}

	start = sparse_base::clocknow();
	std::ofstream file2;
	std::string outname, outname_add("");
	if (program.get<std::string>("--output") == "input_file.rref")
		outname = input_file;
	else
		outname = program.get<std::string>("--output");

	if (program["--output-pivots"] == true) {
		outname_add = ".piv";
		file2.open(outname + outname_add);
		for (auto p : pivots) {
			for (auto ii : p)
				file2 << ii.first + 1 << ", " << ii.second + 1 << '\n';
		}
		file2.close();
	}
	
	if (outname == input_file)
		outname_add = ".rref";
	else 
		outname_add = "";

	file2.open(outname + outname_add);
	if (prime == 0) {
		sparse_mat_write(mat_Q, file2);
	}
	else {
		sparse_mat_write(mat_Zp, file2);
	}
	file2.close();

	if (program["--kernel"] == true) {
		outname_add = ".kernel";
		file2.open(outname + outname_add);
		if (prime == 0) {
			sfmpq_mat_t K;
			auto krank = sparse_mat_rref_kernel(K, mat_Q, pivots, F, pool);
			if (krank > 0)
				sparse_mat_write(K, file2);
			else
				std::cout << "kernel is empty" << std::endl;
		}
		else {
			snmod_mat_t K;
			auto krank = sparse_mat_rref_kernel(K, mat_Zp, pivots, F, pool);
			if (krank > 0)
				sparse_mat_write(K, file2);
			else 
				std::cout << "kernel is empty" << std::endl;
		}
		file2.close();
	}

	end = sparse_base::clocknow();
	printtime("write files");

	field_clear(F);

	// clean is very expansive, leave to OS :(
	// sparse_mat_clear(mat_Q);
	// sparse_mat_clear(mat_Zp);
	// sparse_mat_clear(K);
	return 0;
}