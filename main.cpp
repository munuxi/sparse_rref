// // use mimalloc to replace the default malloc
// #include <cstdlib>
// #include "mimalloc-override.h"
// #include "mimalloc-new-delete.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "argparse.hpp"
#include "sparse_mat.h"

#define printtime(str)                                                         \
    std::cout << (str) << " spent " << std::fixed << std::setprecision(6)      \
              << usedtime(start, end) << " seconds." << std::endl

#define printmatinfo(mat)                                                      \
    std::cout << "nnz: " << sparse_mat_nnz(mat) << " ";                        \
    std::cout << "nrow: " << (mat)->nrow << " ";                               \
    std::cout << "ncol: " << (mat)->ncol << std::endl

int main(int argc, char** argv) {
	argparse::ArgumentParser program("sparserref", "v0.1.3");
	program.add_argument("input_file")
		.help("input file in matrix market format");
	program.add_argument("-o", "--output")
		.help("output file in matrix market format")
		.default_value("input_file.rref")
		.nargs(1);
	program.add_argument("--output-pivots")
		.help("output pivots")
		.default_value(false)
		.implicit_value(true)
		.nargs(0);
	program.add_argument("-k", "--kernel")
		.default_value(false)
		.help("output the kernel")
		.implicit_value(true)
		.nargs(0);
	program.add_argument("-f", "--field")
		.default_value("QQ")
		.help("QQ: rational field\nZp: Z/p for a prime p")
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
	program.add_argument("-sd", "--search_depth")
		.help("the depth of search, default is the max of size_t ")
		.default_value(0)
		.nargs(1)
		.scan<'i', int>();
	program.add_argument("-sm", "--search_min")
		.help("the minimal length to go out of search\nif depth < min, only "
			"search once")
		.default_value(200)
		.nargs(1)
		.scan<'i', int>();
	program.add_argument("-ss", "--sort_step")
		.help("sort the cols when rrefing\nif sort_step = 0, it equals "
			"max(1000,#cols/100)")
		.default_value(0)
		.nargs(1)
		.scan<'i', int>();
	program.add_argument("-pd", "--pivot_direction")
		.help("the direction to select pivots")
		.default_value("row")
		.nargs(1);
	program.add_argument("-V", "--verbose")
		.default_value(false)
		.help("prints information of calculation")
		.implicit_value(true)
		.nargs(0);
	program.add_argument("-ps", "--print_step")
		.help("print step when --verbose is enabled")
		.default_value(100)
		.nargs(1)
		.scan<'i', int>();

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
	else if (program.get<std::string>("--field") == "Zp") {
		if (program.get<std::string>("--prime") == "34534567") {
			prime = 34534567;
		}
		else {
			auto str = program.get<std::string>("--prime");
			fmpz_t prep;
			fmpz_init(prep);
			fmpz_set_str(prep, str.c_str(), 10);
			int is_max;
			if constexpr (FLINT64) {
				is_max = fmpz_cmp_ui(prep, 1ULL << 63);
			}
			else {
				is_max = fmpz_cmp_ui(prep, 1ULL << 31);
			}
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

	sparse_mat_t<fmpq> mat_Q;
	sparse_mat_t<ulong> mat_Zp;

	auto start = clocknow();
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

	auto n_row = (prime == 0) ? mat_Q->nrow : mat_Zp->nrow;
	auto n_col = (prime == 0) ? mat_Q->ncol : mat_Zp->ncol;

	auto end = clocknow();
	std::cout << "-------------------" << std::endl;
	printtime("read");

	if (prime == 0) {
		printmatinfo(mat_Q);
	}
	else {
		printmatinfo(mat_Zp);
	}

	int nthread = program.get<int>("--threads");
	BS::thread_pool pool(nthread);

	std::cout << "-------------------" << std::endl;
	std::cout << "RREFing: " << std::endl;
	std::cout << "using " << nthread << " threads" << std::endl;

	rref_option_t opt;
	opt->verbose = (program["--verbose"] == true);
	opt->print_step = program.get<int>("--print_step");
	opt->sort_step = program.get<int>("--sort_step");
	opt->search_depth = (ulong)program.get<int>("--search_depth");
	opt->search_min = program.get<int>("--search_min");
	opt->pivot_dir = (program.get<std::string>("--pivot_direction") == "row");
	if (opt->search_depth == 0)
		opt->search_depth = INT_MAX;
	if (opt->sort_step == 0)
		opt->sort_step = (int)std::max((ulong)1000, n_col / 100);

	start = clocknow();
	std::vector<std::pair<slong, slong>> pivots;
	if (prime == 0) {
		pivots = sfmpq_mat_rref(mat_Q, pool, opt);
	}
	else {
		pivots = snmod_mat_rref(mat_Zp, p, pool, opt);
	}

	end = clocknow();
	std::cout << "-------------------" << std::endl;
	printtime("RREF");

	std::cout << "rank: " << pivots.size() << " ";
	if (prime == 0) {
		printmatinfo(mat_Q);
	}
	else {
		printmatinfo(mat_Zp);
	}

	std::ofstream file2;
	std::string outname, outname_add("");
	if (program.get<std::string>("--output") == "input_file.rref")
		outname = input_file;
	else
		outname = program.get<std::string>("--output");

	if (outname == input_file)
		outname_add = ".rref";

	file2.open(outname + outname_add);
	if (prime == 0) {
		sparse_mat_write(mat_Q, file2);
	}
	else {
		sparse_mat_write(mat_Zp, file2);
	}
	file2.close();

	if (program["--output-pivots"] == true) {
		outname_add = ".piv";
		file2.open(outname + outname_add);
		for (auto& ii : pivots)
			file2 << ii.first + 1 << ", " << ii.second + 1 << '\n';
		file2.close();
	}

	if (program["--kernel"] == true) {
		outname_add = ".kernel";
		file2.open(outname + outname_add);
		if (prime == 0) {
			sfmpq_mat_t K;
			sfmpq_mat_rref_kernel(K, mat_Q, pivots, pool);
			sparse_mat_write(K, file2);
			// sparse_mat_clear(K);
		}
		else {
			snmod_mat_t K;
			snmod_mat_rref_kernel(K, mat_Zp, pivots, p, pool);
			sparse_mat_write(K, file2);
			// sparse_mat_clear(K);
		}
		file2.close();
	}

	// clean is very expansive
	// sparse_mat_clear(mat_Q);
	// sparse_mat_clear(mat_Zp);
	return 0;
}