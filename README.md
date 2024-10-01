# Sparse RREF
(exact) Sparse Reduced Row Echelon Form (RREF) in C++

---

*Sparse Gaussian elimination is more an art than a science.*  -- [Spasm](https://github.com/cbouilla/spasm) (Charles Bouillaguet)

----

This head only library intends to compute the exact RREF of a sparse matrix over finite field or rational field, which is a common problem in linear algebra, and is widely used for number theory, cryptography, theoretical physics, etc.. The code is based on the FLINT library, which is a C library for number theory. 

Some algorithms are inspired by [Spasm](https://github.com/cbouilla/spasm), but we do not depend on it. The algorithm here is definite (Spasm is random), so once the parameters are given, the result is stable, which is important for some purposes. 

### License

[The MIT License (MIT)](https://raw.githubusercontent.com/munuxi/sparse_mat/master/LICENSE)

### Dependence

The code mainly depends on [FLINT](https://flintlib.org/) to support arithmetic, and [BS::thread_pool](https://github.com/bshoshany/thread-pool) and [argparse](https://github.com/p-ranav/argparse) (they are included) are also used to support thread pool and parse args.

### How to use this code

We now only support the rational field $\mathbb Q$ and the $\mathbb Z/p\mathbb Z$, where $p$ is a prime less than $2^{\texttt{BIT}-1}$ (it's $2^{63}$ on 64-bit machine), but it is possible to generalize to other fields/rings by some small modification.

It is highly recommended to use [mimalloc](https://github.com/microsoft/mimalloc) (or other similar library) to dynamically override the standard malloc, especially on Windows.

We also provide an example, see `mma_link.cpp`, by using the LibraryLink api of Mathematica to compile a library which can used by Mathematica.

Build it, e.g. (also add -lpthread if pthread is required by the compiler)

```bash
g++ main.cpp -o sparserref -O3 -std=c++17 -Iincludepath -Llibpath -lflint -lgmp
```

```bash
g++ mma_link.cpp -fPIC -shared -O3 -std=c++17 -o mathlink.dll -Iincludepath -Llibpath -lflint -lgmp
```


and help is 

```
Usage: sparserref [--help] [--version] [--verbose] [--print_step VAR] [--output VAR] [--kernel] [--output-pivots] [--field VAR] [--prime VAR] [--threads VAR] [--pivot_direction VAR] [--search_depth VAR] input_file

Positional arguments:
  input_file              input file in matrix market format

Optional arguments:
  -h, --help              shows help message and exits
  -v, --version           prints version information and exits
  -V, --verbose           prints information of calculation
  -ps, --print_step       print step when --verbose is enabled [default: 100]
  -o, --output            output file in matrix market format [default: "input_file.rref"]
  -k, --kernel            output the kernel
  --output-pivots         output pivots
  -f, --field             QQ: rational field
                          Zp or Fp: Z/p for a prime p [default: "QQ"]
  -p, --prime             a prime number, only vaild when field is Zp  [default: "34534567"]
  -t, --threads           the number of threads  [default: 1]
  -pd, --pivot_direction  the direction to select pivots [default: "row"]
  -sd, --search_depth     the depth of search, default is the max of int  [default: 0]
```

### BenchMark

We compare it with [Spasm](https://github.com/cbouilla/spasm). Platform and Configuration: 

	CPU: Intel(R) Core(TM) Ultra 9 185H (6P+8E+2LPE)
	MEM: 24.5G 
	OS: Arch Linux x86-64
	Compiler: gcc (GCC) 14.2.1 20240910
	FLINT: v3.1.2
	Prime number: 1073741827 ~ 2^30
	Configuration: 
	  - Spasm: Default configuration for Spasm, first spasm_echelonize and then spasm_rref
	  - SparseRREF: -V -t 16 -f Zp -p 1073741827

First two test matrices come from https://hpac.imag.fr, bs comes from symbol bootstrap, ibp comes from IBP of Feynman integrals:

| Matrix   | (#row, #col, #non-zero-values, rank)   | Spasm (echelonize + rref)    | SparseRREF (-pd row) | SparseRREF (-pd col) |
| -------- | -------------------------------------- | ---------------------------- | -------------------- | -------------------- |
| GL7d24   | (21074, 105054, 593892, 18549)         | 12.6796s + 52.4s             | 4.53s                | 4.40s                |
| M0,6-D10 | (1274688, 616320, 5342400, 493432)     | 101.195s + 13.4s             | 147.75s              | 216.86s              |
| bs-1     | (202552, 64350, 11690309, 62130)       | 5.53596s + 0.9s              | 6.11s                | 4.59s                |
| bs-2     | (709620, 732600, 48819232, 709620)     | too slow                     | 2311.33s             | 411.16s              |
| bs-3     | (10011551, 2958306, 33896262, 2867955) | 484s + 327.1s                | 142.31s              | 88.47s               |
| ibp-1    | (69153, 73316, 1117324, 58252)         | (rank is wrong) 2543.92s + ? | 9.72s                | 8.69s                |
| ibp-2    | (169323, 161970, 2801475, 135009)      | too slow                     | 60.13s               | 50.22s               |

### TODO

* Improve the algorithms.

