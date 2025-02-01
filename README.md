# Sparse RREF
(exact) Sparse Reduced Row Echelon Form (RREF) **with row and column permutations** in C++

---

*Sparse Gaussian elimination is more an art than a science.*  -- [Spasm](https://github.com/cbouilla/spasm) (Charles Bouillaguet)

----

This head only library intends to compute the exact RREF with row and column permutations of a sparse matrix over finite field or rational field, which is a common problem in linear algebra, and is widely used for number theory, cryptography, theoretical physics, etc.. The code is based on the FLINT library, which is a C library for number theory. 

Some algorithms are inspired by [Spasm](https://github.com/cbouilla/spasm), but we do not depend on it. The algorithm here is definite (Spasm is random), so once the parameters are given, the result is stable, which is important for some purposes. 

### License

[The MIT License (MIT)](https://raw.githubusercontent.com/munuxi/sparse_mat/master/LICENSE)

### Dependence

The code mainly depends on [FLINT](https://flintlib.org/) to support arithmetic, and [BS::thread_pool](https://github.com/bshoshany/thread-pool) and [argparse](https://github.com/p-ranav/argparse) (they are included) are also used to support thread pool and parse args.

### What to compute?

For a sparse matrix $M$, the code computes its RREF  $\Lambda$ with row and column permutations. Instead of permute the row and column directly, we keep the row and column ordering of the matrix, i.e. the i-th row/column of $\Lambda$ is the i-th row/column of $M$, and the row and column permutation of this RREF is implicitly given by its pivots, which is a list of pairs of (row,col). In the ordering of pivots, the submatrix $\Lambda[\text{rows in pivots},\text{cols in pivots}]$ of $\Lambda$ is an identity matrix (if `--no-backward-substitution` is enabled, it is upper triangular). 

### How to use compile code

We now only support the rational field $\mathbb Q$ and the $\mathbb Z/p\mathbb Z$, where $p$ is a prime less than $2^{\texttt{BIT}-1}$ (it's $2^{63}$ on a 64-bit machine), but it is possible to generalize to other fields/rings by some small modification.

It is highly recommended to use [mimalloc](https://github.com/microsoft/mimalloc) (or other similar library) to dynamically override the standard malloc, especially on Windows.

We also provide an example, see `mma_link.cpp`, by using the LibraryLink api of Mathematica to compile a library which can used by Mathematica.

Build it, e.g. (also add `-lpthread` if pthread is required by the compiler)

```bash
g++ main.cpp -o sparserref -O3 -std=c++17 -Iincludepath -Llibpath -lflint -lgmp
```

```bash
g++ mma_link.cpp -fPIC -shared -O3 -std=c++17 -o mathlink.dll -Iincludepath -Llibpath -lflint -lgmp
```

### How to use the code

The `main.cpp` is an example to use the head only library, the help is

```
Usage: sparserref [--help] [--version] [--output VAR]
                  [--kernel] [--output-pivots]
                  [--field VAR] [--prime VAR] [--threads VAR]
                  [--pivot_direction VAR]
                  [--verbose] [--print_step VAR]
                  [--no-backward-substitution]
                  input_file

(exact) Sparse Reduced Row Echelon Form v0.2.5

Positional arguments:
  input_file                  input file in matrix market format

Optional arguments:
  -h, --help                  shows help message and exits
  -v, --version               prints version information and exits
  -o, --output                output file in matrix market format [default: "input_file.rref"]
  -k, --kernel                output the kernel
  --output-pivots             output pivots
  -F, --field                 QQ: rational field
                              Zp or Fp: Z/p for a prime p [default: "QQ"]
  -p, --prime                 a prime number, only vaild when field is Zp  [default: "34534567"]
  -t, --threads               the number of threads  [default: 1]
  -pd, --pivot_direction      the direction to select pivots [default: "col"]
  -V, --verbose               prints information of calculation
  -ps, --print_step           print step when --verbose is enabled [default: 100]
  --no-backward-substitution  no backward substitution
```

The main function is `sparse_mat_rref`, its output is its pivots, and it modifies the input matrix $M$ to its RREF $\Lambda$.

One main struct is `sparse_vec_t`, which contains two lists: the first is the list of indices of nonzero entries, and the second is entries. Another main struct is `sparse_mat_t`, which contains an array of `sparse_vec_t` and some other information.

### BenchMark

We compare it with [Spasm](https://github.com/cbouilla/spasm). Platform and Configuration: 

	CPU: Intel(R) Core(TM) Ultra 9 185H (6P+8E+2LPE)
	MEM: 24.5G + SWAP on PCIE4.0 SSD 
	OS: Arch Linux x86-64
	Compiler: gcc (GCC) 14.2.1 20240910
	FLINT: v3.1.2
	SparseRREF: v0.2.4 (with mimalloc)
	Prime number: 1073741827 ~ 2^30
	Configuration: 
	  - Spasm: Default configuration for Spasm, first spasm_echelonize and then spasm_rref
	  - SparseRREF: -V -t 16 -F Zp -p 1073741827

First two test matrices come from https://hpac.imag.fr, bs comes from symbol bootstrap, ibp comes from IBP of Feynman integrals:

| Matrix   | (#row, #col, #non-zero-values, rank)   | Spasm (echelonize + rref)    | SparseRREF (-pd row) | SparseRREF (-pd col) |
| -------- | -------------------------------------- | ---------------------------- | -------------------- | -------------------- |
| GL7d24   | (21074, 105054, 593892, 18549)         | 10.9765s + 51.0s             | 2.82s                | 3.58s                |
| M0,6-D10 | (1274688, 616320, 5342400, 493432)     | 101.195s + 13.4s             | 68.73s               | 101.19s              |
| bs-1     | (202552, 64350, 11690309, 62130)       | 5.53596s + 0.9s              | 2.22s                | 2.10s                |
| bs-2     | (709620, 732600, 48819232, 709620)     | too slow                     | 2210.43s             | 271.89s              |
| bs-3     | (10011551, 2958306, 33896262, 2867955) | 484s + 327.1s                | 77.84s               | 62.41s               |
| ibp-1    | (69153, 73316, 1117324, 58252)         | (rank is wrong) 2543.92s + ? | 4.07s                | 3.87s                |
| ibp-2    | (169323, 161970, 2801475, 135009)      | too slow                     | 32.33s               | 29.29s               |

bs-2 is slow for the option -pd row, since in the calulation of pivots, physical memory is not enough, and the system uses swap, which is slow.
Some tests for Spasm are slow for the same reason, so the SparseRREF is more efficient and uses less memory.

### TODO

* Improve the algorithms.
* Add PLUQ decomposition.
* Add more fields/rings.
* Improve I/O.
* Add document.

