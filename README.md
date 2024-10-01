# Sparse RREF
Sparse Reduced Row Echelon Form (RREF) in C++

### License

[The MIT License (MIT)](https://raw.githubusercontent.com/munuxi/sparse_mat/master/LICENSE)

### Dependence

The code mainly depends on [FLINT](https://flintlib.org/) to support arithmetic, and [BS::thread_pool](https://github.com/bshoshany/thread-pool) and [argparse](https://github.com/p-ranav/argparse) (they are included) are also used to support thread pool and parse args.

Some algorithms are inspired by [Spasm](https://github.com/cbouilla/spasm), but we do not depend on it. The algorithm here is definite (Spasm is random), so once the parameters are given, the result is stable, which is important for some purposes.

### How to use this code

We now only support the rational field $\mathbb Q$ and the $\mathbb Z/p\mathbb Z$, where $p$ is a prime, but it is possible to generalize to other fields/rings by some small modification.

It is highly recommended to use [mimalloc](https://github.com/microsoft/mimalloc) (or other similar library) to dynamically override the standard malloc, especially on Windows.

We also provide an example, see `mma_link.cpp`, by using the LibraryLink api of Mathematica to compile a library which can used by Mathematica.

Build it, e.g. (also add -lpthread if pthread is required by the compiler)

```bash
g++ main.cpp -o sparserref -O3 -std=c++17 -Iincludepath -Llibpath -lflint -lgmp
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

### TODO

* Improve the algorithms.

