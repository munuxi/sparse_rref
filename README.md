# Sparse RREF
Sparse Reduced Row Echelon Form (RREF) in C++

### License

The MIT License (MIT)

### Dependence

Mainly depend on [FLINT](https://flintlib.org/) to support arithmetic of rational field $\mathbb Q$ and $\mathbb Z/p\mathbb Z$, where $p$ is a prime.  And we also use  [BS::thread_pool](https://github.com/bshoshany/thread-pool), [argparse](https://github.com/p-ranav/argparse) (they are included). 

Some algorithms are motivated by [Spasm](https://github.com/cbouilla/spasm), but we do not depend on it. The algorithm here is definite not random, so once the parameters are given, the result is stable, which is important for some purposes.

### How to use this code

Build it; and help is 

```bash
Usage: sparserref [--help] [--version] [--output VAR] [--output-pivots] [--field VAR] [--prime VAR] [--threads VAR] [--search_depth VAR] [--search_min VAR] [--sort_step VAR] [--verbose] [--print_step VAR] input_file

Positional arguments:
  input_file           input file in matrix market format

Optional arguments:
  -h, --help           shows help message and exits
  -v, --version        prints version information and exits
  -o, --output         output file in matrix market format [default: "input_file.rref"]
  --output-pivots      output pivots
  -f, --field          QQ: rational field
                       Zp: Z/p for a prime p [default: "QQ"]
  -p, --prime          a prime number, only vaild when field is Zp  [default: "4194319"]
  -t, --threads        the number of threads  [default: 1]
  -sd, --search_depth  the depth of search, default is the max of size_t  [default: 0]
  -sm, --search_min    the minimal length to go out of search
                       if depth < min, only search once [default: 200]
  -ss, --sort_step     sort the cols when rrefing
                       if sort_step=0, it equals max(1000,#cols/100) [default: 0]
  -V, --verbose        prints information of calculation
  -ps, --print_step    print step when --verbose is enabled [default: 100]
```
