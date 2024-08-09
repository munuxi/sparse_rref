#ifndef UTIL_H
#define UTIL_H

#include "flint/fmpq.h"
#include "flint/nmod.h"
#include "thread_pool.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <execution>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define NULL nullptr

// get the bit at position bit
#define GET_BIT(x, bit) (((x) >> (bit)) & 1ULL)
// set the bit at position bit
#define SET_BIT_ONE(x, bit) ((x) |= (1ULL << (bit)))
#define SET_BIT_NIL(x, bit) ((x) &= ~(1ULL << (bit)))

struct rref_option {
    bool verbose = false;
    bool pivot_dir = true; // true: row, false: col
    int print_step = 100;
    int sort_step = 0;
    int search_min = 200;
    ulong search_depth = ULLONG_MAX;
};
typedef struct rref_option rref_option_t[1];

enum RING {
    FIELD_F2,    // bool
    FIELD_QQ,    // fmpq
    FIELD_Fp,    // ulong
    RING_MulitFp // not implemented now
};

struct field_struct {
    enum RING type;
    ulong len; // the length of the product ring
    nmod_t p;
    // nmod_t* pvec;
};
typedef struct field_struct field_t[1];

template <typename T> inline T *binarysearch(T *begin, T *end, T val) {
    auto ptr = std::lower_bound(begin, end, val);
    if (ptr == end || *ptr == val)
        return ptr;
    else
        return end;
}

// string
inline void DeleteSpaces(std::string &str) {
    str.erase(std::remove_if(str.begin(), str.end(),
                             [](unsigned char x) { return std::isspace(x); }),
              str.end());
}
std::vector<std::string> SplitString(const std::string &s, std::string delim);

// time
inline std::chrono::system_clock::time_point clocknow() {
    return std::chrono::system_clock::now();
}

inline double usedtime(std::chrono::system_clock::time_point start,
                              std::chrono::system_clock::time_point end) {
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return ((double)duration.count() * std::chrono::microseconds::period::num /
           std::chrono::microseconds::period::den);
}

// vector
template <typename T>
void remove_indices(std::vector<T> &vec, std::vector<slong> &indices) {
    auto it = indices.rbegin();
    for (; it != indices.rend(); ++it) {
        if (*it >= 0 && *it < vec.size()) {
            vec.erase(vec.begin() + *it);
        } else {
            std::cerr << "Index out of range: " << *it << std::endl;
        }
    }
}

// Hopcroft-Karp algorithm for maximum cardinality matching in bipartite graphs
class BipartiteMatcher {
  public:
    BipartiteMatcher(slong U, slong V) : U(U), V(V), NIL(0) { init(); }

    void init() {
        adj.resize(U + 1);
        pairU.resize(U + 1, NIL);
        pairV.resize(U + V + 1, NIL);
        dist.resize(U + 1, LLONG_MAX);
    }

    ~BipartiteMatcher() {
        adj.clear();
        pairU.clear();
        pairV.clear();
        dist.clear();
    }

    void clearAdj() {
        for (auto &vec : adj) {
            vec.clear();
        }
        adj.clear();
    }

    void addEdge(slong u, slong v) { adj[u].push_back(v); }

    slong maxMatching() {
        slong matching = 0;
        while (bfs()) {
            for (auto u = 1; u <= U; ++u) {
                if (pairU[u] == NIL && dfs(u)) {
                    ++matching;
                }
            }
        }
        return matching;
    }

    slong maximalMatchingWithEdge(slong u, slong v) {
        pairU[u] = v;
        pairV[v] = u;

        for (auto &neighbors : adj) {
            neighbors.erase(remove(neighbors.begin(), neighbors.end(), v),
                            neighbors.end());
        }
        adj[u].clear();

        slong matching = 1;
        while (bfs()) {
            for (int i = 1; i <= U; ++i) {
                if (pairU[i] == NIL && dfs(i)) {
                    ++matching;
                }
            }
        }
        return matching;
    }

    std::vector<std::pair<slong, slong>> getMatchingPairs() {
        std::vector<std::pair<slong, slong>> pairs;
        for (auto u = 1; u <= U; ++u) {
            if (pairU[u] != NIL) {
                pairs.emplace_back(u, pairU[u] - U);
            }
        }
        return pairs;
    }

  private:
    slong U, V, NIL;
    std::vector<std::vector<slong>> adj;
    std::vector<slong> pairU, pairV, dist;

    bool bfs() {
        std::queue<slong> Q;
        for (auto u = 1; u <= U; ++u) {
            if (pairU[u] == NIL) {
                dist[u] = 0;
                Q.push(u);
            } else {
                dist[u] = LLONG_MAX;
            }
        }
        dist[NIL] = LLONG_MAX;
        while (!Q.empty()) {
            auto u = Q.front();
            Q.pop();
            if (dist[u] < dist[NIL]) {
                for (auto v : adj[u]) {
                    if (dist[pairV[v]] == LLONG_MAX) {
                        dist[pairV[v]] = dist[u] + 1;
                        Q.push(pairV[v]);
                    }
                }
            }
        }
        return dist[NIL] != LLONG_MAX;
    }

    bool dfs(slong u) {
        if (u != NIL) {
            for (auto v : adj[u]) {
                if (dist[pairV[v]] == dist[u] + 1) {
                    if (dfs(pairV[v])) {
                        pairV[v] = u;
                        pairU[u] = v;
                        return true;
                    }
                }
            }
            dist[u] = LLONG_MAX;
            return false;
        }
        return true;
    }
};

class Graph {
  public:
    Graph(slong V) { this->V = V; }
    void addEdge(slong v, slong w) {
        adj[v].push_back(w);
        adj[w].push_back(v);
    }
    void clear() { adj.clear(); }
    std::vector<std::vector<slong>> findMaximalConnectedComponents();
    std::unordered_set<slong> findMaximalCliqueContainingEdge(slong v, slong w);
    std::unordered_set<slong> findMaximalCliqueContainingVertex(slong v);

  private:
    void DFS(slong v, std::unordered_set<slong> &visited,
             std::vector<slong> &component);
    bool isClique(const std::unordered_set<slong> &vertices);
    void expandClique(std::unordered_set<slong> &clique, slong new_vertex);

    slong V;
    std::unordered_map<slong, std::vector<slong>> adj;
};

#endif