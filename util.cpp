#include "util.h"
#include <stack>

void sindex_vec_realloc(sindex_vec_t vec, ulong alloc) {
    if (alloc == vec->alloc)
        return;
    ulong old_alloc = vec->alloc;
    vec->alloc = std::min(alloc, vec->len);
    vec->indices = (slong *)realloc(vec->indices, vec->alloc * sizeof(slong));
}

void _sindex_vec_set_entry(sindex_vec_t vec, slong index) {
    if (index < 0 || (ulong)index >= vec->len)
        return;

    if (vec->nnz == vec->alloc) {
        ulong new_alloc = std::min(2 * (vec->alloc + 1), vec->len);
        sindex_vec_realloc(vec, new_alloc);
    }
    vec->indices[vec->nnz] = index;
    vec->nnz++;
}

std::vector<std::string> SplitString(const std::string &s, std::string delim) {
    auto start = 0ULL;
    auto end = s.find(delim);
    std::vector<std::string> result;
    while (end != std::string::npos) {
        result.push_back(s.substr(start, end - start));
        start = end + delim.length();
        end = s.find(delim, start);
    }
    result.push_back(s.substr(start, end));
    return result;
}

void Graph::DFS(int start, std::unordered_set<int> &visited,
                std::vector<int> &component) {
    std::stack<int> stack;
    stack.push(start);
    visited.insert(start);

    while (!stack.empty()) {
        int v = stack.top();
        stack.pop();
        component.push_back(v);

        for (int neighbor : adj[v]) {
            if (visited.find(neighbor) == visited.end()) {
                stack.push(neighbor);
                visited.insert(neighbor);
            }
        }
    }
}

std::vector<std::vector<int>> Graph::findMaximalConnectedComponents() {
    std::unordered_set<int> visited;
    std::vector<std::vector<int>> components;

    for (const auto &entry : adj) {
        int v = entry.first;
        if (visited.find(v) == visited.end()) {
            std::vector<int> component;
            DFS(v, visited, component);
            components.push_back(component);
        }
    }
    return components;
}

bool Graph::isClique(const std::unordered_set<int> &vertices) {
    for (int v : vertices) {
        for (int u : vertices) {
            if (v != u) {
                auto it = std::find(adj[v].begin(), adj[v].end(), u);
                if (it == adj[v].end()) {
                    return false;
                }
            }
        }
    }
    return true;
}

void Graph::expandClique(std::unordered_set<int> &clique, int new_vertex) {
    clique.insert(new_vertex);
    for (int neighbor : adj[new_vertex]) {
        if (clique.find(neighbor) == clique.end()) {
            std::unordered_set<int> expanded_clique = clique;
            expanded_clique.insert(neighbor);
            if (isClique(expanded_clique)) {
                clique = expanded_clique;
            }
        }
    }
}

std::unordered_set<int> Graph::findMaximalCliqueContainingEdge(int v, int w) {
    std::unordered_set<int> clique = {v, w};
    expandClique(clique, v);
    expandClique(clique, w);

    return clique;
}

std::unordered_set<int> Graph::findMaximalCliqueContainingVertex(int v) {
    std::unordered_set<int> clique = {v};
    for (int neighbor : adj[v]) {
        clique.insert(neighbor);
    }

    for (int neighbor : adj[v]) {
        expandClique(clique, neighbor);
    }

    return clique;
}