#include "util.h"
#include <stack>

std::vector<std::string> SplitString(const std::string& s, std::string delim) {
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

void Graph::DFS(slong start, std::unordered_set<slong>& visited,
	std::vector<slong>& component) {
	std::stack<slong> stack;
	stack.push(start);
	visited.insert(start);

	while (!stack.empty()) {
		slong v = stack.top();
		stack.pop();
		component.push_back(v);

		for (slong neighbor : adj[v]) {
			if (visited.find(neighbor) == visited.end()) {
				stack.push(neighbor);
				visited.insert(neighbor);
			}
		}
	}
}

std::vector<std::vector<slong>> Graph::findMaximalConnectedComponents() {
	std::unordered_set<slong> visited;
	std::vector<std::vector<slong>> components;

	for (const auto& entry : adj) {
		slong v = entry.first;
		if (visited.find(v) == visited.end()) {
			std::vector<slong> component;
			DFS(v, visited, component);
			components.push_back(component);
		}
	}
	return components;
}

bool Graph::isClique(const std::unordered_set<slong>& vertices) {
	for (slong v : vertices) {
		for (slong u : vertices) {
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

void Graph::expandClique(std::unordered_set<slong>& clique, slong new_vertex) {
	clique.insert(new_vertex);
	for (slong neighbor : adj[new_vertex]) {
		if (clique.find(neighbor) == clique.end()) {
			std::unordered_set<slong> expanded_clique = clique;
			expanded_clique.insert(neighbor);
			if (isClique(expanded_clique)) {
				clique = expanded_clique;
			}
		}
	}
}

std::unordered_set<slong> Graph::findMaximalCliqueContainingEdge(slong v, slong w) {
	std::unordered_set<slong> clique = { v, w };
	expandClique(clique, v);
	expandClique(clique, w);

	return clique;
}

std::unordered_set<slong> Graph::findMaximalCliqueContainingVertex(slong v) {
	std::unordered_set<slong> clique = { v };
	for (slong neighbor : adj[v]) {
		clique.insert(neighbor);
	}

	for (slong neighbor : adj[v]) {
		expandClique(clique, neighbor);
	}

	return clique;
}