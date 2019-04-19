#pragma once

#include <vector>
#include <algorithm>
#include <stdio.h>
#include <memory.h>
#include <getopt.h>

#include "datastructure.hpp"

#define MAX_EDGE 100000000
#define DEBUG

#ifdef DEBUG
#include <iostream>
#endif

char short_options[] = "f:p:q:";
struct option long_options[] = {
    {"file", required_argument, 0, 'f'},
    {"p", required_argument, 0, 'p'},
    {"q", required_argument, 0, 'q'}
};

// Node start from index 1
// void readGraph(char *fileName, LL*& adjList, LL*& adjCount, LL& nodes, LL& edges){
//     _HyperEdge* edge = new _HyperEdge[MAX_EDGE];
//     FILE* fd = fopen(fileName, "r");
//     LL startFlag = 1;
//     nodes = 0, edges = 0;
//     while(!feof(fd)){
//         fscanf(fd, "%lld %lld", &edge[edges].from, &edge[edges].to);
//         if(std::min(edge[edges].from, edge[edges].to) < startFlag)
//             startFlag = 0;
//         if(nodes < edge[edges].from || nodes < edge[edges].to)
//             nodes = std::max(edge[edges].from, edge[edges].to);
//         edges++;
//     }
//     adjList = new LL[edges];
//     adjCount = new LL[nodes + 1];
//     memset(adjCount, 0, sizeof(LL) * (nodes + 1));
//     adjCount[0] = 0LL;
//     std::sort(edge, edge + edges);
//     for(LL i = 0;i < edges;i++){
//         adjList[i] = edge[i].to + 1 - startFlag;
//         adjCount[edge[i].from + 1 - startFlag]++;
//     }
//     for(LL i = 1;i <= nodes;i++)
//         adjCount[i] += adjCount[i - 1];
//
//     delete[] edge;
// }

std::vector<_HyperEdge> readGraph(char *fileName, LL& n_nodes, LL& n_edges){
    n_nodes = n_edges = 0;
    FILE* fd = fopen(fileName, "r");
    std::vector<_HyperEdge> hyperEdge;
    fscanf(fd, "%lld", &n_nodes);
    while(!feof(fd)){
        LL n, u, v;
        _HyperEdge edge;
        fscanf(fd, "%lld", &n);
        while(n--){
            fscanf(fd, "%lld %lld", &u, &v);
            edge.vertex.insert(u);
            edge.vertex.insert(v);
            n_edges++;
        }
        hyperEdge.push_back(edge);
    }
    fclose(fd);

    #ifdef DEBUG
    std::cout << ">>> " << n_nodes << " nodes" << std::endl;
    std::cout << ">>> " << hyperEdge.size() << " hyper edges" << std::endl;
    std::cout << ">>> " << n_edges << " edges" << std::endl;
    #endif
    
    return hyperEdge;
}