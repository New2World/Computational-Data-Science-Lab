#pragma once

#include <algorithm>
#include <stdio.h>
#include <memory.h>
#include <getopt.h>

#define MAX_EDGE 100000000
#define LL unsigned long long

typedef struct Edge{
    LL from, to;
    bool operator < (const struct Edge e) const {
        return from < e.from;
    }
}Edge;

char short_options[] = "f:p::t";
struct option long_options[] = {
    {"file", required_argument, 0, 'f'},
    {"probability", optional_argument, 0, 'p'},
    {"thread", no_argument, 0, 't'}
};

void readGraph(char *fileName, LL*& adjList, LL*& adjCount, LL& nodes, LL& edges){
    Edge* edge = new Edge[MAX_EDGE];
    FILE* fd = fopen(fileName, "r");
    LL startFlag = 1;
    nodes = 0, edges = 0;
    while(!feof(fd)){
        fscanf(fd, "%lld %lld", &edge[edges].from, &edge[edges].to);
        if(std::min(edge[edges].from, edge[edges].to) < startFlag)
            startFlag = 0;
        if(nodes < edge[edges].from || nodes < edge[edges].to)
            nodes = std::max(edge[edges].from, edge[edges].to);
        edges++;
    }
    adjList = new LL[edges];
    adjCount = new LL[nodes + 1];
    memset(adjCount, 0, sizeof(LL) * (nodes + 1));
    adjCount[0] = 0LL;
    std::sort(edge, edge + edges);
    for(LL i = 0;i < edges;i++){
        adjList[i] = edge[i].to + 1 - startFlag;
        adjCount[edge[i].from + 1 - startFlag]++;
    }
    for(LL i = 1;i <= nodes;i++)
        adjCount[i] += adjCount[i - 1];

    delete[] edge;
}