#pragma once

#include <stdio.h>
#include <memory.h>
#include <getopt.h>
#include <algorithm>

#define MAX_EDGE 100000000
#define LL unsigned long long

typedef struct Edge{
    LL from, to;
    bool operator < (const struct Edge& e) const {
        if(from == e.from)
            return to < e.to;
        return from < e.from;
    }
}Edge;

char short_options[] = "s:f:p:q::k::";
struct option long_options[] = {
    {"source", required_argument, 0, 's'},
    {"file", required_argument, 0, 'f'},
    {"p", required_argument, 0, 'p'},
    {"q", optional_argument, 0, 'q'},
    {"k", optional_argument, 0, 'k'}
};

int readGraph(char *fileName, LL*& adjList, LL*& adjCount, LL& nodes, LL& edges, bool reversed=false){
    Edge* edge = new Edge[MAX_EDGE];
    FILE* fd = fopen(fileName, "r");
    LL startFlag = 1;
    nodes = 0, edges = 0;
    while(!feof(fd) && ~fscanf(fd, "%lld %lld", &edge[edges].from, &edge[edges].to)){
        if(reversed)
            edge[edges].from ^= edge[edges].to ^= edge[edges].from ^= edge[edges].to;
        if(std::min(edge[edges].from, edge[edges].to) < startFlag)
            startFlag = 0;
        if(nodes < edge[edges].from || nodes < edge[edges].to)
            nodes = std::max(edge[edges].from, edge[edges].to);
        edges++;
    }
    nodes += 1 - startFlag;
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

    return startFlag;
}

void outputAdjInfo(LL* adjList, LL* adjCount, LL nodes, LL edges){
    printf("adjacent count:\n  ");
    for(LL i = 0;i <= nodes;i++)
        printf("%lld ", adjCount[i]);
    printf("\nadjacent list:\n  ");
    for(LL i = 0;i < edges;i++)
        printf("%lld ", adjList[i]);
    putchar('\n');
}