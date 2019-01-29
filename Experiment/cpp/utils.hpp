#pragma once

#include <stdio.h>
#include <memory.h>

#include <map>
#include <algorithm>

#include "datastructure.hpp"

std::map<LL,LL> readGraph(const char *fileName, LL*& adjList, LL*& adjCount, LL& nodes, LL& edges, std::map<LL,LL>& mp, bool reversed=false){
    _Edge* edge = new _Edge[MAXEDGE];
    FILE* fd = fopen(fileName, "r");
    LL u, v, tmp;
    std::map<LL,LL> rmp;
    while(~fscanf(fd, "%lld %lld", &u, &v)){
		if (reversed){
			tmp = u;
            u = v;
            v = tmp;
        }
        if(mp.find(u) == mp.end()){
            mp[u] = ++nodes;
            rmp[nodes] = u;
        }
        if(mp.find(v) == mp.end()){
            mp[v] = ++nodes;
            rmp[nodes] = v;
        }
        edge[edges].from = mp[u];
        edge[edges].to = mp[v];
        edges++;
    }
    adjList = new LL[edges];
    adjCount = new LL[nodes + 1];
    memset(adjCount, 0, sizeof(LL) * (nodes + 1));
    std::sort(edge, edge + edges);
    for(LL i = 0;i < edges;i++){
        adjList[i] = edge[i].to;
        adjCount[edge[i].from]++;
    }
    for(LL i = 1;i <= nodes;i++)
        adjCount[i] += adjCount[i - 1];

    delete[] edge;
    fclose(fd);

    return rmp;
}