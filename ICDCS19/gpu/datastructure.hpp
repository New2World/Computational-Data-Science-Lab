#pragma once

#include <iostream>

#include <memory.h>
#include <vector>
#include <queue>
#include <set>

#define MAXVERTEX 1500000
#define MAXEDGE 3000000
// #define MAXVERTEX 15000000
// #define MAXEDGE 30000000
#define NODESETSIZE 10000
// #define NODESETSIZE 285000

#define LL long long
#define INF 0x3f3f3f3f

typedef struct _Edge{
    LL from, to;
    bool operator < (const struct _Edge& e) const {
        if(from == e.from)
            return to < e.to;
        return from < e.from;
    }
}_Edge;

typedef struct _HyperEdge{
    LL v_size;
    LL vertex[NODESETSIZE / 4];
    _HyperEdge(){
        v_size = 0;
    }
    inline bool operator < (const struct _HyperEdge& e) const {
        return v_size < e.v_size;
    }
} _HyperEdge;

class DSH{
public:
    typedef struct _FlowEdge{
        LL from, to, cap, next;
        LL index;
    } _FlowEdge;

    LL* adjHead, *dis, *pre, *gap;
    LL edgeCount, vertexCount;
    LL tolHyperEdge, tolVertex;
    LL superSrc, superSink;
    _FlowEdge* flowEdge;

    DSH();
    ~DSH();
    void clearAll();
    void buildFlowGraph(LL, std::set<LL>, std::vector<_HyperEdge>, LL);
    void addFlowEdge(LL, LL, LL, LL);
    void maxFlow();
    std::vector<LL> miniCut();
};

DSH::DSH(){
    adjHead = new LL[MAXVERTEX];
    dis = new LL[MAXVERTEX];
    pre = new LL[MAXVERTEX];
    gap = new LL[MAXVERTEX];
    flowEdge = new _FlowEdge[MAXEDGE];
}

DSH::~DSH(){
    try{delete [] adjHead;}catch(...){printf("dsh free adjHead error\n");}
    try{delete [] dis;}catch(...){printf("dsh free dis error\n");}
    try{delete [] pre;}catch(...){printf("dsh free pre error\n");}
    try{delete [] gap;}catch(...){printf("dsh free gap error\n");}
    try{delete [] flowEdge;}catch(...){printf("dsh free flowEdge error\n");}
}

void DSH::clearAll(){
    edgeCount = 0;
    vertexCount = 0;
    tolHyperEdge = 0;
    tolVertex = 0;
    superSrc = superSink = 0;
    memset(adjHead, -1, sizeof(LL) * MAXVERTEX);
    memset(dis, 0, sizeof(LL) * MAXVERTEX);
    memset(pre, -1, sizeof(LL) * MAXVERTEX);
    memset(gap, 0, sizeof(LL) * MAXVERTEX);
    memset(flowEdge, 0, sizeof(_FlowEdge) * MAXEDGE);
}

void DSH::buildFlowGraph(LL V, std::set<LL> edgeSet, std::vector<_HyperEdge> hyperEdge, LL q){
    clearAll();
    tolHyperEdge = edgeSet.size();
    tolVertex = V;
    LL cur = 1, temp = 0;
    std::set<LL>::iterator iter;
    // add super source node(index 0) connecting to all hyperedges
    for(iter = edgeSet.begin();iter != edgeSet.end();iter++, cur++){
        temp += hyperEdge[*iter - 1].v_size;
        addFlowEdge(0, cur, 1, *iter);
    }
    // from index 1, connect all hyperedges with vertices contained
    cur = 1;
    for(iter = edgeSet.begin();iter != edgeSet.end();iter++, cur++)
        for(LL i = 0;i < hyperEdge[*iter - 1].v_size;i++)
            addFlowEdge(cur, hyperEdge[*iter - 1].vertex[i] + tolHyperEdge, INF, 0);
    cur += V;
    // std::cout << ">>>>>>>> " << edgeCount + V * 2 << " - " << std::endl;
    // add super sink node, connecting to all vertices
    superSink = cur;
    vertexCount = superSink + 1;
    for(LL i = 1;i <= V;i++)
        addFlowEdge(i + tolHyperEdge, superSink, q, 0);
    // std::cout << "TolVertex: " << tolVertex << std::endl;
    // std::cout << "TolHyperEdge: " << tolHyperEdge << std::endl;
    // std::cout << "VertexCount: " << vertexCount << std::endl;
    // std::cout << "edgeCount: " << edgeCount << std::endl;
}

void DSH::addFlowEdge(LL u, LL v, LL cap, LL index){
    flowEdge[edgeCount] = {u, v, cap, adjHead[u], index};
    adjHead[u] = edgeCount++;
    flowEdge[edgeCount] = {v, u, 0, adjHead[v], index};
    adjHead[v] = edgeCount++;
}

void DSH::maxFlow(){
    LL flow = 0, aug = INF;
    LL u, v;
    bool flag;
    gap[0] = vertexCount;
    u = pre[0] = 0;
    LL *cur = new LL[MAXVERTEX];
    memcpy(cur, adjHead, sizeof(LL) * MAXVERTEX);
    while(dis[0] < vertexCount){
        flag = false;
        for(LL &j = cur[u];j != -1;j = flowEdge[j].next){
            v = flowEdge[j].to;
            if(flowEdge[j].cap > 0 && dis[u] == dis[v] + 1){
                flag = true;
                if(flowEdge[j].cap < aug)
                    aug = flowEdge[j].cap;
                pre[v] = u;
                u = v;
                if(u == vertexCount - 1){
                    flow += aug;
                    while(u != 0){
                        u = pre[u];
                        flowEdge[cur[u]].cap -= aug;
                        flowEdge[cur[u] ^ 1].cap += aug;
                    }
                    aug = INF;
                }
                break;
            }
        }
        if(flag)
            continue;
        LL mindis = vertexCount;
        for(LL j = adjHead[u];j != -1;j = flowEdge[j].next){
            v = flowEdge[j].to;
            if(flowEdge[j].cap > 0 && dis[v] < mindis){
                mindis = dis[v];
                cur[u] = j;
            }
        }
        if((--gap[dis[u]]) == 0)
            break;
        gap[dis[u] = mindis + 1]++;
        u = pre[u];
    }

    try{
        delete [] cur;
    }
    catch(...){
        printf("maxflow free error\n");
        exit(1);
    }
}

std::vector<LL> DSH::miniCut(){
    maxFlow();
    std::vector<LL> mincut;
    mincut.clear();
    for(LL i = 0;flowEdge[i].from == 0;i += 2)
        if(flowEdge[i].cap <= 0)
            mincut.push_back(flowEdge[i].index);
    return mincut;
}