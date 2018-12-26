#pragma once

#include <iostream>

#include <memory.h>
#include <vector>
#include <queue>
#include <set>

#define LL unsigned long long
#define MAXVERTEX 5000000
#define MAXEDGE 50000000
#define inf (1 << 31)

typedef struct _HyperEdge{
    std::set<LL> vertex;
    inline bool operator < (const struct _HyperEdge& e){
        return vertex.size() < e.vertex.size();
    }
} _HyperEdge;

class DSH{
public:
    typedef struct _FlowEdge{
        LL from, to, cap, next;
    } _FlowEdge;

    LL* adjHead, *dis, *dep, *gap;
    LL edgeCount, vertexCount;
    LL tolHyperEdge, tolVertex;
    LL superSrc, superSink;
    _FlowEdge* flowEdge;

    DSH();
    ~DSH();
    void clearAll();
    void buildFlowGraph(LL, std::set<LL>, std::vector<_HyperEdge>, LL);
    void addFlowEdge(LL, LL, LL);
    void _bfs(LL, LL);
    void maxFlow();
    std::set<LL> miniCut();
};

DSH::DSH(){
    adjHead = new LL[MAXVERTEX];
    dis = new LL[MAXVERTEX];
    dep = new LL[MAXVERTEX];
    gap = new LL[MAXVERTEX];
    flowEdge = new _FlowEdge[MAXEDGE];
}

DSH::~DSH(){
    delete [] adjHead;
    delete [] dis;
    delete [] dep;
    delete [] gap;
    delete [] flowEdge;
}

void DSH::clearAll(){
    edgeCount = 0;
    vertexCount = 0;
    tolHyperEdge = 0;
    tolVertex = 0;
    superSrc = superSink = 0;
    memset(adjHead, -1, sizeof(LL) * MAXVERTEX);
    memset(dis, 0, sizeof(LL) * MAXVERTEX);
    memset(dep, -1, sizeof(LL) * MAXVERTEX);
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
        temp += hyperEdge[*iter - 1].vertex.size();
        addFlowEdge(0, cur, 1);
    }
    // from index 1, connect all hyperedges with vertices contained
    cur = 1;
    for(iter = edgeSet.begin();iter != edgeSet.end();iter++, cur++)
        for(auto i = hyperEdge[*iter - 1].vertex.begin();i != hyperEdge[*iter - 1].vertex.end();i++)
            addFlowEdge(cur, *i + tolHyperEdge, inf);
    cur += V;
    // add super sink node, connecting to all vertices
    superSink = cur;
    vertexCount = superSink + 1;
    for(LL i = 1;i <= V;i++)
        addFlowEdge(i + tolHyperEdge, superSink, q);
}

void DSH::addFlowEdge(LL u, LL v, LL cap){
    flowEdge[edgeCount] = {u, v, cap, adjHead[u]};
    adjHead[u] = edgeCount++;
    flowEdge[edgeCount] = {v, u, 0, adjHead[v]};
    adjHead[v] = edgeCount++;
}

void DSH::_bfs(LL st, LL ed){
    LL u, v;
    gap[0] = 1;
    std::queue<LL> que;
    que.push(ed);
    while(!que.empty()){
        u = que.front();
        que.pop();
        for(LL i = adjHead[u];i != -1;i = flowEdge[i].next){
            v = flowEdge[i].to;
            if(dep[v] != -1)
                continue;
            que.push(v);
            dep[v] = dep[u] + 1;
            gap[dep[v]]++;
        }
    }
}

void DSH::maxFlow(){
    LL maxflow = 0, top = 0;
    _bfs(superSrc, superSink);
    LL u = superSrc, i;
    LL* head = new LL[MAXVERTEX], *S = new LL[MAXEDGE];
    memcpy(head, adjHead, sizeof(LL) * MAXVERTEX);
    while(dep[superSrc] < vertexCount){
        if(u == superSink){
            LL temp = inf;
            LL inser;
            for(i = 0;i < top;i++){
                if(temp > flowEdge[S[i]].cap){
                    temp = flowEdge[S[i]].cap;
                    inser = i;
                }
            }
            for(i = 0;i < top;i++){
                flowEdge[S[i]].cap -= temp;
                flowEdge[S[i] ^ 1].cap += temp;
            }
            maxflow += temp;
            top = inser;
            u = flowEdge[S[top]].from;
        }
        else if(gap[dep[u] - 1] == 0)
            break;
        for(i = head[u];i != -1;i = flowEdge[i].next)
            if(flowEdge[i].cap != 0 && dep[u] == dep[flowEdge[i].to] + 1)
                break;
        if(i != -1){
            head[u] = i;
            S[top++] = i;
            u = flowEdge[i].to;
        }
        else{
            LL min = vertexCount;
            for(LL i = adjHead[u];i != -1;i = flowEdge[i].next){
                if(flowEdge[i].cap == 0)
                    continue;
                if(min > dep[flowEdge[i].to]){
                    min = dep[flowEdge[i].to];
                    head[u] = i;
                }
            }
            gap[dep[u]]--;
            dep[u] = min + 1;
            gap[dep[u]]++;
            if(u != superSrc)
                u = flowEdge[S[--top]].from;
        }
    }
    delete [] head;
    delete [] S;
}

std::set<LL> DSH::miniCut(){
    maxFlow();
    std::set<LL> mincut;
    mincut.clear();
    for(LL i = 0;flowEdge[i].from == 0;i += 2)
        if(flowEdge[i].cap == 0)
            mincut.insert(flowEdge[i].to);
    return mincut;
}