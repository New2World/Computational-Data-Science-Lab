#include <stdlib.h>
#include <memory.h>
#include <time.h>
#include <math.h>
#include <limits.h>

#include <iostream>
#include <vector>
#include <set>

#include "datastructure.hpp"
#include "utils.hpp"

using namespace std;

vector<_HyperEdge> mpu(LL n_nodes, LL n_hedges, LL p, LL q, vector<_HyperEdge> hyperEdge){
    DSH dsh = DSH();

    LL threshold = (LL)(p - sqrt((double)n_hedges)), E_dsize = 0, E_ddsize, rnd = 0;
    set<LL> E, E_dash, E_ddash;
    set<LL>::iterator iter;
    E.clear();
    E_dash.clear();
    E_ddash.clear();
    for(int i = 0;i < n_hedges;i++)
        E.insert((LL)i + 1);

    while(E_dsize < threshold){
        dsh.buildFlowGraph(n_nodes, E, hyperEdge, q);
        E_ddash = dsh.miniCut();
        E_ddsize = E_ddash.size();
        if(E_dsize + E_ddsize <= p)
            E_dash.insert(E_ddash.begin(), E_ddash.end());
        else{
            for(int i = 0;i < p - E_dsize;i++){
                // a better way to select an arbitrary edge from E_ddash?
                rnd = (LL)rand() % E_ddsize;
                iter = E_ddash.begin();
                while(rnd--)
                    iter++;
                E_dash.insert(*iter);
                E_ddash.erase(iter);
            }
        }
        for(iter = E_ddash.begin();iter != E_ddash.end();iter++)
            E.erase(*iter);
        E_dsize = E_dash.size();
    }

    vector<_HyperEdge> cardinality, result;
    cardinality.clear();
    result.clear();
    for(iter = E.begin();iter != E.end();iter++)
        cardinality.push_back(hyperEdge[*iter - 1]);
    sort(cardinality.begin(), cardinality.end());
    for(LL i = 0;cardinality.size() && i < p - E_dash.size();i++)
        result.push_back(cardinality[i]);
    for(iter = E_dash.begin();iter != E_dash.end();iter++)
        result.push_back(hyperEdge[*iter - 1]);

    #ifdef DEBUG
    cout << "MpU size: " << result.size() << endl;
    #endif

    return result;
}

int main(int argc, char** argv){
    // argument parsing
    int source = 0;
    char ch, filePath[256];
    LL p, q = -1, k = 1000;
    while((ch = getopt_long(argc, argv, short_options, long_options, NULL)) != -1){
        switch(ch){
        case 's':
            source = atoi(optarg);
            break;
        case 'f':
            strncpy(filePath, optarg, 256);
            break;
        case 'p':
            p = atoll(optarg);
            break;
        case 'q':
            q = atoll(optarg);
            break;
        case 'k':
            k = atoll(optarg);
            break;
        }
    }

    // read graph from file
    LL totalNodes = 0, totalEdges = 0;
    LL* h_adjCount = NULL, *h_adjList = NULL;
    int startFlag = readGraph(filePath, h_adjList, h_adjCount, totalNodes, totalEdges, true);

    // outputAdjInfo(h_adjList, h_adjCount, totalNodes, totalEdges);

    printf("========= NEW RUN\n");
    printf("This graph contains %lld nodes connected by %lld edges\n", totalNodes, totalEdges);

    double probability;
    int startNode = source + 1 - startFlag, outdegree, nextNode;
    LL nodeCount = 0LL;
    srand(time(NULL));
    vector<_HyperEdge> hyperEdge;
    hyperEdge.clear();
    set<LL> nodeSet;
    while(k--){
        nodeSet.clear();
        while(true){
            outdegree = h_adjCount[startNode] - h_adjCount[startNode - 1];
            probability = 1. * rand() / RAND_MAX * (outdegree + 1);
            nextNode = floor(probability);
            nextNode += h_adjCount[startNode - 1];
            if(nextNode >= h_adjCount[startNode])
                break;
            nodeCount++;
            startNode = h_adjList[nextNode];
            if(nodeSet.find(startNode) != nodeSet.end())
                break;
            nodeSet.insert(startNode);
        }
        if(nodeSet.size() == 0)
            continue;
        hyperEdge.push_back(_HyperEdge{nodeSet});
    }

    delete[] h_adjList;
    delete[] h_adjCount;

    cout << hyperEdge.size() << " hyperedges are added" << endl;
    if(q < 0)
        q = hyperEdge.size() - 1;

    vector<_HyperEdge> E;
    set<LL> vertexSet;
    vertexSet.clear();
    if(hyperEdge.size() > 0)
        E = mpu(totalNodes, hyperEdge.size(), p, q, hyperEdge);

    printf("========= FINISH\n");

    for(int i = 0;i < E.size();i++)
        vertexSet.insert(E[i].vertex.begin(), E[i].vertex.end());

    for(auto i = vertexSet.begin();i != vertexSet.end();i++)
        printf("%lld ", *i);
    putchar('\n');

    return 0;
}
