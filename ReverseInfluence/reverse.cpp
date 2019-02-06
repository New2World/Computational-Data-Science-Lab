#include <stdlib.h>
#include <memory.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <set>

#include <iostream>
#include "utils.h"

#define LL unsigned long long

using namespace std;

float CONSTANT_PROBABILITY = 0.05;

int main(int argc, char** argv){
    // argument parsing
    int source = 0;
    char ch, filePath[256];
    while((ch = getopt_long(argc, argv, short_options, long_options, NULL)) != -1){
        switch(ch){
        case 's':
            source = atoi(optarg);
            break;
        case 'f':
            strncpy(filePath, optarg, 256);
            break;
        case 'p':
            CONSTANT_PROBABILITY = atof(optarg);
            CONSTANT_PROBABILITY = CONSTANT_PROBABILITY > 1 ? 1. : CONSTANT_PROBABILITY;
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
    set<LL> nodeSet;
    nodeSet.clear();
    while(true){
        outdegree = h_adjCount[startNode] - h_adjCount[startNode - 1];
        probability = 1. * rand() / RAND_MAX * (outdegree + 1);
        nextNode = floor(probability);
        nextNode += h_adjCount[startNode - 1];
        if(nextNode >= h_adjCount[startNode])
            break;
        nodeCount++;
        cout << startNode << " --> " << h_adjList[nextNode] << endl;
        startNode = h_adjList[nextNode];
        if(nodeSet.find(startNode) != nodeSet.end())
            break;
        cout << "INSERT" << endl;
        nodeSet.insert(startNode);
    }

    delete[] h_adjList;
    delete[] h_adjCount;

    printf("----------------------\n");
    printf("Total nodes: %lld\n", nodeCount);
    for(auto i = nodeSet.begin();i != nodeSet.end();i++)
        printf(" %lld", *i);
    if(nodeSet.size())
        putchar('\n');
    printf("----------------------\n");

    printf("========= FINISH\n");

    return 0;
}
