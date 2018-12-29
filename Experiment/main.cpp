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

#define DEBUG

using namespace std;

vector<_HyperEdge> mpu(LL n_nodes, LL n_hedges, LL p, LL q, vector<_HyperEdge> hyperEdge){
    DSH dsh = DSH();

    LL threshold = (LL)(p - sqrt((double)n_hedges)), E_dsize = 0, E_ddsize, rnd = 0;
    vector<LL> E_ddash;
    set<LL> E, E_dash;
    set<LL>::iterator iter;
    E.clear();
    E_dash.clear();
    E_ddash.clear();
    vector<LL> overlap;
    for(int i = 0;i < n_hedges;i++)
        E.insert((LL)i + 1);

    while(E_dsize < threshold){
        overlap.clear();
        dsh.buildFlowGraph(n_nodes, E, hyperEdge, q);
        E_ddash = dsh.miniCut();
        E_ddsize = E_ddash.size();
        // printf("get %lld minicut, %ld remain in E,  %lld / %lld\n", E_ddsize, E.size(), E_dsize, threshold);
        set_intersection(E_ddash.begin(), E_ddash.end(), E_dash.begin(), E_dash.end(), back_inserter(overlap));
        if(E_dsize + E_ddsize <= p){
            E_dash.insert(E_ddash.begin(), E_ddash.end());
            for(LL i = 0;i < E_ddsize;i++)
                E.erase(E_ddash[i]);
        }
        else{
            for(int i = 0;i < p - E_dsize;i++){
                rnd = (LL)rand() % E_ddsize;
                E_dash.insert(E_ddash[rnd]);
                E.erase(E_ddash[rnd]);
                E_ddash.erase(E_ddash.begin() + rnd);
                E_ddsize--;
            }
        }
        E_dsize = E_dash.size();
    }

    vector<_HyperEdge> cardinality, result;
    cardinality.clear();
    result.clear();
    for(iter = E.begin();iter != E.end();iter++)
        cardinality.push_back(hyperEdge[*iter - 1]);
    sort(cardinality.begin(), cardinality.end());
    for(LL i = 0;cardinality.size() && i < cardinality.size() && i < p - E_dsize;i++)
        result.push_back(cardinality[i]);
    for(iter = E_dash.begin();iter != E_dash.end();iter++)
        result.push_back(hyperEdge[*iter - 1]);

    return result;
}

void newOutput(LL counter, char* outputFile){
    char str[20] = "wiki_output_";
    sprintf(str + 11, "%lld.txt", counter);
    strcpy(outputFile, str);
}

int main(int argc, char** argv){
    // argument parsing
    LL source = 0, sink = 0;
    char ch, filePath[256];
    LL p = -1, q = -1, k = 1000;
    while((ch = getopt_long(argc, argv, short_options, long_options, NULL)) != -1){
        switch(ch){
        case 's':
            source = atoll(optarg);
            break;
        case 't':
            sink = atoll(optarg);
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
    char outputFile[20];
    LL totalNodes = 0, totalEdges = 0;
    LL* h_adjCount = NULL, *h_adjList = NULL;
    int startFlag = readGraph(filePath, h_adjList, h_adjCount, totalNodes, totalEdges, true);
    FILE* fd = fopen("wiki_input.txt", "r");

    // for(int i = h_adjCount[4123];i < h_adjCount[4124];i++)
    //     printf("%lld ", h_adjList[i]);
    // putchar('\n');

    printf("========= NEW RUN\n");
    printf("This graph contains %lld nodes connected by %lld edges\n\n", totalNodes, totalEdges);

    float alpha, probability, beta, pmax;
    LL startNode = source + 1 - startFlag, outdegree, nextNode;
    LL counter = 0;
    bool flag = true;
    srand(time(NULL));
    vector<_HyperEdge> hyperEdge;
    set<LL> nodeSet;
    vector<_HyperEdge> E;
    while(~fscanf(fd, "s %lld t %lld alpha %f L %lld pmax %f beta %f\n", &sink, &source, &alpha, &k, &pmax, &beta)){
        counter++;
        newOutput(counter, outputFile);
        FILE* wfd = fopen(outputFile, "w");
        hyperEdge.clear();
        for(LL i = 0;i < k;i++){
            startNode = source + 1 - startFlag;
            nodeSet.clear();
            nodeSet.insert(startNode);
            flag = true;
            while(true){
                outdegree = h_adjCount[startNode] - h_adjCount[startNode - 1];
                probability = 1. * rand() / RAND_MAX * outdegree;
                nextNode = floor(probability);
                nextNode += h_adjCount[startNode - 1];
                if(nextNode >= h_adjCount[startNode]){
                    nodeSet.clear();
                    break;
                }

                for(LL i = h_adjCount[startNode - 1];i < h_adjCount[startNode];i++){
                    if(h_adjList[i] == sink){
                        nodeSet.erase(startNode);
                        flag = false;
                        break;
                    }
                }
                if(!flag)
                    break;

                startNode = h_adjList[nextNode];
                if(nodeSet.find(startNode) != nodeSet.end()){
                    nodeSet.clear();
                    break;
                }
                nodeSet.insert(startNode);
            }
            if(nodeSet.size() == 0)
                continue;

            hyperEdge.push_back(_HyperEdge{nodeSet});
        }

        if(q < 0)
            q = 50. * hyperEdge.size() / totalNodes;
        if(p < 0)
            p = (int)(beta * hyperEdge.size());

        // printf("----------\n");
        // printf("s: %lld, t: %lld, hyperedge: %ld\n", sink, source, hyperEdge.size());
        nodeSet.clear();
        if(hyperEdge.size() > 0)
            E = mpu(totalNodes, hyperEdge.size(), p, q, hyperEdge);

        for(int i = 0;i < E.size();i++)
            nodeSet.insert(E[i].vertex.begin(), E[i].vertex.end());

        // printf("Vertex Union:\n");
        for(auto i = nodeSet.begin();i != nodeSet.end();i++)
            fprintf(wfd, "%lld ", *i);
        fprintf(wfd, "\n");
        // printf("\n----------\n");
        // getchar();
        fclose(wfd);
        if(counter >= 5)
            break;
    }

    fclose(fd);
    delete [] h_adjCount;
    delete [] h_adjList;

    printf("\n========= FINISH\n");

    return 0;
}
