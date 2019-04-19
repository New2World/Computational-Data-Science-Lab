#include <bits/stdc++.h>
#include <getopt.h>

#include "utils.hpp"

#define DEBUG

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
    char ch, fileName[256];
    LL p, q;
    while((ch = getopt_long(argc, argv, short_options, long_options, NULL)) != -1){
        switch(ch){
        case 'f':
            strncpy(fileName, optarg, 256);
            break;
        case 'p':
            p = atoll(optarg);
            break;
        case 'q':
            q = atoll(optarg);
            break;
        }
    }
    LL n_nodes = 0, n_edges = 0, n_hedges = 0;
    srand(time(NULL));
    vector<_HyperEdge> hyperEdge = readGraph(fileName, n_nodes, n_edges);
    n_hedges = hyperEdge.size();

    vector<_HyperEdge> E_dash = mpu(n_nodes, n_hedges, p, q, hyperEdge);

    return 0;
}