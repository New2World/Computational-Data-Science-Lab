#include <set>
#include <vector>
#include <string>
#include <utility>
#include <iostream>

#include <random>
#include <chrono>

#include <cstdio>
#include <cstdlib>

#include "network.hpp"
#include "diffusionstate.hpp"
#include "utils.hpp"
#include "tools.hpp"

using namespace std;

void testInfluence(DiffusionState_MIC &diffusionState, const Network &network, Results &result, int k, int span){
    int cindex;
    set<int> seedset;
    bool flag = true;
    for(int i = 0;i < k/span && flag;i++){
        int k = i*span+span;
        seedset = result.seedset[k];
        if(seedset.size() != k)
            flag = false;
        if(seedset.empty()) break;
        cindex = diffusionState.seed(seedset);
        cout << seedset.size() << " " << diffusionState.expInfluenceComplete(network, 3000, cindex) << " " << result.supp[k] << endl;
        diffusionState.removeSeed(cindex);
    }
}

int main(int args, char **argv){
    string name = string(argv[1]);
    string type = string(argv[2]);
    int vnum = atoi(argv[3]);
    int k = atoi(argv[4]);
    int span = atoi(argv[5]);
    string path = "../data/"+name+".txt";
    Network network(path, type, vnum);
    network.setICProb(.1);
    double eps = .3, N = 10000., partial = .005;
    int tenpercent = (int)(vnum * partial);
    mt19937 rand(chrono::high_resolution_clock::now().time_since_epoch().count());
    auto start = chrono::high_resolution_clock::now();

    DiffusionState_MIC diffusionState(network, string(argv[6]), rand);

    vector<int> shuffle_node(vnum);
    for(int i = 0;i < vnum;i++)
        shuffle_node[i] = i;
    shuffle(shuffle_node.begin(), shuffle_node.end(), rand);

    vector<rTuple> rtup;

    double l2;
    Results sandwich_result, reverse_result, highdegree_result;
    sandwich_result = Sandwich_computeSeedSet(network, diffusionState, k, eps, N, rtup, 2, span, &l2);

    cout << "---------- Testing Sandwich ----------" << endl;
    testInfluence(diffusionState, network, sandwich_result, k, span);

    auto end = chrono::high_resolution_clock::now();

    printTime(start, end);
    return 0;
}