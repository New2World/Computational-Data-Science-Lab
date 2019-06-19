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

void testInfluence(DiffusionState_MIC &diffusionState, const Network &network, Results &result, mt19937 &rand, int k, int span){
    int cindex;
    set<int> seedset;
    for(int i = 0;i < k/span;i++){
        int k = i*span+span;
        seedset = result.seedset[k];
        cindex = diffusionState.seed(seedset);
        cout << seedset.size() << " " << diffusionState.expInfluenceComplete(network, 1000, cindex, rand) << " " << result.supp[k] << endl;
        diffusionState.removeSeed(cindex);
    }
}

void test(const Network &network, DiffusionState_MIC &diffu, mt19937 &rand){
    set<int> seedset;
    int tenpercent = 830;
    for(int j = 0;j < 4;j++){
        set<int> seed;
        for(int i = j*tenpercent;i < j*tenpercent+tenpercent;i++)
            seed.insert(i);
        diffu.seed(seed);
    }
    vector<rTuple> rtup;
    cout << "count diff: " << diffu.getRTuples(network, rtup, 10000, rand) << endl;
    // for(rTuple rt: rtup)
    //     rt._stat();

    for(int i = 4000;i < 4100;i++)
        seedset.insert(i);
    int cindex = diffu.seed(seedset);
    cout << diffu.expInfluenceComplete(network, 30000, cindex, rand) << endl;
    diffu.removeSeed(cindex);
    cout << diffu.computeG(seedset, rtup, network.vertexNum, "upper", nullptr, rand) << endl;
    cout << diffu.computeG(seedset, rtup, network.vertexNum, "mid", nullptr, rand) << endl;
    cout << diffu.computeG(seedset, rtup, network.vertexNum, "lower", nullptr, rand) << endl;
}

int main(int args, char **argv){
    string name = string(argv[1]);
    string type = string(argv[2]);
    int vnum = atoi(argv[3]);
    int k = atoi(argv[4]);
    int span = atoi(argv[5]);
    int base = atoi(argv[6]);
    string path = "../data/"+name+".txt";
    Network network(path, type, vnum);
    network.setICProb(.1);
    double eps = .3, N = 10000., partial = .1;
    int tenpercent = (int)(vnum * partial);
    mt19937 rand(chrono::high_resolution_clock::now().time_since_epoch().count());
    auto start = chrono::high_resolution_clock::now();

    DiffusionState_MIC diffusionState(network);

    test(network, diffusionState, rand);
    return 0;
    cout << "seed set: " << partial * 100 << "%" << endl;
    for(int j = 0;j < 4;j++){
        set<int> seed;
        for(int i = j*tenpercent;i < j*tenpercent+tenpercent;i++)
            seed.insert(i);
        diffusionState.seed(seed);
    }
    vector<rTuple> rtup;

    Results sandwich_result, reverse_result, highdegree_result;
    sandwich_result = Sandwich_computeSeedSet(network, diffusionState, k, eps, N, rtup, base, rand, span);

    // set<int> naivegreedy = NaiveGreedy_computeSeedSet(network, diffusionState, k, eps, N, 1, rand);

    reverse_result = ReverseGreedy_computeSeedSet(network, diffusionState, k, rtup, rand, span);

    highdegree_result = HighDegree_computeSeedSet(network, diffusionState, k, span);

    cout << "---------- Testing Sandwich ----------" << endl;
    testInfluence(diffusionState, network, sandwich_result, rand, k, span);

    cout << endl << "---------- Testing Reverse ----------" << endl;
    testInfluence(diffusionState, network, reverse_result, rand, k, span);

    cout << endl << "---------- Testing High Degree ----------" << endl;
    testInfluence(diffusionState, network, highdegree_result, rand, k, span);

    auto end = chrono::high_resolution_clock::now();

    printTime(start, end);
    return 0;
}