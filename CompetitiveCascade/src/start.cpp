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

void testInfluence(DiffusionState_MIC &diffusionState, const Network &network, Results &result, mt19937 &rand){
    int cindex;
    set<int> seedset;
    for(int i = 0;i < 10;i++){
        int k = i*5+5;
        seedset = result.seedset[k];
        cindex = diffusionState.seed(seedset);
        cout << seedset.size() << " " << diffusionState.expInfluenceComplete(network, 1000, cindex, rand) << " " << result.supp[k] << endl;
        diffusionState.removeSeed(cindex);
    }
}

// void test(const Network &network, DiffusionState_MIC &diffu, mt19937 &rand){
//     set<int> seedset;
//     int tenpercent = 830;
//     for(int j = 0;j < 4;j++){
//         set<int> seed;
//         for(int i = j*tenpercent;i < j*tenpercent+tenpercent;i++)
//             seed.insert(i);
//         diffu.seed(seed);
//     }
//     vector<rTuple> rtup;
//     cout << "count diff: " << diffu.getRTuples(network, rtup, 1000000, rand) << endl;

//     for(int i = 4000;i < 4100;i++)
//         seedset.insert(i);
//     int cindex = diffu.seed(seedset);
//     cout << diffu.expInfluenceComplete(network, 30000, cindex, rand) << endl;
//     diffu.removeSeed(cindex);
//     // cout << computeG(diffu, seedset, rtup, network.vertexNum, "upper", rand) << endl;
//     // cout << computeG(diffu, seedset, rtup, network.vertexNum, "mid", rand) << endl;
//     // cout << computeG(diffu, seedset, rtup, network.vertexNum, "lower", rand) << endl;
// }

int main(int args, char **argv){
    string name = string(argv[1]);
    string type = string(argv[2]);
    int vnum = atoi(argv[3]);
    string path = "../data/"+name+".txt";
    Network network(path, type, vnum);
    network.setICProb(.1);
    int k = 50;
    double eps = .3, N = 10000.;
    int tenpercent = (int)(vnum * .1);
    mt19937 rand(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    auto start = std::chrono::high_resolution_clock::now();

    DiffusionState_MIC diffusionState(network);

    for(int j = 0;j < 4;j++){
        set<int> seed;
        for(int i = j*tenpercent;i < j*tenpercent+tenpercent;i++)
            seed.insert(i);
        diffusionState.seed(seed);
    }
    vector<rTuple> rtup;
    Results sandwich_result, reverse_result, highdegree_result;
    sandwich_result = Sandwich_computeSeedSet(network, diffusionState, k, eps, N, rtup, rand);

    // set<int> naivegreedy = NaiveGreedy_computeSeedSet(network, diffusionState, k, eps, N, 1, rand);

    reverse_result = ReverseGreedy_computeSeedSet(network, diffusionState, k, rtup, rand);

    highdegree_result = HighDegree_computeSeedSet(network, diffusionState, k);

    cout << "---------- Testing Sandwich ----------" << endl;
    testInfluence(diffusionState, network, sandwich_result, rand);

    cout << endl << "---------- Testing Reverse ----------" << endl;
    testInfluence(diffusionState, network, reverse_result, rand);

    cout << endl << "---------- Testing High Degree ----------" << endl;
    for(pair<int,set<int>> p: highdegree_result.seedset){
        cout << p.first << endl;
        printContainer(p.second);
    }
    // testInfluence(diffusionState, network, highdegree_result, rand);

    auto end = std::chrono::high_resolution_clock::now();

    printTime(start, end);
    return 0;
}