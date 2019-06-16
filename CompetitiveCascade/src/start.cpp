#include <set>
#include <vector>
#include <string>
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

void testInfluence(DiffusionState_MIC &diffusionState, const Network &network, const set<int> &solution, mt19937 &rand){
    int cindex;
    set<int> seedset;
    set<int>::iterator iter;
    for(int i = 0;i < 10;i++){
        iter = solution.begin();
        for(int j = 0;j < i*5;j++, iter++);
        for(int j = 0;j < 5;j++, iter++)
            seedset.insert(*iter);
        cindex = diffusionState.seed(seedset);
        cout << seedset.size() << " " << diffusionState.expInfluenceComplete(network, 1000, cindex, rand) << endl;
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

void printTime(chrono::high_resolution_clock::time_point start, chrono::high_resolution_clock::time_point end){
    auto duration = end-start;
    auto hour = chrono::duration_cast<chrono::hours>(duration).count();
    auto min = chrono::duration_cast<chrono::minutes>(duration).count();
    auto sec = chrono::duration_cast<chrono::seconds>(duration).count();
    cout << "time elapsed: " << hour << " hours " << min%60 << " minutes " << sec%60 << " seconds" << endl;
}

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
    set<int> sandwich;
    int l = Sandwich_computeSeedSet(network, diffusionState, k, eps, N, sandwich, rand);

    // set<int> naivegreedy = NaiveGreedy_computeSeedSet(network, diffusionState, k, eps, N, 1, rand);

    set<int> reversegreedy = ReverseGreedy_computeSeedSet(network, diffusionState, k, l, rand);

    set<int> highdegree = HighDegree_computeSeedSet(network, diffusionState, k);

    cout << "---------- Testing Sandwich ----------" << endl;
    testInfluence(diffusionState, network, sandwich, rand);

    cout << endl << "---------- Testing Reverse ----------" << endl;
    testInfluence(diffusionState, network, reversegreedy, rand);

    cout << endl << "---------- Testing High Degree ----------" << endl;
    testInfluence(diffusionState, network, highdegree, rand);

    auto end = std::chrono::high_resolution_clock::now();

    printTime(start, end);
}