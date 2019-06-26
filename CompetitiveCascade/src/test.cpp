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

double testInfluence(DiffusionState_MIC &diffusionState, const Network &network, Results &result, int k){
    vector<rTuple> rtup;
    diffusionState.getRTuples(network, rtup, 2000000);
    return diffusionState.expInfluenceComplete_new(network, rtup, result.seedset[k]);
}

int main(int args, char **argv){
    string name = string(argv[1]);
    string type = string(argv[2]);
    int vnum = atoi(argv[3]);
    int k = 50;
    int from = atoi(argv[4]);
    int to = atoi(argv[5]);
    int span = atoi(argv[6]);
    string priority = string(argv[7]);
    string path = "../data/"+name+".txt", fname;
    Network network(path, type, vnum);
    network.setICProb(.1);
    double eps = .3, N = 10000., partial = .05;
    int percent = (int)(vnum * partial);
    mt19937 rand(chrono::high_resolution_clock::now().time_since_epoch().count());
    auto start = chrono::high_resolution_clock::now();

    DiffusionState_MIC diffusionState(network, priority, rand);

    int n, shuffle_node[vnum];
    path = "../data/"+name+"_node.txt";
    FILE *fd = fopen(path.c_str(),"r");
    for(int i = 0;i < vnum;i++){
        fscanf(fd, "%d", &n);
        shuffle_node[i] = n;
    }
    fclose(fd);

    double l2;
    Results sandwich_result, sandwich_empty_result, reverse_result, highdegree_result;
    cout << "seed set: " << partial * 100 << "%" << endl;

    for(int j = 0;j < 4;j++){
        set<int> seed;
        for(int i = j*percent;i < j*percent+percent;i++)
            seed.insert(shuffle_node[i]);
        diffusionState.seed(seed);
    }
    fname = "inner/sandwich_"+name+"_"+priority+".txt";
    fd = fopen(fname.c_str(), "r");
    sandwich_result.readFromFile(fd);
    fclose(fd);
    double sandwich_value = testInfluence(diffusionState, network, sandwich_result, k), reverse_value;
    cout << "l\tsandwich\treverse" << endl;
    for(l2 = from;l2 <= to;l2 += span){
        reverse_result = ReverseGreedy_computeSeedSet_noneoutput(network, diffusionState, k, l2, 5);
        reverse_value = testInfluence(diffusionState, network, reverse_result, k);
        cout << l2 << "\t" << sandwich_value << "\t\t" << reverse_value << endl;
    }

    cout << endl;
    auto end = chrono::high_resolution_clock::now();
    printTime(start, end);

    return 0;
}