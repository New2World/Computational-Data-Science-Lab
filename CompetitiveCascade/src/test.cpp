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

void testRTuple(DiffusionState_MIC &diffusionState, const Network &network, Results &sandwich_result, Results &reverse_result, int k){
    int cindex;
    double sandwich_influence = -1., reverse_influence = -1.;
    set<int> sandwich_seedset, reverse_seedset;
    sandwich_seedset = sandwich_result.seedset[k];
    reverse_seedset = reverse_result.seedset[k];
    if(sandwich_seedset.size() == k){
        cindex = diffusionState.seed(sandwich_seedset);
        sandwich_influence = diffusionState.expInfluenceComplete(network, 3000, cindex);
        diffusionState.removeSeed(cindex);
    }
    if(reverse_seedset.size() == k){
        cindex = diffusionState.seed(reverse_seedset);
        reverse_influence = diffusionState.expInfluenceComplete(network, 3000, cindex);
        diffusionState.removeSeed(cindex);
    }
    cout << sandwich_influence << "    " << reverse_influence << endl;
}

int main(int args, char **argv){
    string name = string(argv[1]);
    string type = string(argv[2]);
    int vnum = atoi(argv[3]);
    string pri = string(argv[4]);
    int k = 50;
    int from = atoi(argv[5]);
    int to = atoi(argv[6]);
    int span = atoi(argv[7]);
    string path = "../data/"+name+".txt";
    Network network(path, type, vnum);
    network.setICProb(.1);
    double eps = .3, N = 10000., partial = .05;
    int tenpercent = (int)(vnum * partial);
    int shuffle_node[vnum];
    mt19937 rand(chrono::high_resolution_clock::now().time_since_epoch().count());
    auto start = chrono::high_resolution_clock::now();

    DiffusionState_MIC diffusionState(network, pri, rand);

    // vector<int> shuffle_node(vnum);
    // for(int i = 0;i < vnum;i++)
    //     shuffle_node[i] = i;
    // shuffle(shuffle_node.begin(), shuffle_node.end(), rand);

    int n;
    path = "../data/"+name+"_node.txt";
    FILE *fd = fopen(path.c_str(),"r");
    for(int i = 0;i < vnum;i++){
        fscanf(fd, "%d", &n);
        shuffle_node[i] = n;
    }
    fclose(fd);

    cout << "seed set: " << partial * 100 << "%" << endl;
    for(int j = 0;j < 4;j++){
        set<int> seed;
        for(int i = j*tenpercent;i < j*tenpercent+tenpercent;i++)
            seed.insert(shuffle_node[i]);
        diffusionState.seed(seed);
    }
    vector<rTuple> rtup;

    Results sandwich_result, reverse_result;

    cout << "sandwich    reverse" << endl;
    for(int l = from;l <= to;l += span){
        sandwich_result = Sandwich_computeSeedSet(network, diffusionState, k, l, rtup, span);
        reverse_result = ReverseGreedy_computeSeedSet(network, diffusionState, k, rtup, span);
        testRTuple(diffusionState, network, sandwich_result, reverse_result, k);
    }

    // FILE *fd;
    // string fname = "inner/sandwich_"+name+"_"+pri+".txt";
    // fd = fopen(fname.c_str(), "w");
    // sandwich_result.writeToFile(fd);
    // fclose(fd);
    // fname = "inner/reverse_"+name+"_"+pri+".txt";
    // fd = fopen(fname.c_str(), "w");
    // reverse_result.writeToFile(fd);
    // fclose(fd);
    // fname = "inner/highdegree_"+name+"_"+pri+".txt";
    // fd = fopen(fname.c_str(), "w");
    // highdegree_result.writeToFile(fd);
    // fclose(fd);

    auto end = chrono::high_resolution_clock::now();
    printTime(start, end);
    return 0;
}