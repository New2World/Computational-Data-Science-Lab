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

void testInfluence2(DiffusionState_MIC &diffusionState, const Network &network, Results &result, int k, int span){
    int cindex;
    set<int> seedset;
    bool flag = true;
    vector<rTuple> rtup;
    diffusionState.getRTuples(network, rtup, 2000000);
    for(int i = 0;i < k/span && flag;i++){
        int k = i*span+span;
        seedset = result.seedset[k];
        if(seedset.size() != k)
            flag = false;
        cout << seedset.size() << " " << diffusionState.expInfluenceComplete_new(network, rtup, result.seedset[k]) << " " << result.supp[k] << endl;
    }
}

void test(const Network &network, DiffusionState_MIC &diffu, int *nodes){
    set<int> seedset;
    int tenpercent = 830;
    for(int j = 0;j < 4;j++){
        set<int> seed;
        for(int i = j*tenpercent;i < j*tenpercent+tenpercent;i++)
            seed.insert(nodes[i]);
        diffu.seed(seed);
    }
    vector<rTuple> rtup;
    cout << "count diff: " << diffu.getRTuples(network, rtup, 100000) << endl;

    for(int i = 4000;i < 4100;i++)
        seedset.insert(nodes[i]);
    int cindex = diffu.seed(seedset);
    cout << diffu.expInfluenceComplete(network, 30000, cindex) << endl;
    diffu.removeSeed(cindex);
    cout << diffu.computeG(seedset, rtup, network.vertexNum, "upper", nullptr) << endl;
    cout << diffu.computeG(seedset, rtup, network.vertexNum, "mid", nullptr) << endl;
    cout << diffu.computeG(seedset, rtup, network.vertexNum, "lower", nullptr) << endl;
}

int main(int args, char **argv){
    string name = string(argv[1]);
    string type = string(argv[2]);
    int vnum = atoi(argv[3]);
    int k = atoi(argv[4]);
    int span = atoi(argv[5]);
    string priority = string(argv[6]);
    bool load = false;
    if(args > 7)
        load = true;
    string path = "../data/"+name+".txt", fname;
    Network network(path, type, vnum);
    network.setICProb(.1);
    double eps = .3, N = 10000., partial = .05;
    int tenpercent = (int)(vnum * partial);
    mt19937 rand(chrono::high_resolution_clock::now().time_since_epoch().count());
    auto start = chrono::high_resolution_clock::now();

    DiffusionState_MIC diffusionState(network, priority, rand);

    int n, shuffle_node[vnum];
    path = "../data/"+name+"_node.txt";
    FILE *fd = fopen(path.c_str(),"r");
    vector<int> temp_node(vnum);
    if(!fd){
        fd = fopen(path.c_str(), "w");
        for(int i = 0;i < vnum;i++)
            temp_node[i] = i;
        shuffle(temp_node.begin(), temp_node.end(), rand);
        for(int i = 0;i < vnum;i++)
            fprintf(fd, "%d ", temp_node[i]);
        fclose(fd);
        fd = fopen(path.c_str(), "r");
    }
    for(int i = 0;i < vnum;i++){
        fscanf(fd, "%d", &n);
        shuffle_node[i] = n;
    }
    fclose(fd);

    // test(network, diffusionState, shuffle_node);
    // return 0;

    double l2;
    vector<rTuple> rtup;
    Results sandwich_result, sandwich_empty_result, reverse_result, highdegree_result;
    cout << "seed set: " << partial * 100 << "%" << endl;

    if(!load){
        sandwich_empty_result = Sandwich_computeSeedSet(network, diffusionState, k, eps, N, rtup, 2, span, &l2);
        fname = "inner/sandwich_no_cascade_"+name+"_"+priority+".txt";
        fd = fopen(fname.c_str(), "w");
        sandwich_result.writeToFile(fd);
        fclose(fd);

        for(int j = 0;j < 4;j++){
            set<int> seed;
            for(int i = j*tenpercent;i < j*tenpercent+tenpercent;i++)
                seed.insert(shuffle_node[i]);
            diffusionState.seed(seed);
        }

        rtup.clear();

        sandwich_result = Sandwich_computeSeedSet(network, diffusionState, k, eps, N, rtup, 2, span, &l2);
        fname = "inner/sandwich_"+name+"_"+priority+".txt";
        fd = fopen(fname.c_str(), "w");
        sandwich_result.writeToFile(fd);
        fclose(fd);

        // set<int> naivegreedy = NaiveGreedy_computeSeedSet(network, diffusionState, k, eps, N, 1);

        reverse_result = ReverseGreedy_computeSeedSet(network, diffusionState, k, l2, span);
        fname = "inner/reverse_"+name+"_"+priority+".txt";
        fd = fopen(fname.c_str(), "w");
        reverse_result.writeToFile(fd);
        fclose(fd);

        highdegree_result = HighDegree_computeSeedSet(network, diffusionState, k, span);
        fname = "inner/highdegree_"+name+"_"+priority+".txt";
        fd = fopen(fname.c_str(), "w");
        highdegree_result.writeToFile(fd);
        fclose(fd);
    } else{
        fname = "inner/sandwich_"+name+"_"+priority+".txt";
        fd = fopen(fname.c_str(), "r");
        sandwich_result.readFromFile(fd);
        fclose(fd);
        fname = "inner/sandwich_no_cascade_"+name+"_"+priority+".txt";
        fd = fopen(fname.c_str(), "r");
        sandwich_result.readFromFile(fd);
        fclose(fd);
        fname = "inner/reverse_"+name+"_"+priority+".txt";
        fd = fopen(fname.c_str(), "r");
        reverse_result.readFromFile(fd);
        fclose(fd);
        fname = "inner/highdegree_"+name+"_"+priority+".txt";
        fd = fopen(fname.c_str(), "r");
        highdegree_result.readFromFile(fd);
        fclose(fd);
    }

    cout << "---------- Testing Sandwich ----------" << endl;
    testInfluence(diffusionState, network, sandwich_result, k, span);
    // cout << endl;
    // testInfluence2(diffusionState, network, sandwich_result, k, span);

    cout << endl << "---------- Testing Sandwich without cascade ----------" << endl;
    testInfluence(diffusionState, network, sandwich_empty_result, k, span);
    // cout << endl;
    // testInfluence2(diffusionState, network, sandwich_empty_result, k, span);

    cout << endl << "---------- Testing Reverse ----------" << endl;
    testInfluence(diffusionState, network, reverse_result, k, span);
    // cout << endl;
    // testInfluence2(diffusionState, network, reverse_result, k, span);

    cout << endl << "---------- Testing High Degree ----------" << endl;
    testInfluence(diffusionState, network, highdegree_result, k, span);
    // cout << endl;
    // testInfluence2(diffusionState, network, highdegree_result, k, span);

    cout << endl;
    auto end = chrono::high_resolution_clock::now();

    printTime(start, end);
    return 0;
}