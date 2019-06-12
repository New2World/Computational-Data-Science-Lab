#pragma once

#include <set>
#include <map>
#include <vector>
#include <utility>
#include <algorithm>

#include <random>

#include <string>
#include <iostream>

#include <cmath>

#include <boost/math/special_functions/binomial.hpp>

#include "rtuple.hpp"
#include "network.hpp"
#include "diffusionstate.hpp"
#include "sortmap.hpp"
#include "tools.hpp"

double computeG(DiffusionState_MIC &diffusionState, std::set<int> &S, std::vector<rTuple> &rtup, int n, std::string type, mt19937 &rand){
    int count = 0;
    for(rTuple rt: rtup){
        switch(type[0]){
        case 'u':
            if(intersection(S, rt.upper)) count++;
            break;
        case 'm':
            if(diffusionState.computeG(S, rt))  count++;
            break;
        case 'l':
            if(intersection(S, rt.lower)) count++;
            break;
        default:
            std::cout << "invalid type" << std::endl;
        }
    }
    return (double)n*count/rtup.size();
}

std::set<int> HighDegree_computeSeedSet(Network &network, DiffusionState_MIC &diffusionState, int k){
    std::cout << "========== High degree running ==========" << std::endl;
    network.sortByDegree();
    int node;
    std::set<int> result;
    for(int i = 0;i < network.vertexNum;i++){
        node = network.sorted_degree[i];
        if(diffusionState.seed_state[node] == -1 && result.find(node) == result.end()){
            result.insert(node);
            if(result.size() == k){
                std::cout << "========== High degree finish ==========" << std::endl << std::endl;
                return result;
            }
        }
    }
    std::cout << "========== High degree finish ==========" << std::endl << std::endl;
    return result;
}

std::set<int> NaiveGreedy_computeSeedSet(const Network &network, DiffusionState_MIC &diffusionState, int k, double epsilon_1, double N, int times, mt19937 &rand){
    std::cout << "========== Naive greedy running ==========" << std::endl;
    std::set<int> solution;
    int cmaxindex, cindex, *state = new int[network.vertexNum];
    double cmaxvalue, tempvalue;
    memcpy(state, diffusionState.seed_state, network.vertexNum*sizeof(int));

    for(int i = 0;i < k;i++){
        std::cout << "Naive greedy #" << i+1 << std::endl;
        cmaxindex = -1;
        cmaxvalue = -1.;
        for(int j = 0;j < network.vertexNum;j++){
            if(state[j] == -1)  solution.insert(j);
            cindex = diffusionState.seed(solution);
            tempvalue = diffusionState.expInfluenceComplete(network, times, cindex, rand);
            if(tempvalue > cmaxvalue){
                cmaxvalue = tempvalue;
                cmaxindex = j;
            }
            diffusionState.removeSeed(cindex);
            solution.erase(j);
        }
    }
    std::cout << "========== Naive greedy finish ==========" << std::endl << std::endl;
    return solution;
}

double Sandwich_greedyMid(const Network &network, DiffusionState_MIC &diffusionState, std::vector<rTuple> &rtup, std::set<int> &solution, int k, mt19937 &rand){
    std::set<int> candidate;
    for(rTuple rt: rtup)
        for(int v: rt.upper)
            if(candidate.find(v) == candidate.end())
                candidate.insert(v);
    int cmaxindex, node;
    double cmaxvalue, tempvalue;
    for(int i = 0;i < k;i++){
        cmaxindex = -1;
        cmaxvalue = -1.;
        for(int node: candidate){
            solution.insert(node);
            tempvalue = computeG(diffusionState, solution, rtup, network.vertexNum, "mid", rand);
            if(tempvalue > cmaxvalue){
                cmaxindex = node;
                cmaxvalue = tempvalue;
            }
            solution.erase(node);
        }
        solution.insert(cmaxindex);
        candidate.erase(cmaxindex);
    }
    return 0.;
}

std::set<int> ReverseGreedy_computeSeedSet(const Network &network, DiffusionState_MIC &diffusionState, int k, int l, mt19937 &rand){
    std::cout << "========== Reverse greedy running ==========" << std::endl;
    std::vector<rTuple> rtup;
    std::cout << "l " << l << " " << diffusionState.getRTuples(network, rtup, l, rand) << std::endl;
    std::set<int> mid_solution;
    std::cout << "working on mid-solution..." << std::endl;
    Sandwich_greedyMid(network, diffusionState, rtup, mid_solution, k, rand);
    std::cout << "========== Reverse greedy finish ==========" << std::endl << std::endl;
    return mid_solution;
}

double Sandwich_greedy(std::vector<rTuple> &rtup, std::set<int> &solution, int k, std::string type){
    std::vector<std::set<int>> rrsets;
    for(rTuple rt: rtup){
        switch(type[0]){
        case 'u':
            rrsets.push_back(rt.upper);
            break;
        case 'l':
            rrsets.push_back(rt.lower);
            break;
        default:
            std::cout << "invalid model" << std::endl;
        }
    }
    
    double coverred = 0.;
    double profit = 0.;
    std::map<int, std::set<int>> nodes_cover_sets;
    bool coverred_rrsets[rrsets.size()];
    for(int i = 0;i < rrsets.size();i++){
        for(int index: rrsets[i]){
            if(nodes_cover_sets.find(index) == nodes_cover_sets.end())
                nodes_cover_sets[index] = std::set<int>();
            nodes_cover_sets[index].insert(i);
        }
        coverred_rrsets[i] = false;
    }
    std::vector<std::pair<int,int>> sortPair;
    for(auto i = nodes_cover_sets.begin();i != nodes_cover_sets.end();i++)
        sortPair.push_back(std::make_pair(i->first, i->second.size()));
    sort(sortPair.begin(), sortPair.end(), [](const std::pair<int,int> a, const std::pair<int,int> b)-> bool {
        return a.second > b.second;
    });
    sortedMap mymap;
    for(std::pair<int,int> p: sortPair)
        mymap.push_back(p.first, p.second);
    int c_bound, t_bound, c_seed;
    bool sign;
    std::set<int> c_seed_cover;
    for(int i = 0;i < k;i++){
        sign = false;
        c_bound = mymap.size();
        while(c_bound > 0){
            c_seed = mymap.get(0);
            c_seed_cover = nodes_cover_sets[c_seed];
            for(int j: c_seed_cover){
                if(coverred_rrsets[j])
                    c_seed_cover.erase(j);
            }
            t_bound = mymap.update(c_seed, c_seed_cover.size());
            if(t_bound == 0){
                solution.insert(c_seed);
                sign = true;
                for(int j: c_seed_cover){
                    if(coverred_rrsets[j]){
                        cout << "greedy update may wrong" << endl;
                        exit(1);
                    }
                    else{
                        coverred_rrsets[j] = true;
                        coverred += 1.;
                    }
                }
                break;
            }
            if(t_bound < c_bound)   c_bound = t_bound;
        }
        if(!sign){
            cout << "greedy lazy: no node selected" << endl;
            exit(1);
        }
    }
    return coverred;
}

double Sandwich_computeLowerBound(const Network &network, DiffusionState_MIC &diffusionState, int k, double eps0, double N, mt19937 &rand){
    int n = network.vertexNum;
    std::vector<rTuple> rtup;
    std::set<int> S;
    double lambda = (n*(2+eps0)*log(N*boost::math::binomial_coefficient<double>(n,k)*log2(n)))/(eps0*eps0);
    double x, l, g_lower;
    for(int i = 1;i < (int)log2(n-1.);i++){
        x = n/pow(2., i);
        l = lambda/x;
        if(rtup.size() < l)
            diffusionState.getRTuples(network, rtup, (int)(l-rtup.size()), rand);
        S.clear();
        Sandwich_greedy(rtup, S, k, "lower");
        g_lower = computeG(diffusionState, S, rtup, n, "lower", rand);
        if(g_lower >= (1+eps0)*x)   return g_lower;
    }
    std::cout << "compute lower bound may wrong, opt too small" << std::endl;
    exit(1);
}

double Sandwich_decideL(int n, int k, double low_bound, double eps1, double eps2, double N){
    double l1 = ((2+eps1)*n*log(N+boost::math::binomial_coefficient<double>(n,k)))/(eps1*eps1);
    double l2 = 2*n*log(N)/(eps2*eps2);
    return (l1>l2?l1:l2)/low_bound;
}

int Sandwich_computeSeedSet(const Network &network, DiffusionState_MIC &diffusionState, int k, double eps1, double N, std::set<int> &solution, mt19937 &rand){
    std::cout << "========== Sandwich running ==========" << std::endl;
    double eps0 = eps1, eps2 = (eps1*log(N))/(log(network.vertexNum)+log(N));
    double low_bound = Sandwich_computeLowerBound(network, diffusionState, k, eps0, N, rand);
    int l = (int)Sandwich_decideL(network.vertexNum, k, low_bound, eps1, eps2, N);
    std::vector<rTuple> rtup;
    std::cout << "l " << l << " " << diffusionState.getRTuples(network, rtup, l, rand) << std::endl;
    std::set<int> upper_solution, lower_solution;
    std::cout << "working on upper solution..." << std::endl;
    Sandwich_greedy(rtup, upper_solution, k, "upper");
    std::cout << "working on lower solution..." << std::endl;
    Sandwich_greedy(rtup, lower_solution, k, "lower");
    std::cout << "calculating upper G... " << std::endl;
    double upper_g = computeG(diffusionState, upper_solution, rtup, network.vertexNum, "mid", rand);
    std::cout << "  upper G = " << upper_g << std::endl;
    std::cout << "calculating lower G... " << std::endl;
    double lower_g = computeG(diffusionState, lower_solution, rtup, network.vertexNum, "mid", rand);
    std::cout << "  lower G = " << lower_g << std::endl;
    if(upper_g > lower_g)
        for(int upper_v: upper_solution)
            solution.insert(upper_v);
    else
        for(int lower_v: lower_solution)
            solution.insert(lower_v);
    std::cout << "========== Sandwich finish ==========" << std::endl << std::endl;
    return l;
}