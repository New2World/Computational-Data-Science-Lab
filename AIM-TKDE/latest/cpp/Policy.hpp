#pragma once
#include <cstdio>
#include <cstdlib>
#include <climits>
#include <vector>
#include <string>
#include <map>
#include <random>

#include "Network.hpp"
#include "DiffusionState.hpp"

class Policy{
public:
    static int rrsets_size, simurest_times;
    string simumethod;

    Policy(){}
    Policy(const Policy &policy){
        simumethod = policy.simumethod;
    }
    ~Policy(){}

    virtual vector<int> computeSeedSet(const Network &, DiffusionState, int, mt19937){
        return {};
    }
};

int Policy::rrsets_size = 100000;
int Policy::simurest_times = 100;

int reSpreadOneRound(const Network & network, vector<int> new_active, bool *state, vector<int> rrset, DiffusionState diffusionState, mt19937 rand){
    int cseed, cseede;
    double prob;
    vector<int> new_active_temp;
    vector<vector<int>> re_neighbor = network.neighbor_reverse;

    for(int i = 0;i < new_active.size();i++){
        cseed = new_active[i];
        for(int j = 0;j < re_neighbor[cseed].size();j++){
            cseede = re_neighbor[cseed][j];
            prob = network.getProb(cseede, cseed);
            if(network.isSuccess(prob, rand))
                new_active_temp.push_back(cseede);
        }
    }
    new_active.clear();
    for(int i: new_active_temp){
        if(find(diffusionState.new_active.begin(), diffusionState.new_active.end(), i) != diffusionState.new_active.end())
            return 1;
        if(!state[i]){
            new_active.push_back(i);
            rrset.push_back(i);
            state[i] = true;
        }
    }
    return 0;
}

int reSpreadOnce(const Network & network, int cindex, vector<int> rrset, DiffusionState diffusionState, mt19937 rand){
    bool *state = new bool[diffusionState.vnum];
    memcpy(state, diffusionState.state, diffusionState.vnum);
    vector<int> new_active;

    state[cindex] = true;
    rrset.push_back(cindex);
    new_active.push_back(cindex);

    int round = 0;
    while(!new_active.empty() && round < diffusionState.round_left){
        if(reSpreadOneRound(network, new_active, state, rrset, diffusionState, rand) == 1)
            return 1;
        round++;
    }
    delete [] state;
    return 0;
}

double getrrset(const Network & network, vector<int> rrset, DiffusionState diffusionState, mt19937 rand){
    int centerIndex;
    while(true){
        centerIndex = (int)(floor((double)rand()/rand.max()*network.vertexNum));
        if(!diffusionState.state[centerIndex])
            break;
    }
    if(network.type == "IC" || network.type == "VIC" || network.type == "WC")
        return reSpreadOnce(network, centerIndex, rrset, diffusionState, rand);
    else if(network.type == "LT")
        return 0.;
    else{
        printf("Invalid model\n");
        return 0.;
    }
}

double getrrsets(const Network & network, vector<vector<int>> rrsets, double size, DiffusionState diffusionState, mt19937 rand){
    double t_set = 0.;
    for(int i = 0;i < size;i++){
        vector<int> rrset;
        if(getrrset(network, rrset, diffusionState, rand) == 0)
            rrsets.push_back(rrset);
    }
    return t_set;
}

void simuGreedy_1(const Network & network, DiffusionState diffusionState, vector<int> result, mt19937 rand){
    int c_index = -1;
    double c_profit = -__DBL_MAX__;
    for(int i = 0;i < network.vertexNum;i++){
        DiffusionState temp = DiffusionState(diffusionState);
        vector<int> seed_set;
        seed_set.push_back(i);
        temp.seed(seed_set);
        double t_profit = temp.expInfluenceComplete(network, 1000, rand);
        if(t_profit > c_profit){
            t_profit = c_profit;
            c_index = i;
        }
    }
    result.push_back(c_index);
}

double reverseGreedyLazy_k(const Network & network, DiffusionState diffusionState, vector<int> &result, int k, mt19937 rand){
    int index;
    double profit = 0.;
    vector<vector<int>> rrsets;
    getrrsets(network, rrsets, Policy::rrsets_size, diffusionState, rand);
    map<int, vector<int>> nodes_cover_sets;
    bool coverred_rrsets[rrsets.size()];
    // bool *coverred_rrsets = new bool[rrsets.size()];
    // bool *nodes_cover_sets_key = new bool[network.vertexNum];
    for(int i = 0;i < rrsets.size();i++){
        for(int j = 0;j < rrsets[i].size();j++){
            index = rrsets[i][j];
            if(nodes_cover_sets.find(index) != nodes_cover_sets.end())
                nodes_cover_sets[index].push_back(i);
            else{
                nodes_cover_sets[index] = vector<int>();
                nodes_cover_sets[index].push_back(i);
            }
        }
        coverred_rrsets[i] = false;
    }

    return profit*(network.vertexNum-diffusionState.anum)/Policy::rrsets_size;
}

double reverseGreedyLazyTime_k(const Network & network, DiffusionState diffusionState, vector<int> &result, int k, mt19937 rand){
    int index;
    double profit = 0.;
    vector<vector<int>> rrsets;
    getrrsets(network, rrsets, Policy::rrsets_size, diffusionState, rand);
    map<int, vector<int>> nodes_cover_sets;
    bool coverred_rrsets[rrsets.size()];
    for(int i = 0;i < rrsets.size();i++){
        for(int j = 0;j < rrsets[i].size();j++){
            index = rrsets[i][j];
            if(nodes_cover_sets.find(index) != nodes_cover_sets.end())
                nodes_cover_sets[index].push_back(i);
            else{
                nodes_cover_sets[index] = vector<int>();
                nodes_cover_sets[index].push_back(i);
            }
        }
        coverred_rrsets[i] = false;
    }
    
    return profit*(network.vertexNum-diffusionState.anum)/Policy::rrsets_size;
}

class GreedyPolicy_kd: public Policy{
public:
    vector<int> computeSeedSet(const Network & network, DiffusionState diffusionState, int k, mt19937 rand){
        vector<int> result;
        reverseGreedyLazy_k(network, diffusionState, result, k, rand);
        return result;
    }
};

class GreedyTime: public Policy{
public:
    vector<int> computeSeedSet(const Network & network, DiffusionState diffusionState, int k, mt19937 rand){
        vector<int> result;
        reverseGreedyLazyTime_k(network, diffusionState, result, k, rand);
        return result;
    }
};

class LocalGreedyPolicy_kd: public Policy{
public:
    vector<int> computeSeedSet(const Network & network, DiffusionState diffusionState, int k, mt19937 rand){
        vector<int> result;
        reverseGreedyLazy_k(network, diffusionState, result, k, rand);
        return result;
    }
};

class RandomPolicy_kd: public Policy{
public:
    vector<int> computeSeedSet(const Network & network, DiffusionState diffusionState, int k, mt19937 rand){
        int index;
        vector<int> result;
        while(result.size() < k){
            index = (int)(rand()*network.vertexNum);
            if(!diffusionState.state[index] && find(result.begin(), result.end(), index) != result.end())
                result.push_back(index);
        }
        return result;
    }
};

class DegreePolicy_kd: public Policy{
public:
    vector<int> computeSeedSet(const Network & network, DiffusionState diffusionState, int k, mt19937 rand){
        vector<int> result;
        for(int i = 0;i < network.vertexNum;i++){
            if(!diffusionState.state[network.sorted_degree[i]]){
                result.push_back(network.sorted_degree[i]);
                if(result.size() == k)
                    return result;
                break;
            }
        }
        return result;
    }
};

class GreedyPolicyDynamic: public Policy{
    double select_k(const Network & network, DiffusionState diffusionState, vector<int> result, int k, mt19937 rand){
        reverseGreedyLazy_k(network, diffusionState, result, k, rand);
        double influence = 0.;
        for(int i = 0;i < simurest_times;i++){
            DiffusionState temp(diffusionState);
            temp.seed(result);
            temp.diffuse(network, 1, rand);
            vector<int> temp_seed;
            reverseGreedyLazy_k(network, temp, temp_seed, temp.budget_left, rand);
            temp.seed(temp_seed);
            temp.diffuse(network, temp.round_left, rand);
            influence += temp.anum;
        }
        return influence / simurest_times;
    }
public:
    vector<int> computeSeedSet(const Network & network, DiffusionState diffusionState, int k, mt19937 rand){
        vector<int> result;
        if(diffusionState.budget_left == 0)
            return result;
        if(!diffusionState.round_limit){
            select_k(network, diffusionState, result, 1, rand);
            return result;
        }
        if(diffusionState.round_left == 1){
            select_k(network, diffusionState, result, diffusionState.budget_left, rand);
            return result;
        }
        
        double profit = -__DBL_MAX__;
        for(int k = 1;k < diffusionState.budget_left;k++){
            vector<int> temp_result;
            double temp = select_k(network, diffusionState, temp_result, k, rand);
            if(temp > profit){
                profit = temp;
                result = temp_result;
            }
        }
        return result;
    }
};