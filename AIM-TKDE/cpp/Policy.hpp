#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <random>
#include <algorithm>
#include <functional>

using namespace std;

#include "sortedMap.hpp"
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

    virtual vector<int> computeSeedSet(const Network &, DiffusionState&, int, mt19937&){
        return {};
    }
};

int Policy::rrsets_size = 100000;
int Policy::simurest_times = 100;

int reSpreadOneRound(const Network & network, vector<int> &new_active, bool *state, vector<int> &rrset, DiffusionState& diffusionState, mt19937& rand){
    int cseede;
    double prob;
    vector<int> new_active_temp;

    for(int cseed: new_active){
        for(int i = 0;i < network.inDegree[cseed];i++){
            cseede = network.getReverseNeighbor(cseed, i);
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

int reSpreadOnce(const Network & network, int cindex, vector<int> &rrset, DiffusionState& diffusionState, mt19937& rand){
    bool state[diffusionState.vnum];
    memcpy(state, diffusionState.state, sizeof(state));
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
    return 0;
}

double getrrset(const Network & network, vector<int> &rrset, DiffusionState& diffusionState, mt19937& rand){
    int centerIndex, vnum = network.vertexNum;
    while(true){
        centerIndex = rand()*vnum/rand.max()-1;
        centerIndex = centerIndex<0?0:centerIndex;
        if(!diffusionState.state[centerIndex])
            break;
    }
    switch(network.type[0]){
    case 'I':    // IC
    case 'V':    // VIC
    case 'W':    // WC
        return reSpreadOnce(network, centerIndex, rrset, diffusionState, rand);
    case 'L':    // LT
        return 0.;
    default:
        cout << "Invalid model" << endl;
        return 0.;
    }
}

double getrrsets(const Network & network, vector<vector<int>> &rrsets, double size, DiffusionState& diffusionState, mt19937& rand){
    double t_set = 0.;
    vector<int> rrset;
    for(int i = 0;i < size;i++){
        rrset.clear();
        if(getrrset(network, rrset, diffusionState, rand) == 0)
            rrsets[i] = rrset;
    }
    return t_set;
}

void simuGreedy_1(const Network & network, DiffusionState& diffusionState, vector<int> &result, mt19937& rand){
    int c_index = -1;
    double c_profit = -__DBL_MAX__, t_profit;
    vector<int> seed_set;
    for(int i = 0;i < network.vertexNum;i++){
        DiffusionState temp = DiffusionState(diffusionState);
        seed_set.clear();
        seed_set.push_back(i);
        temp.seed(seed_set);
        t_profit = temp.expInfluenceComplete(network, 1000, rand);
        if(t_profit > c_profit){
            t_profit = c_profit;
            c_index = i;
        }
    }
    result.push_back(c_index);
}

double reverseGreedyLazy_k(const Network & network, DiffusionState& diffusionState, vector<int> &result, int k, mt19937& rand){
    int index;
    double profit = 0.;
    vector<vector<int>> rrsets(Policy::rrsets_size);
    getrrsets(network, rrsets, Policy::rrsets_size, diffusionState, rand);
    map<int, vector<int>> nodes_cover_sets;
    bool nodes_cover_sets_key[network.vertexNum];
    bool coverred_rrsets[rrsets.size()];
    for(int i = 0;i < rrsets.size();i++){
        for(int j = 0;j < rrsets[i].size();j++){
            index = rrsets[i][j];
            if(!nodes_cover_sets_key[index]){
                nodes_cover_sets[index] = vector<int>();
                nodes_cover_sets_key[index] = true;
            }
            nodes_cover_sets[index].push_back(i);
        }
        coverred_rrsets[i] = false;
    }
    vector<pair<int,int>> sortPair;
    for(auto i = nodes_cover_sets.begin();i != nodes_cover_sets.end();i++)
        sortPair.push_back(make_pair(i->first, i->second.size()));
    sort(sortPair.begin(), sortPair.end(), [](auto a, auto b)-> bool {
        return a.second > b.second;
    });
    sortedMap mymap;
    for(pair<int,int> p: sortPair)
        mymap.push_back(p.first, p.second);
    int c_bound, t_bound, c_seed;
    bool sign;
    vector<int> c_seed_cover;
    for(int i = 0;i < k;i++){
        sign = false;
        c_bound = mymap.size();
        while(c_bound > 0){
            c_seed = mymap.get(0);
            c_seed_cover = nodes_cover_sets[c_seed];
            for(int j = 0;j < c_seed_cover.size();j++){
                if(coverred_rrsets[c_seed_cover[j]]){
                    c_seed_cover.erase(c_seed_cover.begin()+j);
                    j--;
                }
            }
            t_bound = mymap.update(c_seed, c_seed_cover.size());
            if(t_bound == 0){
                result.push_back(c_seed);
                sign = true;
                for(int j = 0;j < c_seed_cover.size();j++){
                    if(coverred_rrsets[c_seed_cover[j]]){
                        cout << "greedy update may wrong" << endl;
                        exit(1);
                    }
                    else
                        coverred_rrsets[c_seed_cover[j]] = true;
                }
                break;
            }
            if(t_bound < c_bound)
                c_bound = t_bound;
        }
        if(!sign){
            cout << "greedy lazy: no node selected" << endl;
            exit(1);
        }
    }
    return profit*(network.vertexNum-diffusionState.anum)/Policy::rrsets_size;
}

double reverseGreedyLazyTime_k(const Network & network, DiffusionState& diffusionState, vector<int> &result, int k, mt19937& rand){
    int index;
    double profit = 0.;
    vector<vector<int>> rrsets(Policy::rrsets_size);
    getrrsets(network, rrsets, Policy::rrsets_size, diffusionState, rand);
    map<int, vector<int>> nodes_cover_sets;
    bool nodes_cover_sets_key[network.vertexNum];
    bool coverred_rrsets[rrsets.size()];
    for(int i = 0;i < rrsets.size();i++){
        for(int j = 0;j < rrsets[i].size();j++){
            index = rrsets[i][j];
            if(!nodes_cover_sets_key[index]){
                nodes_cover_sets[index] = vector<int>();
                nodes_cover_sets_key[index] = true;
            }
            nodes_cover_sets[index].push_back(i);
        }
        coverred_rrsets[i] = false;
    }
    vector<pair<int,int>> sortPair;
    for(auto i = nodes_cover_sets.begin();i != nodes_cover_sets.end();i++)
        sortPair.push_back(make_pair(i->first, i->second.size()));
    sort(sortPair.begin(), sortPair.end(), [](auto a, auto b)-> bool {
        return a.second > b.second;
    });
    sortedMap mymap;
    for(pair<int,int> p: sortPair)
        mymap.push_back(p.first, p.second);
    int c_bound, t_bound, c_seed;
    bool sign;
    vector<int> c_seed_cover;
    for(int i = 0;i < k;i++){
        sign = false;
        c_bound = mymap.size();
        while(c_bound > 0){
            c_seed = mymap.get(0);
            c_seed_cover = nodes_cover_sets[c_seed];
            for(int j = 0;j < c_seed_cover.size();j++){
                if(coverred_rrsets[c_seed_cover[j]]){
                    c_seed_cover.erase(c_seed_cover.begin()+j);
                    j--;
                }
            }
            t_bound = mymap.update(c_seed, c_seed_cover.size());
            if(t_bound == 0){
                result.push_back(c_seed);
                sign = true;
                for(int j = 0;j < c_seed_cover.size();j++){
                    if(coverred_rrsets[c_seed_cover[j]]){
                        cout << "greedy update may wrong" << endl;
                        exit(1);
                    }
                    else
                        coverred_rrsets[c_seed_cover[j]] = true;
                }
                break;
            }
            if(t_bound < c_bound)
                c_bound = t_bound;
        }
        if(!sign){
            cout << "greedy lazy: no node selected" << endl;
            exit(1);
        }
    }
    return profit*(network.vertexNum-diffusionState.anum)/Policy::rrsets_size;
}

class GreedyPolicy_kd: public Policy{
public:
    vector<int> computeSeedSet(const Network & network, DiffusionState& diffusionState, int k, mt19937& rand){
        vector<int> result;
        reverseGreedyLazy_k(network, diffusionState, result, k, rand);
        return result;
    }
};

class GreedyTime: public Policy{
public:
    vector<int> computeSeedSet(const Network & network, DiffusionState& diffusionState, int k, mt19937& rand){
        vector<int> result;
        reverseGreedyLazyTime_k(network, diffusionState, result, k, rand);
        return result;
    }
};

class LocalGreedyPolicy_kd: public Policy{
public:
    vector<int> computeSeedSet(const Network & network, DiffusionState& diffusionState, int k, mt19937& rand){
        vector<int> result;
        reverseGreedyLazy_k(network, diffusionState, result, k, rand);
        return result;
    }
};

class RandomPolicy_kd: public Policy{
public:
    vector<int> computeSeedSet(const Network & network, DiffusionState& diffusionState, int k, mt19937& rand){
        int index;
        vector<int> result;
        while(result.size() < k){
            index = rand()*network.vertexNum/rand.max()-1;
            index = index<0?0:index;
            if(!diffusionState.state[index] && find(result.begin(), result.end(), index) != result.end())
                result.push_back(index);
        }
        return result;
    }
};

class DegreePolicy_kd: public Policy{
public:
    vector<int> computeSeedSet(const Network & network, DiffusionState& diffusionState, int k, mt19937& rand){
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
    double select_k(const Network & network, DiffusionState& diffusionState, vector<int> &result, int k, mt19937& rand){
        reverseGreedyLazy_k(network, diffusionState, result, k, rand);
        double influence = 0.;
        vector<int> temp_seed;
        for(int i = 0;i < simurest_times;i++){
            DiffusionState temp(diffusionState);
            temp.seed(result);
            temp.diffuse(network, 1, rand);
            temp_seed.clear();
            reverseGreedyLazy_k(network, temp, temp_seed, temp.budget_left, rand);
            temp.seed(temp_seed);
            temp.diffuse(network, temp.round_left, rand);
            influence += temp.anum;
        }
        return influence / simurest_times;
    }
public:
    vector<int> computeSeedSet(const Network & network, DiffusionState& diffusionState, int k, mt19937& rand){
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

        double profit = -__DBL_MAX__, temp;
        vector<int> temp_result;
        for(int k = 1;k < diffusionState.budget_left;k++){
            temp_result.clear();
            temp = select_k(network, diffusionState, temp_result, k, rand);
            if(temp > profit){
                profit = temp;
                result = temp_result;
            }
        }
        return result;
    }
};
