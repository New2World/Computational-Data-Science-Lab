#pragma once
#include <cstdlib>
#include <vector>
#include <random>

#include "Policy.hpp"
#include "Network.hpp"

class DiffusionState{
    void diffuseOneRound(const Network &network, mt19937& rand){
        vector<int> new_active_temp;
        double prob;
        for(int cseed: new_active){
            for(int cseede: network.neighbor[cseed]){
                prob = network.getProb(cseed, cseede);
                if((double)rand()/rand.max() < prob){
                    if(!state[cseede]){
                        state[cseede] = true;
                        anum++;
                        new_active_temp.push_back(cseede);
                    }
                }
            }
        }
        new_active.clear();
        for(int i: new_active_temp)
            new_active.push_back(i);
        
        round_left--;
    }

public:
    int round_left, budget_left, anum, vnum;
    bool *state;
    vector<int> new_active;
    bool round_limit, budget_limit;

    DiffusionState(const Network &network, int round_left, int budget_left){
        anum = 0;
        state = nullptr;
        this->round_left = round_left;
        this->budget_left = budget_left;
        round_limit = true;
        budget_limit = true;
        vnum = network.vertexNum;
        state = new bool[vnum];
        memset(state, false, vnum);
        if(round_left == vnum)
            round_limit = false;
        if(budget_left == -1)
            budget_limit = false;
    }

    DiffusionState(const DiffusionState &diffusionState){
        round_left = diffusionState.round_left;
        budget_left = diffusionState.budget_left;
        state = nullptr;
        state = new bool[diffusionState.vnum];
        memcpy(state, diffusionState.state, diffusionState.vnum);
        new_active = diffusionState.new_active;
        round_limit = diffusionState.round_limit;
        budget_limit = diffusionState.budget_limit;
        anum = diffusionState.anum;
        vnum = diffusionState.vnum;
    }

    ~DiffusionState(){
        if(state)
            delete [] state;
    }

    double diffuse(const Network &network, int round, mt19937& rand){
        for(int i = 0;i < round;i++){
            diffuseOneRound(network, rand);
            if(new_active.size() == 0)
                break;
        }
        return anum;
    }

    int diffuse(const Network &network, int round, vector<double> &record, int bound, mt19937& rand){
        double last = record[record.size()-1];
        for(int i = 0;i < round;i++){
            diffuseOneRound(network, rand);
            if(i < record.size())
                record[i] += anum;
            else
                record.insert(record.begin()+i, last+anum);
            if(new_active.size() == 0)
                return i-1;
        }
        return round-1;
    }

    void seed(vector<int> seed_set){
        cout << seed_set.size() << " > " << budget_left << endl;
        if(seed_set.size() > budget_left && budget_limit){
            cout << "diffusionState.seed over budget: " << seed_set.size() << " - " << budget_left << endl;
            exit(1);
        }
        if(budget_left == 0)
            return;
        for(int i: seed_set){
            if(!state[i]){
                state[i] = true;
                new_active.push_back(i);
                anum++;
                budget_left--;
            }
            else
                cout << "diffusionState.seed: seeding an active node" << endl;
        }
    }

    double expInfluenceComplete(const Network &network, int times, mt19937& rand){
        double result = 0.;
        for(int i = 0;i < times;i++){
            DiffusionState temp = DiffusionState(*this);
            result += temp.diffuse(network, temp.vnum, rand);
        }
        return result / times;
    }
};
