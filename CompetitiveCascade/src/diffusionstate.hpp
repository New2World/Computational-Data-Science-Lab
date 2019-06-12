#pragma once

#include <cstdio>
#include <cstdlib>
#include <set>
#include <map>
#include <vector>
#include <string>
#include <random>
#include <iostream>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include "network.hpp"
#include "rtuple.hpp"
#include "tools.hpp"

class DiffusionState_MIC{
    void readPriority(int num){
        std::string path = "../data/perms/"+std::to_string(num)+"_perms.txt";
        char line[50];
        int index = 0;
        FILE *fd = fopen(path.c_str(), "r");
        std::vector<std::vector<int>> temp;
        while((NULL != fgets(line, 50, fd)) && index < _max){
            std::vector<std::string> inStr;
            std::vector<int> tempp;
            boost::split(inStr, line, boost::is_any_of(" "));
            for(std::string str: inStr){
                if(str == "\r\n" || str == "\n")    break;
                tempp.push_back(std::stoi(str));
            }
            temp.push_back(tempp);
            index++;
        }
        caspriority[num] = temp;
        fclose(fd);
    }

    int priority(int a, int b, int node, int shift=0){
        std::vector<int> priority = caspriority[cnum+shift][node%_max];
        if(priority[a] > priority[b])   return a;
        return b;
    }

    void diffuseOneRound(const Network &network, int *state, std::set<int> &new_active, mt19937 &rand){
        int cseede, temp_state[vnum];
        memcpy(temp_state, state, vnum*sizeof(int));
        std::set<int> new_active_temp;
        double prob, rd;
        for(int cseed: new_active){
            for(int i = 0;i < network.outDegree[cseed];i++){
                cseede = network.getNeighbor(cseed, i);
                prob = network.getProb(cseed, cseede);
                rd = (double)rand()/rand.max();
                if(rd < prob && temp_state[cseede] == -1){
                    state[cseede] = priority(state[cseede], state[cseed], cseede);
                    if(new_active_temp.find(cseede) == new_active_temp.end())
                        new_active_temp.insert(cseede);
                }
            }
        }
        new_active.clear();
        for(int i: new_active_temp)
            new_active.insert(i);
    }

    void diffuseOneRound(std::set<int> &new_active, int *state, rTuple &rtup){
        int temp_state[vnum];
        memcpy(temp_state, state, vnum*sizeof(int));
        std::set<int> new_active_temp;
        for(int cseed: new_active){
            for(int cseede: rtup.relations[cseed]){
                if(state[cseede] == -1){
                    state[cseede] = priority(state[cseede], state[cseed], cseede, 1);
                    if(new_active_temp.find(cseede) == new_active_temp.end())
                        new_active_temp.insert(cseede);
                }
            }
        }
        new_active.clear();
        for(int i: new_active_temp)
            new_active.insert(i);
    }

    int reSpreadOneRound(const Network &network, std::set<int> &new_active, int *state, rTuple &rtup, mt19937 &rand){
        int cseede;
        double prob, rd;
        std::set<int> new_active_temp;
        for(int cseed: new_active){
            for(int j = 0;j < network.inDegree[cseed];j++){
                cseede = network.getReverseNeighbor(cseed, j);
                prob = network.getProb(cseede, cseed);
                if((double)rand()/rand.max() < prob && state[cseede] != -2){
                    if(rtup.relations.find(cseede) == rtup.relations.end()){
                        rtup.relations[cseede] = std::set<int>();
                        new_active_temp.insert(cseede);
                    }
                    rtup.relations[cseede].insert(cseed);
                }
            }
        }
        new_active.clear();
        bool islast = false;
        for(int i: new_active_temp){
            if(state[i] >= 0){
                islast = true;
                rtup.seed.insert(i);
            } else {
                new_active.insert(i);
                rtup.upper.insert(i);
                state[i] = -2;
            }
        }
        if(islast){
            new_active.clear();
            rtup.isdiff = true;
        } else
            for(int i: new_active_temp)
                rtup.lower.insert(i);
        return 0;
    }

    int reSpreadOnce(const Network &network, int cindex, rTuple &rtup, mt19937 &rand){
        int state[vnum];
        memcpy(state, seed_state, sizeof(state));
        std::set<int> new_active;
        state[cindex] = -2;
        rtup.upper.insert(cindex);
        rtup.lower.insert(cindex);
        new_active.insert(cindex);
        while(!new_active.empty())
            reSpreadOneRound(network, new_active, state, rtup, rand);
        // printContainer(rtup.upper);
        return 0;
    }

    double getRTuple(const Network &network, rTuple &rtup, mt19937 &rand){
        int cindex = rand()*vnum/rand.max()-1;
        cindex = cindex<0?0:cindex;
        rtup.node_v = cindex;
        rtup.relations[cindex] = std::set<int>();
        if(seed_state[cindex] != -1){
            rtup.seed.insert(cindex);
            return 0;
        }
        switch(network.type[0]){
        case 'I':
        case 'W':
            return reSpreadOnce(network, cindex, rtup, rand);
        default:
            std::cout << "invalid model" << std::endl;
        }
        return 0;
    }

public:
    int cnum, vnum, _max;
    int *seed_state;
    std::set<int> seednodes;
    std::map<int,std::set<int>> seedsets;
    std::map<int,std::vector<std::vector<int>>> caspriority;

    DiffusionState_MIC(const Network &network){
        cnum = 0;
        vnum = network.vertexNum;
        _max = 13;
        seed_state = nullptr;
        seed_state = new int[vnum];
        for(int i = 0;i < vnum;i++) seed_state[i] = -1;
        
        readPriority(1);
        readPriority(3);
        readPriority(5);
        readPriority(10);
    }

    DiffusionState_MIC(const DiffusionState_MIC &diffusionState){
        cnum = diffusionState.cnum;
        _max = diffusionState._max;
        vnum = diffusionState.vnum;
        seed_state = nullptr;
        seed_state = new int[vnum];
        memcpy(seed_state, diffusionState.seed_state, vnum*sizeof(int));
        seednodes = diffusionState.seednodes;
        seedsets = diffusionState.seedsets;
        caspriority = diffusionState.caspriority;
    }

    ~DiffusionState_MIC(){
        if(seed_state)
            delete [] seed_state;
    }

    void diffuse(const Network &network, int *result, int round, mt19937 &rand){
        int state[vnum];
        memcpy(state, seed_state, sizeof(state));
        std::set<int> new_active(seednodes);
        for(int i = 0;i < round;i++){
            diffuseOneRound(network, state, new_active, rand);
            if(new_active.size() == 0)  break;
        }
        for(int i = 0;i < vnum;i++)
            if(state[i] > -1)
                result[state[i]]++;
    }

    int seed(const std::set<int> &seed_set){
        std::set<int> new_seed;
        for(int i: seed_set){
            if(seed_state[i] == -1){
                seed_state[i] = cnum;
                seednodes.insert(i);
                new_seed.insert(i);
            }
            else
                cout << "diffusionState.seed: seeding an active node" << endl;
        }
        seedsets[cnum++] = new_seed;
        return cnum-1;
    }

    void removeSeed(int cindex){
        if(seedsets.find(cindex) == seedsets.end()){
            std::cout << "WARNING: removing cascade does not exist" << std::endl;
            return;
        }
        for(int i: seedsets[cindex]){
            seed_state[i] = -1;
            seednodes.erase(i);
        }
        seedsets.erase(cindex);
        cnum--;
    }

    double getRTuples(const Network &network, std::vector<rTuple> &rtup, double size, mt19937 &rand){
        int countdiff = 0;
        for(int i = 0;i < size;i++){
            rTuple rt;
            getRTuple(network, rt, rand);
            rtup.push_back(rt);
            if(rt.isdiff)   countdiff++;
        }
        return (double)countdiff;
    }

    double expInfluenceComplete(const Network &network, int times, int cindex, mt19937 &rand){
        int c_result[cnum];
        double result[cnum];
        memset(result, 0, sizeof(result));
        for(int i = 0;i < times;i++){
            memset(c_result, 0, sizeof(c_result));
            diffuse(network, c_result, vnum, rand);
            // std::cout << c_result[cindex] << std::endl;
            for(int j = 0;j < cnum;j++)
                result[j] += c_result[j];
        }
        // for(int i = 0;i < cnum;i++)
        //     result[i] /= times;
        return result[cindex]/times;
    }

    bool computeG(const std::set<int> &seed, rTuple &rtup){
        std::set<int> new_active;
        int state[vnum];
        for(int i = 0;i < vnum;i++) state[i] = -1;
        for(int i: rtup.upper){
            state[i] = seed_state[i];
            if(seed_state[i] > -1){
                std::cout << "upper insert" << std::endl;
                new_active.insert(i);
            }
        }
        for(int i: rtup.seed){
            state[i] = seed_state[i];
            if(seed_state[i] > -1)
                new_active.insert(i);
            else{
                std::cout << "ERROR: seed state" << std::endl;
                exit(1);
            }
        }
        int index = cnum;
        for(int i: seed){
            if(rtup.upper.find(i) != rtup.upper.end()){
                state[i] = cnum;
                new_active.insert(i);
            }
        }
        for(int i = 0;i < 2*rtup.relations.size();i++){
            if(new_active.find(rtup.node_v) != new_active.end()){
                if(state[rtup.node_v] == cnum) return true;
                else    return false;
            } else
                if(new_active.empty())  return false;
            diffuseOneRound(new_active, state, rtup);
        }
        std::cout << "ERROR: unusual exit from DiffusionState_MIC.computeG" << std::endl;
        exit(1);
    }
};
