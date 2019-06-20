#pragma once

#include <cstdio>
#include <cstdlib>
#include <set>
#include <map>
#include <vector>
#include <string>
#include <random>
#include <iostream>

#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>
#include <boost/thread.hpp>
#include <boost/bind/bind.hpp>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

#include "network.hpp"
#include "rtuple.hpp"
#include "tools.hpp"

#define THREAD 70

class DiffusionState_MIC{
    short *temp_state_1, *temp_state_2;

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

    void diffuseOneRound(const Network &network, short *state, std::set<int> &new_active, mt19937 &rand, int tid){
        int cseede, base = vnum * tid;
        std::set<int> new_active_temp;
        double prob, rd;
        for(int i = 0;i < vnum;i++) temp_state_2[base+i] = state[i];
        for(int cseed: new_active){
            for(int i = 0;i < network.outDegree[cseed];i++){
                cseede = network.getNeighbor(cseed, i);
                prob = network.getProb(cseed, cseede);
                rd = (double)rand()/rand.max();
                if(rd < prob && temp_state_2[base+cseede] == -1){
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

    void diffuseOneRound(std::set<int> &new_active, short *state, rTuple &rtup, int tid){
        int base = tid * vnum;
        for(int i = 0;i < vnum;i++) temp_state_2[base+i] = state[i];
        std::set<int> new_active_temp;
        for(int cseed: new_active){
            for(int cseede: rtup.relations[cseed]){
                if(temp_state_2[base+cseede] == -1){
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

    int reSpreadOneRound(const Network &network, std::set<int> &new_active, short *state, rTuple &rtup, mt19937 &rand){
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

    int reSpreadOnce(const Network &network, int cindex, rTuple &rtup, mt19937 &rand, int tid){
        int base = vnum * tid;
        for(int i = 0;i < vnum;i++) temp_state_1[base+i] = seed_state[i];
        std::set<int> new_active;
        temp_state_1[base+cindex] = -2;
        rtup.upper.insert(cindex);
        rtup.lower.insert(cindex);
        new_active.insert(cindex);
        while(!new_active.empty())
            reSpreadOneRound(network, new_active, temp_state_1+base, rtup, rand);
        return 0;
    }

    double getRTuple(const Network &network, rTuple &rtup, mt19937 &rand, int tid){
        int cindex = rand()%vnum;
        rtup.clear();
        rtup.node_v = cindex;
        rtup.relations[cindex] = std::set<int>();
        if(seed_state[cindex] != -1){
            rtup.seed.insert(cindex);
            return 0;
        }
        switch(network.type[0]){
        case 'I':
        case 'W':
            return reSpreadOnce(network, cindex, rtup, rand, tid);
        default:
            std::cout << "invalid model" << std::endl;
        }
        return 0;
    }

    template <typename T>
    void freeSpace(T *p){
        if(p)   delete [] p;
    }

public:
    short *seed_state;
    int cnum, vnum, _max;
    std::set<int> seednodes;
    std::map<int,std::set<int>> seedsets;
    std::map<int,std::vector<std::vector<int>>> caspriority;

    DiffusionState_MIC(const Network &network){
        cnum = 0;
        vnum = network.vertexNum;
        _max = 13;
        seed_state = nullptr;
        temp_state_1 = nullptr;
        temp_state_2 = nullptr;
        seed_state = new short[vnum];
        temp_state_1 = new short[vnum*THREAD];
        temp_state_2 = new short[vnum*THREAD];
        memset(seed_state, -1, vnum*sizeof(short));
        
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
        temp_state_1 = nullptr;
        temp_state_2 = nullptr;
        seed_state = new short[vnum];
        memcpy(seed_state, diffusionState.seed_state, vnum*sizeof(short));
        temp_state_1 = new short[vnum*THREAD];
        temp_state_2 = new short[vnum*THREAD];
        seednodes = diffusionState.seednodes;
        seedsets = diffusionState.seedsets;
        caspriority = diffusionState.caspriority;
    }

    ~DiffusionState_MIC(){
        freeSpace(seed_state);
        freeSpace(temp_state_1);
        freeSpace(temp_state_2);
    }

    void diffuse(const Network &network, int *result, int cindex, int round, mt19937 &rand, int tid){
        auto start = std::chrono::high_resolution_clock::now();
        int base = vnum * tid;
        for(int i = 0;i < vnum;i++) temp_state_1[base+i] = seed_state[i];
        std::set<int> new_active(seednodes);
        for(int i = 0;i < round;i++){
            diffuseOneRound(network, temp_state_1+base, new_active, rand, tid);
            if(new_active.empty())  break;
        }
        for(int i = 0;i < vnum;i++)
            if(temp_state_1[base+i] == cindex)
                (*result)++;
        printTime(start, std::chrono::high_resolution_clock::now());
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
        int rtup_size = rtup.size();
        if(rtup.empty())    rtup = std::vector<rTuple>(size);
        else    rtup.resize(size+rtup.size());
        int new_size = rtup.size();
        for(int i = rtup_size;i < new_size;){
            boost::asio::thread_pool pool(THREAD);
            for(int j = 0;j < THREAD && i < new_size;j++, i++){
                auto bind_fn = boost::bind(&DiffusionState_MIC::getRTuple, this, ref(network), ref(rtup[i]), ref(rand), j);
                boost::asio::post(pool, bind_fn);
            }
            pool.join();
        }
        for(int i = rtup_size;i < new_size;i++)
            if(rtup[i].isdiff)
                countdiff++;
        return (double)countdiff;
    }

    double expInfluenceComplete(const Network &network, int times, int cindex, mt19937 &rand){
        int c_result[THREAD];
        double result = 0.;
        memset(c_result, 0, THREAD*sizeof(int));
        for(int i = 0;i < times;){
            boost::asio::thread_pool pool(THREAD);
            for(int j = 0;j < THREAD && i < times;j++,i++){
                auto bind_fn = boost::bind(&DiffusionState_MIC::diffuse, this, ref(network), c_result+j, cindex, vnum, ref(rand), j);
                boost::asio::post(pool, bind_fn);
            }
            pool.join();
            for(int j = 0;j < THREAD;j++){
                result += c_result[j];
                c_result[j] = 0;
            }
        }
        return result/times;
    }

    bool computeMid_g(const std::set<int> &seed, rTuple &rtup, int tid){
        std::set<int> new_active;
        int base = vnum * tid;
        for(int i: rtup.seed){
            temp_state_1[base+i] = seed_state[i];
            if(seed_state[i] > -1)
                new_active.insert(i);
            else{
                std::cout << "ERROR: seed state" << std::endl;
                exit(1);
            }
        }
        for(int i: seed){
            if(rtup.upper.find(i) != rtup.upper.end()){
                temp_state_1[base+i] = cnum;
                new_active.insert(i);
            }
        }
        for(int i = 0;i < 2*rtup.relations.size();i++){
            if(new_active.find(rtup.node_v) != new_active.end()){
                if(temp_state_1[base+rtup.node_v] == cnum)   return true;
                return false;
            } else if(new_active.empty())
                return false;
            diffuseOneRound(new_active, temp_state_1+base, rtup, tid);
        }
        std::cout << "ERROR: unusual exit from DiffusionState_MIC.compute_g" << std::endl;
        exit(1);
    }

    int compute_g(const std::set<int> &seed, rTuple &rtup, const std::string &type, int *result, int tid){
        int temp = 0;
        switch(type[0]){
        case 'u':
            if(intersection(seed, rtup.upper))  temp = 1;
            else    temp = -1;
            break;
        case 'm':
            if(intersection(seed, rtup.lower))  temp = 2;
            else if(intersection(seed, rtup.upper)){
                if(computeMid_g(seed, rtup, tid))    temp = 1;
                else    temp = -1;
            } else  temp = -2;
            break;
        case 'l':
            if(intersection(seed, rtup.lower))  temp = 2;
            else    temp = -2;
            break;
        }
        if(result)  *result = temp;
        return temp;
    }

    double computeG(std::set<int> &S, std::vector<rTuple> &rtup, int n, const std::string &type, double *result, mt19937 &rand){
        int count = 0;
        double output;
        for(rTuple &rt: rtup)
            if(compute_g(S, rt, type, nullptr, 0) > 0)
                count++;
        output = (double)n*count/rtup.size();
        if(result)  *result = output;
        return output;
    }
};
