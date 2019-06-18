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

    void diffuseOneRound(const Network &network, short *state, std::set<int> &new_active, mt19937 &rand){
        int cseede;
        std::set<int> new_active_temp;
        double prob, rd;
        short *temp_state = new short[vnum];
        memcpy(temp_state, state, vnum*sizeof(short));
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

    void diffuseOneRound(std::set<int> &new_active, short *state, rTuple &rtup){
        short *temp_state = new short[vnum];
        memcpy(temp_state, state, vnum*sizeof(short));
        std::set<int> new_active_temp;
        for(int cseed: new_active){
            for(int cseede: rtup.relations[cseed]){
                if(temp_state[cseede] == -1){
                    state[cseede] = priority(state[cseede], state[cseed], cseede, 1);
                    if(new_active_temp.find(cseede) == new_active_temp.end())
                        new_active_temp.insert(cseede);
                }
            }
        }
        delete [] temp_state;
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

    int reSpreadOnce(const Network &network, int cindex, rTuple &rtup, mt19937 &rand){
        short *state = new short[vnum];
        memcpy(state, seed_state, vnum*sizeof(short));
        std::set<int> new_active;
        state[cindex] = -2;
        rtup.upper.insert(cindex);
        rtup.lower.insert(cindex);
        new_active.insert(cindex);
        while(!new_active.empty())
            reSpreadOneRound(network, new_active, state, rtup, rand);
        delete [] state;
        return 0;
    }

    double getRTuple(const Network &network, rTuple &rtup, mt19937 &rand){
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
            return reSpreadOnce(network, cindex, rtup, rand);
        default:
            std::cout << "invalid model" << std::endl;
        }
        return 0;
    }

public:
    int cnum, vnum, _max;
    short *seed_state;
    std::set<int> seednodes;
    std::map<int,std::set<int>> seedsets;
    std::map<int,std::vector<std::vector<int>>> caspriority;

    DiffusionState_MIC(const Network &network){
        cnum = 0;
        vnum = network.vertexNum;
        _max = 13;
        seed_state = nullptr;
        seed_state = new short[vnum];
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
        seed_state = new short[vnum];
        memcpy(seed_state, diffusionState.seed_state, vnum*sizeof(short));
        seednodes = diffusionState.seednodes;
        seedsets = diffusionState.seedsets;
        caspriority = diffusionState.caspriority;
    }

    ~DiffusionState_MIC(){
        if(seed_state)
            delete [] seed_state;
    }

    void diffuse(const Network &network, int *result, int cindex, int round, mt19937 &rand){
        short *state = new short[vnum];
        memcpy(state, seed_state, vnum*sizeof(short));
        std::set<int> new_active(seednodes);
        for(int i = 0;i < round;i++){
            diffuseOneRound(network, state, new_active, rand);
            if(new_active.empty())  break;
        }
        for(int i = 0;i < vnum;i++)
            if(state[i] == cindex)
                (*result)++;
        delete [] state;
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
        rTuple rt;
        if(rtup.empty())    rtup = std::vector<rTuple>(size);
        else    rtup.resize(size+rtup.size());
        int new_size = rtup.size();
        boost::asio::thread_pool pool(10);
        for(int i = rtup_size;i < new_size;i++){
            auto bind_fn = boost::bind(&DiffusionState_MIC::getRTuple, this, ref(network), ref(rtup[i]), ref(rand));
            boost::asio::post(pool, bind_fn);
        }
        pool.join();
        for(int i = rtup_size;i < new_size;i++)
            if(rtup[i].isdiff)
                countdiff++;
        return (double)countdiff;
    }

    double expInfluenceComplete(const Network &network, int times, int cindex, mt19937 &rand){
        int c_result[times];
        double result = 0.;
        boost::asio::thread_pool pool(10);
        memset(c_result, 0, times*sizeof(int));
        for(int i = 0;i < times;i++){
            auto bind_fn = boost::bind(&DiffusionState_MIC::diffuse, this, ref(network), c_result+i, cindex, vnum, ref(rand));
            boost::asio::post(pool, bind_fn);
        }
        pool.join();
        for(int i = 0;i < times;i++)
            result += c_result[i];
        return result/times;
    }

    bool computeMid_g(const std::set<int> &seed, rTuple &rtup){
        std::set<int> new_active;
        short *state = new short[vnum];
        memset(state, -1, vnum*sizeof(short));
        for(int i: rtup.seed){
            state[i] = seed_state[i];
            if(seed_state[i] > -1)
                new_active.insert(i);
            else{
                std::cout << "ERROR: seed state" << std::endl;
                delete [] state;
                exit(1);
            }
        }
        for(int i: seed){
            if(rtup.upper.find(i) != rtup.upper.end()){
                state[i] = cnum;
                new_active.insert(i);
            }
        }
        for(int i = 0;i < 2*rtup.relations.size();i++){
            if(new_active.find(rtup.node_v) != new_active.end()){
                if(state[rtup.node_v] == cnum){
                    delete [] state;
                    return true;
                }
                delete [] state;
                return false;
            } else if(new_active.empty()){
                delete [] state;
                return false;
            }
            diffuseOneRound(new_active, state, rtup);
        }
        std::cout << "ERROR: unusual exit from DiffusionState_MIC.compute_g" << std::endl;
        delete [] state;
        exit(1);
    }

    int compute_g(const std::set<int> &seed, rTuple &rtup, std::string type, int *result){
        int temp = 0;
        switch(type[0]){
        case 'u':
            if(intersection(seed, rtup.upper))  temp = 1;
            else    temp = -1;
            break;
        case 'm':
            if(intersection(seed, rtup.lower))  temp = 2;
            else if(intersection(seed, rtup.upper)){
                if(computeMid_g(seed, rtup))    temp = 1;
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

    double computeG(std::set<int> &S, std::vector<rTuple> &rtup, int n, std::string type, double *result, mt19937 &rand){
        int count = 0;
        double output;
        for(rTuple &rt: rtup)
            if(compute_g(S, rt, type, nullptr) > 0)
                count++;
        output = (double)n*count/rtup.size();
        if(result)  *result = output;
        return output;
    }
};
