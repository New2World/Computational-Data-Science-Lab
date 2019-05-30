#pragma once
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

#include <boost/thread.hpp>
#include <boost/bind/bind.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>

using namespace std;

#include "Policy.hpp"
#include "Network.hpp"
#include "DiffusionState.hpp"

// #define PARALLEL

class SeedingProcessTime{
    static void goDynamic(const Network &network, GreedyPolicyDynamic policy, int round, int budget, vector<double> &record, vector<int> &record_budget, double *result, int tid){
        mt19937 rand(chrono::high_resolution_clock::now().time_since_epoch().count());
        DiffusionState diffusionState(network, round, budget);
        double influence = 0.;
        for(int i = 0;i < round;i++){
            // cout << "dynamic running " << tid+1 << endl;
            vector<int> seed_set;
            seed_set = policy.computeSeedSet(network, diffusionState, 0, rand);
            diffusionState.seed(seed_set);
            influence = diffusionState.diffuse(network, 1, rand);
            record[i] += diffusionState.anum;
            record_budget[i] += seed_set.size();
        }
        *result = influence;
    }

    static void goUniform_d(const Network &network, GreedyPolicy_kd policy, int round, int d, int budget, vector<double> &record, vector<int> &record_budget, double *result, int tid){
        mt19937 rand(chrono::high_resolution_clock::now().time_since_epoch().count());
        DiffusionState diffusionState(network, round, budget);
        double influence = 0.;
        for(int i = 0;i < round;i++){
            // cout << "uniform_d running " << tid+1 << endl;
            vector<int> seed_set;
            if(i == round - 1 && diffusionState.budget_left > 0){
                seed_set = policy.computeSeedSet(network, diffusionState, diffusionState.budget_left, rand);
                diffusionState.seed(seed_set);
            }
            else if(i % d == 0 && diffusionState.budget_left > 0){
                seed_set = policy.computeSeedSet(network, diffusionState, min(budget*d/round, diffusionState.budget_left), rand);
                diffusionState.seed(seed_set);
            }
            influence = diffusionState.diffuse(network, 1, rand);
            record[i] += diffusionState.anum;
            record_budget[i] += seed_set.size();
        }
        *result = influence;
    }

    static void goStatic(const Network &network, GreedyPolicy_kd policy, int round, int budget, vector<double> &record, vector<int> &record_budget, double *result, int tid){
        mt19937 rand(chrono::high_resolution_clock::now().time_since_epoch().count());
        DiffusionState diffusionState(network, round, budget);
        double influence = 0.;
        for(int i = 0;i < round;i++){
            // cout << "static running " << tid+1 << endl;
            vector<int> seed_set;
            if(i == 0){
                seed_set = policy.computeSeedSet(network, diffusionState, budget, rand);
                diffusionState.seed(seed_set);
            }
            influence = diffusionState.diffuse(network, 1, rand);
            record[i] += diffusionState.anum;
            record_budget[i] += seed_set.size();
        }
        *result = influence;
    }

    static void goFull(const Network &network, GreedyPolicy_kd policy, int round, int budget, vector<double> &record, vector<int> &record_budget, double *result, int tid){
        mt19937 rand(chrono::high_resolution_clock::now().time_since_epoch().count());
        DiffusionState diffusionState(network, round, budget);
        double influence = 0.;
        for(int i = 0;i < round;i++){
            // cout << "full running " << tid+1 << endl;
            vector<int> seed_set;
            if(i == round-1 && diffusionState.budget_left > 0){
                seed_set = policy.computeSeedSet(network, diffusionState, diffusionState.budget_left, rand);
                diffusionState.seed(seed_set);
            }
            if(i < round-1 && diffusionState.new_active.size() == 0){
                seed_set = policy.computeSeedSet(network, diffusionState, 1, rand);
                diffusionState.seed(seed_set);
            }
            influence = diffusionState.diffuse(network, 1, rand);
            record[i] += diffusionState.anum;
            record_budget[i] += seed_set.size();
        }
        *result = influence;
    }
public:
    static int round;

    static void MultiGo(const Network &network, int simutimes, int budget, vector<double> &record, vector<int> &record_budget, const string type, int d){
        printf("Multi Go\n");
        if(round == -1){
            throw "round == -1";
            return;
        }
        double result = 0.;
        double results[simutimes];
        vector<vector<double>> records(simutimes, vector<double>(round, 0.));
        vector<vector<int>> records_budget(simutimes, vector<int>(round, 0));
        #ifdef PARALLEL
        boost::asio::thread_pool pool(6);
        for(int i = 0;i < simutimes;i++){
            printf("Simulation number %d\n", i+1);
            switch(type[0]){
            case 'd':
                boost::asio::post(pool, boost::bind(goDynamic, network, GreedyPolicyDynamic(), round, budget, ref(records[i]), ref(records_budget[i]), results+i, i));
                break;
            case 's':
                boost::asio::post(pool, boost::bind(goStatic, network, GreedyPolicy_kd(), round, budget, ref(records[i]), ref(records_budget[i]), results+i, i));
                break;
            case 'u':
                boost::asio::post(pool, boost::bind(goUniform_d, network, GreedyPolicy_kd(), round, d, budget, ref(records[i]), ref(records_budget[i]), results+i, i));
                break;
            case 'f':
                boost::asio::post(pool, boost::bind(goFull, network, GreedyPolicy_kd(), round, budget, ref(records[i]), ref(records_budget[i]), results+i, i));
                break;
            default:
                printf("Invalid model\n");
            }
        }
        pool.join();
        #else
        for(int i = 0;i < simutimes;i++){
            printf("Simulation number %d\n", i+1);
            switch(type[0]){
            case 'd':
                goDynamic(network, GreedyPolicyDynamic(), round, budget, records[i], records_budget[i], results+i, i);
                break;
            case 's':
                goStatic(network, GreedyPolicy_kd(), round, budget, records[i], records_budget[i], results+i, i);
                break;
            case 'u':
                goUniform_d(network, GreedyPolicy_kd(), round, d, budget, records[i], records_budget[i], results+i, i);
                break;
            case 'f':
                goFull(network, GreedyPolicy_kd(), round, budget, records[i], records_budget[i], results+i, i);
                break;
            default:
                printf("Invalid model\n");
            }
        }
        #endif
        for(int i = 0;i < round;i++)
            result += results[i];
        for(int i = 0;i < round;i++){
            for(int j = 0;j < simutimes;j++){
                record[i] += records[j][i];
                record_budget[i] += records_budget[j][i];
            }
            record[i] /= simutimes;
            record_budget[i] /= simutimes;
        }
        printf("%lf\n", result/simutimes);
    }
};

int SeedingProcessTime::round = -1;
