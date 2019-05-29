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

class SeedingProcessTime{
    static void goDynamic(const Network &network, Policy policy, int round, int budget, vector<double> &record, vector<int> &record_budget, double *result, int tid){
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

    static void goUniform_d(const Network &network, Policy policy, int round, int d, int budget, vector<double> &record, vector<int> &record_budget, double *result, int tid){
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

    static void goStatic(const Network &network, Policy policy, int round, int budget, vector<double> &record, vector<int> &record_budget, double *result, int tid){
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

    static void goFull(const Network &network, Policy policy, int round, int budget, vector<double> &record, vector<int> &record_budget, double *result, int tid){
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

    static void MultiGo(const Network &network, Policy policy, int simutimes, int budget, vector<double> &record, vector<int> &record_budget, const string type, int d){
        printf("Multi Go\n");
        if(round == -1){
            throw "round == -1";
            return;
        }
        double result = 0.;
        double *results = new double[simutimes];
        memset(results, 0, simutimes*sizeof(double));
        vector<vector<double>> records(simutimes, vector<double>(round, 0.));
        vector<vector<int>> records_budget(simutimes, vector<int>(round, 0));
        boost::asio::thread_pool pool(6);
        for(int i = 0;i < simutimes;i++){
            printf("Simulation number %d\n", i+1);
            if(type == "dynamic")
                boost::asio::post(pool, boost::bind(goDynamic, network, policy, round, budget, ref(records[i]), ref(records_budget[i]), results+i, i));
            else if(type == "static")
                boost::asio::post(pool, boost::bind(goStatic, network, policy, round, budget, ref(records[i]), ref(records_budget[i]), results+i, i));
            else if(type == "uniform")
                boost::asio::post(pool, boost::bind(goUniform_d, network, policy, round, d, budget, ref(records[i]), ref(records_budget[i]), results+i, i));
            else if(type == "full")
                boost::asio::post(pool, boost::bind(goFull, network, policy, round, budget, ref(records[i]), ref(records_budget[i]), results+i, i));
            else
                printf("Invalid model\n");
        }
        pool.join();
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
        delete [] results;
    }
};

int SeedingProcessTime::round = -1;