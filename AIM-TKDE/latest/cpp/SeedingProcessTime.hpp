#pragma once
#include <cstdio>
#include <cstdlib>
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
    static void goDynamic(Network network, Policy policy, int round, int budget, vector<double> &record, vector<int> &record_budget, double *result, int tid){
        mt19937 rand(chrono::high_resolution_clock::now().time_since_epoch().count());
        DiffusionState diffusionState(network, round, budget);
        double influence = 0.;
        for(int i = 0;i < round;i++){
            vector<int> seed_set;
            seed_set = policy.computeSeedSet(network, diffusionState, 0, rand);
            diffusionState.seed(seed_set);
            influence = diffusionState.diffuse(network, 1, rand);
            record[i] += diffusionState.anum;
            record_budget[i] += seed_set.size();
        }
        result[tid] = influence;
    }

    static void goUniform_d(Network network, Policy policy, int round, int d, int budget, vector<double> &record, vector<int> &record_budget, double *result, int tid){
        mt19937 rand(chrono::high_resolution_clock::now().time_since_epoch().count());
        DiffusionState diffusionState(network, round, budget);
        double influence = 0.;
        for(int i = 0;i < round;i++){
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
        result[tid] = influence;
    }

    static void goStatic(Network network, Policy policy, int round, int budget, vector<double> &record, vector<int> &record_budget, double *result, int tid){
        mt19937 rand(chrono::high_resolution_clock::now().time_since_epoch().count());
        DiffusionState diffusionState(network, round, budget);
        double influence = 0.;
        for(int i = 0;i < round;i++){
            vector<int> seed_set;
            if(i == 0){
                seed_set = policy.computeSeedSet(network, diffusionState, budget, rand);
                diffusionState.seed(seed_set);
            }
            influence = diffusionState.diffuse(network, 1, rand);
            record[i] += diffusionState.anum;
            record_budget[i] += seed_set.size();
        }
        result[tid] = influence;
    }

    static void goFull(Network network, Policy policy, int round, int budget, vector<double> &record, vector<int> &record_budget, double *result, int tid){
        mt19937 rand(chrono::high_resolution_clock::now().time_since_epoch().count());
        DiffusionState diffusionState(network, round, budget);
        double influence = 0.;
        for(int i = 0;i < round;i++){
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
        result[tid] = influence;
    }
public:
    static int round;

    static void MultiGo(Network network, Policy policy, int simutimes, int budget, vector<double> &record, vector<int> &record_budget, string type, int d){
        printf("Multi Go\n");
        if(round == -1){
            throw "round == -1";
            return;
        }
        double result = 0.;
        double *results = new double[round];
        vector<vector<double>> records(simutimes, vector<double>(round, 0.));
        vector<vector<int>> records_budget(simutimes, vector<int>(round, 0));
        boost::asio::thread_pool pool(32);
        for(int i = 0;i < simutimes;i++){
            printf("Simulation number %d\n", i+1);
            // vector<double> _record(round, 0.);
            // vector<int> _record_budget(round, 0);
            // records.push_back(vector<double>(round, 0.));
            // records_budget.push_back(vector<int>(round, 0));
            if(type == "dynamic")
                boost::asio::post(pool, boost::bind(goDynamic, network, policy, round, budget, ref(records[i]), ref(records_budget[i]), results, i));
            else if(type == "static")
                boost::asio::post(pool, boost::bind(goStatic, network, policy, round, budget, ref(records[i]), ref(records_budget[i]), results, i));
            else if(type == "uniform")
                boost::asio::post(pool, boost::bind(goUniform_d, network, policy, round, d, budget, ref(records[i]), ref(records_budget[i]), results, i));
            else if(type == "full")
                boost::asio::post(pool, boost::bind(goFull, network, policy, round, budget, ref(records[i]), ref(records_budget[i]), results, i));
            else
                printf("Invalid model\n");

            // if(type == "dynamic")
            //     results[i] = goDynamic(network, policy, round, budget, records[i], records_budget[i]);
            // else if(type == "static")
            //     results[i] = goStatic(network, policy, round, budget, records[i], records_budget[i]);
            // else if(type == "uniform")
            //     results[i] = goUniform_d(network, policy, round, d, budget, records[i], records_budget[i]);
            // else if(type == "full")
            //     results[i] = goFull(network, policy, round, budget, records[i], records_budget[i]);
            // else
            //     printf("Invalid model\n");
        }
        pool.join();
        for(int i = 0;i < round;i++)
            result += results[i];
        delete [] results;
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