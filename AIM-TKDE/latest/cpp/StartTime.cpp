#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>

#include "Policy.hpp"
#include "Network.hpp"
#include "DiffusionState.hpp"
#include "SeedingProcessTime.hpp"

using namespace std;

template <typename T>
void printVector(vector<T> vec){
    for(T i: vec)
        cout << i << " ";
    cout << endl;
}

void printTime(chrono::high_resolution_clock::time_point start, chrono::high_resolution_clock::time_point end){
    auto duration = end-start;
    auto hour = chrono::duration_cast<chrono::hours>(duration).count();
    auto min = chrono::duration_cast<chrono::minutes>(duration).count();
    auto sec = chrono::duration_cast<chrono::seconds>(duration).count();
    cout << "time elapsed: " << hour << " hours " << min%60 << " minutes " << sec%60 << " seconds" << endl;
}

void run(int simutimes, int k, int vnum, Network network){
    // start timer
    auto launch = chrono::high_resolution_clock::now();

    printf("k = %d\td = %d\n", k, SeedingProcessTime::round);
    printf("simutimes = %d\n", simutimes);
    printf("simu_rest_times = %d\n", Policy::simurest_times);
    printf("rrsets size = %d\n", Policy::rrsets_size);

    vector<double> record_budget;
    vector<double> record;

    cout << "-------------------------------------------" << endl;
    cout << "-------------------------------------------" << endl;
    cout << "dynamic" << endl;
    record = vector<double>(SeedingProcessTime::round, 0.);
    record_budget = vector<double>(SeedingProcessTime::round, 0.);
    auto start = chrono::high_resolution_clock::now();
    SeedingProcessTime::MultiGo(network, simutimes, k, record, record_budget, "dynamic", -1);
    auto end = chrono::high_resolution_clock::now();
    printTime(start, end);
    printVector(record);
    printVector(record_budget);
    cout << "-------------------------------------------" << endl;
    cout << "-------------------------------------------" << endl;

    cout << "static" << endl;
    record = vector<double>(SeedingProcessTime::round, 0.);
    record_budget = vector<double>(SeedingProcessTime::round, 0.);
    start = chrono::high_resolution_clock::now();
    SeedingProcessTime::MultiGo(network, simutimes, k, record, record_budget, "static", -1);
    end = chrono::high_resolution_clock::now();
    printTime(start, end);
    printVector(record);
    printVector(record_budget);
    cout << "-------------------------------------------" << endl;
    cout << "-------------------------------------------" << endl;

    cout << "uniform 1" << endl;
    record = vector<double>(SeedingProcessTime::round, 0.);
    record_budget = vector<double>(SeedingProcessTime::round, 0.);
    start = chrono::high_resolution_clock::now();
    SeedingProcessTime::MultiGo(network, simutimes, k, record, record_budget, "uniform", 1);
    end = chrono::high_resolution_clock::now();
    printTime(start, end);
    printVector(record);
    printVector(record_budget);
    cout << "-------------------------------------------" << endl;
    cout << "-------------------------------------------" << endl;

    cout << "uniform 2" << endl;
    record = vector<double>(SeedingProcessTime::round, 0.);
    record_budget = vector<double>(SeedingProcessTime::round, 0.);
    start = chrono::high_resolution_clock::now();
    SeedingProcessTime::MultiGo(network, simutimes, k, record, record_budget, "uniform", 2);
    end = chrono::high_resolution_clock::now();
    printTime(start, end);
    printVector(record);
    printVector(record_budget);
    cout << "-------------------------------------------" << endl;
    cout << "-------------------------------------------" << endl;

    cout << "full" << endl;
    record = vector<double>(SeedingProcessTime::round, 0.);
    record_budget = vector<double>(SeedingProcessTime::round, 0.);
    start = chrono::high_resolution_clock::now();
    SeedingProcessTime::MultiGo(network, simutimes, k, record, record_budget, "full", -1);
    end = chrono::high_resolution_clock::now();
    printTime(start, end);
    printVector(record);
    printVector(record_budget);
    cout << "-------------------------------------------" << endl;
    cout << "-------------------------------------------" << endl;

    auto terminate = chrono::high_resolution_clock::now();

    printTime(launch, terminate);
}

int main(int args, char **argv){
    SeedingProcessTime::round = 5;
    Policy::simurest_times = 100;
    Policy::rrsets_size = 100000;

    string name(argv[1]);
    string type(argv[2]);
    int vnum = atoi(argv[3]);
    int simutimes = atoi(argv[4]);
    SeedingProcessTime::round = atoi(argv[5]);
    Policy::simurest_times = atoi(argv[6]);
    int k = atoi(argv[7]);

    string path("../data/"+name+".txt");
    Network network(path, type, vnum);
    network.setICProb(.1);

    run(simutimes, k, vnum, network);
}
