#pragma once

#include <iostream>
#include <map>
#include <set>
#include <chrono>
#include <vector>

bool intersection(const std::set<int> &l1, const std::set<int> &l2){
    for(int v: l1)
        if(l2.find(v) != l2.end())
            return true;
    return false;
}

template <typename T>
void printContainer(const T &container){
    if(container.size() == 1)
        std::cout << *container.begin();
    else if(!container.empty()){
        auto it = container.begin();
        std::cout << *(it++);
        for(;it != container.end();it++)
            std::cout << " " << *it;
    } else  std::cout << "empty" << std::endl;
    std::cout << std::endl;
}

void printTime(const std::chrono::high_resolution_clock::time_point &start, const std::chrono::high_resolution_clock::time_point &end){
    auto duration = end-start;
    auto hour = std::chrono::duration_cast<std::chrono::hours>(duration).count();
    auto min = std::chrono::duration_cast<std::chrono::minutes>(duration).count();
    auto sec = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
    std::cout << "time elapsed: " << hour << " hours " << min%60 << " minutes " << sec%60 << " seconds" << std::endl;
}

struct Results{
    std::map<int,std::set<int>> seedset;
    std::map<int,double> supp;

    Results()=default;
    Results(const Results &)=default;
    Results &operator = (const Results &)=default;

    void writeToFile(FILE *fd){
        fprintf(fd, "%d\n", seedset.size());
        for(pair<int,set<int>> p: seedset){
            fprintf(fd, "%d\n", p.first);
            for(int s: p.second)
                fprintf(fd, "%d ", s);
            fprintf(fd, "\n");
        }
        for(pair<int,double> p: supp)
            fprintf(fd, "%llf\n", p.second);
    }

    void readFromFile(FILE *fd){
        int sz, k, v;
        double vv;
        fscanf(fd, "%d", &sz);
        for(int i = 0;i < sz;i++){
            fscanf(fd, "%d", &k);
            seedset[k] = std::set<int>();
            for(int j = 0;j < k;j++){
                fscanf(fd, "%d", &v);
                seedset[k].insert(v);
            }
        }
        for(int i = 0;i < sz;i++){
            fscanf(fd, "%f", &vv);
            supp[k] = vv;
        }
    }
};