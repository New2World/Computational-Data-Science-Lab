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
    }
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
};