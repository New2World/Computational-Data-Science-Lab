#pragma once

#include <iostream>
#include <map>
#include <set>
#include <vector>

bool intersection(const std::set<int> &l1, const std::set<int> &l2){
    for(int v: l1)
        if(l2.find(v) != l2.end())
            return true;
    return false;
}

template <typename T>
void printContainer(const T &container){
    if(container.size() < 0)    return;
    if(container.size() == 1)
        std::cout << *container.begin();
    else{
        auto it = container.begin();
        std::cout << *(it++);
        for(;it != container.end();it++)
            std::cout << " " << *it;
    }
    std::cout << std::endl;
}

void printTime(chrono::high_resolution_clock::time_point start, chrono::high_resolution_clock::time_point end){
    auto duration = end-start;
    auto hour = chrono::duration_cast<chrono::hours>(duration).count();
    auto min = chrono::duration_cast<chrono::minutes>(duration).count();
    auto sec = chrono::duration_cast<chrono::seconds>(duration).count();
    cout << "time elapsed: " << hour << " hours " << min%60 << " minutes " << sec%60 << " seconds" << endl;
}

struct Results{
    std::map<int,std::set<int>> seedset;
    std::map<int,double> supp;

    Results()=default;
    Results(const Results &)=default;
    Results &operator = (const Results &)=default;
};