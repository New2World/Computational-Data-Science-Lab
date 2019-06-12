#pragma once

#include <iostream>
#include <set>

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