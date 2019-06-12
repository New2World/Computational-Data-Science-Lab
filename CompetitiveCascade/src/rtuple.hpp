#pragma once

#include <iostream>
#include <set>
#include <map>
#include <vector>
#include <string>
#include <algorithm>

#include "tools.hpp"

class rTuple{
public:
    int node_v;
    bool isdiff;
    std::set<int> upper, lower, seed;
    std::map<int,std::set<int>> relations;
    rTuple(){
        upper.clear();
        lower.clear();
        seed.clear();
        relations.clear();
        isdiff = false;
    }

    void _stat(){
        std::cout << "node_v: " << node_v << std::endl;
        std::cout << "upper:" << std::endl;
        printContainer(upper);
        std::cout << "lower:" << std::endl;
        printContainer(lower);
    }
};