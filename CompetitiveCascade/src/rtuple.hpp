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
    rTuple(){clear();}
    rTuple(const rTuple &rt){
        node_v = rt.node_v;
        isdiff = rt.isdiff;
        upper = rt.upper;
        lower = rt.lower;
        relations = rt.relations;
    }
    rTuple &operator = (const rTuple &)=delete;

    void clear(){
        node_v = -1;
        upper.clear();
        lower.clear();
        seed.clear();
        relations.clear();
        isdiff = false;
    }

    void _stat(){
        std::cout << "----------" << std::endl;
        std::cout << "node_v: " << node_v << std::endl;
        std::cout << "upper:" << std::endl;
        printContainer(upper);
        std::cout << "lower:" << std::endl;
        printContainer(lower);
        std::cout << "----------" << std::endl;
    }
};