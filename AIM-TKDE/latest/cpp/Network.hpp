#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <map>

using namespace std;

class Network{
    void clearPointers(){
        inDegree = nullptr;
        outDegree = nullptr;
        s_contri_order = nullptr;
        threshold = nullptr;
        c_threshold = nullptr;
        s_contri = nullptr;
    }
public:
    int vertexNum, edgeNum;
    vector<vector<int> > neighbor, neighbor_reverse;
    vector<vector<double> > probability;

    vector<int> sorted_degree;
    int *inDegree, *outDegree;
    int *s_contri_order;
    double *threshold, *c_threshold, *s_contri;
    string type, path, s_contri_path;
    double IC_prob;
    bool is_s_contri;

    Network(){
        neighbor.clear();
        is_s_contri = false;
    }

    Network(string path, string type, int vertexNum){
        clearPointers();
        printf("import %s %s\n", path, type);
        this->path = path;
        this->type = type;
        this->vertexNum = vertexNum;
        is_s_contri = false;
        neighbor.clear();
        neighbor_reverse.clear();
        inDegree = new int[vertexNum];
        outDegree = new int[vertexNum];
        sorted_degree.clear();

        importRelation(path);

        switch(type){
            case "IC":
                IC_prob = .1;
                break;
            case "WC":
                break;
            case "VIC":
                break;
            case "LT":
                threshold = new double[vertexNum];
                c_threshold = new double[vertexNum];
                break;
            default:
                perror("Invalid model");
        }
    }

    void set_ic_prob(double prob){
        IC_prob = prob;
    }

    void sort_by_degree(){
        map<int,int> tempMap;
        for(int i = 0;i < vertexNum;i++)
            tempMap[i] = outDegree[i];
        sort(tempMap.begin(), tempMap.end());
        for(auto i = tempMap.begin();i != tempMap.end();i++)
            sorted_degree.push_back(i->second);
    }

    void set_s_contri(string path){
        s_contri = new double[vertexNum];
        s_contri_order = new int[vertexNum];
        memset(s_contri, 0, vertexNum);
        int index = 0, node;
        double value;
        FILE *fd = fopen(path, "r");
        while(!feof(fd)){
            fscanf(fd, "%d %llf", &node, &value);
            s_contri[node] = value;
            s_contri_order[index] = node;
            index++;
        }
        fclose(fd);
        is_s_contri = true;
        s_contri_path = path;
    }
};
