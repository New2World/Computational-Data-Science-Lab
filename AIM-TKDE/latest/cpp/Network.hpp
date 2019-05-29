#pragma once
#include <cstdio>
#include <cstdlib>

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>

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

    template <typename T>
    void freeSpace(T *ptr){
        if(ptr != nullptr)
            delete [] ptr;
    }

    void addNode(int n1, int n2){
        neighbor[n1].push_back(n2);
        neighbor_reverse[n2].push_back(n1);
        inDegree[n2]++;
        outDegree[n1]++;
    }

public:
    int vertexNum, edgeNum;
    vector<vector<int>> neighbor, neighbor_reverse;
    vector<vector<double>> probability;

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
        cout << "import " << path << " " << type << endl;
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

        if(type == "IC")
            IC_prob = .1;
        else if(type == "LT"){
            threshold = new double[vertexNum];
            c_threshold = new double[vertexNum];
        }
        else if(type != "WC" && type != "VIC")
            printf("Invalid model\n");
    }

    Network(const Network &network){
        clearPointers();
        path = network.path;
        type = network.type;
        vertexNum = network.vertexNum;
        edgeNum = network.edgeNum;
        is_s_contri = network.is_s_contri;
        neighbor = network.neighbor;
        neighbor_reverse = network.neighbor_reverse;
        probability = network.probability;
        sorted_degree = network.sorted_degree;
        IC_prob = network.IC_prob;
        s_contri_path = network.s_contri_path;
        inDegree = new int[vertexNum];
        outDegree = new int[vertexNum];
        memcpy(inDegree, network.inDegree, vertexNum*sizeof(int));
        memcpy(outDegree, network.outDegree, vertexNum*sizeof(int));
        if(network.s_contri != nullptr){
            s_contri_order = new int[vertexNum];
            s_contri = new double[vertexNum];
            memcpy(s_contri_order, network.s_contri_order, vertexNum*sizeof(int));
            memcpy(s_contri, network.s_contri, vertexNum*sizeof(double));
        }
        if(type == "LT"){
            threshold = new double[vertexNum];
            c_threshold = new double[vertexNum];
            memcpy(threshold, network.threshold, vertexNum*sizeof(double));
            memcpy(c_threshold, network.c_threshold, vertexNum*sizeof(double));
        }
    }

    ~Network(){
        freeSpace(inDegree);
        freeSpace(outDegree);
        freeSpace(s_contri);
        freeSpace(s_contri_order);
        freeSpace(threshold);
        freeSpace(c_threshold);
    }

    void importRelation(string path){
        for(int i = 0;i < vertexNum;i++){
            neighbor.push_back(vector<int>());
            neighbor_reverse.push_back(vector<int>());
            probability.push_back(vector<double>());
            inDegree[i] = 0;
            outDegree[i] = 0;
        }
        char line[256];
        int node1, node2;
        double prob;
        FILE *fd = fopen(path.c_str(), "r");
        while(NULL != fgets(line, 256, fd)){
            vector<string> inStr;
            boost::split(inStr, line, boost::is_any_of(" "));
            node1 = stoi(inStr[0]);
            node2 = stoi(inStr[1]);
            if(type == "VIC"){
                prob = stod(inStr[2]);
                addNode(node1, node2);
                probability[node1].push_back(prob);
            }
            else
                addNode(node1, node2);
        }
        fclose(fd);
    }

    void setICProb(double prob){
        IC_prob = prob;
    }

    void sortByDegree(){
        vector<pair<int,int>> tempMap;
        for(int i = 0;i < vertexNum;i++)
            tempMap.push_back(make_pair(i,outDegree[i]));
        sort(tempMap.begin(), tempMap.end(), [](auto a, auto b)-> bool {
            return a.second > b.second;
        });
        for(pair<int,int> p: tempMap)
            sorted_degree.push_back(p.first);
    }

    void setSContri(string path){
        s_contri = new double[vertexNum];
        s_contri_order = new int[vertexNum];
        memset(s_contri, 0, vertexNum);
        char line[256];
        int index = 0, node;
        double value;
        FILE *fd = fopen(path.c_str(), "r");
        vector<string> inStr;
        while(NULL != fgets(line, 256, fd)){
            boost::split(inStr, line, boost::is_any_of(" "));
            s_contri[node] = stod(inStr[1]);
            s_contri_order[index] = stoi(inStr[0]);
            index++;
        }
        fclose(fd);
        is_s_contri = true;
        s_contri_path = path;
    }

    void showSContri(){
        for(int i = 0;i < vertexNum;i++)
            printf("%d %lf\n", i, s_contri[i]);
    }

    void showData(){
        int edgenum = 0;
        for(int i = 0;i < vertexNum;i++)
            for(int j = 0;j < neighbor[i].size();j++,edgenum++)
                printf("%d %d\n", i, neighbor[i][j]);
        printf("Edge number: %d\n", edgenum);
    }

    bool isSuccess(double prob, mt19937 rand) const {
        if((double)rand()/rand.max() < prob)
            return true;
        return false;
    }

    void changeToRelization(mt19937 rand){
        vector<int> temp;
        for(int i = 0;i < neighbor.size();i++){
            for(int j = 0;j < neighbor[i].size();j++){
                if(isSuccess(getProb(i,neighbor[i][j]), rand)){
                    temp.push_back(neighbor[i][j]);
                    neighbor[i].erase(neighbor[i].begin()+j);
                    j--;
                }
            }
            // for(int t: temp)
            //     neighbor[i].erase(find(heighbor[i].begin(), neighbor[i].end(), t));
        }
        neighbor_reverse.clear();
    }

    double getProbByIndex(int cseed, int cseede) const {
        if(type == "IC")
            return IC_prob;
        else if(type == "VIC"){
            int index = [this, cseed, cseede]()->int{
                for(int i = 0;i < neighbor[cseed].size();i++)
                    if(neighbor[cseed][i] == cseede)
                        return i;
                return -1;
            }();
            if(index < 0)
                throw "error occurs in getProbByIndex";
            return probability[cseed][index];
        }
        else if(type == "WC")
            return 1./inDegree[cseede];
        else if(type == "LT")
            return 0.;
        else
            printf("Invalid model\n");
            return 0.;
    }

    double getProb(int cseed, int cseede) const {
        return getProbByIndex(cseed, cseede);
    }
};
