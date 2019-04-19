#include <stdlib.h>
#include <memory.h>
#include <time.h>
#include <math.h>
#include <limits.h>

#include <iostream>
#include <vector>
#include <set>
#include <map>

#include "datastruct.hpp"
#include "utils.hpp"

#define DEBUG

using namespace std;

// delete vertex

vector<LL> nodeSet;
map<LL,LL> mp, rmp;
LL pre[MAXVERTEX];
bool del[MAXVERTEX], vis[MAXVERTEX];

void initMem(){
    memset(del, false, sizeof(del));
    memset(pre, -1, sizeof(pre));
}

bool bfs(LL src, LL snk, LL *adjList, LL *adjCount){
    _queue que;
    que.push(src);
    LL u, v;
    bool flag;
    memset(vis, false, sizeof(vis));
    while(!que.empty()){
        u = que.pop();
        vis[u] = true;
        flag = false;
        for(LL i = adjCount[u-1];i < adjCount[u];i++){
            v = adjList[i];
            if(v < 0 || vis[v] || del[v])
                continue;
            pre[v] = u;
            if(v == snk){
                flag = true;
                while(pre[v] != src)
                    v = pre[v];
                for(LL j = adjCount[src-1];j < adjCount[src];j++)
                    if(adjList[j] == v)
                        adjList[j] = -1;
                break;
            }
            que.push(v);
        }
        if(flag)
            break;
    }
    return flag;
}

bool checkNeighbor(LL n, LL snk, LL* adjList, LL* adjCount){
    for(LL i = adjCount[n - 1];i < adjCount[n];i++)
        if(adjList[i] == snk)
            return true;
    return false;
}

int main(){
    string filePath;
    LL src, snk, nd, counter = 0, lines = 0;
    LL totalNodes = 0, totalEdges = 0;
    LL* h_adjCount = NULL, *h_adjList = NULL;
    LL *c_adjCount = NULL, *c_adjList = NULL;
    int startTime;
    cout << "Choose dataset: ";
    cout.flush();
    cin >> filePath;
    rmp = readGraph(filePath.c_str(), h_adjList, h_adjCount, totalNodes, totalEdges, mp, true);
    c_adjCount = new LL[totalNodes + 1];
    c_adjList = new LL[totalEdges];
    cout << "Choose input file: ";
    cout.flush();
    cin >> filePath;
    FILE *fd = fopen(filePath.c_str(), "r");
    cout << "How many lines: ";
    cout.flush();
    cin >> lines;
    FILE *wfd = fopen("output.txt", "w");
    while (~fscanf(fd, "s %lld t %lld alpha %*f L %*lld pmax %*f beta %*f\n", &src, &snk)){
        counter++;
        printf("Line # %lld: ", counter);
        fflush(stdout);
        initMem();
        snk = mp[snk];
        src = mp[src];
        memcpy(c_adjCount, h_adjCount, (totalNodes+1) * sizeof(LL));
        memcpy(c_adjList, h_adjList, totalEdges * sizeof(LL));
        startTime = clock();
        fprintf(wfd, "%lld", rmp[snk]);
        while(bfs(snk, src, c_adjList, c_adjCount)){
            nd = pre[src];
            if(nd == snk)
                break;
            while(pre[nd] != snk){
                if(!del[pre[nd]]){
                    nodeSet.push_back(pre[nd]);
                    del[pre[nd]] = true;
                }
                nd = pre[nd];
                fprintf(wfd, " %lld", rmp[nd]);
            }
        }
        fputc('\n', wfd);
        startTime = clock() - startTime;
        printf("%ld s %ld ms;\n", startTime / CLOCKS_PER_SEC, startTime % CLOCKS_PER_SEC / 1000);
        if(lines > 0 && counter >= lines)
            break;
    }
    fclose(fd);
    fclose(wfd);
    delete [] h_adjCount;
    delete [] h_adjList;
    delete [] c_adjCount;
    delete [] c_adjList;
    return 0;
}