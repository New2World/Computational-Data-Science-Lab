#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <time.h>
#include <math.h>
#include <limits.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <set>
#include <map>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "utils.hpp"

#define RAND_FACTOR 1e9+7
#define THREAD (9 * 256)

#define DEBUG

using namespace std;

map<LL,LL> mp, rmp;

// Index start from 1 in mapped nodes

vector<_HyperEdge> mpu(LL n_nodes, LL n_hedges, LL p, LL q, vector<_HyperEdge> hyperEdge) {
    DSH dsh = DSH();
	LL threshold = (LL)(p - sqrt((double)n_hedges)), E_dsize = 0, E_ddsize, rnd = 0;
    LL sizeRecord = -1;
    vector<LL> E_ddash;
    set<LL> E, E_dash;
	set<LL>::iterator iter;
	E.clear();
	E_dash.clear();
	E_ddash.clear();
	for (int i = 0; i < n_hedges; i++)
		E.insert((LL)i + 1);
	while (E_dsize < threshold) {
		dsh.buildFlowGraph(n_nodes, E, hyperEdge, q);
		E_ddash = dsh.miniCut();
		E_ddsize = E_ddash.size();
		if (E_dsize + E_ddsize <= p) {
			E_dash.insert(E_ddash.begin(), E_ddash.end());
			for (LL i = 0; i < E_ddsize; i++)
				E.erase(E_ddash[i]);
		}
		else {
			for (int i = 0; i < p - E_dsize; i++) {
				rnd = (LL)rand() % E_ddsize;
				E_dash.insert(E_ddash[rnd]);
				E.erase(E_ddash[rnd]);
				E_ddash.erase(E_ddash.begin() + rnd);
				E_ddsize--;
			}
		}
        E_dsize = E_dash.size();
        if(E_dsize == sizeRecord || E_dsize >= n_hedges){
            printf("ERROR (DEAD) - ");
            break;
        }
        sizeRecord = E_dsize;
	}

	vector<_HyperEdge> cardinality, result;
	cardinality.clear();
	result.clear();
	for (iter = E.begin(); iter != E.end(); iter++)
		cardinality.push_back(hyperEdge[*iter - 1]);
	sort(cardinality.begin(), cardinality.end());
	for (LL i = 0; cardinality.size() && i < cardinality.size() && i < p - E_dsize; i++)
		result.push_back(cardinality[i]);
	for (iter = E_dash.begin(); iter != E_dash.end(); iter++)
		result.push_back(hyperEdge[*iter - 1]);

	return result;
}

void newOutput(LL counter, char* outputFile) {
	char str[20] = "output/";
	sprintf(str + 7, "%lld_%lld.txt", counter / 5, counter % 5);
	strcpy(outputFile, str);
}

__device__ int getIndex(){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int line = blockDim.x * gridDim.x;
    return row * line + col;
}

// __device__ void nd_clearVisit(bool *vis, LL tot, int index){
//     memset(vis + index * tot, false, sizeof(bool) * tot);
// }
//
// __device__ bool nd_isVisited(bool* vis, LL nd, LL tot, int index){
//     return vis[index * tot + nd - 1];
// }
//
// __device__ bool nd_setVisited(bool* vis, LL nd, LL tot, int index){
//     bool oldState = vis[index * tot + nd - 1];
//     vis[index * tot + nd - 1] = true;
//     return oldState;
// }

__device__ void ndset_clear_global(LL *nodeSet, LL *nodeCount, int index){
    nodeSet[index * NODESETSIZE] = 0;
    nodeCount[index] = 0;
}

__device__ void ndset_clear(LL* nodeSet, LL *nodeCount, int index){
    nodeSet[index * NODESETSIZE] = nodeCount[index];
}

__device__ bool ndset_insert(LL* nodeSet, LL value, int index){
    LL cnt = ++nodeSet[index * NODESETSIZE];
    if(cnt >= NODESETSIZE)
        return false;
    nodeSet[index * NODESETSIZE + cnt] = value;
    return true;
}

__device__ int ndset_size_global(LL *nodeSet, int index){
    return nodeSet[index * NODESETSIZE];
}

__device__ int ndset_size(LL* nodeSet, LL *nodeCount, int index){
    return nodeSet[index * NODESETSIZE] - nodeCount[index];
}

__device__ void ndset_erase(LL* nodeSet, LL *nodeCount, int index){
    if(ndset_size(nodeSet, nodeCount, index) > 0)
        nodeSet[index * NODESETSIZE]--;
}

__device__ bool ndset_new(LL *nodeSet, LL *nodeCount, int index){
    if(!ndset_insert(nodeSet, 0, index))
        return false;
    nodeCount[index] = ndset_size_global(nodeSet, index);
    return true;
}

__global__ void setupRandGenerator(float* randSeed, curandState* state){
    int index = getIndex();
    unsigned long seed = (unsigned long)(randSeed[index] * RAND_FACTOR);
    curand_init(seed, index, 0, &state[index]);
}

__global__ void reverseInfluence(LL totalNodes,
                                 LL totalEdges,
                                 LL iter_left,
                                 LL* d_k,
                                 LL* d_adjCount,
                                 LL* d_adjList,
                                 LL* d_nodeSet,
                                 LL* d_nodeCount,
                                 LL source, LL sink,
                                 curandState *state){
    int index = getIndex();
    LL startNode, outdegree, nextNode;
    LL cur_k;
    float probability;
    bool neighbor = false, overflow = false;
    curandState localState = state[index];
    ndset_clear_global(d_nodeSet, d_nodeCount, index);
    bool vis[10000];
    while(!overflow){
        neighbor = false;
        startNode = source;
        memset(vis, false, sizeof(vis));
        ndset_insert(d_nodeSet, startNode, index);
        vis[startNode] = true;
        while (true) {
            outdegree = d_adjCount[startNode] - d_adjCount[startNode - 1];
            probability = curand_uniform(&localState) * outdegree;
            nextNode = floor(probability);
            nextNode += d_adjCount[startNode - 1];
            if (nextNode >= d_adjCount[startNode]) {
                // printf("%lld >= %lld - choose no point - terminate\n", nextNode, d_adjCount[startNode]);
                ndset_clear(d_nodeSet, d_nodeCount, index);
                break;
            }
            // printf("%lld < %lld - choose point\n", nextNode, d_adjCount[startNode]);

            for (LL j = d_adjCount[startNode - 1]; j < d_adjCount[startNode]; j++) {
                if (d_adjList[j] == sink) {
                    // printf("reach sink neighbor - terminate\n");
                    ndset_erase(d_nodeSet, d_nodeCount, index);
                    neighbor = true;
                    break;
                }
            }
            if (neighbor)
                break;

            startNode = d_adjList[nextNode];
            if(vis[startNode]){
                // printf("visited - terminate\n");
                ndset_clear(d_nodeSet, d_nodeCount, index);
                break;
            }
            vis[startNode] = true;
            if(!ndset_insert(d_nodeSet, startNode, index)){
                // printf("overflow - terminate\n");
                overflow = true;
                break;
            }
        }
        cur_k = atomicAdd((unsigned long long*)d_k, 1ULL);

        if(overflow || iter_left <= cur_k){
            ndset_clear(d_nodeSet, d_nodeCount, index);
            if(iter_left <= cur_k)
                atomicSub((unsigned int*)d_k, 1U);
            break;
        }
        if(ndset_size(d_nodeSet, d_nodeCount, index) == 0)
            continue;
        if(!ndset_new(d_nodeSet, d_nodeCount, index))
            break;
    }
    state[index] = localState;
}

int main(int argc, char** argv) {
	LL source = 0, sink = 0, lines;
	string filePath;
	LL p, q, k = 1000;

	// char outputFile[20];
	long startTime;
	LL totalNodes = 0, totalEdges = 0;
	LL* h_adjCount = NULL, *h_adjList = NULL, *h_nodeSet = NULL;
	cout << "Choose dataset: ";
	cout.flush();
	// cin >> filePath;
    filePath = "../../data/wiki/wiki.txt";
	rmp = readGraph(filePath.c_str(), h_adjList, h_adjCount, totalNodes, totalEdges, mp, true);
	cout << "Choose input file: ";
	cout.flush();
	// cin >> filePath;
    filePath = "../../data/wiki/input.txt";
	FILE* fd = fopen(filePath.c_str(), "r");
	cout << "How many lines: ";
	cout.flush();
	// cin >> lines;
    lines = 1;

	printf("========= NEW RUN\n");
	printf("This graph contains %lld nodes connected by %lld edges\n", totalNodes, totalEdges);
    printf("Running on %d threads\n\n", THREAD);

	float alpha, beta, pmax;
    float kmax, dif;
	LL counter = 0, loop, iters, iter_left;
	srand(time(NULL));
	vector<_HyperEdge> hyperEdge;
	set<LL> nodeSet;
	vector<_HyperEdge> E;

    dim3 gridSize(3,3), blockSize(16,16);

    float *d_randSeed;
    curandState *d_randState;
    curandGenerator_t curandGenerator;
    cudaMalloc((void**)&d_randSeed, sizeof(float) * THREAD);
    cudaMalloc((void**)&d_randState, sizeof(curandState) * THREAD);
    curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curandGenerator, time(NULL));
    curandGenerateUniform(curandGenerator, d_randSeed, THREAD);
    setupRandGenerator<<<gridSize, blockSize>>>(d_randSeed, d_randState);

    LL* d_adjCount = NULL, *d_adjList = NULL, *d_nodeSet = NULL, *d_k = NULL;
    LL *d_nodeCount = NULL;
    cudaMalloc((void**)&d_nodeSet, sizeof(LL) * THREAD * NODESETSIZE);
    cudaMalloc((void**)&d_adjCount, sizeof(LL) * (totalNodes + 1));
    cudaMalloc((void**)&d_adjList, sizeof(LL) * totalEdges);
    cudaMalloc((void**)&d_nodeCount, sizeof(LL) * THREAD);
    cudaMalloc((void**)&d_k, sizeof(LL));

    cudaMemcpy(d_adjList, h_adjList, sizeof(LL)*totalEdges, cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjCount, h_adjCount, sizeof(LL)*(totalNodes+1),cudaMemcpyHostToDevice);

    FILE* wfd = fopen("output.txt", "w");
    printf("   ln     %%diff         time     per loop\n");
    while (~fscanf(fd, "s %lld t %lld alpha %f L %lld pmax %f beta %f\n", &sink, &source, &alpha, &k, &pmax, &beta)) {
		counter++;
		printf("#%4lld - ", counter);
        fflush(stdout);
        kmax = k * pmax;

        hyperEdge.clear();
        sink = mp[sink];
        source = mp[source];
		startTime = clock();

        iter_left = k;
        cudaMemset(d_k, 0LL, sizeof(LL));
        h_nodeSet = new LL[THREAD * NODESETSIZE];
        loop = 0;
        LL nodes = 0;
        while(iter_left > 0){
            reverseInfluence<<<gridSize, blockSize>>>(  totalNodes, totalEdges, iter_left,
                                                        d_k,
                                                        d_adjCount,
                                                        d_adjList,
                                                        d_nodeSet,
                                                        d_nodeCount,
                                                        source, sink,
                                                        d_randState );

            cudaMemcpy(h_nodeSet, d_nodeSet, sizeof(LL)*THREAD*NODESETSIZE, cudaMemcpyDeviceToHost);
            cudaMemcpy(&iters, d_k, sizeof(LL), cudaMemcpyDeviceToHost);
            loop++;
            LL setSize;
            iter_left = k - iters;
            for(LL i = 0;i < THREAD;i++){
                setSize = h_nodeSet[i * NODESETSIZE];
                for(LL j = 1;j <= setSize;j++){
                    _HyperEdge item;
                    item.v_size = 0;
                    if(h_nodeSet[j] == 0){
                        if(h_nodeSet[j-1] != 0){
                            if(item.v_size > 0)
                                hyperEdge.push_back(item);
                            nodes += item.v_size;
                        }
                        continue;
                    }
                    item.vertex[item.v_size++] = h_nodeSet[j];
                    if(j == setSize && h_nodeSet[j] != 0){
                        if(item.v_size > 0)
                            hyperEdge.push_back(item);
                        nodes += item.v_size;
                    }
                }
            }
        }
        delete[] h_nodeSet;

		q = ((hyperEdge.size() / totalNodes) + hyperEdge.size()) / 2;
		p = (LL)(beta * hyperEdge.size());

        dif = kmax - hyperEdge.size();
        if(dif < 0) dif = -dif;
        printf("%.4f%% - %.3f\n", dif / kmax * 100, 1. * nodes / hyperEdge.size());
        fflush(stdout);
        break;

		nodeSet.clear();
        E.clear();

		if (hyperEdge.size() > 0)
			E = mpu(totalNodes, (LL)hyperEdge.size(), p, q, hyperEdge);

		for (LL i = 0; i < E.size(); i++)
			nodeSet.insert(E[i].vertex, E[i].vertex + E[i].v_size);

		startTime = clock() - startTime;
        printf("%ld s %3ld ms - %lld s %3lld ms\n",
                startTime / CLOCKS_PER_SEC,
                startTime % CLOCKS_PER_SEC / 1000,
                startTime / loop / CLOCKS_PER_SEC,
                startTime / loop % CLOCKS_PER_SEC / 1000);

		for (auto i = nodeSet.begin(); i != nodeSet.end(); i++)
			fprintf(wfd, "%lld ", rmp[*i]);
		fprintf(wfd, "\n");
		if (lines > 0 && counter >= lines)
			break;
	}

    fflush(wfd);
    fclose(wfd);
	fclose(fd);

    cudaFree(d_randSeed);
    cudaFree(d_randState);
    cudaFree(d_nodeSet);
    cudaFree(d_adjCount);
    cudaFree(d_adjList);
    cudaFree(d_nodeCount);
    cudaFree(d_k);

	delete[] h_adjCount;
	delete[] h_adjList;

	printf("\n========= FINISH\n");

	return 0;
}
