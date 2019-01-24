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

vector<_HyperEdge> mpu(LL n_nodes, LL n_hedges, LL p, LL q, vector<_HyperEdge> hyperEdge) {
    // printf("allocate DSH\n");
    DSH dsh = DSH();
    // printf("finish allocation\n");
	LL threshold = (LL)(p - sqrt((double)n_hedges)), E_dsize = 0, E_ddsize, rnd = 0;
    LL sizeRecord = -1;
    vector<LL> E_ddash;
    set<LL> E, E_dash;
	set<LL>::iterator iter;
	E.clear();
	E_dash.clear();
	E_ddash.clear();
	vector<LL> overlap;
	for (int i = 0; i < n_hedges; i++)
		E.insert((LL)i + 1);
	while (E_dsize < threshold) {
		overlap.clear();
        // printf("  building graph\n");
		dsh.buildFlowGraph(n_nodes, E, hyperEdge, q);
        // printf("  graph built\n");
		E_ddash = dsh.miniCut();
		E_ddsize = E_ddash.size();
		set_intersection(E_ddash.begin(), E_ddash.end(), E_dash.begin(), E_dash.end(), back_inserter(overlap));
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

__device__ bool nd_isVisited(bool* vis, LL nd, LL tot, int index){
    return vis[index * tot + nd - 1];
}

__device__ bool nd_setVisited(bool* vis, LL nd, LL tot, int index){
    bool oldState = vis[index * tot + nd - 1];
    vis[index * tot + nd - 1] = true;
    return oldState;
}

__device__ void ndset_clear(LL* nodeSet, int index){
    nodeSet[index * NODESETSIZE] = 0;
}

__device__ void ndset_insert(LL* nodeSet, LL value, int index){
    LL cnt = ++nodeSet[index * NODESETSIZE];
    if(cnt >= NODESETSIZE){
        printf("\n>>> overflow <<<\n");
    }
    nodeSet[index * NODESETSIZE + cnt] = value;
}

__device__ int ndset_size(LL* nodeSet, int index){
    return nodeSet[index * NODESETSIZE];
}

__device__ void ndset_erase(LL* nodeSet, int index){
    if(ndset_size(nodeSet, index) > 0)
        nodeSet[index * NODESETSIZE]--;
}

__global__ void setupRandGenerator(float* randSeed, curandState* state){
    int index = getIndex();
    unsigned long seed = (unsigned long)(randSeed[index] * RAND_FACTOR);
    curand_init(seed, index, 0, &state[index]);
}

__global__ void reverseInfluence(LL totalNodes,
                                 LL totalEdges,
                                 LL iters,
                                 LL* d_k,
                                 LL* d_adjCount,
                                 LL* d_adjList,
                                 LL* d_nodeSet,
                                 bool* d_visit,
                                 LL source, LL sink,
                                 curandState *state){
    int index = getIndex();
    LL startNode, outdegree, nextNode;
    float probability;
    bool flag = true;
    curandState localState = state[index];
    // while(true){
        flag = true;
        startNode = source;
        ndset_clear(d_nodeSet, index);
        ndset_insert(d_nodeSet, startNode, index);
        while (true) {
            outdegree = d_adjCount[startNode] - d_adjCount[startNode - 1];
            probability = curand_uniform(&localState) * outdegree;
            nextNode = floor(probability);
            nextNode += d_adjCount[startNode - 1];
            if (nextNode >= d_adjCount[startNode]) {
                ndset_clear(d_nodeSet, index);
                break;
            }

            for (LL j = d_adjCount[startNode - 1]; j < d_adjCount[startNode]; j++) {
                if (d_adjList[j] == sink) {
                    ndset_erase(d_nodeSet, index);
                    flag = false;
                    break;
                }
            }
            if (!flag)
                break;

            startNode = d_adjList[nextNode];
            if (nd_isVisited(d_visit, startNode, totalNodes, index)){
                ndset_clear(d_nodeSet, index);
                break;
            }
            nd_setVisited(d_visit, startNode, totalNodes, index);
            ndset_insert(d_nodeSet, startNode, index);
        }
        if(iters <= atomicAdd((unsigned long long*)d_k, 1ULL)){
            ndset_clear(d_nodeSet, index);
            // break;
        }
    //     if (ndset_size(d_nodeSet, index) != 0)
    //         break;
    // }

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
	cin >> filePath;
	rmp = readGraph(filePath.c_str(), h_adjList, h_adjCount, totalNodes, totalEdges, mp, true);
	cout << "Choose input file: ";
	cout.flush();
	cin >> filePath;
	FILE* fd = fopen(filePath.c_str(), "r");
	cout << "How many lines: ";
	cout.flush();
	cin >> lines;

	printf("========= NEW RUN\n");
	printf("This graph contains %lld nodes connected by %lld edges\n", totalNodes, totalEdges);
    printf("Running on %d threads\n\n", THREAD);

	float alpha, beta, pmax;
    float kmax, dif;
	LL counter = 0, iters;
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

    bool *d_visit = NULL;
    LL* d_adjCount = NULL, *d_adjList = NULL, *d_nodeSet = NULL, *d_k = NULL;
    cudaMalloc((void**)&d_visit, sizeof(bool) * THREAD * totalNodes);
    cudaMalloc((void**)&d_nodeSet, sizeof(LL) * THREAD * NODESETSIZE);
    cudaMalloc((void**)&d_adjCount, sizeof(LL) * (totalNodes + 1));
    cudaMalloc((void**)&d_adjList, sizeof(LL) * totalEdges);
    cudaMalloc((void**)&d_k, sizeof(LL));

    cudaMemcpy(d_adjList, h_adjList, sizeof(LL)*totalEdges, cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjCount, h_adjCount, sizeof(LL)*(totalNodes+1),cudaMemcpyHostToDevice);

    FILE* wfd = fopen("output.txt", "w");
    printf("   ln     %%diff         time\n");
    while (~fscanf(fd, "s %lld t %lld alpha %f L %lld pmax %f beta %f\n", &sink, &source, &alpha, &k, &pmax, &beta)) {
		// newOutput(counter, outputFile);
		counter++;
		printf("#%4lld - ", counter);
        fflush(stdout);
        kmax = k * pmax;

		// FILE* wfd = fopen(outputFile, "w");
        hyperEdge.clear();
        sink = mp[sink];
        source = mp[source];
		startTime = clock();

        iters = 0;
        cudaMemset(d_k, iters, sizeof(LL));
        h_nodeSet = new LL[THREAD * NODESETSIZE];
        while(iters < k){
            cudaMemset(d_visit, false, sizeof(bool) * THREAD * totalNodes);
            cudaMemset(d_nodeSet, 0LL, sizeof(LL) * THREAD * NODESETSIZE);
            reverseInfluence<<<gridSize, blockSize>>>(  totalNodes, totalEdges, k,
                                                        d_k,
                                                        d_adjCount,
                                                        d_adjList,
                                                        d_nodeSet,
                                                        d_visit,
                                                        source, sink,
                                                        d_randState );

            cudaMemcpy(h_nodeSet, d_nodeSet, sizeof(LL)*THREAD*NODESETSIZE, cudaMemcpyDeviceToHost);
            cudaMemcpy(&iters, d_k, sizeof(LL), cudaMemcpyDeviceToHost);
            for(LL i = 0;i < THREAD;i++){
                LL setSize = h_nodeSet[i * NODESETSIZE];
                if(setSize <= 0)
                    continue;
                _HyperEdge item;
                item.v_size = setSize;
                memcpy(item.vertex, h_nodeSet + i * NODESETSIZE + 1, sizeof(LL) * setSize);
                hyperEdge.push_back(item);
            }
            // printf("%lld / %lld - %ld\n", iters, k, hyperEdge.size());
        }
        delete[] h_nodeSet;

        // printf("End GPU, get %ld hyperedges\n", hyperEdge.size());
		q = ((hyperEdge.size() / totalNodes) + hyperEdge.size()) / 2;
		p = (LL)(beta * hyperEdge.size());

        dif = kmax - hyperEdge.size();
        if(dif < 0) dif = -dif;
        printf("%.4f%% - ", dif / kmax * 100);
        fflush(stdout);

		nodeSet.clear();
        E.clear();

        // printf("Start MpU\n");
		if (hyperEdge.size() > 0)
			E = mpu(totalNodes, (LL)hyperEdge.size(), p, q, hyperEdge);
        // printf("End MpU\n");

		for (LL i = 0; i < E.size(); i++)
			nodeSet.insert(E[i].vertex, E[i].vertex + E[i].v_size);

		startTime = clock() - startTime;
        printf("%ld s %3ld ms;\n", startTime / CLOCKS_PER_SEC, startTime % CLOCKS_PER_SEC / 1000);

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
    cudaFree(d_visit);
    cudaFree(d_nodeSet);
    cudaFree(d_adjCount);
    cudaFree(d_adjList);
    cudaFree(d_k);

	delete[] h_adjCount;
	delete[] h_adjList;

	printf("\n========= FINISH\n");

	return 0;
}
