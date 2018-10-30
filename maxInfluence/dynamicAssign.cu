#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <getopt.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "cuqueue.cuh"

#define RAND_FACTOR 1e9+7

#define MAX_OUT 2000
#define MAX_NODE 10000
#define THREAD (4 * 256)

float CONSTANT_PROBABILITY = 0.05;
int _adjMat[MAX_NODE][MAX_OUT];

// read graph information from files and generate a adjacent list for representation
int readGraph(const char* filePath,
              int* adjCount,
              int* adjList,
              int& nodes,
              int& outdegree){
    int s, t, count = 0;
    FILE* fd = fopen(filePath, "r");
    if(fd == NULL)
        return -1;
    memset(adjCount, 0, sizeof(int) * MAX_NODE);
    while(!feof(fd)){
        fscanf(fd, "%d %d", &s, &t);
        _adjMat[s - 1][adjCount[s]++] = t - 1;
        count++;
        nodes = nodes > s ? (nodes > t ? nodes : t) : (s > t ? s : t);
    }
    fclose(fd);
    int ptr = 0;
    for(int i = 0;i < nodes;i++){
        memcpy(adjList + ptr, _adjMat[i], sizeof(int) * adjCount[i + 1]);
        ptr += adjCount[i + 1];
        outdegree = outdegree > adjCount[i + 1] ? outdegree : adjCount[i + 1];
        adjCount[i + 1] += adjCount[i];
    }
    return count;
}

// get thread index
__device__ int getIndex(){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int line = blockDim.x * gridDim.x;
    return row * line + col;
}

// find next unused node
__device__ int findVertex(int* nodeSet, int nodes){
    int nextNode = atomicAdd(&nodeSet[nodes], 1);
    if(nextNode >= nodes)
        return -1;
    return nextNode;
}

// judge if node 'nd' is visited once
__device__ bool nd_isVisited(bool* vis, int nd, int tot, int index){
    return vis[index * tot + nd];
}

// set node 'nd' visited
__device__ void nd_setVisited(bool* vis, int nd, int tot, int index){
    vis[index * tot + nd] = true;
}

__device__ void nd_resetVisited(bool* vis, int tot, int index){
    int i = index * tot, end = index * tot + tot;
    for(;i < end;i++)
        vis[i] = false;
}

// initialize random seeds for each thread
__global__ void setupRandGenerator(float* randSeed, curandState* state){
    int index = getIndex();
    unsigned long seed = (unsigned long)(randSeed[index] * RAND_FACTOR);
    curand_init(seed, index, 0, &state[index]);
}

// BFS kernel function in each thread
__global__ void bfs(int totalNodes,
                    int* adjCount,
                    int* adjList,
                    int* nodeSet,
                    int* queue,
                    bool* closed,
                    curandState* state,
                    float constProb){
    int index = getIndex();
    int count = 0, node = index;
    int que_h, que_t;
    int adjNode, from;
    float randProb;
    curandState localState = state[index];
    while((node = findVertex(nodeSet, totalNodes)) != -1){
        count = 0;
        from = node;
        que_init(que_h, que_t, index);
        nd_resetVisited(closed, totalNodes, index);
        if(!que_enque(queue, que_h, que_t, from, index));   // in case queue overflow
        nd_setVisited(closed, from, totalNodes, index);
        while(!que_isEmpty(que_h, que_t, index)){
            node = que_deque(queue, que_h, que_t, index);
            adjNode = adjCount[node];
            while(adjNode < adjCount[node + 1]){
                if(!nd_isVisited(closed, adjList[adjNode], totalNodes, index)){
                    randProb = curand_uniform(&localState);
                    if(randProb < constProb){
                        if(!que_enque(queue, que_h, que_t, adjList[adjNode], index));
                        nd_setVisited(closed, adjList[adjNode], totalNodes, index);
                        count++;
                    }
                }
                adjNode++;
            }
        }
        if(atomicCAS(nodeSet + from, 0, count) != 0);   // theoretically impossiable
    }
    state[index] = localState;
}

// global variables
int h_adjCount[MAX_NODE];
int h_adjList[MAX_NODE * MAX_OUT];
int h_nodeSet[MAX_NODE];

// for argument parsing
char short_options[] = "f:p::c::vto";
struct option long_options[] = {
    {"file", required_argument, 0, 'f'},
    {"probability", optional_argument, 0, 'p'},
    {"constfactory", optional_argument, 0, 'c'},
    {"verbose", no_argument, 0, 'v'},
    {"timeonly", no_argument, 0, 't'},
    {"output", no_argument, 0, 'o'}
};

int main(int argc, char** argv){
    // argument parsing
    char ch, filePath[256];
    bool verbose = false, timeonly = false;
    while((ch = getopt_long(argc, argv, short_options, long_options, NULL)) != -1){
        switch(ch){
        case 'f':
            strncpy(filePath, optarg, 256);
            break;
        case 'p':
            CONSTANT_PROBABILITY = atof(optarg);
            CONSTANT_PROBABILITY = CONSTANT_PROBABILITY > 1 ? 1. : CONSTANT_PROBABILITY;
            break;
        case 'c':
            CONSTANT_PROBABILITY *= atoi(optarg);
            CONSTANT_PROBABILITY = CONSTANT_PROBABILITY > 1 ? 1. : CONSTANT_PROBABILITY;
            break;
        case 'v':
            verbose = true;
            break;
        case 't':
            timeonly = true;
            break;
        case 'o':
            freopen("../outputs/dynamicOutput.txt", "a", stdout);
            break;
        }
    }

    // read graph from file
    int totalNodes = 0, maxOutDegree = 0;
    int totalEdges = readGraph(filePath, h_adjCount, h_adjList, totalNodes, maxOutDegree);
    if(totalEdges < 0)
        return 0;
    if(!timeonly){
        printf("========= NEW RUN\n");
        printf("This graph contains %d nodes connected by %d edges\n", totalNodes, totalEdges);
        printf("Set constant probability: %.2f\n", CONSTANT_PROBABILITY);
    }

    // addresses for GPU memory addresses storage
    bool* d_closed;
    int* d_queue, *d_nodeSet;
    int* d_adjList, *d_adjCount;
    float* d_randSeed;
    float gpu_runtime;

    curandState* d_randState;
    cudaEvent_t start, stop;

    // define GPU thread layout
    dim3 gridSize(2,2), blockSize(16,16);

    // generate random numbers for each thread as random seeds
    curandGenerator_t curandGenerator;
    cudaMalloc((void**)&d_randSeed, sizeof(float) * THREAD);
    cudaMalloc((void**)&d_randState, sizeof(curandState) * THREAD);
    curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curandGenerator, time(NULL));
    curandGenerateUniform(curandGenerator, d_randSeed, THREAD);
    setupRandGenerator<<<gridSize,blockSize>>>(d_randSeed, d_randState);

    // cuda memory allocation and initialization
    cudaMalloc((void**)&d_closed, sizeof(bool) * THREAD * totalNodes);
    cudaMalloc((void**)&d_queue, sizeof(int) * THREAD * QUE_LEN);    // compress?
    cudaMalloc((void**)&d_nodeSet, sizeof(int) * totalNodes + 1);
    cudaMalloc((void**)&d_adjList, sizeof(int) * totalEdges);
    cudaMalloc((void**)&d_adjCount, sizeof(int) * (totalNodes + 1));

    cudaMemset(d_nodeSet, 0, sizeof(int) * totalNodes + 1);
    cudaMemset(d_closed, false, sizeof(bool) * THREAD * totalNodes);
    cudaMemcpy(d_adjList, h_adjList, sizeof(int) * totalEdges, cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjCount,
               h_adjCount,
               sizeof(int) * (totalNodes + 1),
               cudaMemcpyHostToDevice);

    // elapsed time record
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // launch the kernel
    bfs<<<gridSize,blockSize>>>(totalNodes,
                                d_adjCount,
                                d_adjList,
                                d_nodeSet,
                                d_queue,
                                d_closed,
                                d_randState,
                                CONSTANT_PROBABILITY);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_runtime, start, stop);

    // statistics
    if(verbose && !timeonly){
        cudaMemcpy(h_nodeSet,
                   d_nodeSet,
                   sizeof(int) * totalNodes + 1,
                   cudaMemcpyDeviceToHost);
        for(int i = 0;i < totalNodes;i++)
            if(h_nodeSet[i] > 0)
                printf("influence of node %d: %d\n", i, h_nodeSet[i]);
    }
    if(!timeonly)
        printf("========= GPU ELAPSED TIME: %f ms\n\n", gpu_runtime);
    else
        printf("%f\n", gpu_runtime);

    // cuda memory free
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_randSeed);
    cudaFree(d_randState);
    cudaFree(d_closed);
    cudaFree(d_queue);
    cudaFree(d_nodeSet);
    cudaFree(d_adjList);
    cudaFree(d_adjCount);

    return 0;
}
