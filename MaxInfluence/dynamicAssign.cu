#include <stdlib.h>
#include <memory.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "cuqueue.cuh"
#include "utils.h"

#define RAND_FACTOR 1e9+7

#define THREAD (9 * 256)

// get thread index
__device__ int getIndex(){
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int line = blockDim.x * gridDim.x;
    return row * line + col;
}

// find next unused node
__device__ LL findVertex(LL* nodeSet, LL totalNodes){
    int nextNode = atomicAdd(&nodeSet[0], 1LL);
    if(nextNode >= totalNodes)
        return 0;
    return nextNode;
}

// judge if node 'nd' is visited once
__device__ bool nd_isVisited(bool* vis, LL nd, LL tot, int index){
    return vis[index * tot + nd - 1];
}

// set node 'nd' visited
__device__ void nd_setVisited(bool* vis, LL nd, LL tot, int index){
    vis[index * tot + nd - 1] = true;
}

__device__ void nd_resetVisited(bool* vis, LL tot, int index){
    LL i = index * tot, end = index * tot + tot;
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
__global__ void bfs(LL totalNodes,
                    LL* adjCount,
                    LL* adjList,
                    LL* nodeSet,
                    LL* queue,
                    bool* closed,
                    curandState* state,
                    float constProb,
                    bool thread){
    int index = getIndex();
    LL nodeCount = 0, node = index + 1;
    int que_h, que_t;
    LL adjNode, prevNode;
    LL nodeSum = 0;
    float randProb;
    curandState localState = state[index];
    int start, stop;
    start = clock();
    while((node = findVertex(nodeSet, totalNodes)) > 0){
        nodeCount = 0;
        prevNode = node;
        que_init(que_h, que_t, index);
        nd_resetVisited(closed, totalNodes, index);
        if(!que_enque(queue, que_h, que_t, prevNode, index));   // overflow
        nd_setVisited(closed, prevNode, totalNodes, index);
        while(!que_isEmpty(que_h, que_t, index)){
            node = que_deque(queue, que_h, que_t, index);
            adjNode = adjCount[node - 1];
            while(adjNode < adjCount[node]){
                if(!nd_isVisited(closed, adjList[adjNode], totalNodes, index)){
                    randProb = curand_uniform(&localState);
                    if(randProb < constProb){
                        if(!que_enque(queue, que_h, que_t, adjList[adjNode], index));
                        nd_setVisited(closed, adjList[adjNode], totalNodes, index);
                        nodeCount++;
                    }
                }
                adjNode++;
            }
        }
        if(atomicCAS(nodeSet + prevNode, 0, nodeCount) != 0);
        nodeSum += nodeCount;
    }
    stop = clock();
    if(thread)
        printf("%d %lld %f\n", index, nodeSum, 1.f*(stop-start)/CLOCKS_PER_SEC);
    state[index] = localState;
}

float CONSTANT_PROBABILITY = 0.05;

int main(int argc, char** argv){
    // argument parsing
    char ch, filePath[256];
    bool thread = false;
    while((ch = getopt_long(argc, argv, short_options, long_options, NULL)) != -1){
        switch(ch){
        case 'f':
            strncpy(filePath, optarg, 256);
            break;
        case 'p':
            CONSTANT_PROBABILITY = atof(optarg);
            CONSTANT_PROBABILITY = CONSTANT_PROBABILITY > 1 ? 1. : CONSTANT_PROBABILITY;
            break;
        case 't':
            thread = true;
            break;
        }
    }

    // read graph from file
    LL totalNodes = 0, totalEdges = 0;
    LL* h_adjCount = NULL, *h_adjList = NULL;
    readGraph(filePath, h_adjList, h_adjCount, totalNodes, totalEdges);

    if(!thread){
        printf("========= NEW RUN\n");
        printf("This graph contains %lld nodes connected by %lld edges\n", totalNodes, totalEdges);
        printf("Set constant probability: %.2f\n", CONSTANT_PROBABILITY);
        printf("Running on %d threads\n", THREAD);
    }

    // addresses for GPU memory addresses storage
    bool* d_closed;
    LL* d_queue, *d_nodeSet;
    LL* d_adjList, *d_adjCount;
    float* d_randSeed;
    float gpu_runtime;

    curandState* d_randState;
    cudaEvent_t start, stop;

    // define GPU thread layout
    dim3 gridSize(3,3), blockSize(16,16);

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
    cudaMalloc((void**)&d_queue, sizeof(LL) * THREAD * QUE_LEN);    // compress?
    cudaMalloc((void**)&d_nodeSet, sizeof(LL) * (totalNodes + 1));
    cudaMalloc((void**)&d_adjList, sizeof(LL) * totalEdges);
    cudaMalloc((void**)&d_adjCount, sizeof(LL) * (totalNodes + 1));    // sum of edges before current node

    cudaMemset(d_nodeSet, 0LL, sizeof(LL) * (totalNodes + 1));
    cudaMemset(d_closed, false, sizeof(bool) * THREAD * totalNodes);
    cudaMemcpy(d_adjList,
               h_adjList,
               sizeof(LL) * totalEdges,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjCount,
               h_adjCount,
               sizeof(LL) * (totalNodes + 1),
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
                                CONSTANT_PROBABILITY,
                                thread);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_runtime, start, stop);

    // statistics
    if(!thread){
        // LL* h_nodeSet = new LL[totalNodes + 1];
        // cudaMemcpy(h_nodeSet,
        //            d_nodeSet,
        //            sizeof(LL) * (totalNodes + 1),
        //            cudaMemcpyDeviceToHost);
        // for(LL i = 1;i <= totalNodes;i++)
        //     if(h_nodeSet[i] > 0)
        //         printf("Node %lld influence %lld other nodes\n", i, h_nodeSet[i]);
        // delete[] h_nodeSet;
        printf("========= GPU ELAPSED TIME: %f ms\n\n", gpu_runtime);
    }

    if(thread)
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

    delete[] h_adjList;
    delete[] h_adjCount;

    return 0;
}
