#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <getopt.h>

#include <vector>
#include <utility>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define QUE_LEN 1000
#define QUE_ST(x) (x * QUE_LEN)
#define QUE_ED(x) (x * QUE_LEN + QUE_LEN)

#define RAND_FACTOR 1e9+7

#define MAX_OUT 2000
#define MAX_NODE 10000
#define THREAD 1024

float CONSTANT_PROBABILITY = 0.01;
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
__device__ int findVertice(int* nodeSet, int from, int nodes){
    for(int i = from + 1;i < nodes;i++)
        if(atomicCAS(&nodeSet[i], -1, 0) == -1)
            return i;
    return from;
}

// judge if queue overflow
__device__ bool que_isFull(int que_h, int que_t){
    return que_t == que_h;
}

// judge if queue is empty
__device__ bool que_isEmpty(int que_h, int que_t, int index){
    return (que_t == que_h + 1) || (que_t == que_h + 1 - QUE_LEN);
}

// clear all elements in queue and reset queue
__device__ void que_clear(int& que_h, int& que_t, int index){
    que_h = QUE_ST(index);
    que_t = que_h + 1;
}

// enqueue
__device__ bool que_enque(int* queue, int que_h, int& que_t, int val, int index){
    if(que_isFull(que_h, que_t))
        return false;
    int tail = que_t - 1;
    if(tail < QUE_ST(index))
        tail += QUE_LEN;
    queue[tail] = val;
    que_t++;
    if(que_t >= QUE_ED(index))
        que_t -= QUE_LEN;
    return true;
}

// dequeue
__device__ int que_deque(int* queue, int& que_h, int que_t, int index){
    int val = -1;
    if(que_isEmpty(que_h, que_t, index))
        return val;
    val = queue[que_h];
    que_h++;
    if(que_h >= QUE_ED(index))
        que_h -= QUE_LEN;
    return val;
}

// judge if node 'nd' is visited once
__device__ bool nd_isVisited(bool* vis, int nd, int index){
    return vis[index * THREAD + nd];
}

// set node 'nd' visited
__device__ void nd_setVisited(bool* vis, int nd, int index){
    vis[index * THREAD + nd] = true;
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
    int next, prev = -1;
    float randProb;
    int que_h = QUE_ST(index), que_t = que_h + 1;
    curandState localState = state[index];
    while(node != prev){
        prev = node;
        que_clear(que_h, que_t, index);
        if(!que_enque(queue, que_h, que_t, prev, index));   // in case queue overflow
        nd_setVisited(closed, prev, index);
        while(!que_isEmpty(que_h, que_t, index)){
            node = que_deque(queue, que_h, que_t, index);
            next = adjCount[node];
            while(next < adjCount[node + 1]){
                if(!nd_isVisited(closed, adjList[next], index)){
                    randProb = curand_uniform(&localState);
                    if(randProb < constProb){
                        if(!que_enque(queue, que_h, que_t, adjList[next], index));
                        nd_setVisited(closed, adjList[next], index);
                        count++;
                    }
                }
                next++;
            }
        }
        if(atomicCAS(nodeSet + prev, 0, count) != 0);   // theoretically impossiable
        node = findVertice(nodeSet, prev, totalNodes);
    }
    state[index] = localState;
}

// global variables
int h_adjCount[MAX_NODE];
int h_adjList[MAX_NODE * MAX_OUT];
int h_nodeSet[MAX_NODE];

// for argument parsing
char short_options[] = "p::";
struct option long_options[]{
    {"probability", optional_argument, 0, 'p'}
};

int main(int argc, char** argv){
    // argument parsing
    char ch;
    while((ch = getopt_long(argc, argv, short_options, long_options, NULL)) != -1){
        switch(ch){
        case 'p':
            CONSTANT_PROBABILITY = atof(optarg);
            break;
        }
    }

    // read graph from file
    int totalNodes = 0, maxOutDegree = 0;
    int totalEdges = readGraph("../data/wiki.txt", h_adjCount, h_adjList, totalNodes, maxOutDegree);
    if(totalEdges < 0)
        return 0;
    printf("This graph contains %d nodes connected by %d edges\n", totalNodes, totalEdges);

    // addresses for GPU memory addresses storage
    bool* d_closed;
    int* d_queue, *d_nodeSet;
    int* d_adjList, *d_adjCount;
    float* d_randSeed;
    float gpu_runtime;

    curandState* d_randState;
    cudaEvent_t start, stop;

    // define GPU thread layout
    dim3 gridSize(1,1), blockSize(32,32);

    // generate random numbers for each thread as random seeds
    curandGenerator_t curandGenerator;
    cudaMalloc((void**)&d_randSeed, sizeof(float) * THREAD);
    cudaMalloc((void**)&d_randState, sizeof(curandState) * THREAD);
    curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curandGenerator, time(NULL));
    curandGenerateUniform(curandGenerator, d_randSeed, THREAD);
    setupRandGenerator<<<gridSize,blockSize>>>(d_randSeed, d_randState);

    // cuda memory allocation
    cudaMalloc((void**)&d_closed, sizeof(bool) * THREAD * totalNodes);
    cudaMalloc((void**)&d_queue, sizeof(int) * THREAD * QUE_LEN);    // compress?
    cudaMalloc((void**)&d_nodeSet, sizeof(int) * totalNodes);
    cudaMalloc((void**)&d_adjList, sizeof(int) * totalEdges);
    cudaMalloc((void**)&d_adjCount, sizeof(int) * (totalNodes + 1));

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

    cudaMemcpy(h_nodeSet, d_nodeSet, sizeof(int) * totalNodes, cudaMemcpyDeviceToHost);

    // statistics
    for(int i = 0;i < totalNodes;i++)
        if(h_nodeSet[i] > 0)
            printf("influence of node %d: %d\n", i, h_nodeSet[i]);
    printf("========= GPU ELAPSED TIME: %f ms\n", gpu_runtime);

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
