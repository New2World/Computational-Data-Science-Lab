#include <stdio.h>
#include <cuda.h>

int main(){
    struct cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("global Mem.: %ld Mb\nconst Mem.: %ld Kb\nMem. per block: %ld Kb\nwarp size: %d\nmulti processor count: %d\nmax thread per block: %d\n", \
            devProp.totalGlobalMem / 1024 / 1024,
            devProp.totalConstMem / 1024,
            devProp.sharedMemPerBlock / 1024,
            devProp.warpSize,
            devProp.multiProcessorCount,
            devProp.maxThreadsPerBlock);
    printf("max threads dim: [%d %d %d]\n",
            devProp.maxThreadsDim[0],
            devProp.maxThreadsDim[1],
            devProp.maxThreadsDim[2]);
    printf("max grid size: [%d %d %d]\n",
            devProp.maxGridSize[0],
            devProp.maxGridSize[1],
            devProp.maxGridSize[2]);
    return 0;
}