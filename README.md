# Computational Data Science Lab

This is a repository for CISC Computational Data Science Lab

## Task 1

#### Files

- `data`: graph data, for input;  
- `outputs`: output information;  
- `maxInfluence`: source code;

### Experiment 1

Given a directed acyclic graph, for each node in the graph run BFS on the whole graph and find out how many nodes can be reached. To do this parallel, each BFS process should be assigned to a thread in GPU. After one thread finish traversal, the thread will find another node which hasn't been used and start a new BFS from this node. Amount of threads is fixed.

Corresponding file: `dynamicAssign.cu`  
To compile, use following command:

```bash
$ nvcc -lcurand dynamicAssign.cu -o dynamicAssign
```

#### Environment

- Ubuntu 18.04.1  
- Intel i7-6700HQ  
- NVIDIA GTX 970M (Maxwell, 1280 CUDA cores)  
- CUDA Toolkit 9.0

#### Elapsed Time

This following chart shows GPU (2 x 2 blocks and 16 x 16 threads in each block) elapsed time on the given graph in file `wiki.txt`, which contains 8297 nodes connected by 103689 edges.

|const probability|elpsed time in average (ms)|
|:-:|:-:|
|0.05|474.34|
|0.10|1066.60|
|0.15|1420.07|
|0.20|1629.75|
|0.25|1747.10|
|0.30|1828.43|
|0.35|1876.59|
|0.40|1899.53|
|0.45|1933.67|
|0.50|1953.73|

#### Known Bugs

- [x] results are all the same for different independent tries, maybe a bug related to `curand`;  
- [ ] recurrent queue has a fixed size, and if overflow the answer may be wrong, even the program may crash;  

### Experiment 2

Given a directed acyclic graph, assign nodes to threads manually and each thread will run a BFS starting from the given point.

corresponding file: [file_name]
