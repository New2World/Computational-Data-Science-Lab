# Computational Data Science Lab

This is a repository for CISC Computational Data Science Lab

## Task 1

#### Files

- `data`: graph data, for input;  
- `outputs`: output information;  
- `maxInfluence`: source code;

### Experiment 1: Dynamic Assignment

Given a directed acyclic graph, for each node in the graph run BFS on the whole graph and find out how many nodes can be reached. To do this parallel, each BFS process should be assigned to a thread in GPU. After one thread finish traversal, the thread will find another node which hasn't been used and start a new BFS from this node. Amount of threads is fixed.

Corresponding file: `dynamicAssign.cu`  
To compile, use following command:

```bash
$ nvcc -lcurand dynamicAssign.cu -o dynamicAssign
```

#### Known Bugs

- [ ] recurrent queue has a fixed size, and if overflow the answer may be wrong, even the program may crash;  

### Experiment 2: Static Assignment

Given a directed acyclic graph, assign nodes to threads manually and each thread will run a BFS starting from the given point.

corresponding file: `staticAssign.cu`
To compile, use following command:

```bash
$ nvcc -lcurand staticAssign.cu -o staticAssign
```

#### Known Bugs

- [ ] recurrent queue has a fixed size, and if overflow the answer may be wrong, even the program may crash;  

### Comparison

This following charts shows GPU (3 x 3 blocks and 32 x 32 threads in each block) elapsed time on the given graph in file `wiki.txt`, which contains 8297 nodes connected by 103689 edges.

- Ubuntu 18.04.1  
- Intel i7-6700HQ  
- NVIDIA GTX 970M (Maxwell, 1280 CUDA cores)  
- CUDA Toolkit 9.0

|const probability|dynamic|static|
|:-:|:-:|:-:|
|0.05|191.03|196.13|
|0.10|318.48|336.78|
|0.15|378.01|386.64|
|0.20|398.24|410.29|
|0.25|406.88|414.17|
|0.30|414.77|412.66|
|0.35|411.43|414.74|
|0.40|407.20|413.56|
|0.45|407.44|410.52|
|0.50|410.17|410.90|