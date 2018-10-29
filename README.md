# Computational Data Science Lab

This is a repository for CISC Computational Data Science Lab

## Task 1

#### Files

- `data`: graph data, for input;  
- `outputs`: output information;  
- `maxInfluence`: source code

### Experiment 1

Given a directed acyclic graph, for each node in the graph run BFS on the whole graph and find out how many nodes can be reached. To do this parallel, each BFS process should be assigned to a thread in GPU. After one thread finish traversal, the thread will find another node which hasn't been used and start a new BFS from this node. Amount of threads is fixed.

Corresponding file: `dynamicAssign.cu`  
To compile, use following command:

```bash
$ nvcc --std=c++11 -lcurand dynamicAssign.cu -o dynamicAssign
```

#### Elapsed Time

This following chart shows GPU elapsed time on the given graph in file `wiki.txt`, which contains $8297$ nodes connected by $103689$ edges.

|const probability|elpsed time in average|
|:-:|:-:|
|0.01 (1%)|126.0 ms|
|0.1 (10%)|4875.6 ms|

#### Known Bugs

- [x] results are all the same for different independent tries, maybe a bug related to `curand`;  
- [ ] recurrent queue has a fixed size, and if overflow the answer may be wrong, even the program may crash;  

### Experiment 2

Given a directed acyclic graph, assign nodes to threads manually and each thread will run a BFS starting from the given point.

corresponding file: [file_name]
