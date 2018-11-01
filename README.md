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

This following charts shows GPU (2 x 2 blocks and 16 x 16 threads in each block) elapsed time on the given graph in file `wiki.txt`, which contains 8297 nodes connected by 103689 edges.

Conclusion: In average, dynamic assignment is 100ms faster than static assignment.

#### NVIDIA GTX 970M

- Ubuntu 18.04.1  
- Intel i7-6700HQ  
- NVIDIA GTX 970M (Maxwell, 1280 CUDA cores)  
- CUDA Toolkit 9.0

|const probability|dynamic|static|
|:-:|:-:|:-:|
|0.05|473.45|599.15|
|0.10|1081.77|1257.06|
|0.15|1413.74|1620.48|
|0.20|1639.06|1814.76|
|0.25|1748.78|1922.30|
|0.30|1823.14|1990.03|
|0.35|1870.98|2036.42|
|0.40|1901.19|2074.21|
|0.45|1934.53|2090.98|
|0.50|1959.94|2140.40|

#### NVIDIA GTX 1070

> TODO