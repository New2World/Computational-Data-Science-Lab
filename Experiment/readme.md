# Experiment (Reverse + MpU)

In this task, we will combine the reverse traversal and MpU[1] to solve some problems.  
Reverse traversal progress will generate several subsets of vertices, and these subsets are called hyperedges. Then these hyperedges will be the input of MpU problem to find a union of vertices included in hyperedges.

## Compile

- Compile  

```bash
$ g++ main.cpp -std=c++11 -O3 -o main
```

- Run  

```bash
$ ./main
```

after the program start, please input path to your dataset and the input file and # of cases you want (input 0 will output all cases listed in input file).  
The format of input file should be like this

```
s 1051 t 3825 alpha 0.200 L 7398809 pmax 0.024 beta 0.110
```

where `s` and `t` refer to source node and sink node, `alpha` is not used in this program, `L` is # of iterations of reverse traversal, `L * beta` is the value of `p` ,and to check the result, # of hyperedges the program outputs should be close to `L * pmax`.

---
## Reference

[1] Chlamtac, E.; Dinitz, M.; Konrad, C.; Kortsarz, G.; and Rabanca, G. 2018. The densest k-subhypergraph problem. _SIAM Journal on Discrete Mathematics 32(2):1458–1477_.