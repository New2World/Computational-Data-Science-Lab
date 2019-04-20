# Adaptive Influence Maximization

Adaptive influence maximization (AIM) is a method to select seed set after observing certain diffusion results, and it investigates adaptive seeding strategies for maximizing the influence. <sup>[1]</sup>

## Run

```bash
$ javac *.java
$ java adaptive/Start_time <dataset> <type> <vnum> <simutimes> <round> <simurest_times> <k>
```

_example_  

```bash
$ java adaptive/Start_time wiki WC 8300 100 5 100 20
```

---

[1] G. Tong, R. Wang _"On Adaptive Influence Maximization under General Feedback Models"_,