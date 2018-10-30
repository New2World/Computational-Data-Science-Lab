import numpy as np

with open('timeRecord_static.txt') as fdr:
    gpu_runtime = np.array([float(line.strip()) for line in fdr]).reshape((10,10))
    gpu_runtime_avg = np.mean(gpu_runtime, axis=1).tolist()
    with open('timeChart_static.txt','w') as fdw:
        fdw.writelines(["|%.2f|%.2f|\n" % ((i+1)*.05, gpu_runtime_avg[i]) for i in range(len(gpu_runtime_avg))])

with open('timeRecord_dynamic.txt') as fdr:
    gpu_runtime = np.array([float(line.strip()) for line in fdr]).reshape((10,10))
    gpu_runtime_avg = np.mean(gpu_runtime, axis=1).tolist()
    with open('timeChart_dynamic.txt','w') as fdw:
        fdw.writelines(["|%.2f|%.2f|\n" % ((i+1)*.05, gpu_runtime_avg[i]) for i in range(len(gpu_runtime_avg))])