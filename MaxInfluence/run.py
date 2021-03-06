import argparse
import subprocess
import collections

import numpy as np
import matplotlib.pyplot as plt

DYNAMIC_SRC = "./dynamicAssign.cu"
DYNAMIC_EXE = "./dynamicAssign"
STATIC_SRC = "./staticAssign.cu"
STATIC_EXE = "./staticAssign"

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file",
                    type=str,
                    default="../data/wiki.txt",
                    help="load graph from file")
parser.add_argument("-c", "--compile",
                    action="store_true",
                    help="whether need to compile")
parser.add_argument("-p", "--probability",
                    type=float,
                    default=0.01,
                    help="specify probability of edges")
parser.add_argument("-o", "--output",
                    action="store_true",
                    help="output the outcome to file")
parser.set_defaults(compile=False)
parser.set_defaults(output=False)

def compile_src():
    subprocess.call(['nvcc','-lcurand',DYNAMIC_SRC,'-o',DYNAMIC_EXE])
    subprocess.call(['nvcc','-lcurand',STATIC_SRC,'-o',STATIC_EXE])

def exec_prog(probability):
    dynamic_raw = subprocess.check_output([DYNAMIC_EXE,
                                          "-f",DATA_FILE,
                                          "-p"+str(probability),
                                          "--thread"])
    dynamic_results = dynamic_raw.strip().split('\n')
    static_raw = subprocess.check_output([STATIC_EXE,
                                         "-f",DATA_FILE,
                                         "-p"+str(probability),
                                         "--thread"])
    static_results = static_raw.strip().split('\n')
    dynamic_time = dynamic_results[-1]
    static_time = static_results[-1]
    return dynamic_results[:-1], \
           static_results[:-1], \
           float(dynamic_time), \
           float(static_time), \
           dynamic_raw, \
           static_raw

def extract_info(results):
    node_count = [int(item.split()[1]) for item in results]
    time_count = [float(item.split()[2]) for item in results]
    return node_count, time_count

def anaylsis_std(node_count, time_count):
    return np.std(node_count), np.std(time_count)

def draw_statistic(dynamic_node_count,
                   dynamic_node_std,
                   dynamic_time_count,
                   dynamic_time_std,
                   static_node_count,
                   static_node_std,
                   static_time_count,
                   static_time_std):
    plt.subplot(211)
    plt.title("Elapsed Time Comparison of Two Methods")
    plt.plot(range(len(dynamic_time_count)), dynamic_time_count, c='b',
             label="dynamic (std = %f)" % dynamic_time_std)
    plt.plot(range(len(static_time_count)), static_time_count, c='r',
             label="static (std = %f)" % static_time_std)
    plt.legend()

    plt.subplot(212)
    plt.title("Number of Nodes Comparison of Two Methods")
    plt.plot(range(len(dynamic_node_count)), dynamic_node_count, c='b',
             label="dynamic (std = %f)" % dynamic_node_std)
    plt.plot(range(len(static_node_count)), static_node_count, c='r',
             label="static (std = %f)" % static_node_std)
    plt.legend();

    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    DATA_FILE = args.file
    PROBABILITY = args.probability

    if args.compile:
        compile_src()

    dynamic_results, \
    static_results, \
    dynamic_time, \
    static_time, \
    dynamic_raw, \
    static_raw = exec_prog(PROBABILITY)

    dynamic_node_count, \
    dynamic_time_count = extract_info(dynamic_results)
    static_node_count, \
    static_time_count = extract_info(static_results)

    dynamic_node_std, \
    dynamic_time_std = anaylsis_std(dynamic_node_count, dynamic_time_count)
    static_node_std, \
    static_time_std = anaylsis_std(static_node_count, static_time_count)

    if args.output:
        with open("../res/elapsed_time_of_each_thread_dynamic.txt","w") as fd:
            fd.write("thread nodes time\n")
            fd.write(dynamic_raw)
        with open("../res/elapsed_time_of_each_thread_static.txt","w") as fd:
            fd.write("thread nodes time\n")
            fd.write(static_raw)

    print "Total time:"
    print "  dynamic: %f ms" % dynamic_time
    print "  static: %f ms" % static_time

    draw_statistic(dynamic_node_count,
                   dynamic_node_std,
                   dynamic_time_count,
                   dynamic_time_std,
                   static_node_count,
                   static_node_std,
                   static_time_count,
                   static_time_std)