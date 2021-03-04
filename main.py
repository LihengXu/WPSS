#! /usr/bin/env python3
import os

# set the the number of the search
rerun_num = 100

if __name__ == "__main__":

    # init the scenario_pool
    if os.path.exists("./scenario_pool/scenario_pool.csv"):
        os.remove("./scenario_pool/scenario_pool.csv")
    # start the experiments
    for x in range(rerun_num):
        os.system("python train.py")
        print()
        print()
        print("****************************")
        print("****************************")
        print("****************************")
        print("程序已重启第: % d 次..." % (x+1))
        print()
        print()
