import sys
import math

filename = sys.argv[1]
lines = open(filename).readlines()

ln = len(lines)
n = 50 if ln < 2000 else 100

def get_mean_and_std(l):
    mean = sum(l) / len(l)
    nl = [(i-mean)**2 for i in l]
    std = math.sqrt(sum(nl)/len(nl))
    return f"{mean:0.3f}({std:0.3f})"

for start in range(1, ln, n):
    end = start + n
    reward_list = []
    success_list = []
    for l in range(start, end):
        _,_,_,reward,success,collide,timecost = lines[l].split(',')
        reward_list.append(float(reward))
        success_list.append(float(eval(success)))
    print((start-1)// n, get_mean_and_std(reward_list), get_mean_and_std(success_list))
