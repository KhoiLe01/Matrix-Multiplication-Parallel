'''
A simple python code to draw a graph from input1.txt. In that txt file each line has 3 values: the pid of a process, a number of ticket, and the tick 
of that process
'''

import matplotlib.pyplot as plt
import numpy as np

def stripEnding0(x,y):
    count = 0
    for i in range(len(y)-1, 0, -1):    
        if (y[i] != 0):
            length = len(y)-count
            break
        else:
            count += 1;    
    y = [y[i] for i in range(0,length)]
    x = [i for i in range(0,length)]
    return x,y

def fillPrevNum(y):
    for i in range(1,len(y)):
        if y[i] == 0:
            y[i] = y[i-1]

def main():
    lines = []
    with open('time.txt') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = [float(c) for c in line.split()]
            lines.append(line)

    x1 = [i for i in range(1,len(lines)+1)]
    y1 = [250,500,750,100,1250,1500,2000,2250]
    x2 = [i for i in range(1,len(lines)+1)]
    y2 = [250,500,750,100,1250,1500,2000,2250]
    x3 = [i for i in range(1,len(lines)+1)]
    y3 = [250,500,750,100,1250,1500,2000,2250]
    

    # for i in range(len(lines)):
    #     if lines[i][0] == 4:
    #         y1[i] = (lines[i][-1])
    #     elif lines[i][0] == 5:
    #         y2[i] = (lines[i][-1])
    #     elif lines[i][0] == 6:
    #         y3[i] = (lines[i][-1])

    
    x1,y1 = stripEnding0(x1,y1)
    x2,y2 = stripEnding0(x2,y2)
    x3,y3 = stripEnding0(x3,y3)

    # fillPrevNum(y1)
    # fillPrevNum(y2)
    # fillPrevNum(y3)

    xpoints1 = np.array(x1)
    ypoints1 = np.array(y1)
    xpoints2 = np.array(x2)
    ypoints2 = np.array(y2)
    xpoints3 = np.array(x3)
    ypoints3 = np.array(y3)

    plt.plot(xpoints1, ypoints1, label="process 4")
    plt.plot(xpoints2, ypoints2, label="process 5")
    plt.plot(xpoints3, ypoints3, label="process 6")

    plt.xlabel("Time")
    plt.ylabel("Ticks")

    plt.legend()
    plt.show()

main()