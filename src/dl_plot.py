import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import timeit

def test_plot_training_time():
    totalTime = 0
    totalTimes = []
    batchTimes = []
    for i in range(0, 3):
        startTime = timeit.timeit()
        time.sleep(i+1)
        endTime = timeit.timeit()
        elapsed = endTime-startTime
        totalTime = totalTime + elapsed
        batchTimes.append(elapsed)        
        totalTimes.append(totalTime)
    batchTimesNp = np.array(batchTimes)
    totalTimesNp = np.array(totalTimes)
    indices = np.arange(len(batchTimesNp))
    timings = np.hstack((batchTimesNp, totalTimesNp, indices))
    timings = timings.reshape(len(batchTimesNp), 3)
    timings = np.transpose(timings)
    print(timings)
    df = pd.DataFrame(timings, columns=['iteration duration', 'elapsed', 'indexCol']) # column names are compulsory

    sns.lineplot(data = pd.melt(df, ['indexCol']), x = 'indexCol', y = 'value', hue='variable')
    plt.show()    
        
    
def test_plot_costs2():
    costs = []
    loss = np.random.randn(4,50)
    cost = np.divide(np.sum(loss, axis = 1), loss.shape[1])
    costs.append(cost)
    costs.append(cost)
    costs.append(cost)
    costs.append(cost)
    costs.append(cost)


    costsNumpy = np.vstack(costs)#[n for n in costs]
    columnIndexColumn = np.arange(5)
    columnIndexColumn = columnIndexColumn.reshape(columnIndexColumn.shape[0], 1)
    costsNumpy = np.hstack((costsNumpy, columnIndexColumn))
    print(costsNumpy)
    df = pd.DataFrame(costsNumpy, columns=['feature 1', 'feature 2', 'feature 3', 'feature 4', 'indexCol']) # column names are compulsory

    sns.lineplot(data = pd.melt(df, ['indexCol']), x = 'indexCol', y = 'value', hue='variable')
    plt.show()
    
def test_plot_costs():
    # may need [capitalize(n) for n in names] to concatenate 
    sns.set(style="darkgrid")
    costs = np.random.randn(4, 50)
    columnIndexRow = np.arange(50)
    costs = np.vstack([costs, columnIndexRow])
    costs = np.transpose(costs)
    df = pd.DataFrame(costs, columns=['feature 1', 'feature 2', 'feature 3', 'feature 4', 'indexCol']) # column names are compulsory

    sns.lineplot(data = pd.melt(df, ['indexCol']), x = 'indexCol', y = 'value', hue='variable')
    print(df)
    plt.show()
    

if __name__ == "__main__":
    #test_plot_costs()
    test_plot_costs2()
    #test_plot_training_time()
