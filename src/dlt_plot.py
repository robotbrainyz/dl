import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import timeit
import torch

def plot_costs(costs):
    costsNumpy = torch.stack(costs, dim=0)#[n for n in costs]
    columnIndexColumn = torch.arange(len(costs))
    columnIndexColumn = columnIndexColumn.reshape(columnIndexColumn.shape[0], 1)
    costsNumpy = torch.stack((costsNumpy, columnIndexColumn), dim=1)
    indices = torch.arange(costsNumpy.shape[1]-1)
    columnNames = ['feature ' + str(i) for i in indices]
    columnNames.append('indexCol')
    df = pd.DataFrame(costsNumpy, columns=columnNames) # column names are compulsory
    sns.lineplot(data = pd.melt(df, ['indexCol']), x = 'indexCol', y = 'value', hue='variable')
    plt.show()    

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
    batchTimesNp = torch.tensor(batchTimes)
    totalTimesNp = torch.tensor(totalTimes)
    indices = torch.arange(len(batchTimesNp))
    timings = torch.stack((batchTimesNp, totalTimesNp, indices), dim=1)
    timings = timings.reshape(len(batchTimesNp), 3)
    timings = torch.transpose(timings, 0, 1)
    print(timings)
    df = pd.DataFrame(timings, columns=['iteration duration', 'elapsed', 'indexCol']) # column names are compulsory

    sns.lineplot(data = pd.melt(df, ['indexCol']), x = 'indexCol', y = 'value', hue='variable')
    plt.show()    
        

    
def test_plot_costs2():
    costs = []
    loss = torch.randn(4,50)
    cost = torch.true_divide(torch.sum(loss, dim = 1), loss.shape[1])
    costs.append(cost)
    costs.append(cost)
    costs.append(cost)
    costs.append(cost)
    costs.append(cost)


    costsNumpy = torch.stack(costs, dim=0)#[n for n in costs]
    columnIndexColumn = torch.arange(5)
    columnIndexColumn = columnIndexColumn.reshape(columnIndexColumn.shape[0], 1)
    costsNumpy = torch.stack((costsNumpy, columnIndexColumn), dim=1)
    print(costsNumpy)
    df = pd.DataFrame(costsNumpy, columns=['feature 1', 'feature 2', 'feature 3', 'feature 4', 'indexCol']) # column names are compulsory

    sns.lineplot(data = pd.melt(df, ['indexCol']), x = 'indexCol', y = 'value', hue='variable')
    plt.show()
    
def test_plot_costs():
    # may need [capitalize(n) for n in names] to concatenate 
    sns.set(style="darkgrid")
    costs = torch.randn(4, 50)
    columnIndexRow = torch.arange(50)
    costs = torch.stack([costs, columnIndexRow], dim=0)
    costs = torch.transpose(costs, 0, 1)
    df = pd.DataFrame(costs, columns=['feature 1', 'feature 2', 'feature 3', 'feature 4', 'indexCol']) # column names are compulsory

    sns.lineplot(data = pd.melt(df, ['indexCol']), x = 'indexCol', y = 'value', hue='variable')
    print(df)
    plt.show()
    

if __name__ == "__main__":
    #test_plot_costs()
    #test_plot_costs2()
    test_plot_training_time()
