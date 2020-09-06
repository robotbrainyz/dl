import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import timeit
import torch

def plot_costs(costs):
    costsTorch = torch.stack(costs, dim=0)#[n for n in costs]
    columnIndexColumn = torch.arange(len(costs))
    columnIndexColumn = columnIndexColumn.reshape(columnIndexColumn.shape[0], 1)
    costsTorch = torch.cat([costsTorch, columnIndexColumn], dim=1)
    indices = torch.arange(costsTorch.shape[1]-1)
    columnNames = ['feature ' + str(i) for i in indices]
    columnNames.append('indexCol')
    df = pd.DataFrame(costsTorch.numpy(), columns=columnNames) # column names are compulsory
    sns.lineplot(data = pd.melt(df, ['indexCol']), x = 'indexCol', y = 'value', hue='variable')
    plt.show()

def plot_time(totalTime, totalTimings, batchTimings):
    batchTimesNp = torch.tensor(batchTimings)
    totalTimesNp = torch.tensor(totalTimings)
    indices = torch.arange(len(batchTimesNp))
    timings = torch.stack([batchTimesNp, totalTimesNp, indices], dim=0)
    timings = torch.transpose(timings, 0, 1)
    df = pd.DataFrame(timings.numpy(), columns=['iteration duration', 'elapsed', 'indexCol']) # column names are compulsory

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
    plot_time(totalTime, totalTimes, batchTimes)
        

    
def test_plot_costs2():
    costs = []
    loss = torch.randn(4,50)
    cost = torch.true_divide(torch.sum(loss, dim = 1), loss.shape[1])
    costs.append(cost)
    costs.append(cost)
    costs.append(cost)
    costs.append(cost)
    costs.append(cost)

    costsTorch = torch.stack(costs, dim=0)#[n for n in costs]
    columnIndexColumn = torch.arange(5)
    columnIndexColumn = columnIndexColumn.reshape(columnIndexColumn.shape[0], 1)
    costsTorch = torch.cat([costsTorch, columnIndexColumn], dim=1)
    print(costsTorch)
    df = pd.DataFrame(costsTorch.numpy(), columns=['feature 1', 'feature 2', 'feature 3', 'feature 4', 'indexCol']) # column names are compulsory

    sns.lineplot(data = pd.melt(df, ['indexCol']), x = 'indexCol', y = 'value', hue='variable')
    plt.show()
    
    

if __name__ == "__main__":
    #test_plot_costs2()
    test_plot_training_time()
