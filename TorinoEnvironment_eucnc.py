import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class TorinoEnvironment:

    __variabilityParameter = 1000.0 # for the Dirichlet distribution, higher means less variability
    __backlogVsCPUloadCoeffient = 1.0 # how much does backlog cost with respect to CPU

    def __init__(self, location = [40097, 186], maxWorkOf1CPU = 100, minimumNrOfCPUs = 1, maximumNrOfCPUs = 50):
        self.minimumNrOfCPUs = minimumNrOfCPUs
        self.maximumNrOfCPUs = maximumNrOfCPUs
        df = pd.read_csv('cleaned-sorted-torino.csv')
        df = df[['lcd1', 'offset', 'flow']]
        df = df[(df['lcd1']==location[0]) & (df['offset']==location[1])]
        df.pop('lcd1')    
        df.pop('offset')
        self.__work = df.values / maxWorkOf1CPU
        self.__CPUload = np.zeros((minimumNrOfCPUs,), dtype='float')
        self.__backlog = np.zeros((minimumNrOfCPUs,), dtype='float')
        self.__time = 0
        self.stop = False
        self.duration = len(self.__work)

    def resetState(self):
        self.__time = 0
        self.__CPUload = np.zeros((self.minimumNrOfCPUs,), dtype='float')
        self.__backlog = np.zeros((self.minimumNrOfCPUs,), dtype='float')
        self.stop = False        

    def evolveState(self, nrOfCPUs):
        variabilityParameter = self.__variabilityParameter
        work = self.__work[self.__time].copy()
        backlog = self.__backlog
        nrOfCPUs = max(min(nrOfCPUs, self.maximumNrOfCPUs), self.minimumNrOfCPUs) # check if nrOfCPUs is within the bounds
        if(nrOfCPUs > len(backlog)): # if nrOfCPUs increases, additional CPUs are created with empty backlogs
            backlog = np.append(backlog, np.zeros(nrOfCPUs - len(backlog)))
        if(nrOfCPUs < len(backlog)):
            work += np.sum(backlog) # if nrOfCPUs decreases, all backlogs are added to the work first
            backlog = np.zeros(nrOfCPUs)
        backlog += np.random.dirichlet(np.ones(nrOfCPUs)*variabilityParameter)*work
        CPUload = backlog.clip(max=1.0)
        backlog = (backlog - CPUload).clip(min=0.0)
        self.__CPUload = CPUload
        self.__backlog = backlog
        self.__time += 1
        if(self.__time == self.duration): self.stop = True
        return nrOfCPUs

    def getReward(self):
        reward = np.min(self.__CPUload) - self.__backlogVsCPUloadCoeffient*np.max(self.__backlog)
        return reward        

    def monitorState(self, metric):
        if(metric == 'CPUload'): return self.__CPUload
        elif(metric == 'backlog'): return self.__backlog 
        elif(metric == 'work'): return self.__work
        elif(metric == 'time'): return self.__time
        elif(metric == 'instant_work'): return self.__work[max(self.__time-1, 0)][0]
        else: print(metric, 'not defined')
        
    def plotWork(self):
        plt.figure(figsize = (16,4))
        plt.xlabel('time [5m interval]')
        plt.ylabel('work')
        plt.plot(self.__work)
        
    def timeTravel(self, t):
        self.__backlog = np.zeros((self.minimumNrOfCPUs,), dtype='float')
        self.__time = t
        
    def head(self, percentage):
        if percentage < 0 or percentage > 1:
            print(precentage, ' does no belong to [0,1]')
        else:
            self.__work = self.__work[:int(len(self.__work)*percentage)]
            self.duration = len(self.__work)
            print(self.duration)
            
    def tail(self, percentage):
        if percentage < 0 or percentage > 1:
            print(precentage, ' does no belong to [0,1]')
        else:
            self.__work = self.__work[int(len(self.__work)*percentage):]
            self.duration = len(self.__work)
            print(self.duration)
