from math import factorial, exp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class TorinoEnvironment:

    __variabilityParameter = 1000.0 # for the Dirichlet distribution, higher means less variability
    __backlogVsCPUloadCoeffient = 1.0 # how much does backlog cost with respect to CPU

    def __init__(self, location = [40097, 186], maxWorkOf1CPU = 5.82*10**5,
            minimumNrOfCPUs = 1, maximumNrOfCPUs = 50, backlogCoeff=1/3,
            cpuCoeff=1/3, delayCoeff=1/3, opPerVehicle=(2*98)**2,
            rtx=10*10):
        """
        maxWorkOf1CPU: [operations/sec], by default - haversine dist per second
        opPerVehicle: [operations/vehicle], by default - 10 mini-slot messages
                      for redundancy, 10 messages for a second, and computing
                      distances among 2*98 vehicles in the x2 lanes of Orbassano
        rtx: rtx a vehicle does within a second - default x10 for reliability &
                                           x10 Tx [R.5.3-004][22.186 v16.2.0]
        """
        self.__cpuCoeff = cpuCoeff
        self.__delayCoeff = delayCoeff
        self.__maxWorkOf1CPU = maxWorkOf1CPU
        self.__opPerVehicle= opPerVehicle
        self.__backlogVsCPUloadCoeffient = backlogCoeff
        self.__rtx = rtx
        self.minimumNrOfCPUs = minimumNrOfCPUs
        self.maximumNrOfCPUs = maximumNrOfCPUs
        df = pd.read_csv('cleaned-sorted-torino.csv')
        df = df[['lcd1', 'offset', 'flow']]
        df = df[(df['lcd1']==location[0]) & (df['offset']==location[1])]
        df.pop('lcd1')    
        df.pop('offset')
        self.__flowHours = df.values
        self.__flowSecs = df.values / (60*60)
        self.__work = self.__flowSecs * rtx * opPerVehicle / maxWorkOf1CPU
        self.__CPUload = np.zeros((minimumNrOfCPUs,), dtype='float')
        self.__backlog = np.zeros((minimumNrOfCPUs,), dtype='float')
        self.__time = 0
        self.stop = False
        self.duration = len(self.__work)
        self.delay = 0.0


    def __earlang(self, R, A):
        num = A**R/factorial(R) * R/(R-A)
        denom = sum(map(lambda i: A**i/factorial(i), range(R))) + num
        return num / denom

    def __delay_MGR_PS(self, R, r_peak, rho, x):
        return x / r_peak * (1 + self.__earlang(R, R*rho) / (R*(1-rho)))


    def resetState(self):
        self.__time = 0
        self.__CPUload = np.zeros((self.minimumNrOfCPUs,), dtype='float')
        self.__backlog = np.zeros((self.minimumNrOfCPUs,), dtype='float')
        self.stop = False        

    def evolveState(self, nrOfCPUs, forgetBacklog=True, uniform=True,
            cpuAdmission=.99):
        variabilityParameter = self.__variabilityParameter
        work = self.__work[self.__time].copy()
        backlog = self.__backlog if not forgetBacklog else np.zeros(1)
        np.zeros((self.minimumNrOfCPUs))
        nrOfCPUs = max(min(nrOfCPUs, self.maximumNrOfCPUs), self.minimumNrOfCPUs) # check if nrOfCPUs is within the bounds
        if(nrOfCPUs > len(backlog)): # if nrOfCPUs increases, additional CPUs are created with empty backlogs
            backlog = np.append(backlog, np.zeros(nrOfCPUs - len(backlog)))
        if(nrOfCPUs < len(backlog)):
            work += np.sum(backlog) # if nrOfCPUs decreases, all backlogs are added to the work first
            backlog = np.zeros(nrOfCPUs)
        if uniform:
            backlog += np.ones(nrOfCPUs)*work/nrOfCPUs
        else:
            backlog += np.random.dirichlet(np.ones(nrOfCPUs)*variabilityParameter)*work
        CPUload = backlog.clip(max=cpuAdmission)
        backlog = (backlog - CPUload).clip(min=0.0)
        self.__CPUload = CPUload
        self.__backlog = backlog
        self.__time += 1
        if(self.__time == self.duration): self.stop = True
        return nrOfCPUs

    def getReward(self, withDelay=True, delay_th=.1, excessPenalty=False,
            excessTh=.1, excessW=1/3):
        """
        delay_th: in seconds - default 100ms [R.5.3-004][22.186 v16.2.0]
        excessPenalty: penalize the excess of CPUs
        excessTh: penalize if a CPU is below a load of .1
        """
        unusedPenalty = 0
        if excessPenalty:
            unusedPenalty = len(list(filter(lambda C: C < excessTh,
                self.__CPUload))) / self.maximumNrOfCPUs * excessW

        if withDelay:
            delay = self.__delay_MGR_PS(R=len(self.__CPUload),
                      r_peak=self.__maxWorkOf1CPU,
                      rho=np.mean(self.__CPUload),
                      x=self.__opPerVehicle)
            self.delay = delay
            delay_term = exp(-1* (delay/delay_th)**2)
            return self.__cpuCoeff * np.min(self.__CPUload)-\
                    self.__backlogVsCPUloadCoeffient * np.max(self.__backlog)+\
                    self.__delayCoeff * delay_term - unusedPenalty
        # old reward
        reward = np.min(self.__CPUload) - self.__backlogVsCPUloadCoeffient*np.max(self.__backlog)
        return reward - unusedPenalty

    def monitorState(self, metric):
        if(metric == 'CPUload'): return self.__CPUload
        elif(metric == 'flowSecs'): return self.__flowSecs
        elif(metric == 'flowHours'): return self.__flowHours
        elif(metric == 'backlog'): return self.__backlog 
        elif(metric == 'work'): return self.__work
        elif(metric == 'time'): return self.__time
        elif(metric == 'instant_work'): return self.__work[max(self.__time-1, 0)][0]
        elif(metric == 'delay'): return self.__delay_MGR_PS(
                      R=len(self.__CPUload),
                      r_peak=self.__maxWorkOf1CPU,
                      rho=np.mean(self.__CPUload),
                      x=self.__opPerVehicle)
        else: print(metric, 'not defined')
        
    def plotWork(self):
        plt.figure(figsize = (16,4))
        plt.xlabel('time [5m interval]')
        plt.ylabel('work')
        plt.plot(self.__work)
        
    def getWork(self):
        return self.__work
        
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

