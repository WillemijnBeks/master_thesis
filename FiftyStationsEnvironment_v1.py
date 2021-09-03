### 50 station environment, version v1

import pandas as pd
import numpy as np

class stationQueue:
    """list to collect the cars that are currently served at a station"""

    def __init__(self, R, minNrOfCPUs=0, maxNrOfCPUs=50):
        """
        Initializes a list that keeps the cars served at a station
        R: number of servers
        minNrOfCPUs: minimum number of CPUs allowed
        maxNrOfCPUs: maximum number of CPUs allowed
        Internal parameters:
        T: the average time [sec] that a car lingers at the station
        W: work brought by one cars (in fraction of 1 CPU)
        delta: processing time [sec] of one packet sent by a car
        prob: quantile probability
        """
        self.cars = []
        self.R = R
        self.minNrOfCPUs = minNrOfCPUs
        self.maxNrOfCPUs = maxNrOfCPUs
        self.T = 30
        self.W = 0.05
        self.delta = 0.0005
        self.prob = 1e-5

    def placeCar(self, time):
        """
        place an arriving car to the list of served cars by the station
        time: arrival time of the car
        returns: the processing delay the car experiences
        """
        # evolve to the arrival time: purge finished users
        self.cars = [car for car in self.cars if car > time]
        # append the new car in the list with its expiration time
        lingeringTime = self.T*np.random.exponential()
        self.cars.append(time + lingeringTime)
        return self.delay(time)

    def getNrOfCars(self, time):
        """
        get the current number of cars served by the stationQueue, i.e., the 'state' of the system
        time: observation instant
        returns: the number of cars processed by the stationQueue at instant "time"
        """
        # evolve to the new time: purge finished users
        self.cars = [car for car in self.cars if car > time]
        return len(self.cars)
   
    def setNrOfCPUs(self, R, time):
        """
        set the number of CPUs of a stationQueue
        R: number of CPUs
        time: current time
        returns: the delay at that stationQueue just after setting R
        """
        increase = 0.0
        if R - self.R > 0:
            increase = R - self.R
        penalty = 0.002
        
        self.R = R        
        if self.R < self.minNrOfCPUs: self.R = self.minNrOfCPUs 
        if self.R > self.maxNrOfCPUs: self.R = self.maxNrOfCPUs
        return self.delay(time) + increase * penalty

    def getNrOfCPUs(self, R, time):
        """
        get the number of CPUs of a stationQueue
        time: current time
        returns: R = number of CPUs
        """
        return self.R
    
    def delay(self, time):
        # evolve to the new time: purge finished users
        self.cars = [car for car in self.cars if car > time]
        # calculate delay
        if self.R == 0: 
            return np.float('inf')
        load = self.W*len(self.cars)/self.R
        if load == 0: return 0.0
        if load >= 1:
            return np.float('inf')
        return self.delta*(1 - np.log(self.prob) * load / (1 - load) / 2.0)


class FiftyStationsEnvironment:
    """ 
    read the traffic of (50) stations in one (hidden) pandas frame
    initialize as many stationQueues as there are stations + one "cloud" stationQueue
    filename: name of file with trace
    """
    def __init__(self,filename='cars-trace-1.csv'):
        print('reading trace: be patient')
        self.trace = pd.read_csv(filename)
        print('trace length = {:d} entries'.format(len(self.trace)))
        self.duration = self.trace.iloc[-1]['arrival_time'] - self.trace.iloc[0]['arrival_time']
        print('trace timespan = {:.2f} days'.format(self.duration/(24*3600)))
        self.trace['id'] = '(' + self.trace['lat'].apply(lambda x: f'{x:.5f}') + ',' + self.trace['lng'].apply(lambda x: f'{x:.5f}') + ')' 
        self.trace.pop('lat')
        self.trace.pop('lng')
        print('determining stations: be patient')
        self.stations = self.trace['id'].drop_duplicates(keep='first')
        print('there are {:d} stations in the trace'.format(len(self.stations)))
        self.currentIdx = 0
        self.stopIdx = len(self.trace) - 1
        print('setting up one cloud queue and {:d} station queues'.format(len(self.stations)))
        self.Q = {'cloud': stationQueue(0)}
        for station in self.stations:
            self.Q[station] = stationQueue(0)   
        self.delayToRemoteLocation = 0.002
        self.delayTarget = 0.005

    def seek(self, start, stop=1.0):
        """
        jumps to a specific fraction of the trace and set the stop time
        returns: first time of the new staring point
        """
        self.currentIdx = int(len(self.trace)*start)
        self.stopIdx = int(len(self.trace)*stop) - 1
        duration = self.trace.iloc[self.stopIdx]['arrival_time'] - self.trace.iloc[self.currentIdx]['arrival_time']
        print('simulated time: {:.2f} days'.format(duration/(24*3600)))
        return self.trace.iloc[self.currentIdx]['arrival_time']
        
    def step(self):
        """
        step to the next entry of the pandas data frame
        returns: time, id of station id and stop indicator 
        """
        time = self.trace['arrival_time'].iloc[self.currentIdx]
        station = self.trace['id'].iloc[self.currentIdx]
        stop = False
        if self.currentIdx == self.stopIdx: stop = True
        else: self.currentIdx += 1
        return time, station, stop
    
    def monitorState(self, time):
        state = {'cloud': self.Q['cloud'].getNrOfCars(time)}
        for station in self.stations:
            state[station] = self.Q[station].getNrOfCars(time)
        return state
    
    def delayReward(self, delay):
        aux = delay/self.delayTarget
        if aux == np.float('inf'): return 0
        if aux < 0: return 0
        return aux*np.exp(-((aux**2 - 1.0)/2.0))


