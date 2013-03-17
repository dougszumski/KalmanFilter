'''
Created on Mar 16, 2013

@author: Doug Szumski

Helper utilities for Kalman filter examples
'''
from collections import deque
import matplotlib.pyplot as plt
from pylab import rcParams


class MovingAverage:
    """
    Calculates a moving average
    """
    def __init__(self, size):
        """
        Configure the averaging window

        Args:
            size: window size
        """
        self.size = size
        self.stack = deque([])

    def update(self, value):
        """
        Update the moving average

        Args:
            value: latest reading
        """
        if (len(self.stack) < self.size):
            self.stack.append(value)
        else:
            self.stack.append(value)
            self.stack.popleft()

    def getAvg(self):
        """
        Returns the current moving average
        """
        self.avg = 0.0
        for value in self.stack:
            self.avg += value
        self.avg /= self.size
        return self.avg


class Logger:
    """
    Simple logger
    """
    def __init__(self):
        """
        Create a container for the logs
        """
        self.logs = {}

    def new_log(self, item):
        """
        Add a new log

        Args:
            item: log name
        """
        self.logs[item] = []

    def get_log(self, item):
        """
        Returns a log

        Args:
            item: name of log to return
        """
        return self.logs[item]

    def get_all_logs(self):
        """
        Returns all logs
        """
        return self.logs

    def log(self, item, data):
        """
        Log a value to a log

        Args:
            item: log name
            data: value to log
        """
        self.logs[item].append(data)


class KalmanPlotter:
    """
    Plots logged data from Kalman Filter
    """
    def __init__(self):
        """
        Configure the plot
        """
        # Setup a summary figure
        self.fig = plt.figure()
        self.ax1 = plt.subplot2grid((2, 1), (0, 0))
        self.ax2 = plt.subplot2grid((2, 1), (1, 0))
        # Set the legend to auto locate
        rcParams['legend.loc'] = 'best'

    def plot_kalman_data(self, log):
        """
        Plot the system behaviour as a function of time

        Args:
            log: a dictionary containing the keys plotted below each
                 associated with a list of data
        """
        # Plot the evolution of the system state
        self.ax1.set_title("Kalman filter example")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Position (m)")
        self.ax1.plot(log.get_log('time'), log.get_log('measurement'),
                 'o', label='Measured', markersize=5)
        self.ax1.plot(log.get_log('time'), log.get_log('estimate'),
                 '-', label='Estimated', markersize=5)
        self.ax1.plot(log.get_log('time'), log.get_log('actual'),
                 '-', label='Actual', markersize=5)
        self.ax1.plot(log.get_log('time'), log.get_log('moving average'),
                 '-', label='Averaged', markersize=5)
        self.ax1.legend(prop={'size': 10})
        # Plot the evolution of the state covariance
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("State covariance")
        self.ax2.plot(log.get_log('time'), log.get_log('covariance'),
                  '-', label='State covariance', markersize=5)
        plt.show()
