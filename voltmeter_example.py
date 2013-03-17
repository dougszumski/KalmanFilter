'''
Created on Mar 16, 2013

@author: Doug Szumski

A re-implementation of the code by Andrew D. Straw:

http://www.scipy.org/Cookbook/KalmanFiltering

Which is an implementation of the example in:

"An introduction to the Kalman Filter", Greg Welch and Gary Bishop

http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html

A noisy volt meter measuring a voltage of -0.37727V

A moving average is included for comparison
'''
from numpy import matrix, random

from helper_utils import MovingAverage
from helper_utils import Logger
from helper_utils import KalmanPlotter
from kalman_filter import KalmanFilter

# Time step size
dt = 1.0
# Standard deviation of observations
sigma_z = 0.1
# Initial state [voltage level]
X = matrix([0.0])
# Initial state covariance
P = matrix([1.0])
# State transition model
F = matrix([1])
# Initial observation
Z = matrix([0.0])
# Observation model
H = matrix([1])
# Observation covariance
R = matrix([sigma_z ** 2])
# Process noise covariance matrix
Q = matrix([1e-5])

# Initialise the filter
kf = KalmanFilter(X, P, F, Q, Z, H, R)

# Set the actual position
A = matrix([-0.37727])

# Create log for generating plots
log = Logger()
log.new_log('measurement')
log.new_log('estimate')
log.new_log('actual')
log.new_log('time')
log.new_log('covariance')
log.new_log('moving average')

# Moving average for measurements
moving_avg = MovingAverage(15)

# Number of iterations to perform
iterations = 50

for i in range(0, iterations):
    # Predict
    (X, P) = kf.predict(X, P)
    # Update
    (X, P) = kf.update(X, P, Z)
    # Synthesise a new noisy measurement centered around the actual position
    Z = matrix([random.normal(A[0, 0], sigma_z)])
    # Update the moving average with the latest measured position
    moving_avg.update(Z[0, 0])
    # Update the log for plotting later
    log.log('measurement', Z[0, 0])
    log.log('estimate', X[0, 0])
    log.log('actual', A[0, 0])
    log.log('time', i * dt)
    log.log('covariance', P[0, 0])
    log.log('moving average', moving_avg.getAvg())

# Plot the system behaviour
plotter = KalmanPlotter()
plotter.plot_kalman_data(log)
