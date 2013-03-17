'''
Created on Mar 16, 2013

@author: Doug Szumski

An implementation of the technical example
outlined in wikipedia for a Kalman filter:

http://en.wikipedia.org/wiki/Kalman_filter

A truck on straight, frictionless and infinitely long
rails experiences a series of random accelerations

A moving average is included for comparison
'''
from numpy import matrix, diag, random

from helper_utils import MovingAverage
from helper_utils import Logger
from helper_utils import KalmanPlotter
from kalman_filter import KalmanFilter

# Time step size
dt = 0.1
# Standard deviation of random accelerations
sigma_a = 0.2
# Standard deviation of observations
sigma_z = 0.2
# State vector: [[Position], [velocity]]
X = matrix([[0.0], [0.0]])
# Initial state covariance
P = diag((0.0, 0.0))
# Acceleration model
G = matrix([[(dt ** 2) / 2], [dt]])
# State transition model
F = matrix([[1, dt], [0, 1]])
# Observation vector
Z = matrix([[0.0], [0.0]])
# Observation model
H = matrix([[1, 0], [0, 0]])
# Observation covariance
R = matrix([[sigma_z ** 2, 0], [0, 1]])
# Process noise covariance matrix
Q = G * (G.T) * sigma_a ** 2

# Initialise the filter
kf = KalmanFilter(X, P, F, Q, Z, H, R)

# Set the actual position equal to the starting position
A = X

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
iterations = 100

for i in range(0, iterations):
    # Generate a random acceleration
    w = matrix(random.multivariate_normal([0.0, 0.0], Q)).T
    # Predict
    (X, P) = kf.predict(X, P, w)
    # Update
    (X, P) = kf.update(X, P, Z)
    # Update the actual position
    A = F * A + w
    # Synthesise a new noisy measurement distributed around the real position
    Z = matrix([[random.normal(A[0, 0], sigma_z)], [0.0]])
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
