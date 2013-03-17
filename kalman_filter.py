'''
Created on Mar 16, 2013

@author: Doug Szumski

Simple implementation of a Kalman filter based on:

"An introduction to the Kalman Filter", Greg Welch and Gary Bishop

http://www.cs.unc.edu/~welch/kalman/kalmanIntro.html
'''
from numpy.linalg import inv
from numpy import identity


class KalmanFilter:
    """
    Simple Kalman filter

    Control term has been omitted for now
    """
    def __init__(self, X, P, F, Q, Z, H, R):
        """
        Initialise the filter

        Args:
            X: State estimate
            P: Estimate covaConfigureriance
            F: State transition model
            Q: Process noise covariance
            Z: Measurement of the state X
            H: Observation model
            R: Observation noise covariance
        """
        self.X = X
        self.P = P
        self.F = F
        self.Q = Q
        self.Z = Z
        self.H = H
        self.R = R

    def predict(self, X, P, w=0):
        """
        Predict the future state

        Args:
            X: State estimate
            P: Estimate covariance
            w: Process noise
        Returns:
            updated (X, P)
        """
        # Project the state ahead
        X = self.F * X + w
        P = self.F * P * (self.F.T) + self.Q
        return(X, P)

    def update(self, X, P, Z):
        """
        Update the Kalman Filter from a measurement

        Args:
            X: State estimate
            P: Estimate covariance
            Z: State measurement
        Returns:
            updated (X, P)
        """
        K = P * (self.H.T) * inv(self.H * P * (self.H.T) + self.R)
        X += K * (Z - self.H * X)
        P = (identity(P.shape[1]) - K * self.H) * P
        return (X, P)
