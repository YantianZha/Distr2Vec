#!/usr/bin/env python
"""
https://github.com/richard-to/betelbot/blob/master/betelbot/particle.py
"""

import random
from math import atan2, cos, exp, pi, cos, sin, sqrt, tan
import numpy as np
from numpy import inf

def normalizeCmd(cmdTopic, rotation):
    if rotation == 0:
        cmd = cmdTopic.up
    elif rotation > 0:
        cmd = cmdTopic.right
    elif rotation < 0:
        cmd = cmdTopic.left
    return cmd


def convertToMotion(cmdTopic, start, dest, gridsize):
    # In the particle filter we need to know the orientation
    # of the robot. The catch is that it wants the rotation in
    # in radians from the current orientation to the next.
    #
    # Since our robot can only move left, right, up down the angle of
    # rotation 0 or +/- pi/2 radians.
    #
    # 180 degree (pi) rotation is not allowed by particle filter, so
    # that case should never be picked.
    #
    # The particle filter also needs to know distance traveled.
    # Betelbot is designed to move in discrete blocks of distance "gridsize"
    #
    # Expected data output: [orientation, distance]

    rotationDist = {
        cmdTopic.right: 0,
        cmdTopic.down: 1,
        cmdTopic.left: 2,
        cmdTopic.up: 3
    }
    rotation = 0
    dist = rotationDist[dest] - rotationDist[start]

    if dist == 0:
        rotation = 0
    elif dist == 1 or dist == -3:
        rotation = pi/2
    elif dist == 2 or dist == -2:
        rotation = pi
    elif dist == 3 or dist == -1:
        rotation = -pi/2

    return [rotation, gridsize]


class Particle:
    # Represents a single particle in the particle filter.
    # A particle represents the location and orientation of
    # where Betelbot may be located.

    # Error messages
    ERROR_ORIENTATION = 'Orientation must be in [0..2pi]'
    ERROR_MOVE_BACKWARD = 'Robot cant move backwards'

    # Particle to string format
    REPR_STRING = '[y=%.6s x=%.6s orient=%.6s]'

    # Some constants
    ANGLE_2PI_RAD = 2.0 * pi
    DIGITS_ROUND = 15
    DELTA = [[0, 1], [1, 0], [1, 0], [0, 1]]

    def __init__(self, length, grid, lookupTable):
        # Initializes particle with reference to map and lookup table data.
        #
        # - length is not used currently.
        # - delta is used in the sense function and tells whether to multiple y,x values by 0 or 1.
        # - the noise parameters adjust the randomness of the algorithm and helps account for sensor noise.

        self.grid = grid
        self.lookupTable = lookupTable
        self.delta = Particle.DELTA
        self.length = length
        self.forwardNoise  = 0.0
        self.turnNoise = 0.0
        self.senseNoise = 0.0
        self.y = 0.0
        self.x = 0.0

    def randomizePosition(self):
        # Randomly picks an orientation and (y, x) coordinate for particle.
        # Keep choosing coordinates until they fall in an open area.

        self.orientation = random.random() * Particle.ANGLE_2PI_RAD
        gridY = self.grid.shape[0] - 1
        gridX = self.grid.shape[1] - 1
        while True:
            self.y = float(random.randint(0, gridY))
            self.x = float(random.randint(0, gridX))
            if self.grid[self.y][self.x] > 0:
                break

    def set(self, y, x, orientation):
        # Sets the coordinates and orientation of the particle.

        if orientation < 0 or orientation >= Particle.ANGLE_2PI_RAD:
            raise ValueError, Particle.ERROR_ORIENTATION
        self.x = float(x)
        self.y = float(y)
        self.orientation = float(orientation)

    def setNoise(self, forwardNoise, turnNoise, senseNoise):
        # Sets noise parameters. If used all parameters are set at
        # the same time.
        #
        # Parameters get converted floats.
        self.forwardNoise  = float(forwardNoise)
        self.turnNoise = float(turnNoise)
        self.senseNoise = float(senseNoise)

    def move(self, motion):
        # Moves particle.
        #
        # Motionat is a list with two values. Orientation and distance
        #
        # The robot is not allowed to move backwards. ValueError is thrown.
        #
        # The robot's orientation is affected by a gaussian value, if a turn noise
        # is specified.
        #
        # The robot's distance traveled is affected by a forward noise.
        #
        # Resulting y,x values are rounded to ints since the lookup table only
        # handles int values.
        #
        # This function returns a new particle object. The object itself is not modified.

        turn, forward = motion

        if forward < 0:
            raise ValueError, Particle.ERROR_BACKWARD_MOVE

        orientation = self.orientation + float(turn) + random.gauss(0.0, self.turnNoise)
        orientation %= Particle.ANGLE_2PI_RAD

        dist = float(forward) + random.gauss(0.0, self.forwardNoise)
        x = self.x + (round(cos(orientation), Particle.DIGITS_ROUND) * dist)
        y = self.y + (round(sin(orientation), Particle.DIGITS_ROUND) * dist)
        y %= self.grid.shape[0]
        x %= self.grid.shape[1]
        particle = Particle(self.length, self.grid, self.lookupTable)
        particle.set(round(y), round(x), orientation)
        particle.setNoise(self.forwardNoise, self.turnNoise, self.senseNoise)
        return particle

    def sense(self, hasNoise=False):
        # Simulates a sensor on the real robot.
        #
        # Calculates the distance from the 3 of 4 walls (west, north, south, east).
        # The reason for 3 walls is because the robot cannot sense backward.
        #
        # If the y,x value does not fall within the map, then the sensor returns infinity
        # for the distance.
        #
        # The sensor values are affected by sensor noise.

        Z = []
        if (self.y >= 0 and self.y < self.grid.shape[0] and
            self.x >= 0 and self.x < self.grid.shape[1] and
            self.grid[self.y][self.x] > 0):
            count = len(self.delta)
            index = self.y * self.grid.shape[1] * count + self.x * count
            for i in xrange(count):
                value = self.lookupTable[index]
                dy = value * self.delta[i][0]
                dx = value * self.delta[i][1]
                index += 1
                dist = dy or dx
                if hasNoise:
                    dist += random.gauss(0.0, self.senseNoise)
                Z.append(dist)
        else:
            Z = [inf, inf, inf, inf]
        return Z

    def measurementProb(self, measurements):
        # Measures the probability that the particle is close the actual robot position.
        #
        # The probability is measured by calculating the gaussian between actual sensor values
        # and predicated sensor values.

        prob = 1.0
        Z = self.sense()
        for i in xrange(len(Z)):
            if measurements[i] is not None:
                prob *= self.gaussian(float(Z[i]), self.senseNoise, float(measurements[i]))
        return prob

    def gaussian(self, mu, sigma, x):
        # Calculates gaussian for prediced and expected sensor measurement.

        return exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / sqrt(Particle.ANGLE_2PI_RAD * (sigma ** 2))

    def __repr__(self):
        # To string value prints y,x coordinates and orientation.

        return Particle.REPR_STRING % (str(self.y), str(self.x), str(self.orientation))


class ParticleFilter:
    # Particle filter creates N random particles.
    #
    # Every time the robot sends new motion and measurement data,
    # the filter calculates the probability that the particle represents
    # the location of the robot.
    #
    # The probabilities are weighted and then resampled N times. The particles
    # with a higher weight are more likely to be chosen and survive.

    def __init__(self, length, grid, lookupTable, forwardNoise=0.05, turnNoise=0.05, senseNoise=5, N=500):
        self.N = N
        self.length = length
        self.grid = grid
        self.lookupTable = lookupTable
        self.forwardNoise = forwardNoise
        self.turnNoise = turnNoise
        self.senseNoise = senseNoise

    def makeParticles(self, N=None):
        # Creates N particles with random location and noise values.

        self.N = self.N if N is None else N
        self.particles = []
        for i in xrange(self.N):
            p = Particle(self.length, self.grid, self.lookupTable)
            p.randomizePosition()
            p.setNoise(self.forwardNoise, self.turnNoise, self.senseNoise)
            self.particles.append(p)

    def getData(self):
        # Returns a list of y,x values for particles.

        return [[p.y, p.x] for p in self.particles]

    def update(self, motion, measurements):
        # Updates the particles based on motion and measurement values
        #
        # Particles are weighted and resambled.

        updatedParticles = []
        for i in xrange(self.N):
            updatedParticles.append(self.particles[i].move(motion))
        self.particles = updatedParticles
        weight = []
        for i in range(self.N):
            weight.append(self.particles[i].measurementProb(measurements))
        self.particles = self.resample(self.particles, weight, self.N)

    def resample(self, particles, weight, N):
        # Resamples particles. Particles with higher weight have higher probability
        # of surviving.
        #
        # Uses roulette wheel algorithm for resambling.

        sampledParticles = []
        index = int(random.random() * N)
        beta = 0.0
        maxWeight = max(weight)
        for i in xrange(N):
            beta += random.random() * 2.0 * maxWeight
            while beta > weight[index]:
                beta -= weight[index]
                index = (index + 1) % N
            sampledParticles.append(particles[index])
        return sampledParticles

    def getPosition(self, p):
        # Basically gets the average position (y, x) and orientation of
        # all particles.

        x = 0.0
        y = 0.0
        orientation = 0.0
        for i in range(len(p)):
            x += p[i].x
            y += p[i].y
            orientation += (((p[i].orientation - p[0].orientation + pi) % (2.0 * pi))
                            + p[0].orientation - pi)
        return [y / len(p), x / len(p), orientation / len(p)]


class ParticleFilterMethod(object):
    # Supported Particle filter methods.

    UPDATE = 'particles_update'
    STATUS = 'particles_status'




