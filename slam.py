from numpy import *
import copy
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl

class Landmark(object):

    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index

landmarks = []

def getMeasurements(robot_pose):

    measurements = []
    for landmark in landmarks:
        dx = landmark.x - robot_pose[0]
        dy = landmark.y - robot_pose[1]
        r = sqrt(dx**2 + dy**2) + rand
        if r < 5:
            alpha = arctan2(dy, dx)
            measurements.append([r, alpha])

    return measurements

def plotMap():

    for landmark in landmarks:
        plt.plot(landmark.x, landmark.y, 'go')
        plt.text(landmark.x, landmark.y + .05, '(' + str(landmark.index) + ')')
    plt.axis([0, 25, 5, 20])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Map')

def initMap():

    f = open("landmarks.txt", "r")
    i = 0
    for row in f:
        x = float(row.split(',')[0])
        y = float(row.split(',')[1])
        landmarks.append(Landmark(x, y, i))
        i += 1

def particleFilterSlam():

    robot_pose = [7, 10, 0]
    measurements = getMeasurements(robot_pose)
    print(measurements)

    plotMap()

def main():

    initMap()
    particleFilterSlam()

if __name__ == '__main__':
    main()
    plt.show()