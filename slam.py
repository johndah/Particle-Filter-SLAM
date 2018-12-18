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
        r = sqrt(dx**2 + dy**2) + 0.1*random.randn()
        if r < 5:
            alpha = arctan2(dy, dx) + 0.1*random.randn()
            measurements.append([r, alpha])

    return measurements

def plotMap(robot_pose, measurements):

    plt.cla()
    for landmark in landmarks:
        plt.plot(landmark.x, landmark.y, 'go')
        plt.text(landmark.x, landmark.y + .05, '(' + str(landmark.index) + ')')

    for measure in measurements:
        r = measure[0]
        alpha = measure[1]
        plt.plot([robot_pose[0], robot_pose[0] + r*cos(alpha)], [robot_pose[1], robot_pose[1] + r*sin(alpha)], 'b')

    plt.axis([0, 25, 0, 25])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Map')
    plt.pause(1e-3)

def initMap():

    f = open("landmarks.txt", "r")
    i = 0
    for row in f:
        x = float(row.split(',')[0])
        y = float(row.split(',')[1])
        landmarks.append(Landmark(x, y, i))
        i += 1

def particleFilterSlam():

    for i in range(1 ,100):
        robot_pose = [7, 12, 0]
        measurements = getMeasurements(robot_pose)

        plotMap(robot_pose, measurements)

def main():

    initMap()
    particleFilterSlam()

if __name__ == '__main__':
    random.seed(0)
    main()
    plt.show()