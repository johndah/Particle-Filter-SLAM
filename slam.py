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

def plotMap(landmarks):

    landmarks_x = []
    landmarks_y = []

    for landmark in landmarks:
        landmarks_x.append(landmark.x)
        landmarks_y.append(landmark.y)
        plt.plot(landmark.x, landmark.y, 'go')
        plt.text(landmark.x, landmark.y + .05, '(' + str(landmark.index) + ')')
    plt.axis([0, 5, 1, 4])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Map')

def initMap():
    landmarks = []

    f = open("landmarks.txt", "r")
    i = 0
    for row in f:
        x = float(row.split(',')[0])
        y = float(row.split(',')[1])
        landmarks.append(Landmark(x, y, i))
        i += 1
    return landmarks

def main():

    landmarks = initMap()

    plotMap(landmarks)

if __name__ == '__main__':
    main()
    plt.show()