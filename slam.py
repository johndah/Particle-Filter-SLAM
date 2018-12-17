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


    plt.plot(landmarks_x, landmarks_y, 'go')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Map')

def initMap():
    landmarks = []
    for i in range(0, 3):
        landmarks.append(Landmark(i, 2, 1))
    return landmarks

def main():

    landmarks = initMap()

    plotMap(landmarks)

if __name__ == '__main__':
    main()
    plt.show()