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
    plt.figure()
    plt.plot(landmarks[0].x, landmarks[0].y, 'go')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Map')


def main():

    landmarks = []
    landmarks.append(Landmark(1, 2, 1))

    plotMap(landmarks)

if __name__ == '__main__':
    main()
    plt.show()
