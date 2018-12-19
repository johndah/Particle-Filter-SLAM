from numpy import *
import copy
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors

axis = [0, 2.5, 0, 2.5]
landmarks = []


class Landmark(object):

    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index


class OccupancyGrid():

    def __init__(self):
        self.grid_size = 2.5/100
        self.n_cells_x = int((axis[1] - axis[0]) / self.grid_size)
        self.n_cells_y = int((axis[3] - axis[2]) / self.grid_size)
        self.generateGrid()

    def generateGrid(self):
        self.x_grid_vec = linspace(axis[0], axis[1], self.n_cells_x)
        self.y_grid_vec = linspace(axis[2], axis[3], self.n_cells_y)
        '''

        self.grid = []
        for y_grid in self.y_grid_vec:
            row = []
            for x_grid in self.x_grid_vec:
                cell = Cell(x_grid, y_grid)
                if x_grid > 7 and x_grid < 7.4 and y_grid < 12.5:
                    cell.value = 'occupied'
                row.append(cell)
            self.grid.append(row)
        '''

        self.grid = -ones([self.n_cells_x, self.n_cells_y])

    def getCellCoordinates(self, x, y):
        i = int((x - axis[0]) / self.grid_size)
        j = int((y - axis[2]) / self.grid_size)

        return i, j

    def isOccupied(self, x, y):
        i, j = self.getCellCoordinates(x, y)

        # or wall
        return self.grid[j, i]

    def markOccupied(self, x, y):
        i, j = self.getCellCoordinates(x, y)

        self.grid[j, i] = 1

    def markFreeSpace(self, x, y):
        i, j = self.getCellCoordinates(x, y)

        self.grid[j, i] = 0

    def getCoordinates(self, x_grid, y_grid):
        x = axis[0] + x_grid * self.grid_size
        y = axis[2] + y_grid * self.grid_size

        return x, y

'''
class Cell(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.value = 'unknown'
        self.wall = False

'''

def getMeasurements(robot_pose):
    global occ_grid, updated_cells

    measurements = []
    updated_cells = []

    for landmark in landmarks:
        dx = landmark.x - robot_pose[0]
        dy = landmark.y - robot_pose[1]
        r = sqrt(dx ** 2 + dy ** 2) + 0.05 * random.randn()
        if r < 5:
            alpha = arctan2(dy, dx) + 0.05 * random.randn()
            n = int(max(abs(dx), abs(dy))/occ_grid.grid_size)
            x_ray = robot_pose[0]
            y_ray = robot_pose[1]
            add_measurement = True
            for i in range(0, n):
                x_ray += r * cos(alpha) / n
                y_ray += r * sin(alpha) / n

                occ_grid.markFreeSpace(x_ray, y_ray)
                '''   
                rect = mpl.patches.Rectangle((x_ray, y_ray), occ_grid.grid_size, occ_grid.grid_size, edgecolor='none',
                                             facecolor='white')
                ax.add_patch(rect)
                x_grid, y_grid = occ_grid.getCellCoordinates(x_ray, y_ray)
                cell = occ_grid.grid[y_grid][x_grid]
                cell.value = 'free'
                '''

                if occ_grid.isOccupied(x_ray, y_ray): # or occ_grid.grid[y_ray][x_ray].wall:
                    add_measurement = False
                    '''
                    cell.value = 'occupied'
                    occ_grid.markOccupied(x_ray, y_ray)
                    rect = mpl.patches.Rectangle((x_ray, y_ray), occ_grid.grid_size, occ_grid.grid_size, edgecolor='none',
                                                 facecolor='r')
                    ax.add_patch(rect)
                    '''
                    break
            l += 1
                #if not cell in updated_cells:
                #    updated_cells.append(cell)

            if add_measurement:
                measurements.append([r, alpha])

    return measurements


def plotMap(robot_pose, measurements):
    global occ_grid, ax, updated_cells, walls

    # fig, ax = plt.figure()

    plt.cla()
    cmap = colors.ListedColormap(['grey', 'white', 'orange'])
    bounds = [-1, -.1, .1, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    #cmap = mpl.cm.cool
    #norm = mpl.colors.Normalize(vmin=0, vmax=10)
    ax.imshow(occ_grid.grid, cmap=cmap, norm=norm, extent=[0, 2.5, 2.5, 0])

    #rect = mpl.patches.Rectangle((occ_grid.x_grid_vec[i], occ_grid.y_grid_vec[j]), occ_grid.grid_size, occ_grid.grid_size, edgecolor='none',
                    #                             facecolor='grey')
                    #ax.add_patch(rect)

    '''
    for x_grid in occ_grid.x_grid_vec:
        print([x_grid, occ_grid.y_grid_vec[0]])
        print( [x_grid, occ_grid.y_grid_vec[-1]])
        plt.plot([x_grid, x_grid], [occ_grid.y_grid_vec[0], occ_grid.y_grid_vec[-1]], 'k')
    for j in range(0, len(occ_grid.grid)):
        for i in range(0, len(occ_grid.grid[j])):

            x, y = occ_grid.getCoordinates(i, j)
            if occ_grid.grid[j][i].value == 'unknown':
                rect = mpl.patches.Rectangle((x, y), occ_grid.grid_size, occ_grid.grid_size, edgecolor='none',
                                             facecolor='grey')
                ax.add_patch(rect)

    '''
    '''
            if occ_grid.grid[j][i].value == 'occupied':
                rect = mpl.patches.Rectangle((x, y), occ_grid.grid_size, occ_grid.grid_size, edgecolor='none',
                                             facecolor='r')
                ax.add_patch(rect)

            if occ_grid.grid[j][i].wall:
                rect = mpl.patches.Rectangle((x, y), occ_grid.grid_size, occ_grid.grid_size, edgecolor='none',
                                             facecolor='b')
                ax.add_patch(rect)
    print("yes")
    '''
    for measure in measurements:
        r = measure[0]
        alpha = measure[1]
        plt.plot([robot_pose[0], robot_pose[0] + r * cos(alpha)], [robot_pose[1], robot_pose[1] + r * sin(alpha)], 'b', linewidth=.4)

    for wall in walls:
       plt.plot(wall[:2], wall[2:4], 'b', linewidth=2.0)

    '''
    for cell in []:
        if cell.value == 'free':
            rect = mpl.patches.Rectangle((cell.x, cell.y), occ_grid.grid_size, occ_grid.grid_size, edgecolor='none',
                                         facecolor='white')
            ax.add_patch(rect)
        elif cell.value == 'occupied':
            rect = mpl.patches.Rectangle((cell.x, cell.y), occ_grid.grid_size, occ_grid.grid_size, edgecolor='none',
                                         facecolor='r')
            ax.add_patch(rect)

        if cell.wall:
            rect = mpl.patches.Rectangle((cell.x, celly), occ_grid.grid_size, occ_grid.grid_size, edgecolor='none',
                                         facecolor='b')
            ax.add_patch(rect)

    updated_cells = []
    '''

    for landmark in landmarks:
        plt.plot(landmark.x, landmark.y, 'go')
        plt.text(landmark.x, landmark.y + .05, '(' + str(landmark.index) + ')')


    '''
    '''

    plt.axis(axis)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Map')
    plt.pause(1e-5)


def initMap():

    global occ_grid, ax, walls
    f = open("map.txt", "r")
    walls = []
    for row in f:
        coordinates = row.split()
        x_start = float(coordinates[0])
        x_end = float(coordinates[2])
        y_start = float(coordinates[1])
        y_end = float(coordinates[3])

        walls.append([x_start, x_end, y_start, y_end])

        '''
        dx = x_end - x_start
        dy = y_end - y_start
        r = sqrt(dx ** 2 + dy ** 2)

        alpha = arctan2(dy, dx)
        n = int(max(dx, dy))
        x = x_start
        y = y_start
        for i in range(0, n):
            x += r * cos(alpha) / n
            y += r * sin(alpha) / n

            x_grid, y_grid = occ_grid.getCellCoordinates(x, y)
            occ_grid.grid[y_grid][x_grid].wall = True
        '''

    f = open("landmarks.txt", "r")
    i = 0
    for row in f:
        x = float(row.split(',')[0])
        y = float(row.split(',')[1])
        landmarks.append(Landmark(x, y, i))
        i += 1

    for j in range(0, len(occ_grid.grid)):
        for i in range(0, len(occ_grid.grid[j])):
            x, y = occ_grid.getCoordinates(i, j)

    '''
            if not occ_grid.grid[j][i].wall:
                rect = mpl.patches.Rectangle((x, y), occ_grid.grid_size, occ_grid.grid_size, edgecolor='none',
                                                 facecolor='grey')
                ax.add_patch(rect)
            else:
                rect = mpl.patches.Rectangle((x, y), occ_grid.grid_size, occ_grid.grid_size, edgecolor='none',
                                             facecolor='b')
                ax.add_patch(rect)
    '''
    '''
    '''

    plt.axis(axis)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Map')
    plt.pause(1e-5)

def particleFilterSlam():
    for i in range(0, 30):
        robot_pose = [1, 1, 0]
        measurements = getMeasurements(robot_pose)

        plotMap(robot_pose, measurements)


def main():
    global occ_grid, ax
    occ_grid = OccupancyGrid()

    fig, ax = plt.subplots()
    initMap()
    particleFilterSlam()


if __name__ == '__main__':
    random.seed(0)
    main()
    plt.show()
