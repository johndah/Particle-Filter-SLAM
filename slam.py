from numpy import *
import matplotlib.pyplot as plt
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
        self.grid_size = 2.5 / 100
        self.n_cells_x = int((axis[1] - axis[0]) / self.grid_size)
        self.n_cells_y = int((axis[3] - axis[2]) / self.grid_size)
        # self.x_grid_vec = linspace(axis[0], axis[1], self.n_cells_x)
        # self.y_grid_vec = linspace(axis[2], axis[3], self.n_cells_y)

        self.grid = -ones([self.n_cells_x, self.n_cells_y])
        self.true_grid = zeros([self.n_cells_x, self.n_cells_y])


    def getCellCoordinates(self, x, y):
        i = int((x - axis[0]) / self.grid_size)
        j = int((y - axis[2]) / self.grid_size)

        return i, j

    def isOccupied(self, x, y):
        i, j = self.getCellCoordinates(x, y)

        return self.grid[j, i]

    def isWall(self, x, y):
        i, j = self.getCellCoordinates(x, y)

        return self.true_grid[j, i]

    def markOccupiedSpace(self, x, y):
        i, j = self.getCellCoordinates(x, y)

        self.grid[j, i] = 1

    def markFreeSpace(self, x, y):
        i, j = self.getCellCoordinates(x, y)

        self.grid[j, i] = 0

    def getCoordinates(self, x_grid, y_grid):
        x = axis[0] + x_grid * self.grid_size
        y = axis[2] + y_grid * self.grid_size

        return x, y

def getMeasurements(robot_pose):
    global occ_grid, updated_cells

    measurements = []
    updated_cells = []

    n_alphas = 100
    alphas = linspace(-pi, pi, n_alphas) + 0.05 * random.randn(n_alphas)

    for alpha in alphas:
        x_ray = robot_pose[0]
        y_ray = robot_pose[1]

        is_free = True
        n = 1/occ_grid.grid_size

        while is_free:
            x_ray += cos(alpha) / n
            y_ray += sin(alpha) / n

            occ_grid.markFreeSpace(x_ray, y_ray)

            if occ_grid.isWall(x_ray, y_ray):
                is_free = False
                occ_grid.markOccupiedSpace(x_ray, y_ray)

    for landmark in landmarks:
        dx = landmark.x - robot_pose[0]
        dy = landmark.y - robot_pose[1]
        r = sqrt(dx ** 2 + dy ** 2) + 0.05 * random.randn()

        alpha = arctan2(dy, dx) + 0.05 * random.randn()
        n = int(max(abs(dx), abs(dy)) / occ_grid.grid_size)
        x_ray = robot_pose[0]
        y_ray = robot_pose[1]
        add_measurement = True

        for i in range(0, n):
            x_ray += r * cos(alpha) / n
            y_ray += r * sin(alpha) / n

            if occ_grid.isWall(x_ray, y_ray):
                add_measurement = False
                break

        if add_measurement:
            measurements.append([r, alpha])

    return measurements


def plotMap(robot_pose, measurements):
    global occ_grid, ax, updated_cells, walls

    plt.cla()
    cmap = colors.ListedColormap(['grey', 'white', 'orange'])
    bounds = [-1, -.1, .1, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(occ_grid.grid, cmap=cmap, norm=norm, extent=[0, 2.5, 2.5, 0])

    for measure in measurements:
        r = measure[0]
        alpha = measure[1]
        plt.plot([robot_pose[0], robot_pose[0] + r * cos(alpha)], [robot_pose[1], robot_pose[1] + r * sin(alpha)], 'g',
                 linewidth=.4)

    for wall in walls:
        plt.plot(wall[:2], wall[2:4], 'b', linewidth=1.5)

    for landmark in landmarks:
        plt.plot(landmark.x, landmark.y, 'go')
        plt.text(landmark.x, landmark.y + .05, '(' + str(landmark.index) + ')')

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

        dx = x_end - x_start
        dy = y_end - y_start

        n = 10 * int(max(abs(dx), abs(dy)) / occ_grid.grid_size)
        x = x_start
        y = y_start

        for i in range(0, n):
            x += dx / n
            y += dy / n

            i, j = occ_grid.getCellCoordinates(x, y)
            for ii in range(-1, 2):
                for jj in range(-1, 2):
                    occ_grid.true_grid[j+jj, i+ii] = 1

    f = open("landmarks.txt", "r")
    i = 0
    for row in f:
        x = float(row.split(',')[0])
        y = float(row.split(',')[1])
        landmarks.append(Landmark(x, y, i))
        i += 1

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
