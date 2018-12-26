from numpy import *
import matplotlib.pyplot as plt
from matplotlib import colors
import motion
import particle_filter as pf
import matplotlib as mpl

axis = [0, 2.5, 0, 2.5]


class Landmark(object):

    def __init__(self, x, y, index):
        self.x = x
        self.y = y
        self.index = index


class OccupancyGrid:

    def __init__(self):
        self.grid_size = 2.5 / 100
        self.n_cells_x = int((axis[1] - axis[0]) / self.grid_size)
        self.n_cells_y = int((axis[3] - axis[2]) / self.grid_size)

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
    global occ_grid, updated_cells, landmarks

    sigma = 5e-2
    n_landmarks = size(landmarks, 1)
    measurements = zeros((2, n_landmarks))
    updated_cells = []

    n_alphas = 50
    alphas = linspace(-pi, pi, n_alphas) + 0.05 * random.randn(n_alphas)

    for alpha in alphas:
        x_ray = robot_pose[0]
        y_ray = robot_pose[1]

        is_free = True
        n = 1 / occ_grid.grid_size

        while is_free:
            x_ray += cos(alpha) / n
            y_ray += sin(alpha) / n

            occ_grid.markFreeSpace(x_ray, y_ray)

            if occ_grid.isWall(x_ray, y_ray):
                is_free = False
                occ_grid.markOccupiedSpace(x_ray, y_ray)

    for j in range(n_landmarks):
        dx = landmarks[0, j] - robot_pose[0]
        dy = landmarks[1, j] - robot_pose[1]
        r = sqrt(dx ** 2 + dy ** 2)
        alpha = arctan2(dy, dx)

        n = int(max(abs(dx), abs(dy)) / occ_grid.grid_size)
        x_ray = robot_pose[0]
        y_ray = robot_pose[1]
        add_measurement = True
        first_wall = False
        wall_count = 0
        wall_count_inc = 0

        for i in range(0, n):
            x_ray += r * cos(alpha) / n
            y_ray += r * sin(alpha) / n

            if occ_grid.isWall(x_ray, y_ray):

                if first_wall and wall_count > 10:
                    add_measurement = False
                    break
                first_wall = True
                wall_count_inc = 1

            wall_count += wall_count_inc

        if add_measurement:
            measurements[:, j] = [r + sigma * random.randn(), alpha - robot_pose[2] + sigma * random.randn()]

    return measurements


def plotMap(robot_poses, measurements, pose_index, S, W):
    global occ_grid, ax, updated_cells, walls, landmarks

    plt.cla()
    robot_pose = robot_poses[:, pose_index]

    cmap = colors.ListedColormap(['grey', 'white', 'orange'])
    bounds = [-1, -.1, .1, 1]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax.imshow(occ_grid.grid, cmap=cmap, norm=norm, extent=[0, 2.5, 2.5, 0])

    for i in range(size(measurements, 1)):
        if measurements[0, i].any():
            r = measurements[0, i]
            alpha = measurements[1, i]
            plt.plot([robot_pose[0], robot_pose[0] + r * cos(alpha + robot_pose[2])], [robot_pose[1], robot_pose[1] + r * sin(alpha + robot_pose[2])],
                     'g', linewidth=.7)

    for wall in walls:
        plt.plot(wall[:2], wall[2:4], 'b', linewidth=1.5)

    for j in range(size(landmarks, 1)):
        plt.plot(landmarks[0, j], landmarks[1, j], 'go', markersize=10)
        plt.text(landmarks[0, j], landmarks[1, j] + .05, '(' + str(j) + ')')

    pf.plot_particle_set(S)
    pf.plot_landmark_particle_set(W)

    plt.plot(robot_poses[0, :-2], robot_poses[1, :-2], 'gx')
    t = mpl.markers.MarkerStyle(marker='>')
    t._transform = t.get_transform().rotate_deg(robot_pose[2] * 180 / pi)
    plt.scatter(robot_pose[0], robot_pose[1], marker=t, s=40, color='g')

    plt.axis(axis)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Map')
    plt.pause(1e-5)


def initMap():
    global occ_grid, ax, walls, path, landmarks, n_landmarks
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
                    occ_grid.true_grid[j + jj, i + ii] = 1

    f = open("landmarks.txt", "r")
    i = 0
    for row in f:
        if '#' not in row:
            x = float(row.split(',')[0])
            y = float(row.split(',')[1])
            if i == 0:
                landmarks = array([[x], [y]])
            else:
                landmarks = concatenate((landmarks, array([[x], [y]])), axis=1)
            i += 1
    n_landmarks = i

    f = open("path.txt", "r")
    distances = []
    angular_velocities = []
    for row in f:
        if '#' not in row:
            distances.append(float(row.split(',')[0]))
            angular_velocities.append(float(row.split(',')[1]))
    path = [distances, angular_velocities]


def getLandmarkParticles(S, measurements, Q, W):
    M = size(S, 1)

    particle_measurements = zeros((2 * size(measurements, 1), M))
    particle_measurements[:, :] = reshape(measurements, (size(measurements), 1), order='F')
    seen_landmarks_indices = where(reshape(tile(measurements.any(axis=0), (2, 1)), (1, 2 * n_landmarks), order='F'))
    noise = tile(diag(Q), (M, sum(measurements.any(axis=0)))).T * random.rand(
        sum(measurements.any(axis=0)) * size(diag(Q)), M)
    particle_measurements[seen_landmarks_indices, :] += noise
    s = where(reshape(tile(measurements.any(axis=0), (2, 1)), (1, 2 * n_landmarks), order='F')[0])[0]
    s = intersect1d(s, where(1 - W.any(axis=1)))
    feature1_indices = s[where(mod(s, 2) == 0)[0]]
    feature2_indices = feature1_indices + 1
    W[feature1_indices, :] = S[0, :] + particle_measurements[feature1_indices, :] * cos(
        S[2, :] + particle_measurements[feature2_indices, :])
    W[feature2_indices, :] = S[1, :] + particle_measurements[feature1_indices, :] * sin(
        S[2, :] + particle_measurements[feature2_indices, :])

    return W


def getOdometry(start_pose, dt):
    distances = path[0]
    a_velocities = path[1]
    n_path = int(sum(distances) / dt)

    robot_poses = zeros([3, n_path + 1])
    robot_poses[:, 0] = start_pose

    velocities = ones((1, n_path))
    angular_velocities = zeros((1, n_path))
    start = 0
    for path_index in range(len(a_velocities)):
        n = min(int(distances[path_index] / dt), n_path - start)
        end = start + n
        angular_velocities[0, start:end] = a_velocities[path_index] * ones((1, n))
        start = end

    return robot_poses, velocities, angular_velocities


def init_parameter():  # Initialization fo parameters in particle fitler
    x0, y0, theta0 = 0.25, .25, pi / 2
    Q = 1e-2 * eye(2)  # Measurement noise
    R = diag([1e-2, 1e-2, 1e-1])  # Prediction noise
    lambda_Psi = 0.01  # Outlier threshold
    M = 100  # Number of particles
    start_pose = [x0, y0, theta0]
    S = pf.particle_init(axis, M, start_pose)  # Particle set
    W = zeros((2 * n_landmarks, M))
    dt = 0.1

    return start_pose, Q, R, lambda_Psi, S, W, dt


def particleFilterSlam():
    global path, n_landmarks

    start_pose, Q, R, lambda_Psi, S, W, dt = init_parameter()
    robot_poses, velocities, angular_velocities = getOdometry(start_pose, dt)

    for i in range(0, 10):  # n_path):

        robot_poses = motion.motion_model(velocities[0, i], angular_velocities[0, i], robot_poses, dt, i)
        S = motion.motion_model_prediction(S, velocities[0, i], angular_velocities[0, i], R, dt)

        measurements = getMeasurements(robot_poses[:, i])
        W = getLandmarkParticles(S, measurements, Q, W)

        plotMap(robot_poses, measurements, i, S, W)

        psi, outlier = pf.associate_known(S, measurements, W, lambda_Psi, Q)
        S = pf.weight(S, psi, outlier)

        S, W = pf.systematic_resample(S, W)

    print('Done')


def main():
    global occ_grid, ax
    occ_grid = OccupancyGrid()

    fig, ax = plt.subplots()
    initMap()
    particleFilterSlam()


if __name__ == '__main__':
    main()
    plt.show()
