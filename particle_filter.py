from numpy import *
import matplotlib.pyplot as plt
from motion import *
import matplotlib as mpl

def particle_init(window, M, start_pose=[]):
    # initializes the particle set of M particles inside given window
    # window should be  [x_min, x_max, y_min, y_max]
    sigma_xy = 1e-1  # Variance in starting position for known pose for x and y
    sigma_theta = 1e-1  # Variance in starting position for known pose for theta
    S = zeros([4, M])
    if not start_pose:  # If start_pose is empty, wont be used
        S[0, :] = random.uniform(window[0], window[1], [1, M])
        S[1, :] = random.uniform(window[2], window[3], [1, M])
        S[2, :] = random.uniform(-pi, pi, [1, M])
    else:
        S = zeros([4, M])  # If start_pose is given the particle set will be gaussians around the starting position
        S[0, :] = start_pose[0] + random.randn(1, M) * sigma_xy
        S[1, :] = start_pose[1] + random.randn(1, M) * sigma_xy
        S[1, :] = start_pose[2] + random.randn(1, M) * sigma_theta
    return S


def associate_known(S, measurements, W, lambda_Psi, Q):
    M = size(S, 1)
    n_landmarks = size(measurements, 1)

    z_hat = measurement_model(S, W)
    particle_measurements = zeros((2 * n_landmarks, M))
    particle_measurements[:, :] = reshape(measurements, (2 * n_landmarks, 1), order='F')

    nu = particle_measurements - z_hat
    feature1_indices = arange(0, 2 * n_landmarks, 2)
    feature2_indices = feature1_indices + 1
    nu[feature2_indices, :] = mod(nu[feature2_indices, :] + pi, 2 * pi) - pi

    q = tile(diag(Q), (M, n_landmarks)).T
    # q = tile(flip(array([diag(Q)]), axis=1), [1, M])
    nu2 = nu ** 2 / q  # Assuming Q is 2x2
    d = nu2[feature1_indices, :] + nu2[feature2_indices, :]

    psi = 1 / (2 * pi * linalg.det(Q) ** .5) * exp(-.5 * d)
    seen_landmarks_indices = where(1 - measurements.any(axis=0))
    psi[seen_landmarks_indices, :] = 0
    outlier = mean(psi, axis=1) <= lambda_Psi

    '''
    z_i = tile(measurements[:, i], (1, M, N)) 
    nu[:, :, :] = z_i - z_hat
    nu[2, :, :] = mod(nu[2, :, :] + pi, 2*pi) - pi
    q = flip(tile(diag(Q), [1, M, N]), axis=1)
    d = sum(nu**2/q)  # Assuming Q is 2x2
    psi[:, :] = 1/(2*pi*linalg.det(Q)**.5)
    Psi[i, :] = max(psi, [], [2])
    '''

    return psi, outlier


def plot_particle_set(S):
    #  Plots particle set S in figure figure
    #  S has dimensions 4xM where M in the number of particles
    for i in range(size(S, 1)):
        t = mpl.markers.MarkerStyle(marker='>')
        t._transform = t.get_transform().rotate_deg(S[2, i]*180/pi)
        plt.scatter(S[0, i], S[1, i], marker=t, s=20, color='b')

def plot_landmark_particle_set(W):
    #  Plots particle set S in figure figure
    #  S has dimensions 4xM where M in the number of particles
    s = where(W.any(axis=1))[0]
    feature1_indices = s[where(mod(s, 2) == 0)[0]]
    feature2_indices = feature1_indices + 1

    plt.scatter(W[feature1_indices, :], W[feature2_indices, :], marker='o', s=5, color=[.05, .3, .05])

def systematic_resample(S, W, Qw):  # Must include map resample
	''' Qw is the noise that get added to the maps in the resampling step '''
    M = S.shape[1]
    cdf = cumsum(S[3, :])

    # print('rand')
    # print(rand)!
    random.seed(0)
    rand = random.uniform(0, 1 / M, 1)
    S_new = zeros(S.shape)
    W_new = zeros(W.shape)

    for i in range(M):
        # print(cdf >= rand + (i ) / M)
        # c = argmax(cdf >= rand + (i) / M)
        c = where(cdf >= rand + (i) / M)[0][0]
        # print('c')
        # print(cdf)
        # print(where(cdf >= rand + (i) / M)[0])
        # print(c)
        S_new[:, i] = S[:, c]
        S_new[3, i] = 1/M
        W_new[:, i] = W[:, c] + dot(random.randn(2,1), Qw)

    return S_new, W_new

'''
M = size(S_bar, 2);
cdf = cumsum(S_bar(4, :));
S = zeros(size(S_bar));
r0 = rand()/M;
for j = 1:M
    [~, i] = find(cdf >= r0 + (j-1)/M, 1);
    S(:,j) = S_bar(:,i);
    S(4,j) = 1/M;
end

'''

'''
W = zeros((2 * size(measurements, 1), M))
    particle_measurements = zeros((2 * size(measurements, 1), M))
    particle_measurements[:, :] = reshape(measurements, (size(measurements), 1), order='F')
    seen_landmarks_indices = where(reshape(tile(measurements.any(axis=0), (2, 1)), (1, 2 * n_landmarks), order='F'))
    noise = tile(diag(Q), (M, sum(measurements.any(axis=0)))).T * random.rand(
        sum(measurements.any(axis=0)) * size(diag(Q)), M)
    particle_measurements[seen_landmarks_indices, :] += noise
    s = where(reshape(tile(measurements.any(axis=0), (2, 1)), (1, 2 * n_landmarks), order='F')[0])[0]
    feature1_indices = s[where(mod(s, 2) == 0)[0]]
    feature2_indices = feature1_indices + 1
    W[feature1_indices, :] = S[0, :] + particle_measurements[feature1_indices, :] * cos(
        S[2, :] + particle_measurements[feature2_indices, :])
    W[feature2_indices, :] = S[1, :] + particle_measurements[feature1_indices, :] * sin(
        S[2, :] + particle_measurements[feature2_indices, :])
'''

def measurement_model(S, W):
    # W is the location of the landmarks on each particles map. Shape [2*landmarks, particles]
    # S is the particle set. Shape [4, particles]
    # h is predicted measurements. Shape [2*landmarks, particles]
    no_landmarks = int(W.shape[0] / 2)
    M = S.shape[1]
    # xindices = arange(0, no_landmarks, 2)
    # yindices = xindices + 1

    s = where(W.any(axis=1))[0]
    feature1_indices = s[where(mod(s, 2) == 0)[0]]
    feature2_indices = feature1_indices + 1

    h = zeros((2 * no_landmarks, M))
    h[feature1_indices, :] = sqrt(square(W[feature1_indices, :] - S[0, :]) + square(W[feature2_indices, :] - S[1, :]))
    h[feature2_indices, :] = arctan2(W[feature2_indices, :] - S[1, :], W[feature1_indices, :] - S[0, :]) - S[2, :]

    h[feature2_indices, :] = mod(h[feature2_indices, :] + pi, 2 * pi) - pi

    '''
    # inputa varannan
    h = zeros((2 * no_landmarks, M))
    h[xindices, :] = sqrt(square(W[xindices, :] - S[0, :]) + square(W[yindices, :] - S[1, :]))
    h[yindices, :] = arctan2(W[yindices, :] - S[1, :], W[xindices, :] - S[0, :]) - S[2, :]
    # print(shape(sqrt(square(W[xindices, :] - S[0,:]) + square(W[yindices, :] - S[1,:]) )))
    # print(shape(h))

    h[yindices, :] = mod(h[yindices, :] + pi, 2 * pi) - pi
    '''

    return h


def weight(S, Psi, outlier):
    # Adds weights to the last row in S
    # Psi in on the [n, M] where n is the number of measurements an M is the number of particles
    # outlier contains information about measurement outliers

    pz = prod(Psi[where(1 - outlier)[0], :], axis=0)  # Non normalized weights without outliers
    w = pz / sum(pz)  # Normalization
    S[3, :] = w

    return S

def measurement_model_test():
    S = array([array([1, 2, 3, 4]), array([-1, -2, -3, -4]), array([0.1, 0.2, 0.3, 0.4]), array([1, 1, 1, 1])])
    W = array([array([1, 2, 3, 4]), array([1, 2, 3, 4]), array([-1, -2, -3, -4]),
               array([-1, -2, -3, -4])])  # 4 particles and 2 landmark
    W = W.T

    h = measurement_model(W, S)


def motion_model_prediction_test():
    window = [0, 5, 0, 5]
    S = particle_init(window, 100)
    fig1 = plt.figure()
    plot_particle_set(S, fig1)

    R = diag([0, 0, 0])
    delta_t = 1
    S = motion_model_prediction(1, 0, S, R, delta_t)
    fig2 = plt.figure()
    plot_particle_set(S, fig2)
    plt.show(fig1)


def systematic_resample_test():
    S = array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [0, 1 / 6, 2 / 6, 3 / 6]])
    W = array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [0, 1 / 6, 2 / 6, 3 / 6]])
    # print(S)
    S, W = systematic_resample(S, W)
    print('S')
    print(S)
    print('W')
    print(W)



def main():
    S = array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [0, 1 / 6, 2 / 6, 3 / 6]])
    # print(S)
    S = systematic_resample(S)
    # print(S)


if __name__ == '__main__':
    main()

'''
def main():
    window = [0,5,0,5]
    S = particle_init(window, 100)
    fig1 = plt.figure()
    plot_particle_set(S, fig1)

    R = diag([0,0,0])
    delta_t = 1
    S = motion_model_prediction(1, 0, S, R, delta_t)
    fig2 = plt.figure()
    plot_particle_set(S, fig2)
    plt.show(fig1)
    '''