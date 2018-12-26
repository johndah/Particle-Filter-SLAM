from numpy import *
from motion import *
from particle_filter import *

def particle_filter_main():
	''' How does the map W update its number of landmarks?
		How should we add noise to the maps?  '''

	''' Initialization '''
	Q, R, lambda_Psi, S = init_parameter()  # Parameter initialization
	print(Q)
	''' Prediction '''
	S = motion_model_prediction(v, omega, S, R, delta_t)
	print(S)

	''' Measurement '''
	measurement = getMeasurement(robot_pose)  # Measurement from the robot
	W = getLandmarkParticles(S, measurements, Q)  #funktion som lägger till nya landmarks 
	h = measurement_model(W, S)  # Predicted measurement
	 
	''' Association '''
	outlier, Psi = associate_known(S, measurements, W, lambda_Psi, Q)  # This function should initialize the new landmarks 	
	 
	''' Weigthing '''
	S = weight(S, Psi, outlier)
	 
	''' Resampling '''
	S, W = Systematic_resample(S, W)  # W is the map, figure out how to noise the landmarks


'''
Need v and omega vectors
Need a good way to update W
'''



def init_parameter():  # Initialization fo parameters in particle fitler 
	Q = diag([1e-2, 1e-2, 1e-1])  # Measurement noise
	R = diag([1e-2, 1e-2, 1e-1])  # Prediction noise
	lambda_Psi = 0.01  # Outlier threshold
	M = 1e3  # Number of particles
	start_pose = array([[0,0,0]])
	S = particle_init(M, start_pose)  # Particle set

	return Q, R, lambda_Psi, S


#particle_filter_main()