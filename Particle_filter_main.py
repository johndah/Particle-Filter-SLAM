from numpy import *
from motion import *
from particle_filter import *

def particle_filter_main():

	''' Parameter initialization '''
	start_pose, Q, Qw, R, lambda_Psi, S, W, dt= init_parameter()  # Parameter initialization
	print(Q)

	for step in some_vector:

		''' Prediction '''
		S = motion_model_prediction(v[step], omega[step], S, R, delta_t)
		print(S)

		''' Measurement '''
		measurement = getMeasurement(robot_pose[step])  # Measurement from the robot
		W = getLandmarkParticles(S, measurements, Q)  #funktion som lägger till nya landmarks 
		h = measurement_model(W, S)  # Predicted measurement
		 
		''' Association '''
		outlier, Psi = associate_known(S, measurements, W, lambda_Psi, Q)  # This function should initialize the new landmarks 	
		 
		''' Weigthing '''
		S = weight(S, Psi, outlier)
		 
		''' Resampling '''
		S, W = Systematic_resample(S, W, Qw)  # W is the map, figure out how to noise the landmarks


'''
Need v and omega vectors
Need a good way to update W
'''





#particle_filter_main()