from numpy import *
from motion import *
from particle_filter import *

def particle_filter_main():
	''' How does the map W update its number of landmarks?
		How should we add noise to the maps?  '''

	''' Initialization '''
	Q, R, lambda_Psi, S = init_parameter():  # Parameter initialization
    W = []

	''' Prediction '''
	S = motion_model_prediction(v, omega, S, R, delta_t)

	''' Measurement '''
    measurement = getMeasurement(robot_pose)  # Measurement from the robot
	h = measurement_model(W, S)  # Predicted measurement
	 
	''' Association '''
	outlier, Psi = associate_known(S, measurements, W, lambda_Psi, Q, known_associations)  # This function should initialize the new landmarks 	
	 
	''' Weigthing '''
	S = weight(S, Psi, outlier)
	 
	''' Resampling '''
	S, W = Systematic_resample(S, W)  # W is the map




