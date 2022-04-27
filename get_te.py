import numpy as np

def get_te(prediction_data, test_data, eps=0.1, dt=0.02, error_version='normsquare', verbose=True):
    '''
    Compute the time t_e when the prediction exceeds the tolerance error eps.
    
    INPUT
    prediction_data ((3 x N) array):
        predicted time series of x, y, and z coordinates
        N is the number of time steps in the test interval
    test_data ((3 x N) array):
        true time series of x, y, and z coordinates
    eps (float):
        error threshold
    dt (float):
        time step
    error_version (str):
        Definition of the error
        * 'normsquare': square of the norm of the difference (prediction - truth)
        * 'normnorm': normalized norm of the difference (prediction - truth), see eq. 14 in Pathak et al. 2018
        
    RETURNS
    te (float):
        time t_e in given time units
    '''
    
    if error_version == 'normsquare':
        error = np.linalg.norm(prediction_data-test_data,axis=0)**2
    
    elif error_version == 'normnorm':
        error = np.linalg.norm(prediction_data-test_data, axis=0)/np.sqrt(np.mean(np.linalg.norm(test_data, axis=0)**2))
        
    N_te = np.where(error > eps)[0][0]
    
    if verbose:
        print('Error exceeds threshold value {} after {} time steps --> t_e = {:.3f}.'.format(eps, N_te, N_te*dt))
    
    return N_te * dt
