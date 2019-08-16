import numpy as np
import sys
from utils import *

def SAFeatures1(y, states, nu_t, params):
    
    u      = states[0:np.shape(y)[0]*2:2]
    nu_SA  = states[1:np.shape(y)[0]*2:2]

    f1     = params['nu'] / (nu_t + params['nu'])
    
    return np.vstack((f1)).T

def SAFeatures2(y, states, nu_t, params):
    
    u      = states[0:np.shape(y)[0]*2:2]
    nu_SA  = states[1:np.shape(y)[0]*2:2]

    f1     = params['nu'] / (nu_t + params['nu'])
    f2     = nu_t * diff(y, u) / (params['nu'] * params['Retau'])**2
    f3     = diff(y,u) * (0.41 * y)**2 / params['nu']
    
    return np.vstack((f1,f2,f3))

def SAFeatures3(y, states, nu_t, params):
    
    u      = states[0:np.shape(y)[0]*2:2]
    nu_SA  = states[1:np.shape(y)[0]*2:2]

    f1     = params['nu'] / (nu_t + params['nu'])
    f2     = nu_t * diff(y, u) / (params['nu'] * params['Retau'])**2
    
    return np.vstack((f1,f2))

def KOmFeatures1(y, states, nu_t, params):
    
    u      = states[0:np.shape(y)[0]*3:3]
    k      = states[1:np.shape(y)[0]*3:3]
    w      = states[2:np.shape(y)[0]*3:3]

    f1     = params['nu'] / (nu_t + params['nu'])
    f2     = -diff(y, u) * params['nu'] / params['dpdx']
    #f3     = diff2(y,u) * params['nu']**2 / np.abs(params['dpdx'])**1.5

    #f1 = (f1-np.mean(f1))/(np.mean(f1**2)-np.mean(f1)**2)
    #f2 = (f2-np.mean(f2))/(np.mean(f2**2)-np.mean(f2)**2)
    #f3 = (f3-np.mean(f3))/(np.mean(f3**2)-np.mean(f3)**2)

    return np.vstack((f1, f2))

Features_Dict = {"SAFtrCombo1" : SAFeatures1, "SAFtrCombo2" : SAFeatures2, "SAFtrCombo3" : SAFeatures3,
                 "KOmFtrCombo1" : KOmFeatures1}
