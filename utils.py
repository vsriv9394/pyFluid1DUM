import numpy as np

def diff(y, q):

    qy       = np.zeros_like(q, dtype=q.dtype)
    qy[1:-1] = 0.5 * ( (q[2:]-q[1:-1])/(y[2:]-y[1:-1]) + (q[1:-1]-q[0:-2])/(y[1:-1]-y[0:-2]) )
    qy[0]    = (q[1]-q[0])/(y[1]-y[0])
    qy[-1]   = 0.0
    
    return qy

def diff2(y, q):

    qyy       = np.zeros_like(q, dtype=q.dtype)
    qyy[1:-1] = 2.0 * ( (q[2:]-q[1:-1])/(y[2:]-y[1:-1]) - (q[1:-1]-q[0:-2])/(y[1:-1]-y[0:-2]) ) / (y[2:]-y[0:-2])
    qyy[0]    = 0.0
    qyy[-1]   = 2.0 * (q[-2] - q[-1]) / (y[-1] - y[-2])**2
    
    return qyy