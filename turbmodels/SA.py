import numpy as np
import sys
sys.path.append("..")
from utils import diff, diff2

class SA_Equation:

    # K-Omega turbulence model
    # =========================
    # Inputs - States and Model Augmentation
    # Outputs - Residuals

    #----------------------------------------------------

    def __init__(self):

        self.cb1   = 0.1355
        self.cb2   = 0.622
        self.sigma = 2./3.
        self.kappa = 0.41
        self.cw2   = 0.3
        self.cw3   = 2.0
        self.cv1   = 7.1

    #-----------------------------------------------------

    def getEddyViscosity(self, nu_SA, nu, beta_inv):

        chi = beta_inv * nu_SA/nu
        fv1 = chi**3/(chi**3 + self.cv1**3)

        return nu_SA*fv1

    def getDestruction(self, S, y, nu_SA, beta_inv, cw1):

        r   = nu_SA / (S * self.kappa**2 * y**2 + 1e-10)
        r   = np.minimum(r, 10.0)
        g   = r + self.cw2 * (r**6 - r)
        f_w = g * ((1.0 + self.cw3**6)/(g**6 + self.cw3**6))**(1./6.)

        return (- cw1 * f_w * nu_SA**2 / (y**2 + 1e-10))

    #-----------------------------------------------------

    def evalResidual(self, y, nu, states, beta_inv, res):

        u     = states[0::2]
        nu_SA = states[1::2]

        uy     = diff(y, u)
        nu_SAy = diff(y, nu_SA)

        uyy     = diff2(y, u)
        nu_SAyy = diff2(y, nu_SA)

        nu_t  = self.getEddyViscosity(nu_SA, nu, beta_inv)
        nu_ty = diff(y, nu_t)

        cw1   = self.cb1 / self.kappa**2 + (1.0+self.cb2)/self.sigma 

        S     = np.abs(uy)
        S[1:] = S[1:] + nu_SA[1:]/(self.kappa**2 * y[1:]**2) * (1.0 - beta_inv[1:]*nu_SA[1:]/(nu + beta_inv[1:]*nu_t[1:]))

        nu_SA_res =   self.cb1 * S * nu_SA \
                    + self.getDestruction(S, y, nu_SA, beta_inv, cw1) \
                    + (1.0/self.sigma) * ((nu+nu_SA) * nu_SAyy + (self.cb2 + 1.0) * nu_SAy**2)

        gridrat = (y[-1] - y[-2])/(y[-1] - y[-3])
        nu_SA_res[0]   = -nu_SA[0]
        nu_SA_res[-1]  = (nu_SA[-1]*(1-gridrat*gridrat) -\
                          nu_SA[-2] +\
                          nu_SA[-3]*gridrat*gridrat)/(gridrat*(y[-3] - y[-2]))

        res[1::2] = nu_SA_res

        tau12y = nu_t*uyy + nu_ty*uy

        return tau12y, nu_t
    
    #-----------------------------------------------------
