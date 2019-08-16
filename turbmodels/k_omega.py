import numpy as np
import sys
sys.path.append("..")
from utils import diff, diff2

class K_Omega_Equation:

    # K-Omega turbulence model
    # =========================
    # Inputs - States and Model Augmentation
    # Outputs - Residuals

    #----------------------------------------------------

    def __init__(self):

        self.sigma_k   = 0.6
        self.sigma_w   = 0.5
        self.beta_star = 0.09
        self.gamma     = 0.52
        self.C_lim     = 0.875
        self.beta      = 0.0708

    #-----------------------------------------------------

    def getEddyViscosity(self, uy, k, om):

        #return k / np.maximum(om+1e-16, self.C_lim * (0.5*uy*uy/self.beta_star)**0.5 )
        return k/(om+1e-16)

    #-----------------------------------------------------

    def evalResidual(self, y, nu, states, beta_inv, res):

        u    = states[0::3]
        k    = states[1::3]
        om   = states[2::3]

        uy   = diff(y, u)
        ky   = diff(y, k)
        omy  = diff(y, om)

        uyy  = diff2(y, u)
        kyy  = diff2(y, k)
        omyy = diff2(y, om)
        
        nu_T  = self.getEddyViscosity(uy, k, om)
        nu_Ty = diff(y, nu_T)

        P     = uy*uy

        #k_res  =   beta_inv*nu_T*P - self.beta_star*om*k + (nu + self.sigma_k*k/(om+1e-16))*kyy  + self.sigma_k*k_by_om_y*ky
        #om_res = self.gamma*om/(k+1e-16)*P - self.beta*om*om     + (nu + self.sigma_w*k/(om+1e-16))*omyy + self.sigma_w*k_by_om_y*omy + np.maximum(ky*omy/(om+1e-16), 0.0)*0.125

        #beta_inv[y>30./180.] = 1.

        self.gamma = self.beta/self.beta_star - self.sigma_w*0.41*0.41/(beta_inv*self.beta_star)**0.5

        k_res  =   P*nu_T - beta_inv*self.beta_star*om*k + (nu + self.sigma_k*nu_T)*kyy  + self.sigma_k*nu_Ty*ky
        om_res =   P*self.gamma - beta_inv*self.beta*om*om     + (nu + self.sigma_w*nu_T)*omyy + self.sigma_w*nu_Ty*omy + np.maximum(ky*omy/(om+1e-16), 0.0)*0.125

        gridrat = (y[-1] - y[-2])/(y[-1] - y[-3])
        k_res[0]   = -k[0]
        k_res[-1]  = (k[-1]*(1-gridrat*gridrat) -\
                      k[-2] +\
                      k[-3]*gridrat*gridrat)/(gridrat*(y[-3] - y[-2]))
        om_res[0]  = -(om[0]-5000000*nu/0.005**2)
        om_res[-1] = (om[-1]*(1-gridrat*gridrat) -\
                      om[-2] +\
                      om[-3]*gridrat*gridrat)/(gridrat*(y[-3] - y[-2]))

        res[1::3] = k_res
        res[2::3] = om_res

        tau12y = nu_T*uyy + nu_Ty*uy

        return tau12y, nu_T
    
    #-----------------------------------------------------
