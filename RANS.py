import sys
sys.path.append(".")
sys.path.append("../pyModelAugmentationUM")

from os import path
from subprocess import call
import numpy as np
import pickle
import adolc as ad

from utils import *
from Features import *
from turbmodels.k_omega import K_Omega_Equation
from turbmodels.SA import SA_Equation
from plotting import *
from Neural_Network import nn
from copy import copy

def create_mesh(Retau, numpoints):
    gf_old = 1.0001
    gf_cur = 1.1
    gf_new = 1.3
    max_old = 1./float(Retau) * (gf_old**(numpoints-1) - 1) / (gf_old - 1)
    max_cur = 1./float(Retau) * (gf_cur**(numpoints-1) - 1) / (gf_cur - 1)
    max_new = 1./float(Retau) * (gf_new**(numpoints-1) - 1) / (gf_new - 1)
    while(abs(max_cur-1.)>1e-10):
        if max_cur>1.:
            gf_new = copy(gf_cur)
            gf_cur = 0.5*(gf_old+gf_new)
        else:
            gf_old = copy(gf_cur)
            gf_cur = 0.5*(gf_old+gf_new)
        max_old = 1./float(Retau) * (gf_old**(numpoints-1) - 1) / (gf_old - 1)
        max_cur = 1./float(Retau) * (gf_cur**(numpoints-1) - 1) / (gf_cur - 1)
        max_new = 1./float(Retau) * (gf_new**(numpoints-1) - 1) / (gf_new - 1)
    return 1./float(Retau) * (gf_cur**np.linspace(0.,numpoints-1,numpoints)-1) / (gf_cur - 1)

class RANS_Channel_Equation:

    #---------------------------------------------------------------------------------------------------------------------

    def __init__(self, y=np.linspace(0.,1.,201)**2, Retau=550, nu=1e-4, model="SA", dt=1.0, n_iter=100,\
                       restart_file=None, verbose=True, ftr_fn="", evalFtr=False, NN=None, tol=1e-8, lambda_reg=1e-5,
                       cfl_start=100., cfl_end=50000., cfl_ramp=2.0, urlxfac=0.5):

        # Set the parameters required for the simulation
        self.params     = {'nu'            : nu,
                           'Retau'         : Retau,
                           'dpdx'          : 0,
                           'model'         : model,
                           'neq'           : 1,
                           'cfl'           : [cfl_start, cfl_end, cfl_ramp],
                           'dt'            : dt,
                           'n_iter'        : n_iter,
                           'verbose'       : verbose,
                           'tol'           : tol,
                           'lambda_reg'    : lambda_reg,
                           'evalFtr'       : evalFtr,
                           'urlxfac'       : urlxfac}

        # Get the machine learning parameters
        self.nn_params  = NN
        
        # Update the pressure gradient (parameter) for the simulation
        self.params['dpdx'] = -(self.params['Retau'] * self.params['nu'])**2

        # Initialize the feature function, turbulence model and the feature array for the problem
        self.ftr_fn     = ftr_fn
        self.model      = None
        self.features   = None
        
        # Initialize the grid, eddy viscosity and augmentation field for the problem
        self.y          = create_mesh(self.params['Retau']*5, np.shape(y)[0])
        #self.y          = y
        self.nu_t       = np.zeros_like(y)
        self.beta       = np.ones_like(y)

        # Depending upon the chosen turbulence model, initialize accordingly
        # K-Omega model
        if self.params['model']=="K_Omega":

            # Initialize the K Omega model
            self.model         = K_Omega_Equation()

            # Set the number of equations to 3
            self.params['neq'] = 3

            # If no restart file has been provided, initialize using default values as mentioned below
            if restart_file==None:
                self.states              = np.zeros((np.shape(y)[0]*self.params['neq']))
                self.states[1::self.params['neq']] = 1e-9 * np.ones_like(self.states[1::self.params['neq']])
                self.states[2::self.params['neq']] = np.zeros_like(self.states[2::self.params['neq']])
                self.nu_t                = np.zeros_like(self.y)
            
            # Otherwise initialize the states and eddy viscosity from file
            else:
                self.states              = np.loadtxt(restart_file)[0:np.shape(self.y)[0]*self.params['neq']]
                self.nu_t                = np.loadtxt(restart_file)[np.shape(self.y)[0]*self.params['neq']:]
                if self.params['evalFtr']==True:
                    self.features = Features_Dict[self.ftr_fn](self.y, self.states, self.nu_t, self.params)

        
        # Spalart Allmaras model
        elif self.params['model']=="SA":

            # Initialize the Spalart Allmaras model
            self.model         = SA_Equation()

            # Set the number of equations to 2
            self.params['neq'] = 2

            # If no restart file has been provided, initialize using default values as mentioned below
            if restart_file==None:
                self.states              = np.zeros((np.shape(y)[0]*self.params['neq']))
                self.states[1::self.params['neq']] = 1e-9 * np.ones_like(self.states[1::self.params['neq']])
                self.nu_t                = np.zeros_like(self.y)
            
            # Otherwise initialize the states and eddy viscosity from file
            else:
                self.states              = np.loadtxt(restart_file)[0:np.shape(self.y)[0]*self.params['neq']]
                self.nu_t                = np.loadtxt(restart_file)[np.shape(self.y)[0]*self.params['neq']:]
                if self.params['evalFtr']==True:
                    self.features = Features_Dict[self.ftr_fn](self.y, self.states, self.nu_t, self.params)

        
        # Error message if turbulence model not available
        else:
            print("\n\nWrong option selected for turbulence model. Available options are:")
            print("\t- K_Omega")
            print("\t- SA")
            sys.exit()
        
        # Set local dt if cfl specified as greater than zero otherwise specify constant dt
        if self.params['cfl'][0]>0.0:
            self.dt       = np.zeros_like(self.states)
            for i_eq in range(self.params['neq']):
                self.dt[i_eq]                                                           =       self.params['cfl'][0] * (self.y[1]-self.y[0])
                self.dt[i_eq+self.params['neq']:-self.params['neq']:self.params['neq']] = 0.5 * self.params['cfl'][0] * (self.y[2:]-self.y[0:-2])
                self.dt[-i_eq-1]                                                        =       self.params['cfl'][0] * (self.y[-1]-self.y[-2])
        else:
            self.dt = self.params['dt']
        
    #---------------------------------------------------------------------------------------------------------------------
    
    def evalResidual(self, states, beta_inv):
        
        # Initialize the residual array
        res  = np.zeros_like(states)

        # Evaluate the second derivative of velocity
        uyy  = diff2(self.y, states[0::self.params['neq']])
        
        # Apply the turbulence model and update residuals from turbulent quantities
        # Also, obtain the gradient of Reynolds Stress and eddy viscosity field
        tau12y, nu_t = self.model.evalResidual(self.y, self.params['nu'], states, beta_inv, res)

        # Store the eddy viscosity field if the run is direct
        if nu_t.dtype=='float64':
            self.nu_t = nu_t
        
        # Specify residuals at internal nodes
        res[0::self.params['neq']] = self.params['nu'] * uyy + tau12y - self.params['dpdx']
        
        # Specify residuals at boundary nodes (non-uniform stencil used)
        gridrat        = (self.y[-1]-self.y[-2])/(self.y[-1]-self.y[-3])
        res[0]         = -states[0]
        res[-self.params['neq']] = (states[-self.params['neq']]*(1-gridrat*gridrat) -\
                          states[-self.params['neq']*2] +\
                          states[-self.params['neq']*3]*gridrat*gridrat)/(gridrat*(self.y[-3]-self.y[-2]))
        
        return res
    
    #---------------------------------------------------------------------------------------------------------------------

    def evalJacobian(self, states, beta_inv):
        
        # Evaluate jacobian of residuals w.r.t. states

        ad.trace_on(1)

        ad_states   = ad.adouble(states)

        ad.independent(ad_states)

        ad_res = self.evalResidual(ad_states, beta_inv)

        ad.dependent(ad_res)
        ad.trace_off()

        return ad.jacobian(1, states)
    
    #---------------------------------------------------------------------------------------------------------------------

    def implicit_euler_update(self, beta_inv):

        # Obtain the residuals and corresponding jacobian matrix
        res = self.evalResidual(self.states, beta_inv)
        jac = self.evalJacobian(self.states, beta_inv)
        
        # Advance the states based on the implicit Euler method
        self.states = self.states + np.linalg.solve(np.eye(np.shape(self.states)[0])/self.dt - jac, res)

        # Get the square root of the number of points in the domain
        N = np.shape(self.y)[0]**0.5

        # Create a list of root mean square residuals for all state variables
        res_out = []
        for i_eq in range(self.params['neq']):
            res_out.append(np.linalg.norm(res[i_eq::self.params['neq']])/N)

        return res_out

    #---------------------------------------------------------------------------------------------------------------------

    def direct_solve(self):
        
        if self.params['verbose']==True:
            print("\n=============================================================")

        # Iterate over solver iterations
        for iteration in range(self.params['n_iter']):

            # Evaluate features if the feature evaluation is set to True or if a machine learning model has been provided
            if self.params['evalFtr']==True or self.nn_params!=None:
                self.features = Features_Dict[self.ftr_fn](self.y, self.states, self.nu_t, self.params)

            # Obtain the correction field from the machine learning model if provided
            if self.nn_params!=None:
                beta = nn.nn.nn_predict(np.asfortranarray(self.nn_params["network"]),
                       
                                        self.nn_params["act_fn"],
                                        self.nn_params["loss_fn"],
                                        self.nn_params["opt"],
                                        
                                        np.asfortranarray(self.nn_params["weights"]),
                                        np.asfortranarray(self.features),

                                        np.asfortranarray(self.nn_params["opt_params_array"]))

                self.beta = self.params['urlxfac']*beta + (1.0-self.params['urlxfac'])*self.beta
                x = self.params['nu'] / (self.nu_t + self.params['nu'])
                self.beta = ( 0.95*np.tanh(10.0*(1-x)) * (np.exp(5.0*(x-0.9))*1.1-0.1) - 0.1*np.exp(6.5*(0.2-x)) ) * (1./(1.+np.exp(9.0-150.0*x))) + 1

            # Update states based on implicit Euler method and obtain the root mean square values of residuals
            res_out = self.implicit_euler_update(self.beta)
            
            # Write the root mean square values to the terminal if verbose is set to true
            if self.params['verbose']==True:
                sys.stdout.write("%9d"%iteration)
                for i_eq in range(self.params['neq']):
                    sys.stdout.write("\t%E"%res_out[i_eq])
                sys.stdout.write("\n")

            # Check if the required tolerance is reached at every iteration after the 20th iteration
            if iteration>20:
                
                if self.params['cfl'][0]<self.params['cfl'][1]:
                    
                    self.params['cfl'][0] = self.params['cfl'][0] * self.params['cfl'][2]
                    self.dt = self.dt * self.params['cfl'][2]
                    
                    if self.params['cfl'][0]>self.params['cfl'][1]:
                        self.dt = self.dt * self.params['cfl'][1] / self.params['cfl'][0]

                if res_out[0]<self.params['tol']:
                    break
        
        if self.params['verbose']==True:
            print("-------------------------------------------------------------")

        return self.states

    #---------------------------------------------------------------------------------------------------------------------

    def adjoint_solve(self, data, weight_sens=False):
        
        # Evaluate the jacobian of residuals w.r.t. states and augmentation field

        ad.trace_on(1)

        ad_states   = ad.adouble(self.states)
        ad_beta     = ad.adouble(self.beta)

        ad.independent(ad_states)
        ad.independent(ad_beta)

        ad_res = self.evalResidual(ad_states, ad_beta)

        ad.dependent(ad_res)
        ad.trace_off()

        jacres = ad.jacobian(1, np.hstack((self.states, self.beta)))

        Rq = jacres[:,0:np.shape(self.states)[0]]
        Rb = jacres[:,np.shape(self.states)[0]:]
        
        # Obtain the jacobian of objective function w.r.t. states and augmentation field
        Jq, Jb = self.getObjJac(data)
        
        # Solve the discrete adjoint system to obtain sensitivity
        psi  = np.linalg.solve(Rq.T,Jq)
        sens = Jb - np.matmul(Rb.T,psi)

        # Obtain the sensitivity of the objective function w.r.t. NN weights
        if weight_sens==True:

            d_weights = nn.nn.nn_get_weights_sens(np.asfortranarray(self.nn_params["network"]),
                                                  
                                                  self.nn_params["act_fn"],
                                                  self.nn_params["loss_fn"],
                                                  self.nn_params["opt"],
                                                  
                                                  np.asfortranarray(self.nn_params["weights"]),
                                                  np.asfortranarray(self.features),
                                                  
                                                  1,
                                                  np.shape(self.beta)[0],
                                                  
                                                  np.asfortranarray(sens),
                                                  np.asfortranarray(self.nn_params["opt_params_array"]))

            return d_weights

        else:
            
            return sens

    #---------------------------------------------------------------------------------------------------------------------

    def getObjRaw(self, states, data, beta):

        return np.mean((states[0::self.params['neq']]/(-self.params['dpdx'])**0.5-data)**2) + self.params['lambda_reg'] * (np.mean((beta-1)**2)) 

    #---------------------------------------------------------------------------------------------------------------------

    def getObj(self, data):

        return self.getObjRaw(self.states, data, self.beta)

    #---------------------------------------------------------------------------------------------------------------------

    def getObjJac(self, data):
        
        ad.trace_on(1)

        ad_states   = ad.adouble(self.states)
        ad_beta     = ad.adouble(self.beta)

        ad.independent(ad_states)
        ad.independent(ad_beta)

        ad_obj = self.getObjRaw(ad_states, data, ad_beta)

        ad.dependent(ad_obj)
        ad.trace_off()

        jacobj = ad.jacobian(1, np.hstack((self.states, self.beta)))

        Jq = jacobj[:,0:np.shape(self.states)[0]]
        Jb = jacobj[:,np.shape(self.states)[0]:]

        return Jq[0,:], Jb[0,:]






if __name__=="__main__":

    plotbeta = False
    useML    = False

    Retau     = 5200
    urlxfac   = 0.5

    cfl_start = 35*5200/Retau
    cfl_end   = 5000
    cfl_ramp  = 1.001

    n_iter = 2000
    
    model  = "SA"
    tol    = 1e-10
    ftr_fn = "SAFtrCombo1"

    NN = None

    if useML==True:
        with open("nn_model", "rb") as f:
            NN = pickle.load(f)

    restart_file = "solution_%s/solution_%d" % (model, Retau)

    #===================================================================================================================

    rans = RANS_Channel_Equation(Retau=Retau, cfl_start=cfl_start, cfl_end=cfl_end, cfl_ramp=cfl_ramp, n_iter=n_iter, 
                                 model=model, ftr_fn=ftr_fn, NN=None, restart_file=restart_file, tol=tol, urlxfac=urlxfac)

    np.savetxt("mesh/y_%d"%Retau, rans.y)

    states = rans.direct_solve()
    y      = rans.y * rans.params['Retau']
    vel    = rans.states[0::rans.params['neq']] / (-rans.params['dpdx'])**0.5

    mysemilogx(rans.params['Retau']*10+1, y,                                                                             vel, '-r', 2.0, 'Baseline')
    mysemilogx(rans.params['Retau']*10+2, y,                                                                  y*diff(y, vel), '-r', 2.0, 'Baseline')
    mysemilogx(rans.params['Retau']*10+3, y, -rans.nu_t*diff(rans.y, rans.states[0::rans.params['neq']])/rans.params['dpdx'], '-r', 2.0, 'Baseline')

    #===================================================================================================================
    
    if useML==True:
        rans.nn_params = NN
        states = rans.direct_solve()
        y      = rans.y * rans.params['Retau']
        vel    = rans.states[0::rans.params['neq']] / (-rans.params['dpdx'])**0.5
        
        mysemilogx(rans.params['Retau']*10+1, y,                                                                             vel, '-g', 2.0, 'Prediction')
        mysemilogx(rans.params['Retau']*10+2, y,                                                                  y*diff(y, vel), '-g', 2.0, 'Prediction')
        mysemilogx(rans.params['Retau']*10+3, y, -rans.nu_t*diff(rans.y, rans.states[0::rans.params['neq']])/rans.params['dpdx'], '-g', 2.0, 'Prediction')
        np.savetxt("figs/beta.%d"%rans.params['Retau'], rans.beta)

    if useML==False:
        savearr = np.hstack((rans.states,rans.nu_t))
        np.savetxt(restart_file, savearr)

    #===================================================================================================================

    y_DNS    =  np.loadtxt("DNS/DNS_%d/DNSsol.dat"%rans.params['Retau'])[:,0]*rans.params['Retau']
    u_DNS    =  np.loadtxt("DNS/DNS_%d/DNSsol.dat"%rans.params['Retau'])[:,2]
    tau_DNS  = -np.loadtxt("DNS/DNS_%d/DNSsol.dat"%rans.params['Retau'])[:,10]
    
    mysemilogx(rans.params['Retau']*10+1, y_DNS[::3],                              u_DNS[::3], '.k', 2.0, 'DNS')
    mysemilogx(rans.params['Retau']*10+2, y_DNS[::3], y_DNS[::3]*diff(y_DNS[::3], u_DNS[::3]), '.k', 2.0, 'DNS')
    mysemilogx(rans.params['Retau']*10+3, y_DNS[::3],                            tau_DNS[::3], '.k', 2.0, 'DNS')

    #===================================================================================================================
    
    myfig(rans.params['Retau']*10+1, "$y^+$", "$u^+$", "Velocity profile (Re=%d)"%rans.params['Retau'], legend=True)
    myfig(rans.params['Retau']*10+2, "$y^+$", "$y^+\\frac{du^+}{dy^+}$", "Velocity gradient profile (Re=%d)"%rans.params['Retau'], legend=True)
    myfig(rans.params['Retau']*10+3, "$y^+$", "$\\tau_{12}$", "Reynolds stress profile (Re=%d)"%rans.params['Retau'], legend=True)
    
    if plotbeta==True:
        mysemilogx(4, rans.y*rans.params['Retau'], rans.beta, '-g', 2.0, 'ML')
        myfig(4, "$y^+$", "$\\beta$", "Augmentation profile", legend=True)
    
    call("mkdir -p figs", shell=True)
    myfigsave(".", rans.params['Retau']*10+1)
    myfigsave(".", rans.params['Retau']*10+2)
    myfigsave(".", rans.params['Retau']*10+3)
    myfigshow()
