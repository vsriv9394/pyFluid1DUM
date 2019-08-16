import numpy as np
import sys
sys.path.append("..")
sys.path.append("/home/vsriv/root/opt/pyModelAugmentationUM")
from subprocess import call
from RANS import RANS_Channel_Equation
from FIML import FIML
from plotting import *
from Features import *

#========================================================================== PROBLEM DEFINITION =========================================================================#

# Set names of as many variables as required (Features names will be used to search in a dictionary 
#                                             of user-defined functions in the companion file of Features.py
#                                             and FIML_type is mandatory with options "Classic", "Direct" or "Embedded")

FIML_type      = "Classic"
Problem        = "Channel"
Features_name  = "SAFtrCombo1"
Model          = "SA"

# Set name of the folder where the files will be saved

folder_name    = "%s_%s_%s_%s"%(FIML_type, Problem, Features_name, Model)

# Create a list of parameters corresponding to the different cases that will be used in augmenting the model

#Retau_list     = [180, 395, 550, 950, 2000, 4200, 5200]
Retau_list     = [950]

#============================================================================ PRE PROCESSING ===========================================================================#

# Create a list of equations which will contain python classes containing solvers for all the cases mentioned above

problems   = []

# Create a list of truth data arrays that will be used in objective function evaluation

data       = []

# Populate the above lists based on the parameter list above

for i_problem in range(len(Retau_list)):

    Retau = Retau_list[i_problem]

    cfl_start = 5000
    cfl_end   = 5000
    cfl_ramp  = 1.2

    n_iter = 200
    
    tol    = 1e-8

    restart_file = "../solution_%s/solution_%d" % (Model, Retau)

    problems.append(RANS_Channel_Equation(Retau=Retau, model=Model, cfl_start=cfl_start, cfl_end=cfl_end, cfl_ramp=cfl_ramp, lambda_reg=1e-1,
                                          n_iter=n_iter, verbose=False, evalFtr=True, ftr_fn=Features_name, tol=tol, restart_file=restart_file))
    
    #------------------------------------------------------------------------------------------------------------------------------------------------------

    data_temp1 = np.loadtxt("../DNS/DNS_%d/DNSsol.dat"%Retau)
    data_temp2 = np.interp(problems[i_problem].y, data_temp1[:,0], data_temp1[:,2])
    data.append(data_temp2)

del data_temp1
del data_temp2

# Define step length for optimization along with the adaptive parameters optpar1 and optpar2

step_length = 1e-2
optpar1     = 0.1
optpar2     = 0.35

# Define the restart iteration for the inverse solve

restart     = 1000

# Define the maximum iteration number (including that before restart to be reached)

maxiter     = 1000

# Whether to apply post-processing

postprocess = True

# Check whether the adjoint formulation of the direct solver is accurate

check_sens_problem_list = [950]

#====================================================================== CONFIGURE NEURAL NETWORK =======================================================================#

# Define the number of neurons in the hidden layers as a list

Hidden_Layers = [7,7]

# Choose an optimizer for the neural network and edit any default values by an update dictionary

nn_opt               = 'adam'
nn_opt_params_update = {'alpha':0.01, 'beta_1':0.8}      # A possible update would look like --->  nn_opt_params_update = {'alpha':0.01, 'beta_1':0.99, 'beta_2':0.9999, 'eps':1e-9}

# Choose an activation function, number of epochs to be trained on and batch size

act_fn               = 'sigmoid'       # Current options are - 'relu', 'sigmoid'
n_epochs_long        = 10000           # Used for FIML-Classic and to initialize FIML-Embedded and Direct
n_epochs_short       = 1000            # Used during training between iterations for FIML-Embedded
batch_size           = 21
weight_factor        = 1.00

#=========================================================================== INVERSE SOLVER ============================================================================#

fiml = FIML(kind           = FIML_type,
            problems       = problems,
            data           = data,
            problem_names  = Retau_list,
            n_iter         = maxiter, 
            folder_name    = folder_name,
            restart        = restart,
            step_length    = step_length,
            optpar1        = optpar1,
            optpar2        = optpar2,
            FD_derivs      = False,
            FD_step_length = 1e-2,
            sav_iter       = 10)

fiml.configure_nn(Hidden_Layers)
fiml.set_nn_optimizer(nn_opt, update_values=nn_opt_params_update)
fiml.nn_params['act_fn']         = act_fn
fiml.nn_params['n_epochs_long']  = n_epochs_long
fiml.nn_params['n_epochs_short'] = n_epochs_short
fiml.nn_params['batch_size']     = batch_size
fiml.nn_params['weights']        = fiml.nn_params['weights'] * weight_factor

for (problem, data, problem_name) in zip(fiml.problems, fiml.data, fiml.problem_names):
    if problem_name in check_sens_problem_list:
        fiml.get_sens(problem, data, check_sens=True)

fiml.inverse_solve()

#=========================================================================== POST PROCESSING ===========================================================================#

if postprocess==True:
    
    call("mkdir -p %s/figs"%fiml.folder_name, shell=True)
    
    mysemilogy(0, np.linspace(0., fiml.n_iter, fiml.n_iter+1), fiml.optim_history, '-ob', 2.0, None)
    myfig(0, "Iterations", "Objective Function", "Optimization convergence")
    myfigsave(fiml.folder_name, 0)
    
    for i_problem in range(len(Retau_list)):

        call("mkdir -p %s/dataset_%d/figs"%(fiml.folder_name, Retau_list[i_problem]), shell=True)
    
        problem = problems[i_problem]
        np.savetxt("%s/features.%d"%(fiml.folder_name, Retau_list[i_problem]), 1e-4/(problem.nu_t+1e-4))

        yp = problem.y * Retau_list[i_problem]
    
        DNS_data      = np.loadtxt("../DNS/DNS_%d/DNSsol.dat"%(Retau_list[i_problem]))
        baseline_data = np.loadtxt("../solution_%s/solution_%d"%(Model, Retau_list[i_problem]))
    
        if fiml.kind=="Classic":
            legend_str = "Inverse"
        else:
            legend_str = "Inverse-ML"
    
        up           = DNS_data[:,2]
        ydudyp       = np.zeros_like(up)
        ydudyp[1:-1] = DNS_data[1:-1,0] * (up[2:]-up[0:-2]) / (DNS_data[2:,0]-DNS_data[0:-2,0])
        upvp         = -DNS_data[:,10]
        
        mysemilogx(i_problem+1,        DNS_data[:,0]*problem.params['Retau'], up,       '.k', 2.0, 'DNS')
        mysemilogx((i_problem+1)*10,   DNS_data[:,0]*problem.params['Retau'], ydudyp,   '.k', 2.0, 'DNS')
        mysemilogx((i_problem+1)*100,  DNS_data[:,0]*problem.params['Retau'], upvp,     '.k', 2.0, 'DNS')
        
        ydudyp       = np.zeros_like(yp)
        upvp         = np.zeros_like(yp)
        up           = baseline_data[0:problem.params['neq']*np.shape(problem.y)[0]:problem.params['neq']] / (-problem.params['dpdx'])**0.5
        ydudyp[1:-1] = yp[1:-1] * (up[2:]-up[0:-2]) / (yp[2:]-yp[0:-2])
        upvp[1:-1]   = baseline_data[problem.params['neq']*np.shape(problem.y)[0]+1:-1] * (up[2:]-up[0:-2]) / (problem.y[2:]-problem.y[0:-2]) / (-problem.params['dpdx'])**0.5
        upvp[-1]     = baseline_data[-1]                                        * (up[-1]-up[-2])   / (problem.y[-1]-problem.y[-2])   / (-problem.params['dpdx'])**0.5
        
        mysemilogx(i_problem+1,        yp, up,       '-r', 2.0, 'Baseline')
        mysemilogx((i_problem+1)*10,   yp, ydudyp,   '-r', 2.0, 'Baseline')
        mysemilogx((i_problem+1)*100,  yp, upvp,     '-r', 2.0, 'Baseline')

        problem.direct_solve()
        
        ydudyp       = np.zeros_like(yp)
        upvp         = np.zeros_like(yp)
        up           = problem.states[0::problem.params['neq']] / (-problem.params['dpdx'])**0.5
        ydudyp[1:-1] = yp[1:-1] * (up[2:]-up[0:-2]) / (yp[2:]-yp[0:-2])
        upvp[1:-1]   = problem.nu_t[1:-1] * (up[2:]-up[0:-2]) / (problem.y[2:]-problem.y[0:-2]) / (-problem.params['dpdx'])**0.5
        upvp[-1]     = problem.nu_t[-1]   * (up[-1]-up[-2])   / (problem.y[-1]-problem.y[-2])   / (-problem.params['dpdx'])**0.5
        
        mysemilogx(i_problem+1,        yp, up,       '-b', 2.0, legend_str)
        mysemilogx((i_problem+1)*10,   yp, ydudyp,   '-b', 2.0, legend_str)
        mysemilogx((i_problem+1)*100,  yp, upvp,     '-b', 2.0, legend_str)
        mysemilogx((i_problem+1)*1000, yp, problem.beta, '-b', 2.0, legend_str)
        
        if fiml.kind=="Classic":
            fiml.problems[i_problem].states = np.loadtxt("../solution_%s/solution_%d"%(Model, Retau_list[i_problem]))[0:np.shape(problems[-1].states)[0]]
            fiml.problems[i_problem].params['cfl'][0] = 10.0
            fiml.problems[i_problem].params['cfl'][1] = 10.0
            fiml.problems[i_problem].params['cfl'][2] = 1.0
            fiml.problems[i_problem].params['verbose'] = True
            fiml.problems[i_problem].params['nn_model_file'] = "%s/model_%s_%d"%(folder_name, FIML_type, maxiter)
            fiml.problems[i_problem].direct_solve()

            problem = fiml.problems[i_problem]

            ydudyp       = np.zeros_like(yp)
            upvp         = np.zeros_like(yp)
            up           = problem.states[0::problem.params['neq']] / (-problem.params['dpdx'])**0.5
            ydudyp[1:-1] = yp[1:-1] * (up[2:]-up[0:-2]) / (yp[2:]-yp[0:-2])
            upvp[1:-1]   = problem.nu_t[1:-1] * (up[2:]-up[0:-2]) / (problem.y[2:]-problem.y[0:-2]) / (-problem.params['dpdx'])**0.5
            upvp[-1]     = problem.nu_t[-1]   * (up[-1]-up[-2])   / (problem.y[-1]-problem.y[-2])   / (-problem.params['dpdx'])**0.5
            
            #mysemilogx(i_problem+1,        yp, up,       '-g', 2.0, 'ML')
            #mysemilogx((i_problem+1)*10,   yp, ydudyp,   '-g', 2.0, 'ML')
            #mysemilogx((i_problem+1)*100,  yp, upvp,     '-g', 2.0, 'ML')
            #mysemilogx((i_problem+1)*1000, yp, problem.beta, '-g', 2.0, 'ML')
        
        myfig(i_problem+1,        "$$y^+$$", "$$u^+$$",                                     "Velocity Profile ($Re_\\tau$=%d)"%Retau_list[i_problem],          legend=True)
        myfig((i_problem+1)*10,   "$$y^+$$", "$$y^+\\frac{\\partial u^+}{\\partial y^+}$$", "Velocity gradient profile ($Re_\\tau$=%d)"%Retau_list[i_problem], legend=True)
        myfig((i_problem+1)*100,  "$$y^+$$", "$$\\frac{-u'v'}{u_\\tau^2}$$",                "Reynolds stress profile ($Re_\\tau$=%d)"%Retau_list[i_problem],   legend=True)
        myfig((i_problem+1)*1000, "$$y^+$$", "$$\\beta$$",                                  "Augmentation profile ($Re_\\tau$=%d)"%Retau_list[i_problem],      legend=True)
    
        myfigsave("%s/dataset_%d"%(fiml.folder_name, Retau_list[i_problem]), i_problem+1)
        myfigsave("%s/dataset_%d"%(fiml.folder_name, Retau_list[i_problem]), (i_problem+1)*10)
        myfigsave("%s/dataset_%d"%(fiml.folder_name, Retau_list[i_problem]), (i_problem+1)*100)
        myfigsave("%s/dataset_%d"%(fiml.folder_name, Retau_list[i_problem]), (i_problem+1)*1000)
        
        myfigshow()
