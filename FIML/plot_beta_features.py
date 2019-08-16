import numpy as np
from plotting import *

folder = "Classic_Channel_SAFtrCombo1_SA"

casenames = ['395','550','950','2000','4200','5200']
Retau_list = [395,550,950,2000,4200,5200]

iteration = 1000

for (casename, Retau) in zip(casenames, Retau_list):
    y = np.loadtxt("../mesh/y_%d"%Retau)
    features = np.loadtxt("%s/features.%s"%(folder, casename))
    beta = np.loadtxt("%s/dataset_%s/beta_%d"%(folder, casename, iteration))
    myplot(1, features, beta, '-o',  2.0, casename)
    #mysemilogx(2, y*Retau, 1./features-1., '-o', 2.0, casename)
    #mysemilogx(3, y*Retau, beta, '-o', 2.0, casename)

x = np.linspace(0.,1.,101)
myplot(1, x, ( 0.95*np.tanh(10.0*(1-x)) * (np.exp(5.0*(x-0.9))*1.1-0.1) - 0.1*np.exp(6.5*(0.2-x)) ) * (1./(1.+np.exp(9.0-150.0*x))) + 1, '-r', 4.0, "Aug. fn.")
#myplot(1, x, ( 1.5*np.tanh(5.0*(1-x)) * (np.exp(8.0*(x-0.9))) - 0.1*np.exp(6.5*(0.2-x)) ) * (1./(1.+np.exp(9.0-150.0*x))) + 1, '-r', 4.0, "Aug. fn.")
myfig(1, "Feature ($\\nu/(\\nu_t+\\nu)$)", "Augmentation ($\\beta$)", "Field Inversion", legend=True)
#myfig(2, "$y^+$", "$\\nu_t/\\nu$", "Field Inversion", legend=True)
#myfig(3, "$y^+$", "$\\beta$", "Field Inversion", legend=True)
myfigsave(".", 1)
myfigshow()
