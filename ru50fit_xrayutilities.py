import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.constants import hbar
import os
import lmfit


import xrayutilities as xu

# or using the more flexible function
sample="XRR_Step_"
om, tt, psd = xu.io.getxrdml_scan(sample + '%d.xrdml', 'tt', scannrs=[1,2,3], 
                                 path='.\data')
#tt,psd,om = tt[om<4.25],psd[om<4.25],om[om<4.25]

I0 = 6807430
bckg=0.17877705
#bckg=0.08613659
# define layers
# Si / SiO2 / Ru
ρSi   ,wSi    ,σSi    =2328.89,    np.inf,   0
##Si/Ru interface
ρSiO2 ,wSiO2  ,σSiO2  =2645.65,    18.54,    4.31
##Ru layer
ρRu   ,wRu    ,σRu    =12590.03,   291.1,    4.32
##Ru/Air interface
ρRuO2 ,wRuO2  ,σRuO2  =2698.65,    18.07,    2.48


lSi = xu.simpack.Layer(xu.materials.Amorphous("Si",ρSi), wSi, roughness=σSi)
l1SiO2 = xu.simpack.Layer(xu.materials.Amorphous("SiO2",ρSiO2), wSiO2, roughness=σSiO2)
    
lRu = xu.simpack.Layer(xu.materials.Amorphous("Ru",ρRu), wRu, roughness=σRu)
lAir = xu.simpack.Layer(xu.materials.Amorphous("RuO2",ρRuO2), wRuO2, roughness= σRuO2)


m = xu.simpack.SpecularReflectivityModel(lSi,l1SiO2,lRu,lAir, energy='CuKa1', sample_width=20,\
                                         beam_width=0.07,I0 = I0,background=bckg, beam_shape='gaussian', resolution_width = 0.012)
sims = m.simulate(om)
plt.plot(om, np.log(sims))
plt.plot(om, np.log(psd))
#%%
# embed model in fit code
fitm = xu.simpack.FitModel(m, plot=True, verbose=False, elog=True)

# set some parameter limitations
fitm.set_param_hint('resolution_width', vary=False)
fitm.set_param_hint('background', vary=False)
fitm.set_param_hint('I0', vary=False)

fitm.set_param_hint('Si_density', vary=False)#, max = xu.materials.Si.density)
fitm.set_param_hint('Si_roughness', vary=False)
#fitm.set_param_hint('SiO2_density', vary=False)
#fitm.set_param_hint('SiO2_roughness', vary=False)
#fitm.set_param_hint('SiO2_thickness', vary=False)

#fitm.set_param_hint('Ru_thickness', vary=False)
#fitm.set_param_hint('Ru_roughness', vary=False)

#fitm.set_param_hint('Ru_density', vary=False)

#fitm.set_param_hint('RuO2_roughness', vary=False)
#fitm.set_param_hint('Ru_density', min=0,
#                    max=xu.materials.Ru.density)
p = fitm.make_params()
fitm.set_fit_limits(xmin=0.39, xmax=10) # give it free range

# perform the fit
res = fitm.fit(psd, p, om)
lmfit.report_fit(res, min_correl=0.7)
# export the fit result for the full data range (Note that only data between
# xmin and xmax were actually optimized)
# numpy.savetxt(
#     "xrrfit.dat",
#     numpy.vstack((ai, res.eval(res.params, x=ai))).T,
#     header="incidence angle (deg), fitted intensity (arb. u.)",
# )
# m.densityprofile(500, plot=True)  # 500 number of points
#%%




