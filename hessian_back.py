#hessian calculation

import xrayutilities as xu
import numpy as np
#extraction of the experimental data
    #om - omega values
    #tt - 2theta values
    #psd- XRR intensity
sample="./data/XRR_Ru30nmv5_S"
om, tt, psd = xu.io.getxrdml_scan(sample + '%d.xrdml', 'tt', scannrs=[1, 2, 3],
                                 )
om, tt, psd = om[om<5.5], tt[om<5.5], psd[om<5.5]


beam_div=0.07
def calc_intens(strt,profile_return=False):
    #calculate the intensity using the best-fitted model with variation of some parameters (using the passed string)
    I0 = 6807430
    bckg=0.17877705
    ρSi=2328.89
    ρSiO2=2645.6534097798835
    ρRuO2=2698.6455701232426
    σSi=0
    σSiO2=4.30955895
    σRuO2=2.48009795
    ρRu,wRu,σRu,wSiO2,wRuO2 =strt
    
    lSi = xu.simpack.Layer(xu.materials.Amorphous("Si",ρSi), np.inf, roughness=σSi)
    lSiO2 = xu.simpack.Layer(xu.materials.Amorphous("SiO2",ρSiO2), wSiO2, roughness=σSiO2)
    lRu = xu.simpack.Layer(xu.materials.Amorphous("Ru",ρRu), wRu, roughness=σRu)
    lRuO2 = xu.simpack.Layer(xu.materials.Amorphous("RuO2",ρRuO2), wRuO2, roughness=σRuO2)    


    m = xu.simpack.SpecularReflectivityModel(lSi, lSiO2, lRu, lRuO2, energy='CuKa1', sample_width=20,\
                                         beam_width=0.07, background=bckg,I0 = I0, beam_shape='gaussian', resolution_width = 0.012)
    Y_sim  = m.simulate(om)

    if profile_return:
        return m.densityprofile(500)
    else:
        return Y_sim



def minfunc(Y_sim):
    #chi2 function
    N = len(psd)
    error=np.abs(np.interp((om+beam_div),om,psd)-np.interp((om-beam_div),om,psd))/2
    return np.sum((psd-Y_sim)**2.0/(psd**2.0+error**2.0)**0.5/(N))



def calc_hesiian_comp(strt,det):
    #calculate the pertubated profiles chi2 for derivative calculation
        #strt - best fitted profile
        #det - relative step size for the calculation of the derivatives
    plen = len(strt)
    #chi2 of best fitted profile
    chi0=0
    #chi2 of best fitted profile + delta_i
    chi1=np.zeros(plen)
    #chi2 of best fitted profile - delta_i
    chi_1=np.zeros(plen)
    #chi2 of best fitted profile + delta_i,k
    chi2=np.zeros((plen,plen))
    #chi2 of best fitted profile - delta_i,k
    chi_2=np.zeros((plen,plen))

    #calculation of the components:
    chi0=minfunc(calc_intens(strt))
    for i in range(plen):
        strt0=strt+np.zeros(len(strt))
        strt0[i]=strt[i]*(1.+det)
        chi1[i]=minfunc(calc_intens(strt0))
        strt0=strt+np.zeros(len(strt))
        strt0[i]=strt[i]*(1.-det)
        chi_1[i]=minfunc(calc_intens(strt0))
    for i in range(plen):
        for k in range(plen):
            
            strt0=strt+np.zeros(len(strt))
                
            strt0[i]=strt[i]*(1.+det)
            strt0[k]=strt[k]*(1.+det)
            chi2[i,k]=minfunc(calc_intens(strt0))
            strt0=strt+np.zeros(len(strt))
            strt0[i]=strt[i]*(1.-det)
            strt0[k]=strt[k]*(1.-det)
            chi_2[i,k]=minfunc(calc_intens(strt0))

    #return hessian chi2 values around the best fit for hessian calculation
    return chi0,chi1,chi_1,chi2,chi_2 



def calc_hessian(strt,det=10**-6):
    #hessian components calculation
        #strt - best fitted profile
        #det - relative step size for the calculation of the derivatives
    plen=len(strt)
    hess=np.zeros((plen,plen))
    comp0,comp1,comp_1,comp2,comp_2=calc_hesiian_comp(strt,det)

    #derivatives calculation 
    for i in range(plen):
        for k in range(plen):
            if i==k:
                h=strt[i]*det
                hess[i,k]=(comp1[i]+comp_1[i]-2*comp0)/(h*h)
            else:
                h=strt[i]*det
                d=strt[k]*det
                hess[i,k]=(comp2[i,k]+comp_2[i,k]+2*comp0-comp1[i]-comp1[k]-comp_1[i]-comp_1[k])/(2*h*d)
    
    #return hessian
    return np.matrix(hess)