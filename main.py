import hessian_back as hess
import numpy as np
import matplotlib.pyplot as plt
import xrayutilities as xu
omega = hess.om
measurments = hess.psd

#define parameters for the model
##Si substrate
ρSi   ,wSi    ,σSi    =2328.89,    np.inf,   0
##Si/Ru interface
ρSiO2 ,wSiO2  ,σSiO2  =2645.65,    18.54,    4.31
##Ru layer
ρRu   ,wRu    ,σRu    =12590.03,   291.1,    4.32
##Ru/Air interface
ρRuO2 ,wRuO2  ,σRuO2  =2698.65,    18.07,    2.48

fitted_parameters=[ρRu,wRu,σRu,wSiO2,wRuO2]
#fitted_parameters=[ρRu,wRu,σRu,wSiO2,wRuO2,σRuO2]

names=["ρRu","wRu","σRu", "wSiO2","wRuO2"]
#names=["ρRu","wRu","σRu", "wSiO2","wRuO2","σRuO2"]


#plot the measured and fitted XRR
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111)
ax1.set_xlabel("omega, deg")
ax1.set_ylabel("intensity, cps")
ax1.plot(omega,np.log10(measurments),label="measurments")
ax1.plot(omega,np.log10(hess.calc_intens(fitted_parameters)),label="simulation")
ax1.legend()

#calculate hessian and covariation
hessian=hess.calc_hessian(fitted_parameters)
covariation = (hessian)**-1

#plot the table of the trust intervals
delta = 2*np.diagonal(covariation)**0.5
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111)
fig2.patch.set_visible(False)
ax2.set_ylabel("density, g/cm3")
ax2.set_xlabel("depth, nm")
x_min, y_min = hess.calc_intens(fitted_parameters-delta,True)
x_max, y_max = hess.calc_intens(fitted_parameters+delta,True)
x_0, y_0 = hess.calc_intens(fitted_parameters,True)

y_max=np.interp(x_min,x_max,y_max)
y_0=np.interp(x_min,x_0,y_0)
ax2.plot(-x_min,y_0,"k",linewidth=0.6)
ax2.plot(-x_min,y_min,"k--",linewidth=0.6)
ax2.plot(-x_min,y_max,"k--",linewidth=0.6)
ax2.fill_between(-x_min,y_min,y_max)
values=[]
for i in range(len(delta)):
     values = np.append(values,str(fitted_parameters[i])[0:5]+"±"+str(delta[i])[0:4]+" ("+str(delta[i]/fitted_parameters[i]*100)[0:4]+"%)")
the_table = ax2.table(cellText=np.rot90([values],-1),
                      rowLabels=names, colLabels=["95% trust interval"],loc='bottom',bbox=[0.0,-0.45,1,.28])



fig2.tight_layout()


#calculation of correlation matrix
leng=len(fitted_parameters)
correlation = np.zeros((leng,leng))
for i in range(leng):
        for k in range(leng):
            if k>i:
                correlation[i,k]=covariation[i,k]/np.sqrt(covariation[i,i]*covariation[k,k])
            if k==i:
                correlation[i,k]=None
            if k<i:
                correlation[i,k]=correlation[k,i]

#plot correlation matrix
fig3 = plt.figure(figsize=(8, 6))
ax3 = fig3.add_subplot(111)
cax = ax3.matshow(correlation)
fig3.colorbar(cax)
alpha = names
plt.title("Correlation matrix")
ax3.set_xticklabels(np.append([' '],alpha))
ax3.set_yticklabels(np.append([' '],alpha))
plt.show()



