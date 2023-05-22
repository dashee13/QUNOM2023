import hessian_back as hess
import numpy as np
import matplotlib.pyplot as plt
omega = hess.om
measurments = hess.psd

#define parameters for the model
##Si substrate
ρSi  ,wSi  ,σSi  =2328.89,    np.inf,        0
##Ru layer
ρRu,  wRu,  σRu  = 12590.0313,    291.430586,    4.43368330
##Ru/Air interface
ρRuO, wRuO, σRuO =1469.81758 ,    17.7836146,    0.87128395

fitted_parameters=[ρRu,wRu,σRu,ρRuO,wRuO,σRuO]
names=["ρRu","wRu","σRu","ρRuO","wRuO","σRuO"]


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
covariation = (-hessian)**-1

#plot the table of the trust intervals
delta = np.diagonal(covariation)**0.5
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111)
fig2.patch.set_visible(False)
ax2.axis('off')
ax2.axis('tight')
values=[]
for i in range(len(delta)):
     values = np.append(values,str(fitted_parameters[i])[0:7]+"±"+str(delta[i])[0:6]+" ("+str(delta[i]/fitted_parameters[i]*100)[0:4]+"%)")
the_table = ax2.table(cellText=np.rot90([values],-1),
                      rowLabels=names, colLabels=["95% trust interval"],loc='center')
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



