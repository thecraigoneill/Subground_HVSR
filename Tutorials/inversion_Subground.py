import numpy as np
import matplotlib.pyplot as plt
from Subground_HVSR import *

Vs = np.array([250, 750, 1500, 2500])
Vp = 1000*(Vs/1000 + 1.164)/0.902
ro=(310*(Vp*1000)**0.25)/1000

h = np.array([10, 10,20, 1e3])
#Damping
Dp=np.array( [0.1, 0.05, 0.01, 0.01])
Ds=np.array( [0.1, 0.05, 0.01,0.01])

f1=0.0001
f2=150.


mod1 = HVSRForwardModels(fre1=f1,fre2=f2,Ds=Ds,Dp=Dp,h=h,ro=ro,Vs=Vs,Vp=Vp,ex=0.0)
f, hvsr = mod1.Transfer_Function()
print("VS",mod1.Vs)

#print("mod1",mod1.Transfer_Function())

Vs_init=np.array([500, 500, 500, 500])
Vp_init= 1000*(Vs_init/1000 + 1.164)/0.902
ro_init=(310*(Vp*1000)**0.25)/1000
h_init = np.array([15,15,15,1e3])

mod0 = HVSRForwardModels(fre1=f1,fre2=f2,Ds=Ds,Dp=Dp,h=h_init,ro=ro_init,Vs=Vs_init,Vp=Vp_init,ex=0.0)
print("VS", mod0.Vs)
f_init, hvsr_init = mod0.Transfer_Function()

#########################################################################################################
# Amoeba inversion first
#########################################################################################################

#print("mod0",mod0.Transfer_Function())

#print("init",np.c_[f_init,hvsr_init,f,hvsr])

run1 = HVSR_inversion(hvsr=hvsr,hvsr_freq=f,n=1000,n_burn=1000,fre1=f1,fre2=f2,Ds=Ds,Dp=Dp,h=h_init,ro=ro_init,Vs=Vs_init,Vp=Vp_init)
Vs_best,h_best = run1.Amoeba_crawl()
#print("Amoeba:",results1)

Vp_best=1000*(Vs_best/1000 + 1.164)/0.902
ro_best=(310*(Vp_best*1000)**0.25)/1000

h1=h_best
mod2 = HVSRForwardModels(fre1=f1,fre2=f2,Ds=Ds,Dp=Dp,h=h_best,ro=ro_best,Vs=Vs_best,Vp=Vp_best,ex=0.0)
hvsr_best_f, hvsr_best = mod2.Transfer_Function()

#  End amoeba, create new model for MCMC 
###################################################################################################
#print("Results",np.shape(results))
run2 = HVSR_inversion(hvsr=hvsr,hvsr_freq=f,n=6000,n_burn=1,step_size=0.019,step_floor=0.0,alpha=0.17,beta=1.09,fre1=f1,fre2=f2,Ds=Ds,Dp=Dp,h=h_best,ro=ro_best,Vs=Vs_best,Vp=Vp_best)
results = run2.MCMC_walk()
h_best2 = results[0]
Vs_ens2 = results[1]
h_ens2 = results[2]
Vs_best2=results[3]
L1_best2=results[4]
hvsr_best2=results[5]
hvsr_best_f2 = results[6]


Vp_best2=1000*(Vs_best2/1000 + 1.164)/0.902
ro_best2=(310*(Vp_best2*1000)**0.25)/1000

#print(h1)
print(Vs_ens2)
#print(Vs_best)
#print(L1_best)
#print(hvsr_best)
#print(hvsr_best_f)




plt.plot(f,hvsr,color="xkcd:black",label="Original",linewidth=2)
plt.plot(f_init,hvsr_init,color="xkcd:midnight blue",label="Initial Condition",linewidth=0.8)
plt.plot(hvsr_best_f,hvsr_best,color="xkcd:blood red",label="Inversion NM",linestyle="--")
plt.plot(hvsr_best_f2,hvsr_best2,color="xkcd:kelly green",label="Inversion MCMC",linestyle="--")



plt.legend()
plt.savefig("hvsr_inversion_test.png")
plt.show()

p4p =  HVSR_plotting_functions(h=h,ro=ro,Vs=Vs,Vp=Vp)
VS, D = p4p.profile_4_plot()
p5p = HVSR_plotting_functions(h=h_best,ro=ro_best,Vs=Vs_best,Vp=Vp_best)
VS5,D5 = p5p.profile_4_plot()
p0p = HVSR_plotting_functions(h=h_init,ro=ro_init,Vs=Vs_init,Vp=Vp_init)
VS0,D0 = p0p.profile_4_plot()
p2p = HVSR_plotting_functions(h=h_best2,ro=ro_best2,Vs=Vs_best2,Vp=Vp_best2)
VS2,D2 = p2p.profile_4_plot()

plt.clf()


for i in range(len(Vs_ens2[:,0])):
    p4pa =  HVSR_plotting_functions(h=h_ens2[i,:],ro=ro,Vs=Vs_ens2[i,:],Vp=Vp)
    VS, D = p4pa.profile_4_plot()
    plt.plot(VS,D,color="xkcd:raw umber",alpha=0.09,linewidth=0.4)

p4p =  HVSR_plotting_functions(h=h,ro=ro,Vs=Vs,Vp=Vp)
VS, D = p4p.profile_4_plot()
plt.plot(VS,D,color="xkcd:black",label="Original")
plt.plot(VS0,D0,color="xkcd:midnight blue",linewidth=0.8,label="Initial Condition")
plt.plot(VS5,D5,color="xkcd:blood red",label="Inversion NM",linestyle="--")
plt.plot(VS2,D2,color="xkcd:kelly green",label="Inversion MCMC",linestyle="--")

plt.legend()

plt.ylim(np.max(D[-3])+40,0)
#plt.xlim(np.min(VS)-50,np.max(VS)+50)
plt.savefig("model1.png")
plt.show()








