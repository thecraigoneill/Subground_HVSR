import numpy as np
import matplotlib.pyplot as plt
from Subground_HVSR import *

Vs = np.array([650, 650, 1800, 1800])
Vp = 1000*(Vs/1000 + 1.164)/0.902
ro=(310*(Vp*1000)**0.25)/1000

h = np.array([10, 1,50, 1e3])
#Damping
Dp=np.array( [0.01, 0.01, 0.01, 0.01])
Ds=np.array( [0.05, 0.01, 0.01,0.01])

f1=0.01
f2=100.


# Test HVSR forward models - Transer_Function
mod1 = HVSRForwardModels(fre1=f1,fre2=f2,Ds=Ds,Dp=Dp,h=h,ro=ro,Vs=Vs,Vp=Vp,ex=0.0)

f, hvsr = mod1.Transfer_Function()

# Test HVSR forward models - HV
f2, hvsr2 = mod1.HV()

#print("F2",f2)
#print("HVSR2:",hvsr2[:,0])
#print(np.c_[f2[1:],hvsr2[:,0]])

plt.plot(f,hvsr,color="xkcd:blood red")
plt.plot(f2[1:],hvsr2[:,0],color="xkcd:baby shit brown")
plt.savefig("test1.png")
plt.show()


p4p =  HVSR_plotting_functions(h=h,ro=ro,Vs=Vs,Vp=Vp)

VS, D = p4p.profile_4_plot()

plt.clf()
plt.plot(VS,D,color="xkcd:raw umber")
plt.ylim(np.max(D),0)
plt.xlim(np.min(VS)-50,np.max(VS)+50)
plt.savefig("model1.png")
plt.show()



