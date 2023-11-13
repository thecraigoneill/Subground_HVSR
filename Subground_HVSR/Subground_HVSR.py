# HVSR analysis and inversion for subsurface structure
# Developed by Craig O'Neill 2023
# Distributed under the accompanying MIT licence
# Use at own discretion

import numpy as np
import numpy as _np
import matplotlib.pyplot as plt
from numpy.fft import rfft, irfft, fft, ifft, fftfreq
from numba import jit
from scipy.optimize import minimize
from scipy.signal import periodogram, welch,lombscargle
import pandas as pd
from scipy.interpolate import interp1d
from scipy.interpolate import griddata, Rbf
import glob

class HVSRForwardModels(object):
    def __init__(self,
				 fre1 = 10,
				 fre2 = 100,
				 Ds = np.array( [0.05, 0.01, 0.01,0.01]),
                 Dp = np.array( [0.05, 0.01, 0.01,0.01]),
				 h = np.array([10, 200,100, 1e3]),
				 ro = np.array([1000, 2000, 2000, 2000])/1000,
				 Vs = np.array([500, 1500, 1500, 1500]),
				 Vp = np.array([500, 3000, 3000, 3000]),
                 ex = 0.0,
				 filename = None
					):
            self.fre1 = fre1
            self.fre2 = fre2
            self.Ds = Ds
            self.Dp = Dp
            self.h = h
            self.ro = ro
            self.Vs = Vs
            self.Vp = Vp
            self.ex = ex
            

    def Transfer_Function(self,):
        """

        Adopted from Transfer Function Tool
        Weaponised by Craig O'Neill for inversion of HVSR
        Assumes the transfer function approximates HVSR response (which it turns out it not bad, see Nakamura).
        Note uses damping D, not Qs. 

        Compute the SH-wave transfer function using Knopoff formalism
        (implicit layer matrix scheme). Calculation can be done for an
        arbitrary angle of incidence (0-90), with or without anelastic
        attenuation (qs is optional).

        It return the displacements computed at arbitrary depth.
        If depth = -1, calculation is done at each layer interface
        of the profile.

        NOTE: the implicit scheme is simple to understand and to implement,
        but is also computationally intensive and memory consuming.
        For the future, an explicit (recursive) scheme should be implemented.

        :param float or numpy.array freq:
        array of frequencies in Hz for the calculation

        :param numpy.array hl:
            array of layer's thicknesses in meters (half-space is 0.)

        :param numpy.array vs:
            array of layer's shear-wave velocities in m/s

        :param numpy.array dn:
            array of layer's densities in kg/m3

        :param numpy.array qs:
            array of layer's shear-wave quality factors (adimensional)

        :param float inc_ang:
            angle of incidence in degrees, relative to the vertical
            (default is vertical incidence)

        :param float or numpy.array depth:
            depths in meters at which displacements are calculated
            (default is the free surface)

        :return numpy.array dis_mat:
            matrix of displacements computed at each depth (complex)
        """

        freq = np.logspace(np.log10(self.fre1),np.log10(self.fre2),500)
        hl = self.h
        vs = self.Vs
        dn = self.ro
        Ds = self.Ds
        qs = np.ones_like(vs)*(1.0/(2.0*Ds))
        inc_ang=0. 
        depth=0.

        # Precision of the complex type
        CTP = 'complex128'

        # Check for single frequency value
        if isinstance(freq, (int, float)):
            freq = _np.array([freq])

        # Model size
        lnum = len(hl)
        fnum = len(freq)

        # Variable recasting to numpy complex
        hl = _np.array(hl, dtype=CTP)
        vs = _np.array(vs, dtype=CTP)
        dn = _np.array(dn, dtype=CTP)

        # Attenuation using complex velocities
        if qs is not None:
            qs = _np.array(qs, dtype=CTP)
            vs *= ((2.*qs*1j)/(2.*qs*1j-1.))

        # Conversion to angular frequency
        angf = 2.*_np.pi*freq

        # Layer boundary depth (including free surface)
        #print("hl",hl,hl.dtype)
        bounds = self.interface_depth(self)

        # Check for depth to calculate displacements
        if isinstance(depth, (int, float)):
            if depth < 0.:
                depth = _np.array(bounds)
            else:
                depth = _np.array([depth])
        znum = len(depth)

        # -------------------------------------------------------------------------
        # Computing angle of propagation within layers

        iD = _np.zeros(lnum, dtype=CTP)
        iM = _np.zeros((lnum, lnum), dtype=CTP)

        iD[0] = _np.sin(inc_ang)
        iM[0, -1] = 1.

        for nl in range(lnum-1):
            iM[nl+1, nl] = 1./vs[nl]
            iM[nl+1, nl+1] = -1./vs[nl+1]

        iA = _np.linalg.solve(iM, iD)
        iS = _np.arcsin(iA)

        # -------------------------------------------------------------------------
        # Elastic parameters

        # Lame Parameters : shear modulus
        mu = dn*(vs**2.)

        # Horizontal slowness
        ns = _np.cos(iS)/vs

        # -------------------------------------------------------------------------
        # Data vector initialisation

        # Layer's amplitude vector (incognita term)
        amp_vec = _np.zeros(lnum*2, dtype=CTP)

        # Layer matrix
        lay_mat = _np.zeros((lnum*2, lnum*2), dtype=CTP)

        # Input motion vector (known term)
        inp_vec = _np.zeros(lnum*2, dtype=CTP)
        inp_vec[-1] = 1.

        # Output layer's displacement matrix
        dis_mat = _np.zeros((znum, fnum), dtype=CTP)

        # -------------------------------------------------------------------------
        # Loop over frequencies

        for nf in range(fnum):

            # Reinitialise the layer matrix
            lay_mat *= 0.

            # Free surface constraints
            lay_mat[0, 0] = 1.
            lay_mat[0, 1] = -1.

            # Interface constraints
            for nl in range(lnum-1):
                row = (nl*2)+1
                col = nl*2

                exp_dsa = _np.exp(1j*angf[nf]*ns[nl]*hl[nl])
                exp_usa = _np.exp(-1j*angf[nf]*ns[nl]*hl[nl])

                # Displacement continuity conditions
                lay_mat[row, col+0] = exp_dsa
                lay_mat[row, col+1] = exp_usa
                lay_mat[row, col+2] = -1.
                lay_mat[row, col+3] = -1.

                # Stress continuity conditions
                lay_mat[row+1, col+0] = mu[nl]*ns[nl]*exp_dsa
                lay_mat[row+1, col+1] = -mu[nl]*ns[nl]*exp_usa
                lay_mat[row+1, col+2] = -mu[nl+1]*ns[nl+1]
                lay_mat[row+1, col+3] = mu[nl+1]*ns[nl+1]

            # Input motion constraints
            lay_mat[-1, -1] = 1.

            # Solving linear system of wave's amplitudes
            try:
                amp_vec = _np.linalg.solve(lay_mat, inp_vec)
            except:
                amp_vec[:] = _np.nan

            # ---------------------------------------------------------------------
            # Solving displacements at depth

            for nz in range(znum):

                # Check in which layer falls the calculation depth
                if depth[nz] <= hl[0]:
                    nl = 0
                    dh = depth[nz]
                elif depth[nz] > bounds[-1]:
                    nl = lnum-1
                    dh = depth[nz] - bounds[-1]
                else:
                    # There might be a more python way to do that...
                    nl = map(lambda x: x >= depth[nz], bounds).index(True) - 1
                    dh = depth[nz] - bounds[nl]

                # Displacement of the up-going and down-going waves
                exp_dsa = _np.exp(1j*angf[nf]*ns[nl]*dh)
                exp_usa = _np.exp(-1j*angf[nf]*ns[nl]*dh)

                dis_dsa = amp_vec[nl*2]*exp_dsa
                dis_usa = amp_vec[nl*2+1]*exp_usa

                dis_mat[nz, nf] = dis_dsa + dis_usa

        return freq, np.abs(dis_mat[0,:])


# =============================================================================

    def interface_depth(self, dtype='complex128'):
        """
        Utility to calcualte the depth of the layer's interface
        (including the free surface) from a 1d thickness profile.

        :param numpy.array hl:
            array of layer's thicknesses in meters (half-space is 0.)

        :param string dtype:
            data type for variable casting (optional)

        :return numpy.array depth:
            array of interface depths in meters
        """
        CTP = 'complex128'
        hl2 = self.h
        hl = _np.array(hl2, dtype=CTP)
        #print("In interface, hl:",hl2)
        depth = np.array([sum(hl[:i]) for i in range(len(hl))])
        depth = _np.array(depth.real, dtype="float64")

        return depth


    def HV(self,):
        # Needs a cross-check against the matlab code for high freq shift.
        # ============================================================================
        # Code courtesy of OpenHVSR/Model HVSR.
        # Converted in Python by CONeill
        # Assumes body wave hum in transfer function (not Rayleigh ellipticity)
        # Note uses quality factor Qs, not damping. Does the conversion internally. 
        # INPUT:
        # vp   = P-wave velocity vector (m/s)   (for all layers + half-space)
        # vs   = S-wave velocity vector (m/s)   (for all layers + half-space)
        # ro   = Density vector (g/cm^3)        (for all layers + half-space)
        # h    = Layer thickness vector (m)     (for all layers + half-space)
        # qp   = Quality factor vector, P-waves (for all layers + half-space)
        # qs   = Quality factor vector, S-waves (for all layers + half-space)
        # ex   = Exponent in Q(f) = Q(1.0 Hz)*f^ex 
        #        (ex = 0 --> no frequency-dependence of Q)
        # fref = Frequency at which velocity is measured (Hz) 
        #        (fref=0 --> no body-wave velocity dispersion)
        # f    = Frequency vector (Hz)

        #  OUTPUT:
        # HVSR = AMP_S./AMP_P;
        # AMP_P= Amplification spectrum for vertically incident P-waves
        # AMP_S= Amplification spectrum for vertically incident S-waves
        #Vp,Vs,ro,h,qp,qs,ex,fref,f
        Ds = self.Ds
        Dp = self.Dp
        h =  self.h
        ro = self.ro
        Vs = self.Vs
        Vp = self.Vp
        ex = self.ex
        f = np.logspace(np.log10(self.fre1),np.log10(self.fre2),500)
        fref = 1. # Frequency at which vel is measured. Assumed to be 1. 
        qp = np.ones_like(Vp)*(1.0/(2.0*Dp))
        qs = np.ones_like(Vs)*(1.0/(2.0*Ds))

        h[-1]=9e4;
        ns=len(Vp)
        nf=len(f)
        nsl=ns-1
        HVSR=np.array([]) 
        AMP_P=np.array([]) 
        AMP_S=np.array([])
        for m in (1,2):           #m=1 for P-waves, m=2 for S-waves
            TR=np.zeros((50,1),dtype=complex);
            AR=np.zeros((50,1),dtype=complex);
            qf=np.zeros((ns,nf),dtype=complex);
            T=np.zeros((nsl,1),dtype=complex);
            A=np.zeros((nsl,1),dtype=complex);
            FI=np.zeros((nsl,1),dtype=complex);
            Z=np.zeros((nsl,1),dtype=complex);
            X=np.zeros((nsl,1),dtype=complex);
            FAC=np.zeros((ns,nf),dtype=complex);  #Increased from nsl to ns - indexing required it below...
    
            if (m==1):
                c=Vp 
                q=qp
            else:
                c=Vs
                q=qs

            for j in np.arange(0,ns,1):   #Note unlike matlab, arange will stay one short of end value
                for i in np.arange(0,nf,1):
                    #print(i,j,q[j],f[i])
                    qf[j,i]=q[j]*(f[i]**ex)

            #print(qf)
    
            idisp=0
            if fref >0:
                idisp=1
        
            for I in np.arange(0,ns-1,1):
                TR[I]=h[I]/c[I]    # distance/vel=transit time for layer
                AR[I]=ro[I]*c[I]/ro[I+1]/c[I+1]  # This is an impedance contrast
    
            TOTT=0
    
            for I in np.arange(0,nsl,1):
                TOTT=TOTT+TR[I]  # Sum of time - total transit time for section
    
    
            X[0]=np.complex(1.,0.)
            Z[0]=np.complex(1.,0.)
        
            if (idisp==0):  
                FJM1=1
                FJ=1
    
            for J in np.arange(0,nsl,1):
                for i in np.arange(0,nf,1):
                    FAC[J,i]=2./(1.+np.sqrt(1.+qf[J,i]**(-2)))*(1.-np.complex(0.,1.)/qf[J,i])
                    FAC[J,i]=np.sqrt(FAC[J,i])
    
            FAC[nsl,0:nf-1]=np.complex(1.,0.) #nsl +1 in matlab - overloading?
            qf[nsl,0:nf-1]=999999.
    
            jpi=1/np.pi
    
            AMP=np.zeros( (len(np.arange(0,nf,1)),1) )
            #print("FAC",np.shape(FAC),np.shape(X),X)

            for k in np.arange(0,nf,1):
                ALGF=np.log(f[k]/fref)
        
                for J in np.arange(1,nsl,1): #2:(nsl+1)
            
                    if (idisp != 0):
                        FJM1=1.+jpi/qf[J-1,k]*ALGF
                        FJ  =1.+jpi/qf[J,k]  *ALGF
            
                    T[J-1]=TR[J-1]*FAC[J-1,k]/FJM1
                    A[J-1]=AR[J-1]*FAC[J,k]/FAC[J-1,k]*FJM1/FJ
                    FI[J-1]=(2*np.pi)*f[k]*T[J-1]
                    ARG=np.complex(0.,1.)*FI[J-1]
                    CFI1=np.exp(ARG)
                    CFI2=np.exp(np.complex(-1.,0.)*ARG)
                    Z[J]=(1.+A[J-1])*CFI1*Z[J-1]+(1.-A[J-1])*CFI2*X[J-1] #complex impedance
                    Z[J]=Z[J]*np.complex(0.5,0.0)
                    X[J]=(1.-A[J-1])*CFI1*Z[J-1]+(1.+A[J-1])*CFI2*X[J-1]
                    X[J]=X[J]*np.complex(0.5,0.0)
                    #print(J,nsl,Z[0],Z[J])

                #print(k,nsl,np.shape(Z),np.size(Z))    
                AMP[k]=(1./abs(Z[nsl-1]))

            if (m==1): 
                AMP_P=AMP 
            else: 
                AMP_S=AMP


        HVSR=AMP_S/AMP_P
        #print(np.c_[HVSR,AMP_S,AMP_P])
        return(f,HVSR[0:-1])
    
        
class HVSR_plotting_functions(object):
    def __init__(self,
                    filename = None,
                    h = np.array([10, 200,100, 1e3]),
				    ro = np.array([1000, 2000, 2000, 2000])/1000,
				    Vs = np.array([500, 1500, 1500, 1500]),
				    Vp = np.array([500, 3000, 3000, 3000])
                        ):

        self.filename = filename
        self.ro = ro
        self.Vs = Vs
        self.Vp = Vp
        self.h = h

    def profile_4_plot(self,):
        Vs = self.Vs
        h = self.h
        # Build profiles for plotting
        DEPTH = np.array([])
        VS = np.array([])
    
        VSd = np.zeros((6))
        DEPTHd = np.zeros((6))
    
        VS=np.array([])
        DEPTH = np.array([])
    
        for i in range(len(Vs)-1):
            if (i==0):
                cdepth = 0.0
                DEPTH = np.append(DEPTH,cdepth)
                ndepth = cdepth+h[i]
                DEPTH = np.append(DEPTH,ndepth)
            if (i>0):
                cdepth = DEPTH[-1] 
                ndepth = cdepth+h[i]
                DEPTH = np.append(DEPTH,cdepth)
                DEPTH = np.append(DEPTH,ndepth)

 
            VS = np.append(VS,Vs[i])
        VSd[:] =  np.repeat(VS,2) 
        DEPTHd[:] =  DEPTH
        return VSd, DEPTHd

class HVSR_Processing(object):
    def __init__(self,
                 filename = None,
                 header_lines = 34,
                 E_col = 1,
                 N_col = 0,
                 Z_col = 2,
                 time1 = 0.0,
                 time2 = 1.0,
                 freq = 1024,
                 Vs=750,
                 freq_win = np.array([ [0,200], [300,500] ]),
                 win_sec = 6,
                 E = np.zeros(10),
                 N = np.zeros(10),
                 Z = np.zeros(10),
                 time = np.zeros(10),
                 EW_filt = np.array([]),
                 NS_filt = np.array([]),
                 Z_filt = np.array([]),
                 time_filt = np.array([]),
                 PZ = np.array([]),
                 PE = np.array([]),
                 PN = np.array([]),
                 fZ = np.array([]),
                 fE = np.array([]),
                 fN = np.array([]),
                 HVSR = np.array([]),
                 d = np.array([]),
                 HVpower = np.array([]),
                 normalised = 1
                    ):
        self.filename = filename
        self.header_lines = header_lines
        self.E_col = E_col
        self.N_col = N_col
        self.Z_col = Z_col
        self.time1 = time1
        self.time2 = time2
        self.freq = freq
        self.Vs=Vs
        self.freq_win = freq_win
        self.win_sec = win_sec
        self.E = E
        self.N = N
        self.Z = Z
        self.EW_filt = EW_filt
        self.NS_filt = NS_filt
        self.Z_filt = Z_filt
        self.time_filt = time_filt
        self.PN = PN
        self.PE=PE
        self.PZ=PZ
        self.HVSR = HVSR
        self.fN = fN
        self.fE = fE
        self.fZ = fZ
        self.d = d
        self.HVpower = HVpower
        self.normalised = normalised



    def import_HVSR(self,):
        """ Reads a typicaly Tromino data file, or equivalent
        Note the header, as well as position of frequency and times vary in this file format
        We need to minimally human read the header for this info an give it to the routine, 
        as well as the column positions of the EW, NS and Z motions
        """
        file= self.filename
        header = self.header_lines

        EW=np.genfromtxt(file,skip_header=header,usecols=self.E_col,encoding= 'unicode_escape')
        NS=np.genfromtxt(file,skip_header=header,usecols=self.N_col,encoding= 'unicode_escape')
        Z=np.genfromtxt(file,skip_header=header,usecols=self.Z_col,encoding= 'unicode_escape')
   
        time1=np.fromstring(str(self.time1),sep=':')
        time2=np.fromstring(str(self.time2),sep=':')
        seconds = 3600*(time2[0] - time1[0]) + 60*(time2[1]-time1[1]) + (time2[2] - time1[2])
        samples=seconds*self.freq
        time = np.linspace(0.0,seconds,int(samples))
        self.E = EW
        self.N = NS
        self.Z = Z
        self.time = time
        return

    def plot_raw(self,):
        plt.plot(self.time,self.Z,label="Z",color="xkcd:forest green")
        plt.xticks(np.arange(np.min(self.time),np.max(self.time),100))
        plt.legend(framealpha=0.5)
        plt.grid()
        fig_file = str(self.filename)+"_raw.png"
        plt.xlabel("Time (ms)")
        plt.ylabel("Displacement (mm)")
        plt.savefig(fig_file,dpi=300)
        #plt.show()


    def filter_HVSR_time(self,):
        """ Filters HVSR windows according to user supplied tuple matrix of 
        optimal values
        """
        EW_filt=np.array([])
        NS_filt = np.array([])
        Z_filt=np.array([])
        time_filt=np.array([])
        for i in range(len(self.freq_win[:,0])):
            start_window = self.freq_win[i,0]
            end_window = self.freq_win[i,1]
            filt = (self.time>start_window)&(self.time<end_window)
            EW_filt = np.append(EW_filt,self.E[filt])
            NS_filt = np.append(NS_filt,self.N[filt])
            Z_filt = np.append(Z_filt,self.Z[filt])
            time_filt = np.append(time_filt,self.time[filt])
    
        self.EW_filt = EW_filt
        self.NS_filt = NS_filt
        self.Z_filt = Z_filt
        self.time_filt = time_filt
        return
 
    def plot_filtered(self,):
        plt.plot(self.time_filt,self.Z_filt,color="xkcd:blood red",label="Z - Filtered data")
        plt.xticks(np.arange(np.min(self.time),np.max(self.time),100))
        plt.legend(framealpha=0.5)
        plt.grid()
        fig_file = str(self.filename)+"_filt.png"
        plt.xlabel("Time (ms)")
        plt.ylabel("Displacement (mm)")
        plt.savefig(fig_file,dpi=300)
        #plt.show()


    def HVSR_periodogram(self,):
        """ Performs a Welch filter on the spectrogram
        Needs a window size, the rest is hard coded.
        Returns the HVSR as well. 
        """
        win_sec=self.win_sec
        nps=self.freq*win_sec
        nol=nps-10
        window="triang"
        fN, PN = welch(self.NS_filt[:],fs=self.freq,nperseg=nps,scaling="spectrum",average='median',noverlap=nol,window=window)
        fE, PE = welch(self.EW_filt[:],fs=self.freq,nperseg=nps,scaling="spectrum",average='median',noverlap=nol,window=window)
        fZ, PZ = welch(self.Z_filt[:],fs=self.freq,nperseg=nps,scaling="spectrum",average='median',noverlap=nol,window=window) 

        PN=np.sqrt(PN)
        PE=np.sqrt(PE)
        PZ=np.sqrt(PZ)
        H = (PN+PE)/2
        HVSR = H/PZ
        self.PN = PN
        self.PE=PE
        self.PZ=PZ
        self.HVSR = HVSR
        self.fN = fN
        self.fE = fE
        self.fZ = fZ
        return

    def plot_HVSR(self,):
        # Now we plot the EW, NS and Z data triaxial geophone data.
        plt.figure(figsize=(10,8),dpi=600)
        #plt.figure(dpi=600)

        ax=plt.subplot(2,1,1)
        plt.loglog(self.fN[:],self.PN[:],color="xkcd:royal blue",alpha=0.8,label="NS")
        plt.loglog(self.fE[:],self.PE[:],color="xkcd:kelly green",alpha=0.8,label="EW")
        plt.loglog(self.fZ[:],self.PZ[:],color="xkcd:raw umber",alpha=0.8,label="Z")
        plt.xlim(0.1,250)
        plt.ylabel("Spectral power")
        #ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
        plt.grid()
        ax.xaxis.grid(True, which='minor')
        plt.legend(framealpha=0.5)

        ax2=plt.subplot(2,1,2)
        plt.semilogx(self.fZ,self.HVSR,color="xkcd:blood red",linewidth=2,alpha=0.8,label="H/V")
        plt.xlim(0.1,250)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("HVSR Intensity")
        plt.grid()
        ax2.xaxis.grid(True, which='minor')
        ts=pd.Series(self.HVSR,index=self.fZ)
        var=ts.rolling(window=6).std()
        def_var =var.loc[var.first_valid_index()]
        var = var.fillna(def_var)

        plt.plot(self.fZ,self.HVSR+var,color='black', linestyle="--",alpha=0.5,linewidth=0.7)
        plt.plot(self.fZ,self.HVSR-var,color='black', linestyle="--",alpha=0.5,linewidth=0.7)
        #print(np.c_[fZ,var])
        fig2=str(self.filename)+"_HVSR.png"
        plt.savefig(fig2)
        #plt.show()
        return

    def virtual_Borehole(self,):
        #################################################################################
        # Convert to depths and then borehole
        Vs=self.Vs
        # Break up into freq chunks
        d = Vs/(4*np.concatenate(([1e-6],self.fZ[1:])))
        #print("d",d,fZ)
        hv_norm = np.log(self.HVSR) / (np.linalg.norm(np.log(self.HVSR),np.inf) + 1e-16)
        plt.figure(figsize=(10,8))
        plt.plot(hv_norm,d)
        plt.ylim(120,0)

        x = np.linspace(0,3,50)
        y = np.linspace(0,120,240)
        X,Y = np.meshgrid(x,y)
        HV2 = np.zeros_like(X)
        fun = interp1d(d,hv_norm,kind="linear",fill_value="extrapolate")  
        HV1a=fun(Y[:,0])
        flt = (HV1a) > np.max(np.abs(hv_norm))
        HV1a[flt] = np.max(hv_norm)
        flt = (HV1a) < np.min((hv_norm))
        HV1a[flt] = np.min(hv_norm)
        #plt.plot(HV1a,Y[:,0],color="xkcd:blood red")
        datafile = str(self.filename)+"_depth_"+str(chainage)+"m.dat"
        np.savetxt(datafile,np.c_[HV1a,Y[:,0]])
        self.d = Y[:,0]
        self.HVpower = HV1a

        print("Creating borehole")
        for i in range(len(x)):
            HV2[:,i] = HV1a + np.random.normal(0,HV1a.std(),np.size(Y[:,i]))*0.25
        plt.figure(figsize=(3,12),dpi=600)
        plt.rcParams.update({'font.size': 8})
        plt.imshow(HV2,cmap=plt.cm.plasma,extent=[0,10,120,0])
        plt.xticks([])
        fig3=str(self.filename)+"_hvsr_borelog.png"
        plt.savefig(fig3,dpi=600,bbox_inches="tight")

        return

    def sortKeyFunc(s):
        return int(s.split("_")[2].split("m")[0])

    def plot_HVSR_section(self,):
        """ Requires formatting to the output files of HVSR vs depth from the 
        routine "virtual_Borehole". These were named (self.filename)+"_depth_"+str(chainage)+"m.dat".
        This routine looks for a list of them, eg. blah_depth_0m.dat, blah_depth_10m.dat
        and attempts to build a normalised profile.
        """
        DEPTH = np.array([])
        DIST = np.array([])
        HVS = np.array([])

        normalised = self.normalised

        di=np.linspace(0,60,60)
        files = sorted(glob.glob(str(self.filename)+"_depth_"+"*m.dat"),key=lambda s: int(s.split("_")[2].split("m")[0]))
        n=len(files)
        for file in files:
            dist=file.split("_")[2].split("m")[0]
            print(file,"\t",dist)
            HV=np.genfromtxt(file,usecols=0)
            d=np.genfromtxt(file,usecols=1)
            if np.max(d) > 60:
                f = interp1d(d,HV,fill_value='extrapolate',kind='cubic')
                HVi = f(di)
                if (normalised < 1):
                    filt2=di>0
                    hv=np.log10(np.abs(HVi[filt2]))
                    di = di[filt2]
                    HVi = hv / (np.linalg.norm(hv) + 1e-16) 
                distance = np.float(dist)*(np.ones_like(di))
                DEPTH = np.append(DEPTH,di)
                DIST = np.append(DIST,distance)
                HVS = np.append(HVS,HVi)
                #plt.subplot(int(n/7),7,i)
                #plt.plot(hv_norm, di2)
                #plt.ylim(60,0)
                #plt.annotate(str(dist)+" m",xy=(0,0),xytext=(0.5,0.2),xycoords='axes fraction',textcoords='axes fraction')
                #i +=1
            else:
                f = interp1d(d,HV,fill_value='extrapolate',kind='nearest')
                HVi = f(di)
    
        max_dist = np.max(DIST)+5
        min_dist = np.min(DIST)-5
        max_depth = np.max(DEPTH)+5

        np.savetxt(str(self.filename)+"_alldata.dat", np.c_[DIST,DEPTH,HVS])
        arr1 = np.vstack((DIST, DEPTH, HVS)).T
        df = pd.DataFrame(arr1, columns=['X','Z','HV'])
        df.to_csv("HVSR.csv",index=False)
        x= np.arange(min_dist,max_dist,1)
        y = np.arange(0,max_depth,1)
        X,Y = np.meshgrid(x,y)

        filt1 = ~np.isnan(HVS)
        #Z = griddata( (DIST[filt1], DEPTH[filt1]), HVS[filt1], (X, Y), method='cubic')
        rbf = Rbf(DIST[filt1],DEPTH[filt1],HVS[filt1],function="multiquadric", smooth=3.9) #,epsilon=0.5)
        Z = rbf(X,Y)
        #plt.rcParams["figure.figsize"] = (8, 6)
        #plt.rcParams["figure.dpi"] = 300
        arr2 = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
        df2 = pd.DataFrame(arr2, columns=['X','Z','HV'])
        df2.to_csv("HVSR_gridded2.csv",index=False)
        #plt.subplot(211)
        plt.figure(figsize=(8,3),dpi=300)
        plt.imshow(np.log(Z),extent=[min_dist,max_dist,max_depth,0],cmap="plasma")
        plt.xticks(np.arange(min_dist, max_dist, 10.0),fontsize=6)
        plt.yticks(fontsize=6)
        plt.colorbar()
        plt.xlabel("Chainage (m)")
        plt.ylabel("Depth (m)")
        plt.tight_layout()
        plt.savefig("HVSR_gridded2.png")
        plt.close()
        #plt.clim(0,0.4)

