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
from scipy.optimize import optimize
import glob
import copy

class HVSRForwardModels(object):
    def __init__(self,
				 fre1 = 1,
				 fre2 = 60,
				 Ds = np.array( [0.05, 0.01, 0.01,0.01]),
                 Dp = np.array( [0.05, 0.01, 0.01,0.01]),
				 h = np.array([10, 200,100, 1e3]),
				 ro = np.array([1000, 2000, 2000, 2000])/1000,
				 Vs = np.array([500, 1500, 1500, 1500]),
				 Vp = np.array([500, 3000, 3000, 3000]),
                 ex = 0.0,
                 fref = 1.0,
                 f = np.logspace(np.log10(1),np.log10(60),500),
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
            self.fref=fref
            self.f = f
            

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

        #freq=np.linspace(self.fre1,self.fre2,500)
        #print("Initial freq!!",freq)
        hl = self.h
        vs = self.Vs
        dn = self.ro
        Ds = self.Ds
        freq=self.f
        qs = np.ones_like(vs)*(1.0/(2.0*Ds))
        inc_ang=0. 
        depth=0.
        #print("Transfer function: Vs:",vs)

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
        
        s_amp=self.HV3(self.Vs, self.ro, self.h, self.Ds, self.ex, self.fref, self.f)
        p_amp=self.HV3(self.Vp, self.ro, self.h, self.Dp, self.ex, self.fref, self.f)
        hvsr = s_amp/p_amp
        return(self.f,hvsr)
    
    def HV3(self,c, ro, h, d, ex, fref, f):
        # Code adopted From Albarello el al. Suppl. Mat/BW, following from Model HVSR (Herak) and OpenHVSR.
        # Migrated to pure python and benchmarked by Craig O'Neill Nov 2023.
        q = 1/(2*d)
        ns = len(c)
        nf = len(f)
        TR=np.zeros((50,1),dtype=complex);
        AR=np.zeros((50,1),dtype=complex);
        qf=np.zeros((ns,nf),dtype=complex);
        T=np.zeros((ns,1),dtype=complex);
        A=np.zeros((ns,1),dtype=complex);
        FI=np.zeros((ns,1),dtype=complex);
        Z=np.zeros((ns,1),dtype=complex);
        X=np.zeros((ns,1),dtype=complex);
        FAC=np.zeros((ns,nf),dtype=complex); 
        frref = fref
        frkv = f
        qf = np.zeros((ns, nf))
        for j in range(ns):
            for i in range(nf):
                qf[j, i] = q[j] * frkv[i] ** ex

        idisp = 0
        if frref > 0:
            idisp = 1

        TR = np.zeros(ns - 1)
        AR = np.zeros(ns - 1)

        for I in range(ns - 1):
            TR[I] = h[I] / c[I]
            AR[I] = ro[I] * c[I] / ro[I + 1] / c[I + 1]

        NSL = ns - 1
        TOTT = sum(TR)

        #X = np.zeros(NSL + 1, dtype=np.complex128)
        #Z = np.zeros(NSL + 1, dtype=np.complex128)
        X[0] = 1.0+0.0j
        Z[0] = 1.0 + 0.0j
        II = 1j

        korak = 1
        if idisp == 0:
            FJM1 = 1
            FJ = 1

        FAC = np.zeros((NSL + 2, nf), dtype=np.complex128)

        for J in range(1, NSL + 1):
            for ii in range(nf):
                FAC[J - 1, ii] = 2 / (1 + np.sqrt(1 + qf[J - 1, ii] ** (-2))) * (1 - 1j / qf[J - 1, ii])
                FAC[J - 1, ii] = np.sqrt(FAC[J - 1, ii])

        FAC[NSL, :nf] = 1
        qf[NSL, :nf] = 999999

        jpi = 1 / 3.14159

        AMP = np.zeros(nf)

        for k in range(0, nf, korak):
            ALGF = np.log(frkv[k] / frref)

            for J in range(2, NSL + 2):
                if idisp != 0:
                    FJM1 = 1 + jpi / qf[J - 2, k] * ALGF
                    FJ = 1 + jpi / qf[J - 1, k] * ALGF

                T[J - 2] = TR[J - 2] * FAC[J - 2, k] / FJM1
                A[J - 2] = AR[J - 2] * FAC[J - 1, k] / FAC[J - 2, k] * FJM1 / FJ
                FI[J - 2] = 6.283186 * frkv[k] * T[J - 2]
                ARG = 1j * FI[J - 2]

                CFI1 = np.exp(ARG)
                CFI2 = np.exp(-ARG)

                Z[J - 1] = (1 + A[J - 2]) * CFI1 * Z[J - 2] + (1 - A[J - 2]) * CFI2 * X[J - 2]
                Z[J - 1] = Z[J - 1] * 0.5

                X[J - 1] = (1 - A[J - 2]) * CFI1 * Z[J - 2] + (1 + A[J - 2]) * CFI2 * X[J - 2]
                X[J - 1] = X[J - 1] * 0.5

            AMP[k] = 1 / abs(Z[NSL])

        return AMP

        
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
    
        VSd = np.zeros((np.size(h)*2))
        DEPTHd = np.zeros(np.size(h)*2)
    
        VS=np.array([])
        DEPTH = np.array([])
    
        for i in range(len(Vs)):
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
                 normalised = 1,
                 chainage = 0
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
        self.chainage = chainage

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
        datafile = str(self.filename)+"_depth_"+str(self.chainage)+"m.dat"
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


class HVSR_inversion(object):
    def __init__(self,               
                    fre1 = 0.01,
				    fre2 = 100,
				    Ds = np.array( [0.05, 0.01, 0.01,0.01]),
                    Dp = np.array( [0.05, 0.01, 0.01,0.01]),
				    h = np.array([10, 200,100, 1e3]),
				    ro = np.array([1000, 2000, 2000, 2000])/1000,
				    Vs = np.array([500, 1500, 1500, 1500]),
				    Vp = np.array([500, 3000, 3000, 3000]),
                    ex = 0.0,
                    Poisson = 0.4,
				    filename = None,
                    hvsr=None,
                    hvsr_freq=None,
                    step_size=0.01, #Recommended 0.25 (there is a multiplier above)
                    step_floor = 0.002, #Recommended 0.15 (150m/s)
                    alpha=0.12,  #Recommended 0.5
                    beta=1.06,  #Recommended 1.1
                    Vs_ensemble = None,
                    h_ensemble = None,
                    Vs_best = None,
                    hvsr_best = None,
                    hvsr_best_f = None,
                    L1_best = 1e6,
                    hvsr_c = None,
                    hvsr_c_f = None,
                    n=1,
                    n_burn = 1,
                    L1_old = None,
                    Hfac = 100,
                    Vfac = 2000


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
            self. Poisson = Poisson
            self.filename = filename
            self.hvsr = hvsr
            self.hvsr_freq = hvsr_freq
            self.hvsr_c = hvsr_c
            self.hvsr_c_f = hvsr_c_f
            self.L1_best = L1_best
            self.step_size= step_size 
            self.step_floor = step_floor 
            self.alpha= alpha
            self.beta= beta  
            self.Vs_ensemble = Vs_ensemble
            self.h_ensemble = h_ensemble
            self.Vs_best = Vs_best
            self.hvsr_best = hvsr_best,
            self.hvsr_best_f = hvsr_best_f
            self.n = n
            self.n_burn = n_burn
            self.L1_old = L1_old
            self.Hfac = Hfac
            self.Vfac = Vfac

    def moving_average(self, a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def L1_norm(self,):
        Vs = self.Vs
        h=self.h
        #To invert for Vs and h, we use empirical relationships to get consistent Vp and ro.
        if (np.all(Vs > 0)):
            #Vp = 1000*(Vs/1000 + 1.164)/0.902
            #ro=(310*(Vp*1000)**0.25)/1000
            Vp = Vs * np.sqrt( (1-self.Poisson)/(0.5 - self.Poisson))
            ro= 1740 * (Vp/1000)**0.25

        else:
            Vp = 1000*np.ones_like(Vs)
            #ro=(310*(Vp*1000)**0.25)/1000
            ro= 1740 * (Vp/1000)**0.25
        hvsr=self.hvsr
        hvsr_freq = self.hvsr_freq
        
        mod1 = HVSRForwardModels(ro=ro,Vs=Vs,Vp=Vp,fre1=self.fre1,fre2=self.fre2, f=hvsr_freq, Ds=self.Ds,Dp=self.Dp,h=self.h)
        try:
            #mod1 = HVSRForwardModels(ro=ro,Vs=Vs,Vp=Vp,fre1=self.fre1,fre2=self.fre2, Ds=self.Ds,Dp=self.Dp,h=self.h)
            f_c, hvsr_c = mod1.HV()  #Change to try against Transfer function
            #print("L1, transfer function -1:",hvsr_c[-1])
            self.hvsr_c = hvsr_c
            self.hvsr_c_f = f_c
            fm=interp1d(f_c,hvsr_c,fill_value="extrapolate")
            hvsr_2=fm(hvsr_freq)
            # Scale to HVSR data range
            #shvsr_2 *= np.max(hvsr)/np.max(hvssr_c)
            #print(" .... L1   worked")
        except:
            hvsr_2=np.ones_like(hvsr) * 1e9
            #to=t
            #print(" .... L1 bombed")
        if (np.any(Vs > 5000)):
            hvsr_2=np.ones_like(hvsr) * 1e9
        if (np.any(Vs < 100)):
            hvsr_2=np.ones_like(hvsr) * 1e9
        #print("   h:",h)
        if (np.any(h < 0)):
            #print("hi") #Not seeing this - why?
            hvsr_2=np.ones_like(hvsr) * 1e9
        std1 = np.std(hvsr[ ( hvsr_freq > 10)&( hvsr_freq < 50)])
        filt = hvsr > (np.mean(hvsr) + 0.6*std1)
        L1_peak = np.sum(np.abs(hvsr_2[filt] - hvsr[filt])**2)
        L1_all = np.sum((np.abs(hvsr_2 - hvsr))**1)
        L1_grad = np.sum(np.abs(np.gradient(hvsr_2) - np.gradient(hvsr))**2)
        L1 = 0.6*(0.5*L1_peak + 0.5*L1_all) + 0.4*L1_grad
        return L1


    def MCMC_step(self,i):
        Vfac=self.Vfac
        hfac=self.Hfac
        x_start = np.append(self.Vs/Vfac, self.h/hfac)
        dim2=np.size(self.Vs)


        Vs = self.Vs
        h = self.h

        L1 = self.L1_norm()
        L1_old = self.L1_old
        if (i==0):
            L1_old = L1
            self.L1_best = L1
        elif (L1 < L1_old):
            #Accept
            L1_old = L1
            Vs = np.copy(self.Vs)
            h = np.copy(self.h)
        elif (L1 >= L1_old):
            rdm = np.random.uniform(0.0,1.0,1)[0]
            # Want to randomly reward if only a little bigger
            # This dice term can blow out, but < should catch silly increases
            # If L1 close to L1_old, 2nd term should be small, and dice ~1
            dice =  ((1 - ( np.abs(L1 - L1_old)/np.max((L1_old,L1)))))
            if (  dice < rdm):
                #Accept anyway
                L1_old = L1
                Vs = np.copy(self.Vs)
                h = np.copy(self.h)
        self.Vs = Vs
        self.h = h
        if (L1 <= self.L1_best):
            self.L1_best = L1
            self.Vs_best = Vs
            self.hvsr_best = self.hvsr_c
            self.hvsr_best_f = self.hvsr_c_f
        print(i,L1,self.L1_best,Vs[0])

        # Now move
        x_start = np.append(self.Vs/Vfac, self.h/hfac)

        self.Vs_ensemble[i] = self.Vs
        self.h_ensemble[i]=self.h
        self.L1_old = L1_old
        Vs0a=Vs
        try:
            stdv = self.step_floor + self.alpha*x_start**self.beta
        except:
            stdv = 10.0

        # CALCULATE THE NOISE TO ADD
        #local_mean = self.moving_average(Vs0a,3)
        #means = -0.5*(Vs0a[1:-1] - local_mean)/local_mean
        #means2 = np.hstack(([0],means,means[-1])) #NOTE THESE ARE PERTURBATIONS TO LOCAL MEAN!
        noise = np.random.normal(0,stdv,np.size(x_start))*self.step_size
        #noise = noise_deadener(Vs0a,noise) #Works out local mean, makes sure noise is within +/- 3 stdv

        self.Vs = np.abs(x_start[:dim2] + noise[:dim2]) * Vfac
        self.h=np.abs(x_start[dim2:] + noise[dim2:])*hfac

        #print("...",Vs0a,noise)
        #self.Vp = (Vs0a + 1.164)/0.902   # Garnders relationships etc
        #self.ro=(310*(self.Vp*1000)**0.25)/1000
        self.Vp = Vs0a * np.sqrt( (1-self.Poisson)/(0.5 - self.Poisson))
        self.ro= 1740 * (self.Vp/1000)**0.25

    def MCMC_walk(self,):
        n = self.n
        nburn = self.n_burn
        Vs_ensemble = np.zeros((n, np.shape(self.Vs)[0]))
        h_ensemble = np.zeros((n, np.shape(self.h)[0]))

        self.Vs_ensemble = Vs_ensemble
        self.h_ensemble = h_ensemble

        for i in range(nburn):
            self.MCMC_step(i)
            #print(i)
        for i in range(n):
            #print(i)
            self.MCMC_step(i)

        return(self.h,self.Vs_ensemble,self.h_ensemble,self.Vs_best,self.L1_best,self.hvsr_best,self.hvsr_best_f)

    def nelder_mead(self,
                step=125.1, no_improve_thr=1e-4,
                no_improv_break=40, max_iter=0,
                alpha=11.5, gamma=17., rho=0.55, sigma=0.55):
        '''
            @param f (function): function to optimize, must return a scalar score
            and operate over a numpy array of the same dimensions as x_start
            @param x_start (numpy array): initial position
            @param step (float): look-around radius in initial step
            @no_improv_thr,  no_improv_break (float, int): break after no_improv_break iterations with
            an improvement lower than no_improv_thr
             @max_iter (int): always break after this number of iterations.
            Set it to 0 to loop indefinitely.
            @alpha, gamma, rho, sigma (floats): parameters of the algorithm
            (see Wikipedia page for reference)

            return: tuple (best parameter array, best score)
        '''

        # init
        Vfac=self.Vfac
        hfac=self.Hfac
        x_start = np.append(self.Vs/Vfac, self.h/hfac)

        dim2=np.size(self.Vs)
        dim = len(x_start)
        prev_best = self.L1_norm() #(x_start)
        no_improv = 0
        res = [[x_start, prev_best]]

        for i in range(dim):
            x = copy.copy(x_start)
            x[i] = x[i] + step
            self.Vs =x[:dim2]*Vfac
            self.h=x[dim2:]*hfac
            score = self.L1_norm() #f(x)
            res.append([x, score])

        # simplex iter
        iters = 0
        while 1:
            # order
            res.sort(key=lambda x: x[1])
            best = res[0][1]

            # break after max_iter
            if max_iter and iters >= max_iter:
                return res[0]
            iters += 1

            # break after no_improv_break iterations with no improvement
            print( '...best so far:', best, self.h)

            if best < prev_best - no_improve_thr:
                no_improv = 0
                prev_best = best
            else:
                no_improv += 1

            if no_improv >= no_improv_break:
                return res[0]

            # centroid
            x0 = [0.] * dim
            for tup in res[:-1]:
                for i, c in enumerate(tup[0]):
                    x0[i] += c / (len(res)-1)

            # reflection
            xr = x0 + alpha*(x0 - res[-1][0])
            #self.Vs, self. h =xr
            self.Vs =xr[:dim2]*Vfac
            self.h=xr[dim2:]*hfac

            rscore = self.L1_norm() #f(xr)
            if res[0][1] <= rscore < res[-2][1]:
                del res[-1]
                res.append([xr, rscore])
                continue

            # expansion
            if rscore < res[0][1]:
                xe = x0 + gamma*(x0 - res[-1][0])
                #self.Vs, self.h =xe
                self.Vs =xe[:dim2]*Vfac
                self.h=xe[dim2:]*hfac

                escore = self.L1_norm() #f(xe)
                if escore < rscore:
                    del res[-1]
                    res.append([xe, escore])
                    continue
                else:
                    del res[-1]
                    res.append([xr, rscore])
                    continue

            # contraction
            xc = x0 + rho*(x0 - res[-1][0])
            #self.Vs, self.h =xc
            self.Vs =xc[:dim2]*Vfac
            self.h=xc[dim2:]*hfac

            cscore = self.L1_norm() #f(xc)
            if cscore < res[-1][1]:
                del res[-1]
                res.append([xc, cscore])
                continue

            # reduction
            x1 = res[0][0]
            nres = []
            for tup in res:
                redx = x1 + sigma*(tup[0] - x1)
                #self.Vs,self.h =redx
                self.Vs =redx[:dim2]*Vfac
                self.h=redx[dim2:]*hfac

                score = self.L1_norm() #f(redx)
                nres.append([redx, score])
            res = nres



    def Amoeba_crawl(self,):
        Vfac=self.Vfac
        hfac=self.Hfac
        dim2 = np.size(self.Vs)
        result = self.nelder_mead()
        Vs = result[0][:dim2]*Vfac
        h = result[0][dim2:]*hfac
        print("Amoeba results:",np.c_[Vs,h])
        return(Vs,h)

        
    

