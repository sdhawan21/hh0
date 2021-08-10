"""
#####------- Linear Model for cepheid H0 ------ ##### ------ ##### 

##N.B.: soon becoming the default script for me!
differences from Edvard's code
- Fixing R as of now
- Only focussing on the maser calibration
- Can easily turn the calibrators on or off
"""

#Current problems: not mapping the equations to the correct A matrix

import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import leastsq 
import pandas as pd

from astropy.io import ascii 
#get the data and organise the hosts + setup assumptions
tab4 = ascii.read('Riess16_Table4.txt', format='cds')

#-----------------------------------------------------------------------
# Set parameters

#From Edvard's code:
# S.D. I dont understand what these numbers are. Read!
#-----------------------------------------------------------------------
intcolaLMC0 = 0.675 #LMC Cepheid intrinsic colour V_0-I-0= intcolaLMC + intcolbLMC*(logP-1)
intcolbLMC0 = 0.201 #for logP-1<-0.1
intcolaLMC1 = 0.676 #LMC Cepheid intrinsic colour V_0-I-0= intcolaLMC + intcolbLMC*(logP-1)
intcolbLMC1 = 0.345 #for logP-1>-0.1
intcolaMW = 0.753 #MW Cepheid intrinsic colour V_0-I-0= intcolaMW + intcolbMW*(logP-1)
intcolbMW = 0.256
#intcola = 0.75 #Extragalactic Cepheid intrinsic colour V_0-I-0= intcola + intcolb*(logP-1)
#intcolb = 0.1
#intcolaMW = 0.708 #From Pejcha et al
#intcolbMW = 0.266

intcola = intcolaMW #Extragalactic Cepheid intrinsic colour V_0-I-0= intcola + intcolb*(logP-1)
intcolb = intcolbMW

#plerror = 0.0267 #Scatter in P-L relation. Should be set to give chi2/dof=1 iteratively.
#plerror =  0.057 #Scatter in P-L relation. Should be set to give chi2/dof=1 iteratively.
plerror = 0.065#27


mu4258 = 29.397  	#Reid+2019
sigmu4258 = 0.032
muM31 = 24.36		#where is this distance even from?
sigmuM31 = 0.08
muLMC = 18.477		#Pietrynski+2019
sigmuLMC = 0.0263

sigab = 0.00176

####- Since this code is focussed on reproducing Riess et al. 2016, fix the value of R here (and change Npar accordingly) ---- ###
#####- R=0.386
R = 0.386#0.386 #details in R16 + Mortsell+21


#-----------------------------------------------------------------------
#https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/instrument-science-reports-isrs/_documents/2019/WFC3-2019-01.pdf

### These are corrections to the filter functions? 
#-----------------------------------------------------------------------
crnl = 0.0075 #mag/dex to correct CRNL for 5 to 6.5 dex between MW and extragalactic Cepheids
sigcrnl = 0.0006
crnl2 = 0.03
F160W0 = 21. #zero-point for non-linear correction

zp0 = -18.2e-3 #Value for GAIA off-set to use for nin-linar corrections. Can be fitted iteratively.

FeHLMC = -0.3 #Assumed [Fe/H] in LMC
koh = 0.#[O/H]=(1+k)[Fe/H]. Should be -1<koh<0  #is the default, k = 0? #setting the default to k = 0
ZMW = 8.824 #Z = 12+log(O/H) in MW

rescorrLMC = 0.0074 #Correction between resolved and not resolved Cepheids

##------------##------------##------------##------------##------------
#define what to skip ##------------##------------##------------##-----
#case 1 is always only maser distance ##------------##------------##--
##------------##------------##------------##------------##------------

skipMW = True
skipN4258dis = True
skipN4258 = True
skipM31  = True
skipM31dis = True
skipLMC = False
skipLMCdis = False

#load the data and define the galaxy array
mw = pd.read_csv('../BHM_FMD/riess21_phot_parallax.ascii.txt', delim_whitespace=True)
lmc = ascii.read('../BHM_FMD/riess19_phot_LMC.txt', format='cds')

galaxyarr = ['M101','N1015','N1309','N1365','N1448','N2442','N3021','N3370','N3447',
	'N3972','N3982','N4038','N4424','N4536','N4639','N5584','N5917','N7250','U9391','N4258','M31']
ng = len(galaxyarr)

mb = [13.310,17.015,16.756,15.482,15.765,15.840,16.527,16.476,16.265,16.048,15.795,15.797,15.110,
	15.177,15.983,16.265,16.572,15.867,17.034]
sigmb = [0.117,0.123,0.116,0.125,0.116,0.142,0.117,0.115,0.124,0.116,0.115,0.114,0.109,
	0.124,0.115,0.115,0.115,0.115,0.114]

#write these as a single array instead of writing them as two individual points along with the 19 calibrators
rhprior = np.zeros(ng + 2) + 0.386
rhpriorvec = rhprior
rhpriorsig = np.zeros(ng+2) + 1.e-10
rhpriorsigvec = rhpriorsig#[1.e-10, rhpriorsig, 1.e-10]

tab_host = tab4[(tab4['Field'] != 'M31') & (tab4['Field'] != 'N4258')]

#define the vector lengths
nMW = len(mw['logP'])
nexgal = len(tab4['Per'])
nLMC = len(lmc['logP'])
nCalib = nMW + nLMC + len(tab4['Per'][tab4['Field'] == galaxyarr[-2]]) + len(tab4['Per'][tab4['Field'] == galaxyarr[-1]])
nsn = len(mb)
#these next three are to "chunk up" the 
nM31 = len(tab4['Per'][tab4['Field'] == 'M31'])
nN4258 = len(tab4['Per'][tab4['Field'] == 'N4258'])
nHost = len(tab_host)
ntot = nMW+nexgal+nLMC
print("Total number of Cepheids incl. hosts MW, LMC, M31, N4258 ", ntot)

#define the periods
#the split to long / short period is explicit here
Per = np.concatenate([np.array(mw['logP']) - 1.,np.log10(np.array(tab4['Per'])) -1.,np.array(lmc['logP']) - 1.])
Per1 = np.concatenate([np.array(mw['logP']) - 1.,np.log10(np.array(tab4['Per'])) -1.,np.array(lmc['logP']) - 1.])
Per2 = np.concatenate([np.array(mw['logP']) - 1.,np.log10(np.array(tab4['Per'])) -1.,np.array(lmc['logP']) - 1.])
Per1[Per1 >= 0] = 0.
Per2[Per2 < 0] = 0.
print(Per1, Per2)

#Colour corrections
intcolMW = intcolaMW + intcolbMW*(mw['logP']-1.)
intcol = intcola + intcolb*(np.log10(tab4['Per'])-1.)
intcolLMC = np.zeros(nLMC)
intcolLMC[lmc['logP'] < -1.1] = intcolaLMC0 + intcolbLMC0*(lmc['logP'][lmc['logP'] < -1.1]-1.)
intcolLMC[lmc['logP'] >= -1.1] = intcolaLMC1 + intcolbLMC1*(lmc['logP'][lmc['logP'] >= -1.1]-1.)
print("Int Col LMC ", len(intcolLMC))

VIMW = mw['F555W'] - mw['F814W'] 
VI = tab4['F555W-F814W']
VILMC = lmc['F555Wmag']-lmc['F814Wmag']

OH = np.concatenate([np.array(mw['[Fe/H]'] * (1 + koh)), np.zeros(nLMC) + FeHLMC * (1+koh), np.array(tab4['[O/H]']) - ZMW])

#instead of Edvard's code where the data are as mag, implement the wesenheit correction here
HMW = mw['F160W'] - R * VIMW + crnl/2.5*(F160W0-mw['F160W']) 
H = tab4['F160W'] - R * VI + crnl/2.5*(F160W0 - tab4['F160W']) 
HLMC =   lmc['mWH']#lmc['F160Wmag'] - R * VILMC - lmc['Geo']  + crnl/2.5*(F160W0-lmc['F160Wmag']) - rescorrLMC#  #lmc['F160Wmag'] - R * VILMC #lmc['mWH']

#these are the uncertainties to fix to a high value for removing the anchors
sigHMW2 = (mw['sigma160W']*(1.-crnl/2.5)) ** 2 + (sigcrnl*(F160W0-mw['F160W'])) ** 2
sigH2 = tab4['{sigma}tot'] ** 2 
sigHLMC2 = lmc['e_mWH']**2. #(lmc['e_mWH']*(1.-crnl/2.5)) ** 2. + (sigcrnl*(F160W0-lmc['mWH'])) ** 2.
	
if skipMW: 
    sigHMW2  += 1.e12
if skipN4258dis:
    sigmu4258 += 1.e12
#for cases when the maser galaxy is to be skipped altogether
if skipN4258:				
    sigH2[tab4['Field'] == 'N4258'] = 1.e12
if skipM31:
    sigH2[tab4['Field'] == 'M31'] = 1.e12
if skipM31dis:
    sigmuM31 = 1.e12
if skipLMC:
    sigHLMC2 += 1.e12
if skipLMCdis:
    sigmuLMC = 1.e12


##### --- Define the vectors --------- #### ------------ ### 
mwpar = HMW-10.+5.*np.log10(mw['pi_EDR3'])-5./np.log(10.)*((zp0/mw['pi_EDR3']) ** 2/2.-(zp0/mw['pi_EDR3']) ** 3/3.+(zp0/mw['pi_EDR3']) ** 4/4.)
Y = np.concatenate([np.array(mwpar), np.array(H),  np.array(HLMC), np.array([mu4258]), np.array([muM31]), np.array([muLMC]), np.array(mb)])
#print(Y, mwpar, H)
#	H,HLMC,mu4258,muM31,muLMC,mb,rhpriorvec], axis=1)
sigpiEDR3 = mw['sigma_piEDR3']
insigmw = np.array(1./(sigHMW2+(5./np.log(10.)*sigpiEDR3/mw['pi_EDR3']) ** 2+ plerror ** 2))
insigH = np.array(1./(sigH2))
insigLMC  = np.array(1./(sigHLMC2+plerror ** 2))
insigN4258 = np.array([1./sigmu4258 ** 2])
insigM31 = np.array([1./sigmuM31 ** 2.])
insigLMCdis = np.array([1./sigmuLMC ** 2.])
insigmb = 1./(np.array(sigmb) ** 2.)
insigrh = 1./(np.array(rhpriorsigvec, dtype=object) ** 2.)
#some issues with the way the diagonal matrix is setup so parsing the inverse variance as the elements to np.diag
tt = np.concatenate([insigmw, insigH, insigLMC, insigN4258, insigM31, insigLMCdis, insigmb])
Cinv = np.diag(list(tt))
	
ndata = len(Y)
npar = ng+7#8+ng+1 #(ng includes 19 hosts + M31 + N4258)
dof = ndata - npar

#A = np.zeros([npar,ndata]) 
#the idl fltarr and the np.zeros works differently
nLeaveHost = nMW
A_alt = np.zeros([ndata, npar])

#below is for the host cepheids and the SN peak magnitude
for i, galval in enumerate(galaxyarr[:-2]):
    host_ceph = len(tab_host[tab_host['Field'] == galval])
    slice_len = len(A_alt[:,i][nLeaveHost:nLeaveHost+host_ceph])

    #this is to set all the cepheids in the host to have mu_i as the distance
    A_alt[:,i][nLeaveHost:nLeaveHost+host_ceph] = np.zeros(slice_len)+1.
    #this is to set the SN in the host to have the same distance for computing MB
    A_alt[:,i][ntot+i+3] = 1.
    nLeaveHost += host_ceph 

#this is for the MB which is unrelated to the primary calibrators and only set for the SN (not even the host cepheids)
A_alt[:,-1] = np.concatenate([np.zeros(ntot+3), np.zeros(nsn)+1.])
A_alt[:,nsn][nMW + nHost:nMW+nHost+nN4258] = 1
A_alt[:,nsn+1][nMW + nHost + nN4258:nMW+nHost+nN4258+nM31] = 1
#need the mu_LMC for the distances to the LMC cepheids so its an "nLMC" number of ones
#this is in the order of N4258, M31 and LMC
A_alt[:,nsn+2] = np.concatenate([np.zeros(nMW), np.zeros(nexgal), np.zeros(nLMC)+1., np.zeros(3+nsn)])
#this is for the cepheid absolute magnitude
A_alt[:,nsn+3] = np.concatenate([np.zeros(nMW)+1, np.zeros(nexgal)+1, np.zeros(nLMC)+1.,  np.zeros(3+nsn)])

#these three equations are to "fit" the fixed distances for the 3 primary anchors that are not the MW
A_alt[ntot][nsn] = 1
A_alt[ntot+1][nsn+1] = 1
A_alt[ntot+2][nsn+2] = 1

A_alt[:,nsn+4] = np.concatenate([Per1, np.zeros(3+nsn)])
A_alt[:,nsn+5] = np.concatenate([Per2, np.zeros(3+nsn)])
A_alt[:,nsn+6] = np.concatenate([OH, np.zeros(3+nsn)])
#to multiply the gaia parallax zero point, everything after the MW is completely independent of that
A_alt[:,nsn+7] = np.concatenate([-5./np.log(10.)/mw['pi_EDR3'], np.zeros(nexgal), np.zeros(nLMC), np.zeros(3+nsn)])


Sig = np.linalg.inv(np.dot(A_alt.T, np.dot(Cinv, A_alt)))

X = np.dot(Sig, np.dot(A_alt.T, np.dot(Cinv, Y)))
delta = Y - np.dot(A_alt, X)
chi2 = np.dot(delta, np.dot(Cinv, delta))

errvec = np.sqrt(np.diag(Sig))



mbabs = X[ng+6]
sigmbabs = errvec[ng+6]
h0 = np.exp(np.log(10.)/5.*(mbabs+25.))
sigh0 = np.log(10.)/5.*sqrt(sigmbabs ** 2+25*sigab ** 2)*h0

#rharr = X[ng+5:2*ng+7]
#sigrharr = errvec[ng+5:2*ng+7]

#print the outputs 

print('')
print('chi2 =', chi2)
print('chi2/dof =', chi2/dof)
print('')
print('AIC =', 2.*npar+chi2)
print('BIC =', npar*np.log(ndata)+chi2)
print('')
print('MHW =', X[ng+1], ' +/-', errvec[ng+1])
print, 'bW1 =', X[ng+2], ' +/-', errvec[ng+2]
print, 'bW2 =', X[ng+3], ' +/-', errvec[ng+3]
print, 'ZW =', X[ng+4], ' +/-', errvec[ng+4]
#print, 'RH =', rharr
#print, 'sigRH =', sigrharr
print, 'zp =', X[ng+5]*1.e3, ' +/-', errvec[ng+5]*1.e3
print, 'Mb =', mbabs, ' +/-', sigmbabs
print, 'H_0 = ',h0, ' +/-',sigh0

