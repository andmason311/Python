'''
NENG 612 Lab 1 
Data Analysis
Andy Mason, Evan Threlkeld, Anthony Hagey
3/3/2022
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lmfit.models import GaussianModel
from scipy import stats
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.stats import norm
import seaborn as sns

###############################################################################
#%% Part 1: Read Data 
    
# Assign folder and file: 
path = r"C:\Users\Admin\OneDrive\Documents\School\AFIT_Classes\Quarter_4\NENG_612\Lab_1"
file1 = r"DataF_CH0@DT5790N_1811_threlkeld_hagey_mason_lab1_cs_calibration_300S_sorted.xlsx"
file2 = r"DataF_CH1@DT5790N_1811_threlkeld_hagey_mason_lab1_cs_calibration_300S_sorted.xlsx"
file3 = r"Data_ThrelkeldHageyMason_AmBe Source_PSD1_Measurement_sorted.xlsx"
file4 = r"DataF_ThrelkeldHageyMason_AmBe Source_PSD2_Measurement_CoincidenceWaves1_sorted.xlsx"
cal_1 = pd.read_excel(file1)#read energy column
cal_2 = pd.read_excel(file2)#read energy column
cal_3 = pd.read_excel(file3)#read energy column
cal_4 = pd.read_excel(file4)#read energy column
cal_1_Energy=cal_1.loc[:,"ENERGY"]
cal_2_Energy=cal_2.loc[:,"ENERGY"]
cal_3_Energy=cal_3.loc[:,"ENERGY"]
cal_4_Energy=cal_4.loc[:,"ENERGY"]
#%% Part 2: Plot Histogram of Bined Energies For Calibration
cal_1_Energy=cal_4_Energy
def gaussian(x, mu, sig, A):#Calable Function for Plotting Gaussian
    y=1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)/A
    return y
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
def CompEdge(gaussian, n, bins):
     CompEdge=find_nearest(n,np.max(gaussian(x_gaus, mu, sig, A))/2)
     return CompEdge    
    
#Histogram of Raw Data and Graph Details
(n, bins, patches)=plt.hist(cal_1_Energy, 1000, rwidth=1, color='#607c8e', density=True, stacked=True, label='Raw Data')
plt.legend()  
ax1=cal_1_Energy.plot.kde(bw_method=.05, label='Polynomial Fit') # Fiting Polynomial to Raw Histogram Data
plt.title('Cs-137 Source Spectrum: Detector 2'), plt.xlabel('Light Units (Arbitrary)'), plt.ylabel('Integral Density Normalized (Arbitrary)'), plt.grid(axis='y', alpha=0.1)
ax1.legend()
#Viewing Field of Plot
plt.xlim(1000,5000) #Detector 1: Cs=-100-1500 AmBe=, Detector 2: Cs=-100-1500 AmBe=
plt.ylim(0,.0002) #Detector 1: Cs=0-.005 AmBe=, Detector 2: Cs=0-.005 AmBe=
#Gaussian Fit Parameters
x_gaus_range=[200,600]
x_gaus = np.arange(x_gaus_range[0], x_gaus_range[1], 0.001)
mu=350 #Detector 1: Cs=375 AmBe=, Detector 2: Cs=350 AmBe=
sig=30 #Detector 1: Cs=45 AmBe=, Detector 2: Cs=30 AmBe=
A=9.5  #Detector 1: Cs=3.55 AmBe=, Detector 2: Cs=9.5 AmBe=
plt.plot(x_gaus, gaussian(x_gaus, mu, sig, A), label='Gaussian Fit') #Plot Gaussian Curve
plt.legend()
plt.vlines(bins[CompEdge(gaussian, n, bins)],.003,0,colors='black',linestyles='solid',label='Compton Edge')
plt.legend()
print("Bin", CompEdge(gaussian, n, bins),"is 662 KeV") #Cs-137 Compton Edge Value=662 keV

#%% Categorizing PSD Events
#x_vals0, y_vals0
#x_vals1, y_vals1
Finding Gamma Events
    Set Threshold in x and y 
    Find Points below threshold
    Store Indices of Points in array
    Reference timestamp with same index values
    Store timestamps as Gamma Events in Time
    
def Event_Categorization(x_vals0,y_vals0,x_vals1,y_vals1)
    x = np.linspace(0, 2000, 1000)
    threshold0=np.piecewise(x, [x < 300, 300 <= x < 1000, 1000 < x], [0, .15,x*.5-1000])
    threshold00=np.piecewise(x, [x < 300, 300 <= x < 1000, 1000 < x], [1, .15,x*.5-1000])
    threshold1=np.piecewise(x, [x < 300, 300 <= x < 1000, 1000 < x], [1, .15,x*.5-1000])
    for i in range(len(y_vals0))::
        if y_vals0[i] < threshold0:
            y_Gamma0=np.append(y_Gamma0, y_vals0[i])
            x_Gamma0=np.append(x_Gamma0, x_vals0[i])
        elif y_vals0[i] > threshold00:
            y_Neu0=np.append(y_Neu0, y_vals0[i])
            x_Neu0=np.append(x_Neu0, x_vals0[i])
    for i in range(len(y_vals1))::
        if y_vals0[i] > threshold1:
            y_Neu0=np.append(y_Neu0, y_vals0[i])
            x_Neu0=np.append(x_Neu0, x_vals0[i])
    
#
#%%Time of Flight Calculations
'''
#Statistical Error of Detectors Function [ns]
def t_offset(timestamp1_Gamma, timestamp2_Gamma):
    c=3e8 #Speed of light [m/s] 
    d1=1.3716 #Distance between detectors [m] =54 [in]
    Dt_GammaGamma=timestamp2_Gamma-timestamp1_Gamma #[ns] Time of Flight between Coincident Gamma Events Detector 1 and Gamma Events Detector 2
    t_offset_spectrum=TOF_GammaGamma-d1/c #Data for TOF Gamma Gamma events
    (GammaGamma_n, GammaGamma_bins, GammaGamma_patches)=plt.hist(t_offset_spectrum, 1000, rwidth=1, color='#607c8e', density=False, stacked=False)  
    (n_GG, bins_GG, patches_GG)=plt.title('Neutron Energy Spectrum'), plt.xlabel('Delta(time) [ns]'), plt.ylabel('Counts'), plt.grid(axis='y', alpha=0.1)
    t_offset_spectrum.plot.kde(bw_method=.05) #Fiting Polynomial to Raw Histogram Data
    t_offset_index=find_nearest(n_GG,np.max(y)) #Finding index of Maxium Polynomial value
    t_offset=toffset_spectrum[t_offset_index] #Get statistical time difference value
    return t_offset

Energy of Neutron Function
#Kinetic Energy of Nuetron Function [MeV] 
def EnergyofNeutron(timestamp1_Gamma, timestamp2_Neu): 
    m_n=1.66e(−27) #Mass of Neutron [kg]
    d1=1.3716 #Distance between detectors [m] =54 [in]
    TOF_GammaNuetron=timestamp2_Neu-timestamp1_Gamma #[ns] Time of Flight between Coincident Gamma Events Detector 1 and Neutron Events Detector 2
    V_n=d1/(TOF_GammaNeutron-t_offset(timestamp1_Gamma,timestamp2_Gamma))*1e9 #[m/s] Velocity of Neutron
    E_n=(1/2*m_n*(V_n)^2)/(1e6*1.6e(−19)) #Kinetic Energy of Neutron [MeV]
    return E_n 
   
#Histogram of Nuetron Energies
(Neutron_n, Neutron_bins, Neutron_patches)=plt.hist(EnergyofNeutron(timestamp1, timestamp2), 1000, rwidth=1, color='#607c8e', density=False, stacked=False)  
plt.title('Neutron Energy Spectrum'), plt.xlabel('E_n [MeV]'), plt.ylabel('Counts'), plt.grid(axis='y', alpha=0.1)
plt.xlim(-1,11)
plt.ylim(0,1000)
        
'''




