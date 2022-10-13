# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:11:47 2022

@author: Evan
"""


import pandas as pd
import numpy as np
#from scipy.interpolate import interp1d
#from scipy.integrate import quad
from scipy.integrate import trapz
import matplotlib.pyplot as plt
#import FOM



class data_start:
    
    def __init__ (self, filename, delimiter=','):#',' or ';'
        #sample_length = 940#ns
        #self.dataframe = pd.read_csv(filename, delimiter=';')
        if delimiter == ';':
            
            timecolumns = [*range(7,254,1)]
            column_headers = ['BOARD', 'CHANNEL', 'TIMETAG', 'ENERGY', 'ENERGYSHORT', 'FLAGS', 'SAMPLES'] + timecolumns
            self.dataframe = pd.read_csv(filename, delimiter = delimiter, names=column_headers, header=0)
        else:
            self.dataframe = pd.read_csv(filename)
        #Data
        self.dataframe = self.dataframe.drop(['BOARD', 'FLAGS'], axis = 1)
        
        #Useful values referenced throughout functions, changing here will adjust all instances of data
        self.optimal_short_gate = False#Flag for whether or not optimal short gate has been set
        self.wave_time_list = np.linspace(-48, 940, num = 248)#builds assumed time list
        self.cal_slope_0 = 1
        self.cal_offset_0 = 0
        self.cal_slope_1 = 1
        self.cal_offset_1 = 0
        self.delay_gate_index = 12 #12 is where 0 starts
        self.short_gate_index = 32 #32 is index of 80ns
        self.long_gate_index300ns = 87 #87 is the index of 300 ns
        
        #PSD values for each row when optimum short gate is set separated by channel
        self.optPSD0 = []
        self.optPSD1 = []
        self.optPSDboth = []
        
        #Calibrated ENERGY value for each row, separated by channel: UNFINISHED@@@@@@
        self.calibrated_light0 = []
        self.calibrated_light1 = []
        self.calibrated_lightboth = []
        
        #Discriminator values of interest for each channel
        self.discriminator_ch0 = False
        self.discriminator_ch1 = False
        
        self.energy_unit = 'Arbitrary'
    
    #Can input 1 or 2 energy calibration points (pick bin, set energy for bin) and define energy unit "KeVee", "MeVee", etc.
    def calibrate(self, energy_bin1, energy_value1, energy_unit, energy_bin2 = False, energy_value2 = False, channel = -1):
        self.energy_unit = energy_unit
        if energy_bin2 == False and energy_value2 == False:
            if channel == 0:
                self.cal_slope_0 = energy_value1 / energy_bin1
            elif channel == 1:
                self.cal_slope_1 = energy_value1 / energy_bin1
            else:
                self.cal_slope_0 = energy_value1 / energy_bin1
                self.cal_slope_1 = energy_value1 / energy_bin1
        else:
            if channel == 0:
                self.cal_slope_0 = abs(energy_value1 - energy_value2) / abs(energy_bin1 - energy_bin2)
                self.cal_offset_0 = energy_value1 - self.cal_slope_0 * energy_bin1
            elif channel == 1:
                self.cal_slope_1 = abs(energy_value1 - energy_value2) / abs(energy_bin1 - energy_bin2)
                self.cal_offset_1 = energy_value1 - self.cal_slope_1 * energy_bin1
            else:
                self.cal_slope_0 = abs(energy_value1 - energy_value2) / abs(energy_bin1 - energy_bin2)
                self.cal_offset_0 = energy_value1 - self.cal_slope_0 * energy_bin1
                self.cal_slope_1 = abs(energy_value1 - energy_value2) / abs(energy_bin1 - energy_bin2)
                self.cal_offset_1 = energy_value1 - self.cal_slope_1 * energy_bin1
        
        
        
        
        return
    
    #reads the 'ENERGY' and 'ENERGYSHORT' values and returns long - short / long
    def getPSD (self, row_index):
        long = self.dataframe['ENERGY'][row_index]
        short = self.dataframe['ENERGYSHORT'][row_index]
        PSD_out = (long - short) / long
        return PSD_out
    
    
    #reads the 'ENERGY' value and returns it
    def getEnergy(self, row_index):
        energy_out = self.dataframe['ENERGY'][row_index]
        return energy_out
    
    #builds a list of the various values at each timestep and subtracts it from the set zero values.
    def readPulse (self, row_index, zero0 = 3565, zero1 = 3650):
        raw_pulse = self.dataframe.loc[row_index, 'SAMPLES':].values.flatten().tolist()
        pulse_out = []
        if self.dataframe['CHANNEL'][row_index]==0:
            for vals in raw_pulse:
                pulse_out.append(zero0 - vals)
        else:
            for vals in raw_pulse:
                pulse_out.append(zero1 - vals)
        return pulse_out
        
    
    #Uses trapezoid integration from t = 0 to t = short gate index (long gate)
    #interpolation/quadrature takes too long and since interp is linear anyways, it isn't worth doing
    def calcPSD (self, short_gate, row_index):
        digital_list = self.readPulse(row_index)
        """
        working_interp = interp1d(self.wave_time_list, digital_list)
        start = self.wave_time_list[self.delay_gate_index]
        end_short = start + short_gate
        end_long = self.wave_time_list[-1]
        long = quad(working_interp, start, end_long)[0]
        short = quad(working_interp, start, end_short)[0]
        """
        
        long = trapz(digital_list[self.delay_gate_index:self.long_gate_index300ns], x = self.wave_time_list[self.delay_gate_index:self.long_gate_index300ns])
        short = trapz(digital_list[self.delay_gate_index:self.delay_gate_index+short_gate], x = self.wave_time_list[self.delay_gate_index:self.delay_gate_index+short_gate])
        #"""
        PSD_out = (long - short) / long
        return PSD_out
    
    #Uses trapezoid integration from t = 0 to t = 300ns (long gate)
    def calcEnergy (self, row_index):
        digital_list = self.readPulse(row_index)
        """
        working_interp = interp1d(self.wave_time_list, digital_list)
        start = self.wave_time_list[self.delay_gate_index]
        end_long = self.wave_time_list[-1]
        energy_out = quad(working_interp, start, end_long)[0]
        """
        energy_out = trapz(digital_list[self.delay_gate_index:self.long_gate_index300ns], x = self.wave_time_list[self.delay_gate_index:self.long_gate_index300ns])
        #"""
        return energy_out
    
    #After finding the best PSD values, the short gate indexes can be set here 23 and 25 were found to be best for our settings/detectors
    def setOptimumPSDvals (self, short0, short1, plot = True):
        PSDchan1 = []
        PSDchan0 = []
        PSDchanBoth = []
        chan1Light = []
        chan0Light = []
        chanBothLight = []
        
        row_index = 0
        while row_index < len(self.dataframe):
            if self.dataframe['CHANNEL'][row_index]==0:#Only updates ch0 lists and entire list
                PSDchan0.append(self.calcPSD(short0, row_index))
                PSDchanBoth.append(self.calcPSD(short0, row_index))
                chan0Light.append(self.cal_slope_0 * self.getEnergy(row_index) + self.cal_offset_0)
                chanBothLight.append(self.cal_slope_0 * self.getEnergy(row_index) + self.cal_offset_0)
            else:#Only updates ch1 lists and entire list
                PSDchan1.append(self.calcPSD(short1, row_index))
                PSDchanBoth.append(self.calcPSD(short1, row_index))
                chan1Light.append(self.cal_slope_1 * self.getEnergy(row_index) + self.cal_offset_1)
                chanBothLight.append(self.cal_slope_1 * self.getEnergy(row_index) + self.cal_offset_1)
            row_index+=1
        
        
        self.optPSD0 = PSDchan0
        self.optPSD1 = PSDchan1
        self.optPSDboth = PSDchanBoth
        self.calibrated_light0 = chan0Light
        self.calibrated_light1 = chan1Light
        self.calibrated_lightboth = chanBothLight
        
        
        #Plots the PSD vs calibrated energy
        if plot == True:
            plt.scatter(chan0Light, PSDchan0, s = 0.1, label = 'Ch 0')
            plt.scatter(chan1Light, PSDchan1, s = 0.1, label = 'Ch 1')
            plt.ylim(0, 0.4)
            plt.title('PSD: Ch0 short gate: ' + str(short0 * 4 - 48) + 'ns; Ch1 short gate: ' + str(short1 * 4 -48) + 'ns')
            plt.xlabel('Light ('+ self.energy_unit + ')')
            plt.legend()
            plt.grid('both')
            plt.show()
        
        return
    
    
    
    #Plots pulse with gates overplotted and hatched areas under the integrals
    def plotPulse (self, row_index):
        test_pulse = self.readPulse(row_index)
        short_gate_list = [-48, 0, self.wave_time_list[self.short_gate_index], 300]
        short_gate_height_list = [10, 10, max(test_pulse), 10]
        long_gate_list = [-48, 0, self.wave_time_list[self.long_gate_index300ns], 300]
        long_gate_height_list = [-10, -10, max(test_pulse)/2, -10]
        plt.step(self.wave_time_list, test_pulse, 'k')
        plt.step(short_gate_list, short_gate_height_list, 'b')
        plt.step(long_gate_list, long_gate_height_list, 'r')
        plt.fill_between(self.wave_time_list[self.delay_gate_index:self.long_gate_index300ns], test_pulse[self.delay_gate_index:self.long_gate_index300ns], step='pre', color = 'r' ,hatch = '/', alpha = 0.5)
        plt.fill_between(self.wave_time_list[self.delay_gate_index:self.short_gate_index+1], test_pulse[self.delay_gate_index:self.short_gate_index+1], step='pre',color = 'b' , hatch = '\\', alpha = 0.5)
        plt.grid('both')
        plt.title('Pulse '+ str(row_index) + ' with long and short gates')
        plt.ylabel('Arbitrary units')
        plt.xlabel('Time (ns)')
        plt.legend(['Pulse', 'Short Gate', 'Long Gate'])
        plt.xlim(-48,312)
        plt.show()
        return

    
    #No longer used--but plots the PSD vs light by integrating (or pulling ENERGYSHORT if short_gate = False)
    def plotPSD(self, short_gate = False):
        x_vals0 = []
        y_vals0 = []
        x_vals1 = []
        y_vals1 = []
        
        row = 0
        
        if short_gate == False:
            while row < len(self.dataframe):
                if self.dataframe['CHANNEL'][row] == 0:
                    x_vals0.append(self.cal_slope_0 * self.getEnergy(row) + self.cal_offset_0)
                    y_vals0.append(self.getPSD(row))
                else:
                    x_vals1.append(self.cal_slope_1 * self.getEnergy(row) + self.cal_offset_1)
                    y_vals1.append(self.getPSD(row))
                row+=1
        
        else:#This is for when a short gate is defined in the call
            while row < len(self.dataframe):
                if self.dataframe['CHANNEL'][row] == 0:
                    #x_vals0.append(self.calcEnergy(row))
                    x_vals0.append(self.cal_slope_0 * self.getEnergy(row) + self.cal_offset_0)
                    y_vals0.append(self.calcPSD(short_gate, row))
                else:
                    #x_vals1.append(self.calcEnergy(row))
                    x_vals1.append(self.cal_slope_1 * self.getEnergy(row) + self.cal_offset_1)
                    y_vals1.append(self.calcPSD(short_gate, row))
                row+=1
        
        
        
        plt.scatter(x_vals0, y_vals0, s = 0.1)
        plt.scatter(x_vals1, y_vals1, s = 0.1)
        plt.ylim(0, 0.4)
        plt.title('PSD')
        plt.xlabel('Light ('+ self.energy_unit + ')')
        plt.show()
        return
        
    """
    #Not necessary--done by hand for this project
    def optimizeShortGate(self, channel, skip = False):
        if skip != False:
            self.short_gate_index = skip
            return
        else:
            
            return
    """
    
    #Unused, was intended for FOM vs short gate--done by hand instead
    def getList(self, channel, listname):
        desired_list = []
        row = 0
        if listname == 'Light':
            while row < len(self.dataframe):
                if self.dataframe['CHANNEL'][row] == channel:
                    desired_list.append(self.dataframe['ENERGY'][row])
                row+=1
        elif listname == 'PSD':
            if channel == 0:
                desired_list = self.PSDchan0
            else:
                desired_list = self.PSDchan1
        
        return desired_list
    
    #Sets the horizontal line that cuts between neutron and gamma events, a turning point and then a second point to account for curving light stuff
    def set_discriminator (self, psd_h_cut, turn_point, rise, run, channel):
        slope = rise / run
        if channel == 0:
            self.discriminator_ch0 = [psd_h_cut, turn_point, slope]
            line_x = [0, turn_point, max(self.calibrated_light0)]
            line_y = [psd_h_cut, psd_h_cut, psd_h_cut + (max(self.calibrated_light0) - turn_point) * slope]
            
            plt.scatter(self.calibrated_light0, self.optPSD0, s = 0.5, label = 'PSD Ch 0')
            plt.plot(line_x, line_y, 'k', label = 'Cutoff')
            plt.legend()
            plt.title('Cutoff lines for Discrimination')
            plt.grid('both')
            plt.xlabel('Light Measured (arbitrary)')
            plt.ylim(0, 0.4)
            plt.show()
        else:
            self.discriminator_ch1 = [psd_h_cut, turn_point, slope]
            line_x = [0, turn_point, max(self.calibrated_light1)]
            line_y = [psd_h_cut, psd_h_cut, psd_h_cut + (max(self.calibrated_light1) - turn_point) * slope]
            plt.scatter(self.calibrated_light1, self.optPSD1, s = 0.5, label = 'PSD Ch 1')
            plt.plot(line_x, line_y, 'k', label = 'Cutoff')
            plt.legend()
            plt.title('Cutoff lines for Discrimination')
            plt.grid('both')
            plt.xlabel('Light Measured (arbitrary)')
            plt.ylim(0, 0.4)
            plt.show()
        return
    
    #Must use set_discriminator first, but this returns a boolian for whether it is below discriminator line(s) True => gamma false => neutron
    def is_gamma (self, light_val, PSD_val, channel):
        
        if channel == 0:
            if self.discriminator_ch0 == False:
                print('Discriminator Parameters Not Set for Ch 0')
                return
            effective_slope = (PSD_val - self.discriminator_ch0[0]) / (light_val - self.discriminator_ch0[1])
            if PSD_val < self.discriminator_ch0[0]:#if below horizontal line, event is gamma
                is_gamma = True
                
            elif light_val > self.discriminator_ch0[1] and effective_slope < self.discriminator_ch0[2]:
                is_gamma = True#if below the sloped line, event is gamma
                
            else:
                is_gamma = False
        if channel == 1:
            if self.discriminator_ch1 == False:
                print('Discriminator Parameters Not Set for Ch 1')
                return
            effective_slope = (PSD_val - self.discriminator_ch1[0]) / (light_val - self.discriminator_ch1[1])
            if PSD_val < self.discriminator_ch1[0]:
                is_gamma = True
                
            elif light_val > self.discriminator_ch1[1] and effective_slope < self.discriminator_ch1[2]:
                is_gamma = True
                
            else:
                is_gamma = False
        
        return is_gamma
    
    
    #This breaks down the dataframe to just the necessary info (optimized PSD, timetag, channel, energy/light, event number)
    #Event number is per coincidence event, not the row_index!!!!
    def buildCoincEvent (self):
        timecolumns = [*range(7,254,1)]
        sorted_out = self.dataframe.drop(['SAMPLES'] + timecolumns, axis = 1)
        sorted_out.insert(2, "OptPSD", self.optPSDboth)
        sorted_out = sorted_out.sort_values(by='TIMETAG')
        sorted_out = sorted_out.reset_index(drop = True)
        
        event_list = [1]
        iterator = 1
        event_tracker = 1
        trigger_time = sorted_out['TIMETAG'][0]
        while iterator < len(sorted_out):
            if sorted_out['TIMETAG'][iterator] < trigger_time + 400000:
                event_list.append(event_tracker)
            else:
                trigger_time = sorted_out['TIMETAG'][iterator]
                event_tracker+=1
                event_list.append(event_tracker)
            iterator += 1
        
        sorted_out.insert(0, 'Event', event_list)
        print('There are '+str(event_tracker)+'coincidence events in this dataset\n')
        return sorted_out
    
        
        








#%%
#Builds Dataframe
test = data_start('CoincidenceWaveSemi.csv', delimiter = ';')
#%%
#FOMcheck is used for setting the short gate for FOM calculations in another file
#FOMcheck = 0


#Sets the optimum short gate indexes 23,25
test.setOptimumPSDvals(23, 25)

#%%

#sets discrimination parameters
test.set_discriminator(0.16, 2500, 0.04, 900, 0)

test.set_discriminator(0.18, 2800, 0.04, 900, 1)

#calibrates the light channels
test.calibrate(420.4, 478, 'keVee', channel = 0)
test.calibrate(381.4, 478, 'keVee', channel = 1)

#Resets short gate indices to plot PSD vs Light instead of PSD vs Arbitrary
test.setOptimumPSDvals(23, 25)



#%%

#Builds coincidence dataframe for further analysis
coincidence = test.buildCoincEvent()


#Gamma gamma and gamma neutron events for times of flight to plot histograms of
deltaT_gam_gam = []
deltaT_gam_neut = []

#various counters to adjust for non-coincidence measurements like single detector or multiple events within a trigger window
single_counter = 0
multi_counter = 0
inverse_counter = 0


#Starting parser for logic
event_sorter = 1
row_index = 0

while event_sorter <= 29200:#last events aren't coincidence anyways, so giving a buffer
    if coincidence['Event'][row_index] != coincidence['Event'][row_index + 1]:#checks for non-matching event in further detector: there are about 965
        single_counter += 1
        while coincidence['Event'][row_index] < event_sorter + 1:#skips to next event
            row_index += 1
    elif coincidence['Event'][row_index] == coincidence['Event'][row_index + 2]:#checks for 2 chanel 
        multi_counter += 1
        while coincidence['Event'][row_index] < event_sorter + 1:#skips to next event
            row_index += 1
    
    #elif coincidence['CHANNEL'][row_index] == 0 and coincidence['CHANNEL'][row_index + 1] == 1:
     #   inverse_counter += 1
      #  while coincidence['Event'][row_index] < event_sorter + 1:#skips to next event
       #     row_index += 1
    
    #If at this point, event is seen as candidate of interest
    else:
        light_trig = coincidence['ENERGY'][row_index]#trigger event light value
        light_second = coincidence['ENERGY'][row_index + 1]#coincident event light value
        psd_trig = coincidence['OptPSD'][row_index]#trigger event PSD
        psd_second = coincidence['OptPSD'][row_index]#coincident event PSD
        #This represents gamma-gamma events and adds to the gam-gam list
        if test.is_gamma(light_trig, psd_trig, 1) == True and test.is_gamma(light_second, psd_second, 0) == True:
            deltaT_gam_gam.append((coincidence['TIMETAG'][row_index + 1] - coincidence['TIMETAG'][row_index]) / 1000)
            row_index += 2
        #This represents gamma-neutron events and adds the delta t to gam-neut list
        elif test.is_gamma(light_trig, psd_trig, 1) == True and test.is_gamma(light_second, psd_second, 0) == False:
            deltaT_gam_neut.append((coincidence['TIMETAG'][row_index + 1] - coincidence['TIMETAG'][row_index]) / 1000)
            row_index += 2
        #this is for n-n and n-gam events
        else:
            row_index += 2#skips events that are not gam-gam or gam-neut
    
    
    
    event_sorter += 1
    if coincidence['Event'][row_index] != event_sorter:
        print('INDEX MISMATCH')
        break
#plots the gam-gam time of flight hist
plt.hist(deltaT_gam_gam, bins = 100)
plt.title('Gamma-Gamma time of flight')
plt.annotate('12 ns', [20, 700])
plt.xlabel('Time of Flight-Measured (ns)')
plt.ylabel('Counts')
plt.show()



#Kinetic Energy of Nuetron Function [MeV] 
def EnergyofNeutron(delta_T, t_offset): 
    m_n = 1.66E-27 #Mass of Neutron [kg]
    d1=1.524 #Distance between detectors [m] =60 [in]
    V_n=d1/((delta_T-t_offset)*1E-9) #[m/s] Velocity of Neutron
    E_n=(1/2*m_n*(V_n)**2)/(1E6*1.6E-19) #Kinetic Energy of Neutron [MeV]
    return E_n 

#builds a list of energies, cuts off energies higher than ~11 MeV
neut_energies = []
for dT in deltaT_gam_neut:
    if dT > 40.:
        neut_energies.append(EnergyofNeutron(dT, 7))
#Plots g-n tof
plt.hist(deltaT_gam_neut, bins = 20)
plt.title('Gamma-Neutron time of flight')
plt.show()
#plots n energy
plt.hist(neut_energies, bins = 100)
plt.title('Neutron Energies')
plt.xlabel('Energy (MeV)')
plt.ylabel('Counts')
plt.xlim(0, 10)
plt.grid('both')
plt.show()





#these lines plot a random pulse from the dataset.
#pulse = np.random.randint(0, len(test.dataframe))
#test.plotPulse(pulse)



#For writing text files to go into the FOM stats final script from winter quarter
"""

filename0 = str(FOMcheck) + '0.txt'
filename1 = str(FOMcheck) + '1.txt'

with open(filename0, 'w') as out:
    i = 0
    while i < len(test.optPSD0):
        psd = '{insert:.4f}'
        out.write(str(test.calibrated_light0[i]) + ' ' + psd.format(insert = test.optPSD0[i]) + '\n')
        i += 1
    print('done')


with open(filename1, 'w') as out:
    i = 0
    while i < len(test.optPSD1):
        psd = '{insert:.4f}'
        out.write(str(test.calibrated_light1[i]) + ' ' + psd.format(insert = test.optPSD1[i]) + '\n')
        i += 1
    print('done')

"""




























