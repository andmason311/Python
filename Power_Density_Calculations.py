"""
Author: Andy Mason
Project: Power Density Calculations of Thulium Impregnated Carbon Composite
"""
# importing mplot3d toolkits, numpy and matplotlib Libraries
import numpy as np
import math as m
#from dash import Dash, dcc, html, Input, Output
import chart_studio.plotly as plotly
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='browser'
 
# defining geometry parameters
D = np.linspace(5, 500, 100) # Pellet Diameter ranging from 5 to 500 um
L = np.linspace(5, 500, 100) # Pellet Distance ranging from 5 to 500 um 
D,L=np.meshgrid(D,L) 
mesh=np.meshgrid(D,L)

# Variables

# Conversions
Na=6.022e23 # Avagadros Number [amu/g]
MassToEng=931.494 # Mass to Energy Conversion [MeV/amu*c^2]
MeVtoJ=6.2415e12 # [MeV/J]
# Molecular Weights
MW_169Tm=168.9342 # [amu/atom]
MW_170Tm=169.936 # [amu/atom]
MW_171Tm=170.936 # [amu/atom]
MW_12C=168.9342 # [amu/atom]
# Densities
rho_Tm=9.37688596 # [g/cm3]
rho_12C=2.265 # [g/cm3]
rho_170Tm=9.321071252 # [g/cm3]
rho_171Tm=9.432600381 # [g/cm3]
rho_170Yb=6.980 # [g/cm3]
rho_169Tm=rho_Tm*Na/MW_169Tm # [atoms/cm3]
# Unit Cell Geometry
r_Tm=174 # atomic radius [pm]
z_Tm=6 # number of atoms per unit volume
z_C=3 # number of atoms per unit volume
Vc_Tm=24*m.sqrt(2)*pow(r_Tm,3)*1e-30 # Unit Volume [cm^3]
# Decay Constants
lambda_170Tm=6.23836E-08 # decay consant [1/s]
lambda_171Tm=1.14477E-08 # decay consant [1/s]
# Decay Fractions
BetaFrac1_170Tm=.9987*.78 # fraction of decays
BetaFrac2_170Tm=.9987*.22 # fraction of decays
ECFrac_170Tm=.0013 # fraction of decays
BetaFrac_171Tm=1 # fraction of decays
# Decay Energies
BetaEng1_170Tm=.9686 # energy of beta particle [MeV]
BetaEng2_170Tm=.884 # energy of beta particle [MeV]
GammaEng2_170Tm=.084 # energy of gamma ray released after beta decay 2 [MeV]
ECEng_170Tm=.3122 # energy of x ray released after electron capture [MeV]
BetaEng_171Tm=.0965 # energy of beta particle [MeV]

# Calculations

# Number of Tm atoms per pellet
No_Tm=(4/3*np.pi*pow(D*1e-4/2,3))*(rho_169Tm) # Vsphere[cm^3]*rho_Tm[atoms/cm^3]

# Number of Tm isotope atoms per pellet after activation
Frac_170Tm=.11 # Fraction of 170Tm immediately after irradiation
Frac_171Tm=.505 # Fraction of 171Tm immediately after irradiation
Frac_170Yb=.294 # Fraction of 294Tm immediately after irradiation
Frac_169Tm=1-Frac_170Tm-Frac_171Tm-Frac_170Yb
N1_170Tm=No_Tm*Frac_170Tm # Number of 170Tm atoms/pellet
N1_171Tm=No_Tm*Frac_171Tm # Number of 171Tm atoms/pellet

# Number of Pellets per 1 cm^3 CCC
z_Cube=1 # Number of pellets per unit volume of cubic lattice
    #z_HCP=6 # Number of pellets per unit volume of Hexagonal Close Packed lattice
N_Pellets_Cube=1/(pow(L*1e-4,3)) # varying pellet distance [Pellets/1cm^3]
    #N_Pellets_HCP=6/(24*np.sqrt(2)*pow(L*1e-4,3)) # varying pellet distance [Pellets/1cm^3]1/((L*1e-4)^3) # varying pellet distance [Pellets/1cm^3]

# Number of Tm isotope atoms per 1 cm^3 CCC after activation
N1_170Tm_Cube=N1_170Tm*N_Pellets_Cube # Number of [170Tm atoms/1cm^3 CCC]
N1_171Tm_Cube=N1_171Tm*N_Pellets_Cube # Number of [171Tm atoms/1cm^3 CCC]
    #N1_170Tm_HCP=N1_170Tm*N_Pellets_HCP # Number of [170Tm atoms/1cm^3 CCC]
    #N1_171Tm_HCP=N1_171Tm*N_Pellets_HCP # Number of [171Tm atoms/1cm^3 CCC]

# Activity of Each Geometric Arrangement
A1_170Tm_Cube=(N1_170Tm_Cube*lambda_170Tm)# Activity of all 170Tm in 1 cm^3 CCC [Bq]
A1_171Tm_Cube=(N1_171Tm_Cube*lambda_171Tm)# Activity of all 171Tm in 1 cm^3 CCC [Bq]
A1_Cube=(A1_170Tm_Cube+A1_171Tm_Cube)/3.7e10 # Activity of all Tm in 1 cm^3 CCC [Ci]
    #A1_170Tm_HCP=(N1_170Tm_HCP*lambda_170Tm)# Activity of all 170Tm in 1 cm^3 CCC [Bq]
    #A1_171Tm_HCP=(N1_171Tm_HCP*lambda_171Tm)# Activity of all 171Tm in 1 cm^3 CCC [Bq]
    #A1_HCP=(A1_170Tm_HCP+A1_171Tm_HCP)/3.7e10 # Activity of all Tm in 1 cm^3 CCC [Ci]

# Activity over time
t=np.linspace(0,12,13) # [weeks]
A1_170Tm_t=np.empty((len(t),len(A1_170Tm_Cube),len(A1_170Tm_Cube))) #Creating blank array to place values in
A1_171Tm_t=np.empty((len(t),len(A1_170Tm_Cube),len(A1_170Tm_Cube))) #Creating blank array to place values in
P_Cube=np.empty((len(t),len(A1_170Tm_Cube),len(A1_170Tm_Cube))) #Creating blank array to place values in
Rad_Int=np.empty((len(t),len(A1_170Tm_Cube),len(A1_170Tm_Cube))) #Creating blank array to place values in
m_required=np.empty((len(t),len(A1_170Tm_Cube),len(A1_170Tm_Cube))) #Creating blank array to place values in
    #A1_HCP_t=np.empty((len(t),len(A1_170Tm_Cube),len(A1_170Tm_Cube)))
for i in range(len(t)):
    for j in range(len(A1_170Tm_Cube[0])):
        for k in range(len(A1_170Tm_Cube[0])):               
            A1_170Tm_t[i,j,k]=(A1_170Tm_Cube[j,k])*m.exp(-lambda_170Tm*t[i]*7*24*3600)# Activity over time [Bq]
            A1_171Tm_t[i,j,k]=(A1_171Tm_Cube[j,k])*m.exp(-lambda_171Tm*t[i]*7*24*3600)# Activity over time [Bq]
           # A1_HCP_t[j,k,i]=((A1_170Tm_HCP)*m.exp(-lambda_170Tm*i*30*24*3600)+(A1_171Tm_HCP)*m.exp(-lambda_171Tm*i*30*24*3600)) # Activity over time [Ci]

# Density of Each Geometry
rho_Pellet=Frac_169Tm*rho_Tm+Frac_170Tm*rho_170Tm+Frac_171Tm*rho_171Tm+Frac_170Yb*rho_170Yb # Average density of pellet [g/cm^3]
V_Pellet= (4/3*np.pi*pow(D*1e-4/2,3)) # Volume of each pellet [cm^3]
rho_Composite= N_Pellets_Cube*V_Pellet*rho_Pellet+(1-N_Pellets_Cube*V_Pellet)*rho_12C# Density of Composite varying D and L of pellets [g/cm^3]

# Power Density of Aviation Fuel in a Predator MQ-1
E_Fuel=43e6# [J/kg]
Range=1250# [km]
Max_Fuel=387# [kg]
Max_Speed=170# [km/hr]
Consumption=Range/Max_Fuel#[km/kg]
Fuel_Density=.00083# [kg/cm3]
P_Fuel=(E_Fuel/Consumption)*Max_Speed/3600 #[J/kg]*[kg/km]*[km/hr]*[hr/s]=[J/s]=[W]

# Power Density of Each Geometry and Mass Required for Geometry to provide equivalent Power as Aviation Fuel
for i in range(len(t)):
    P_Cube[i]=(A1_170Tm_t[i]*(BetaFrac1_170Tm*BetaEng1_170Tm+BetaFrac2_170Tm*BetaEng2_170Tm)+A1_171Tm_t[i]*(BetaFrac_171Tm*BetaEng_171Tm))/MeVtoJ # Power Density [W/cm^3]
    Rad_Int[i]=(A1_170Tm_t[i]*(BetaFrac2_170Tm*GammaEng2_170Tm+ECFrac_170Tm*ECEng_170Tm))/MeVtoJ # Inidial Radiation Density [W/cm^3] from EC and beta production. No bremstrahlung
    m_required[i]=(P_Fuel*rho_Composite/(P_Cube[i]*1000)) # mass required to provide same energy to Predator as Avaition Fuel or P_Fuel [kg]
for i in range(len(t)):
    for j in range(len(L[0])):
        for k in range(len(D[0])):
            if L[j,k]<D[j,k]:
                P_Cube[i,j,k]= 0 #np.max(P_Cube[i])
                Rad_Int[i,j,k]= 0 #np.max(Rad_Int[i])
                rho_Composite[j,k]= 0#np.max(rho_Composite[j])
                #m_required[i,j,k]=0 #np.min(m_required[i])

for i in range(len(t)):
    for j in range(len(L[0])):
        for k in range(len(D[0])):
            if L[j,k]<D[j,k]:
                P_Cube[i,j,k]=np.max(P_Cube[i])
                Rad_Int[i,j,k]=np.max(Rad_Int[i])
                rho_Composite[j,k]=np.max(rho_Composite[i])
                m_required[i,j,k]=m_required[i,0,0]                             
'''
# Identifiying possible pellet geometries baced upon mass requirement constraints
indices=np.transpose(np.where(np.logical_and(m_required[0]>0, m_required[0]<= Max_Fuel)))
DLconfigs=mesh[indices]
print(DLconfigs)
'''
#Plotting

# Plotting Power Density over Time
fig1 = go.Figure(data=[go.Surface(x=D,y=L,z=P_Cube[0],colorscale=[[0, 'rgb(192,192,192)'], [1, 'rgb(0,100,0)']],cmin=0,cmax=np.max(P_Cube[0]),
    name="Power Density of Tm Composite", contours = {
    "x": {"show": True, "start": 0, "end": 500, "size": 100, "color":"white"},
    "y": {"show": True, "start": 0, "end": 500, "size": 100, "color":"white"},
    "z": {"show": True, "start": 0, "end": 21, "size": 5, "color":"white"}})], 
    layout=go.Layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Play",method="animate", 
    args=[None, {"frame": {"duration": 1, "redraw": True},"fromcurrent": True, "transition": {"duration": 0}}])])]),
    frames=[go.Frame(data=[go.Surface(z=k)], name=str(i)) for i, k in enumerate(P_Cube)])
fig1.update_layout(title='Power Density', autosize=False, width=1000, height=1000, margin=dict(l=65, r=50, b=65, t=90), scene=dict(
    xaxis_title=dict(text = 'Pellet Diameter (D)'), yaxis_title=dict(text = 'Pellet Distance (L)'), zaxis_title=dict(text = 'Power Density (W/cm^3)'),
    annotations=(dict(showarrow=False,x=350,y=150,z=10,text="Fully Dense Tm Composite",xanchor="center",xshift=0,opacity=0.7),),),)
fig1.update_layout(scene=dict(zaxis=dict(range=[0,22],tickvals= [5,10,15,20])))
fig1.update_traces(colorbar_title_text='[W/cm^3]', selector=dict(type='surface'))
def frame_args(duration):
    return {"frame": {"duration": duration},"mode": "immediate","fromcurrent": True,"transition": {"duration": duration, "easing": "linear"},}
sliders = [{"pad": {"b": 10, "t": 60}, "len": 0.9, "x": 0.1, "y": 0, "steps": [{"args": [[f.name], frame_args(0)], "label":('Week '+str(k)),"method": "animate",}
    for k, f in enumerate(fig1.frames)],}]    
fig1.update_layout(sliders=sliders)
fig1.to_html( include_plotlyjs="cdn", full_html=False)
pio.write_html(fig1, file="Power_Density_Tm_Composite.html", auto_open=True)

# Plotting Density of Each Geometry
fig2 = go.Figure(data=[go.Surface(x=D,y=L,z=rho_Composite,colorscale=[[0, 'rgb(192,192,192)'], [1, 'rgb(100,0,0)']],
    name="Mass Density of Tm Composite", contours = {
    "x": {"show": True, "start": 0, "end": 500, "size": 100, "color":"white"},
    "y": {"show": True, "start": 0, "end": 500, "size": 100, "color":"white"},
    "z": {"show": True, "start": 0, "end": 10, "size": 1, "color":"white"}})],)
fig2.update_layout(title='Mass Density', autosize=False, width=1000, height=1000, margin=dict(l=65, r=50, b=65, t=90), scene=dict(
    xaxis_title=dict(text = 'Pellet Diameter (D)'), yaxis_title=dict(text = 'Pellet Distance (L)'), zaxis_title=dict(text = 'Mass Density (g/cm^3)'),
    annotations=(dict(showarrow=False,x=350,y=150,z=0,text="Fully Dense Tm Composite",xanchor="center",xshift=0,opacity=0.7),),),)
fig1.update_layout(scene=dict(zaxis=dict(tickvals= [1,2,3,4,5,6,7])))
fig2.update_traces(colorbar_title_text='[g/cm^3]', selector=dict(type='surface'))
fig2.to_html( include_plotlyjs="cdn", full_html=False)
pio.write_html(fig2, file="Mass_Density_Tm_Composite.html", auto_open=True)

# Plotting Mass of Tm Composite for Equivalent Power Output as Aviation Fuel
axis_ref3=np.full((len(D),len(L)),Max_Fuel)
fig3 = go.Figure(data=[go.Surface(x=D,y=L,z=m_required[0],colorscale=[[0, 'rgb(192,192,192)'], [1, 'rgb(0,0,100)']],cmin=0,cmax=500,
    name="Mass of Tm Composite for Equivalent Power Output as Aviation Fuel", contours = {
    "x": {"show": True, "start": 0, "end": 500, "size": 100, "color":"white"},
    "y": {"show": True, "start": 0, "end": 500, "size": 100, "color":"white"},
    "z": {"show": True, "start": 0, "end": 1000, "size": 100, "color":"white"}}),go.Surface(x=D,y=L,z=axis_ref3,opacity=.7,showscale=False)], 
    layout=go.Layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Play",method="animate", 
    args=[None, {"frame": {"duration": 1, "redraw": True},"fromcurrent": True, "transition": {"duration": 0}}])])]),
    frames=[go.Frame(data=[go.Surface(z=k)], name=str(i)) for i, k in enumerate(m_required)])
fig3.update_layout(title='Mass of Tm Composite for Equivalent Power Output as Aviation Fuel', autosize=False, width=1000, height=1000, margin=dict(l=65, r=50, b=65, t=90), 
    scene=dict(xaxis_title=dict(text = 'Pellet Diameter (D)'), yaxis_title=dict(text = 'Pellet Distance (L)'), zaxis_title=dict(text = 'Mass Required (kg)'),
    annotations=(dict(showarrow=False,x=350,y=150,z=100,text="Fully Dense Tm Composite",xanchor="center",xshift=0,opacity=0.7),
                 dict(showarrow=False,x=350,y=150,z=387,text="Maximum Mass Possible",xanchor="center",xshift=0,opacity=0.7),),))
fig3.update_layout(scene=dict(zaxis=dict(range=[0,500],tickvals= [100,200,300,387,400,500])))
fig3.update_traces(colorbar_title_text='[kg]', selector=dict(type='surface'))
def frame_args(duration):
    return {"frame": {"duration": duration},"mode": "immediate","fromcurrent": True,"transition": {"duration": duration, "easing": "linear"},}
sliders = [{"pad": {"b": 10, "t": 60}, "len": 0.9, "x": 0.1, "y": 0, "steps": [{"args": [[f.name], frame_args(0)], "label":('Week '+str(k)),"method": "animate",}
    for k, f in enumerate(fig3.frames)],}]    
fig3.update_layout(sliders=sliders)
fig3.to_html( include_plotlyjs="cdn", full_html=False)
pio.write_html(fig3, file="Required_Mass_Tm_Composite.html", auto_open=True)




