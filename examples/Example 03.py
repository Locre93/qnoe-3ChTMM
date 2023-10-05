from qnoe_3ChTMM import * 
from cycler import cycler
import matplotlib.pyplot as plt

# Section 4. Simulation of graphene plasmons

nm = 1e-9

# Graphene
Sheet = Graphene()

Ef = 0.55														# Fermi energy [eV]
ϵ = [1,1]														# Dielectric permittivities of the environment [Above,Below]
k = np.arange(500,2500,3) 										# Wavenumber [cm⁻¹] - 1/λ
ϵ_eff = Sheet.mode_effective_permittivity(k,Ef,ϵ)	

# Structure initialization
SHEET = Chunk(name="Suspended graphene",k=k,ϵ=ϵ_eff,length=200,units=nm)
BOUNDARY = Interface(name="Boundary",k=k).set(reflecting_interface(k=k,ϕ=-3*np.pi/4,r=0.99))		# Phese ϕ upon reflection 

TMM_system = TMM([BOUNDARY,SHEET,BOUNDARY])
system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm)

# Standard Transfer Matrix Method
TMM_system.global_scattering_matrix()

plt.plot(np.abs(TMM_system.S.S21),k)
plt.xlabel('Transmission', size=16)
plt.ylabel('Wavenumber, cm⁻¹', size=16)
plt.show()

# Near-field response calculation
x,MAP = system.scan(sites=[1],resolution=10,units=nm)
X,K = np.meshgrid(x,k)

plt.contourf(X,K,np.abs(MAP),100)
plt.xlabel('X, nm',size=16)
plt.ylabel('Wavenumber, cm⁻¹',size=16)
plt.title('Near-field',size=16)
plt.show()

# Near-field comparison
k = np.arange(900,1300,1) 										# Wavenumber [cm⁻¹] - 1/λ
ϵ_eff = Sheet.mode_effective_permittivity(k,Ef,ϵ)	

# Structure initialization
SHEET = Chunk(name="Suspended graphene",k=k,ϵ=ϵ_eff,length=200,units=nm)
BOUNDARY = Interface(name="Boundary",k=k).set(reflecting_interface(k=k,ϕ=-3*np.pi/4,r=0.99))		# Phese ϕ upon reflection 

TMM_system = TMM([BOUNDARY,SHEET,BOUNDARY])
TMM_system.global_scattering_matrix()

system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm)

top = 5

fig, ax = plt.subplots()
ax.plot(k,top + np.abs(TMM_system.S.S21)/np.max(np.abs(TMM_system.S.S21)),color="royalblue")

# Pseudo-heterodyne
system.near_field(Eᴮᴳ=0,harm=4)
ax.plot(k,top+np.abs(system.O)/np.max(np.abs(system.O)),color="black")
plt.show()

# Near-field resonses at different coupling strengths
fig, ax = plt.subplots()

system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm,coupling=0.01)
system.near_field(Eᴮᴳ=0,harm=4)
ax.plot(k,top+2.2+np.abs(system.O)/np.max(np.abs(system.O)),color="black")

system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm,coupling=0.05)
system.near_field(Eᴮᴳ=0,harm=4)
ax.plot(k,top+1.1+np.abs(system.O)/np.max(np.abs(system.O)),color="black")

system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm,coupling=0.1)
system.near_field(Eᴮᴳ=0,harm=4)
ax.plot(k,top+np.abs(system.O)/np.max(np.abs(system.O)),color="black")

system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm,coupling=0.125)
system.near_field(Eᴮᴳ=0,harm=4)
ax.plot(k,top-1.1+np.abs(system.O)/np.max(np.abs(system.O)),color=(0.3,0,0))

system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm,coupling=0.15)
system.near_field(Eᴮᴳ=0,harm=4)
ax.plot(k,top-2.2+np.abs(system.O)/np.max(np.abs(system.O)),color=(0.35,0,0))

system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm,coupling=0.175)
system.near_field(Eᴮᴳ=0,harm=4)
ax.plot(k,top-3.3+np.abs(system.O)/np.max(np.abs(system.O)),color=(0.4,0,0))

system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm,coupling=0.2)
system.near_field(Eᴮᴳ=0,harm=4)
ax.plot(k,top-4.4+np.abs(system.O)/np.max(np.abs(system.O)),color=(0.45,0,0))

system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm,coupling=0.225)
system.near_field(Eᴮᴳ=0,harm=4)
ax.plot(k,top-5.5+np.abs(system.O)/np.max(np.abs(system.O)),color=(0.5,0,0))

system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm,coupling=0.25)
system.near_field(Eᴮᴳ=0,harm=4)
ax.plot(k,top-6.6+np.abs(system.O)/np.max(np.abs(system.O)),color=(0.55,0,0))

system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm,coupling=0.275)
system.near_field(Eᴮᴳ=0,harm=4)
ax.plot(k,top-7.7+np.abs(system.O)/np.max(np.abs(system.O)),color=(0.6,0,0))

system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm,coupling=0.3)
system.near_field(Eᴮᴳ=0,harm=4)
ax.plot(k,top-8.8+np.abs(system.O)/np.max(np.abs(system.O)),color=(0.65,0,0))

system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm,coupling=0.4)
system.near_field(Eᴮᴳ=0,harm=4)
ax.plot(k,top-9.9+np.abs(system.O)/np.max(np.abs(system.O)),color=(0.7,0,0))

system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm,coupling=0.5)
system.near_field(Eᴮᴳ=0,harm=4)
ax.plot(k,top-11+np.abs(system.O)/np.max(np.abs(system.O)),color=(0.75,0,0))

system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm,coupling=0.6)
system.near_field(Eᴮᴳ=0,harm=4)
ax.plot(k,top-12.1+np.abs(system.O)/np.max(np.abs(system.O)),color=(0.8,0,0))

system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm,coupling=0.7)
system.near_field(Eᴮᴳ=0,harm=4)
ax.plot(k,top-13.2+np.abs(system.O)/np.max(np.abs(system.O)),color=(0.85,0,0))

plt.show()

# Self-homodyne

fig, ax = plt.subplots()

system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm,coupling=0.1)

system.near_field(Eᴮᴳ=100*np.exp(1j*0),harm=4)
ax.plot(k,top-0.5+np.real(system.O)/np.max(np.abs(system.O)),color=(0,0,1))

system.near_field(Eᴮᴳ=100*np.exp(1j*np.pi/4),harm=4)
ax.plot(k,top-1+np.real(system.O)/np.max(np.abs(system.O)),color=(0,0,0.9))

system.near_field(Eᴮᴳ=100*np.exp(1j*np.pi/3),harm=4)
ax.plot(k,top-1.5+np.real(system.O)/np.max(np.abs(system.O)),color=(0,0,0.8))

system.near_field(Eᴮᴳ=100*np.exp(1j*np.pi/2),harm=4)
ax.plot(k,top-2+np.real(system.O)/(1.5*np.max(np.abs(system.O))),color=(0,0,0.7))

system.near_field(Eᴮᴳ=100*np.exp(2j*np.pi/3),harm=4)
ax.plot(k,top-2.5+np.real(system.O)/np.max(np.abs(system.O)),color=(0,0,0.6))

system.near_field(Eᴮᴳ=100*np.exp(3j*np.pi/4),harm=4)
ax.plot(k,top-3+np.real(system.O)/np.max(np.abs(system.O)),color=(0,0,0.5))

system.near_field(Eᴮᴳ=100*np.exp(1j*np.pi/1),harm=4)
ax.plot(k,top-3.5+np.real(system.O)/np.max(np.abs(system.O)),color=(0,0,0.4))

plt.show()