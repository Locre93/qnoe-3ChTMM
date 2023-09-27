from qnoe_3ChTMM import * 
import matplotlib.pyplot as plt

# Section 4. Simulation of graphene plasmons
# Constants
# Anomalous reflection phase of graphene plasmons and its influence on resonators

μm = 1e-6

# Graphene
Sheet = Graphene()

Ef = 0.4														# Fermi energy [eV]
ϵ = [1,1]														# Dielectric permittivities of the environment [Above,Below]
k = np.arange(800,1800,10) 										# Wavenumber [cm⁻¹] - 1/λ (4.497THz to 10.493THz)
ϵ_eff = Sheet.ModeEffectivePermittivity(k,Ef,ϵ)	

# Structure initialization
SHEET = Chunk(name="Suspended graphene",k=k,ϵ=ϵ_eff,length=2,units=μm)
BOUNDARY = Interface(name="Boundary",k=k).set(ReflectingInterface(k=k,ϕ=np.pi))		# Phese ϕ upon reflection

# System initialization
system = TMM_sSNOM_Simple([SHEET,BOUNDARY],position=1,site=1,units=μm)

# Near-field response calculation
x,MAP = system.Scan(sites=[1],resolution=0.01,units=μm)
X,K = np.meshgrid(x,k)

plt.contourf(X,K,np.abs(MAP),100)
plt.xlabel('X, μm',size=16)
plt.ylabel('Wavenumber, cm⁻¹',size=16)
plt.title('Near-field',size=16)
plt.show()

# System initialization
system = TMM_sSNOM_Advanced([SHEET,BOUNDARY],position=1,site=1,units=μm)

# Near-field response calculation
x,MAP = system.Scan(sites=[1],resolution=0.01,units=μm)
X,K = np.meshgrid(x,k)

plt.contourf(X,K,np.abs(MAP),100)
plt.xlabel('X, μm',size=16)
plt.ylabel('Wavenumber, cm⁻¹',size=16)
plt.title('Near-field',size=16)
plt.show()


# Missing: linecut and absorption loss scaling. Plot a relevant case study for plasmon fringes. Plot the scan with different phase upon reflection
# and compare with Rainer result. Maybe hBN fringes are better because we can directly compare with expermiental data and do a comparison 
# with realistic tip geometry.