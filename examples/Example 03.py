from qnoe_3ChTMM import * 
import matplotlib.pyplot as plt

# Section 4. Simulation of graphene plasmons

nm = 1e-9

# Graphene
Sheet = Graphene()

Ef = 0.55														# Fermi energy [eV]
ϵ = [1,1]														# Dielectric permittivities of the environment [Above,Below]
k = np.arange(500,3000,10) 										# Wavenumber [cm⁻¹] - 1/λ
ϵ_eff = Sheet.mode_effective_permittivity(k,Ef,ϵ)	

# Structure initialization
SHEET = Chunk(name="Suspended graphene",k=k,ϵ=ϵ_eff,length=200,units=nm)
BOUNDARY = Interface(name="Boundary",k=k).set(reflecting_interface(k=k,ϕ=-3*np.pi/4,r=0.99))		# Phese ϕ upon reflection

# System initialization
system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm)

# Near-field response calculation
x,MAP = system.scan(sites=[1],resolution=10,units=nm)
X,K = np.meshgrid(x,k)

plt.contourf(X,K,np.abs(MAP),100)
plt.xlabel('X, μm',size=16)
plt.ylabel('Wavenumber, cm⁻¹',size=16)
plt.title('Near-field',size=16)
plt.show()

# # System initialization
# system = TMM_sSNOM_Advanced([SHEET,BOUNDARY],position=1,site=1,units=μm)

# # Near-field response calculation
# x,MAP = system.scan(sites=[1],resolution=0.01,units=μm)
# X,K = np.meshgrid(x,k)

# plt.contourf(X,K,np.abs(MAP),100)
# plt.xlabel('X, μm',size=16)
# plt.ylabel('Wavenumber, cm⁻¹',size=16)
# plt.title('Near-field',size=16)
# plt.show()


# # Missing: linecut and absorption loss scaling. Plot a relevant case study for plasmon fringes. Plot the scan with different phase upon reflection
# # and compare with Rainer result. Maybe hBN fringes are better because we can directly compare with expermiental data and do a comparison 
# # with realistic tip geometry.