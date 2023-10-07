from qnoe_3ChTMM import * 
from cycler import cycler
import matplotlib.pyplot as plt

# Section 4. Simulation of graphene plasmons photonic lattice

nm = 1e-9

# Graphene
Sheet = Graphene()

Ef_A = 0.55														# Fermi energy [eV]
Ef_B = 0.15														# Fermi energy [eV]
ϵ = [1,1]														# Dielectric permittivities of the environment [Above,Below]
k = np.arange(900,1350,0.1) 									# Wavenumber [cm⁻¹] - 1/λ
ϵ_eff_A = np.real(Sheet.mode_effective_permittivity(k,Ef_A,ϵ))	
ϵ_eff_B = np.real(Sheet.mode_effective_permittivity(k,Ef_B,ϵ))

# Structure initialization
INTERFACE_LEFT = EffectiveInterfaceLeft(name="Suspended graphene",k=k,ϵ=ϵ_eff_B)
SHEET_A = EffectiveChunk(name="Suspended graphene",k=k,ϵ=ϵ_eff_A,length=200,units=nm)
SHEET_B = EffectiveChunk(name="Suspended graphene",k=k,ϵ=ϵ_eff_B,length=80,units=nm)
INTERFACE_RIGHT = EffectiveInterfaceRight(name="Suspended graphene",k=k,ϵ=ϵ_eff_B)

# BAND STRUCTURE
system = TMM([SHEET_A,SHEET_B])
Ka,Ea,Kb,Eb = system.band_structure()

plt.scatter(np.real(Ka),Ea,color="firebrick",s=2)
plt.scatter(np.real(Kb),Eb,color="firebrick",s=2)
plt.ylim([900,1350])
plt.show()

# REFLECTION SPECTRUM
finite_lattice = [INTERFACE_LEFT,SHEET_B,SHEET_A,SHEET_B,SHEET_A,SHEET_B,SHEET_A,SHEET_B,SHEET_A,SHEET_B,SHEET_A,SHEET_B,SHEET_A,SHEET_B,SHEET_A,SHEET_B,SHEET_A,SHEET_B,INTERFACE_RIGHT]

system = TMM(finite_lattice)
system.global_scattering_matrix()

plt.plot(abs(system.S.S12)**2,k,color="forestgreen")
plt.ylim([900,1350])
plt.show()

# NEAR-FIELD SPECTRAL SCAN
k = np.arange(900,1350,1) 											# Wavenumber [cm⁻¹] - 1/λ
ϵ_eff_A = np.real(Sheet.mode_effective_permittivity(k,Ef_A,ϵ)) + 1j*np.imag(Sheet.mode_effective_permittivity(k,Ef_A,ϵ))
ϵ_eff_B = np.real(Sheet.mode_effective_permittivity(k,Ef_B,ϵ)) + 1j*np.imag(Sheet.mode_effective_permittivity(k,Ef_B,ϵ))

INTERFACE_LEFT = EffectiveInterfaceLeft(name="Suspended graphene",k=k,ϵ=ϵ_eff_B)
SPACING_B = EffectiveChunk(name="Suspended graphene",k=k,ϵ=ϵ_eff_B,length=390,units=nm)
SHEET_A = EffectiveChunk(name="Suspended graphene",k=k,ϵ=ϵ_eff_A,length=200,units=nm)
SHEET_B = EffectiveChunk(name="Suspended graphene",k=k,ϵ=ϵ_eff_B,length=80,units=nm)
INTERFACE_RIGHT = EffectiveInterfaceRight(name="Suspended graphene",k=k,ϵ=ϵ_eff_B)

finite_lattice = [INTERFACE_LEFT,SPACING_B,SHEET_B,SHEET_A,SHEET_B,SHEET_A,SHEET_B,SHEET_A,SHEET_B,SHEET_A,SHEET_B,SHEET_A,SHEET_B,SHEET_A,SHEET_B,SHEET_A,SHEET_B,SHEET_A,SHEET_B,SPACING_B,INTERFACE_RIGHT]

system = TMM_sSNOM_Simple(array_of_sections=finite_lattice,position=100,site=1,units=nm)

x, MAP = system.scan(sites=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19],resolution=5)
X,K = np.meshgrid(x,k)
plt.contourf(X,K,np.abs(MAP),100)
plt.ylim([900,1350])
plt.xlabel('X, nm', size=16)
plt.title('Near-field spectral scan', size=16)
plt.show()
