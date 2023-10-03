from qnoe_3ChTMM import * 
import matplotlib.pyplot as plt

# Section 4. Simulation of graphene plasmons

nm = 1e-9

# Graphene
Sheet = Graphene()

Ef = 0.55														# Fermi energy [eV]
ϵ = [1,1]														# Dielectric permittivities of the environment [Above,Below]
k = np.arange(500,3000,5) 										# Wavenumber [cm⁻¹] - 1/λ
ϵ_eff = Sheet.mode_effective_permittivity(k,Ef,ϵ)	

# Structure initialization
SHEET = Chunk(name="Suspended graphene",k=k,ϵ=ϵ_eff,length=200,units=nm)
BOUNDARY = Interface(name="Boundary",k=k).set(reflecting_interface(k=k,ϕ=-3*np.pi/4,r=0.99))		# Phese ϕ upon reflection 

# Standard Transfer Matrix Method
TMM_system = TMM([BOUNDARY,SHEET,BOUNDARY])
TMM_system.global_scattering_matrix()

plt.plot(k,np.abs(TMM_system.S.S21))
plt.ylabel('Transmission', size=16)
plt.xlabel('Wavenumber, cm⁻¹', size=16)
plt.show()

# 1100
# 1680
# 2110
# 2460
# 2770

# System initialization
system = TMM_sSNOM_Simple([BOUNDARY,SHEET,BOUNDARY],position=100,site=1,units=nm)

system.near_field(Eᴮᴳ=0,harm=4)
O4_pseudo = system.O

system.near_field(Eᴮᴳ=np.exp(1j*np.pi/2),harm=4)
O4_pseudo = system.O



plt.plot(k,np.abs(system.O),label="Pseudo-heterodyne")


plt.plot(k,np.real(system.O),'r',label="Self-homodyne")

plt.title("Near-field spectrum at the cavity centre", size=16)
plt.xlabel("Wavenumber, cm⁻¹", size=16)
plt.ylabel("Signal, a.u.", size=16)
plt.legend()
plt.show()




# # Near-field response calculation
# x,MAP = system.scan(sites=[1],resolution=10,units=nm)
# X,K = np.meshgrid(x,k)

# plt.contourf(X,K,np.abs(MAP),100)
# plt.xlabel('X, μm',size=16)
# plt.ylabel('Wavenumber, cm⁻¹',size=16)
# plt.title('Near-field',size=16)
# plt.show()