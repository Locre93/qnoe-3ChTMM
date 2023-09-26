from qnoe_3ChTMM import * 
import matplotlib.pyplot as plt

# Section 4. Simulation of graphene plasmons
# Main text

Sheet = Graphene()

Ef = [0.1,0.2,0.3,0.4]				# Fermi energy [eV]
k = np.arange(100,350,0.1) 			# Wavenumber [cm⁻¹] - 1/λ (4.497THz to 10.493THz)
ϵ = [1,1]							# Dielectric permittivities of the environment [Above,Below]


ϵ_eff = np.zeros(shape = (len(k),len(Ef)), dtype = complex)
Q = np.zeros(shape = (len(k),len(Ef)), dtype = complex)
λp = np.zeros(shape = (len(k),len(Ef)), dtype = complex)

for i in range(len(Ef)):
	Q[:,i] = Sheet.ModeWavevector(k,Ef[i],ϵ)							# Plasmon wavevector [m⁻¹]
	ϵ_eff[:,i] = np.real(Sheet.ModeEffectivePermittivity(k,Ef[i],ϵ))	# Plasmon's effective permittivity
	λp[:,i] = 2*np.pi/np.real(Q[:,i])									# Plasmon wavelength [m]

SHEET = Chunk("Ef 0.4eV",k,ϵ_eff[:,3],10,1e-6)

BOUNDARY = Interface("",k)
BOUNDARY.set(ReflectingInterface(k=k,ϕ=0))

system = TMM_sSNOM_Simple([SHEET,BOUNDARY],position=1,site=1,units=1e-6)

