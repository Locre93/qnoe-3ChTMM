from qnoe_3ChTMM import * 
import matplotlib.pyplot as plt

# Section X. Transfer matrix method framework for a 3-channel scattering process
# Supplement 1

Sheet = Graphene()

Ef = [0.1,0.2,0.3,0.4]				# Fermi energy [eV]
k = np.arange(100,350,0.1) 			# Wavenumber [cm⁻¹] - 1/λ (4.497THz to 10.493THz)
ϵ = [1,1]							# Dielectric permittivities of the environment [Above,Below]


ϵ_eff = np.zeros(shape = (len(k),len(Ef)), dtype = complex)
Q = np.zeros(shape = (len(k),len(Ef)), dtype = complex)
λp = np.zeros(shape = (len(k),len(Ef)), dtype = complex)

for i in range(len(Ef)):
	Q[:,i] = Sheet.mode_wavevector(k,Ef[i],ϵ)							# Plasmon wavevector [m⁻¹]
	ϵ_eff[:,i] = np.real(Sheet.mode_effective_permittivity(k,Ef[i],ϵ))	# Plasmon's effective permittivity
	λp[:,i] = 2*np.pi/np.real(Q[:,i])									# Plasmon wavelength [m]

plt.plot(np.real(Q),k)
plt.title("Plasmon dispersion relation",size=16)
plt.ylabel('Wavenumber, cm⁻¹',size=16)
plt.xlabel('Momentum, m⁻¹',size=16)
plt.show()

plt.plot(np.real(λp*1e6),k)
plt.title("Plasmon dispersion relation",size=16)
plt.ylabel('Wavenumber, cm⁻¹',size=16)
plt.xlabel('Wavelength, μm',size=16)
plt.show()

# For a 7μm Fabry-Peròt gate defined cavity:
# EFFECT OF COUPLING COEFFICIENT

LEFT = EffectiveInterfaceLeft("Ef 0.2eV",k,ϵ_eff[:,0])
CAVITY = EffectiveChunk("Ef 0.4eV",k,ϵ_eff[:,3],7,1e-6)
RIGHT = EffectiveInterfaceRight("Ef 0.2eV",k,ϵ_eff[:,0])

system_2Ch = TMM([LEFT,CAVITY,RIGHT])
system_2Ch.global_scattering_matrix()

system_3Ch = TMM_3PD([LEFT,CAVITY,RIGHT],position=3.5,site=1,units=1e-6)
system_3Ch_3PD = system_3Ch.global_scattering_matrix(c=[0.,0.2,0.5],γ=0)

plt.plot(k,np.abs(system_2Ch.S.S12),'b',label="TMM WO the 3-port device")
plt.plot(k,np.abs(system_3Ch_3PD[2,1,:,0]),'r--',dashes=(5, 2),label="TMM W the 3-port device - no coupling")		#  (c = 0)
plt.plot(k,np.abs(system_3Ch_3PD[2,1,:,1]),'g--',dashes=(5, 2),label="TMM W the 3-port device - weak coupling")		#  (c = 0.2)
plt.plot(k,np.abs(system_3Ch_3PD[2,1,:,2]),'black',label="TMM W the 3-port device - strong coupling")				#  (c = 0.5)
plt.title('Effect of coupling strength', size=16)
plt.xlabel('Wavenumber, cm⁻¹', size=16)
plt.ylabel('Transmission in the near-field channel', size=16)
plt.legend()
plt.show()

# EFFECT OF PHASE COEFFICIENT

SG_11 = np.zeros(shape = (1,len(k)), dtype = complex)
SG_32 = np.zeros(shape = (1,len(k)), dtype = complex)

phase = np.arange(0,2*np.pi,0.1)

for γ in phase:

	system_3Ch.S3x3_update = False
	SG_3PD = system_3Ch.global_scattering_matrix(c = 0.1,γ = γ)

	SG_32 = np.vstack([SG_32,SG_3PD[2,1,:,0]])
	SG_11 = np.vstack([SG_11,SG_3PD[0,0,:,0]])

SG_32 = np.delete(SG_32,0,0)
SG_11 = np.delete(SG_11,0,0)

SG_32 = np.transpose(SG_32)
SG_11 = np.transpose(SG_11)

PHASE,K = np.meshgrid(phase,k)

# Print
plt.contourf(PHASE,K,np.abs(SG_32),100)
plt.title("Transmission amplitude")
plt.ylabel('Wavenumber, cm⁻¹', size=16)
plt.xlabel('Coupling phase γ', size=16)
plt.show()

plt.contourf(PHASE,K,np.angle(SG_32),100,cmap="seismic")
plt.title("Transmission phase")
plt.ylabel('Wavenumber, cm⁻¹', size=16)
plt.xlabel('Coupling phase γ', size=16)
plt.show()

plt.contourf(PHASE,K,np.abs(SG_11),100)
plt.title("s-SNOM scattering amplitude")
plt.ylabel('Wavenumber, cm⁻¹', size=16)
plt.xlabel('Coupling phase γ', size=16)
plt.show()

plt.contourf(PHASE,K,np.angle(SG_11),100,cmap="seismic")
plt.title("s-SNOM scattering phase")
plt.ylabel('Wavenumber, cm⁻¹', size=16)
plt.xlabel('Coupling phase γ', size=16)
plt.show()