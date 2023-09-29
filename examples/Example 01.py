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

# plt.plot(np.real(Q),k)
# plt.title("Plasmon dispersion relation",size=16)
# plt.ylabel('Wavenumber, cm⁻¹',size=16)
# plt.xlabel('Momentum, m⁻¹',size=16)
# plt.show()

# plt.plot(np.real(λp*1e6),k)
# plt.title("Plasmon dispersion relation",size=16)
# plt.ylabel('Wavenumber, cm⁻¹',size=16)
# plt.xlabel('Wavelength, μm',size=16)
# plt.show()

# For a 7μm Fabry-Peròt gate defined cavity:
# EFFECT OF COUPLING COEFFICIENT

LEFT = EffectiveInterfaceLeft("Ef 0.2eV",k,ϵ_eff[:,0])
CAVITY = EffectiveChunk("Ef 0.4eV",k,ϵ_eff[:,3],7,1e-6)
RIGHT = EffectiveInterfaceRight("Ef 0.2eV",k,ϵ_eff[:,0])

system_2Ch = TMM([LEFT,CAVITY,RIGHT])
system_2Ch.global_scattering_matrix()

system_3Ch = TMM_3PD([LEFT,CAVITY,RIGHT],position=1.5,site=1,units=1e-6)
system_3Ch_3PD = system_3Ch.global_scattering_matrix(c=[0.,0.2,0.5],γ=0)

plt.plot(k,np.abs(system_2Ch.S.S12),'b',label="TMM WO the 3-port device")
plt.plot(k,np.abs(system_3Ch_3PD[2,1,:,0]),'r--',dashes=(5, 2),label="TMM W the 3-port device - no coupling")			#  (c = 0)
plt.plot(k,np.abs(system_3Ch_3PD[2,1,:,1]),'g--',dashes=(5, 2),label="TMM W the 3-port device - weak coupling")		#  (c = 0.2)
plt.plot(k,np.abs(system_3Ch_3PD[2,1,:,2]),'black',label="TMM W the 3-port device - strong coupling")					#  (c = 0.5)
plt.title('Effect of coupling strength', size=16)
plt.xlabel('Wavenumber, cm⁻¹', size=16)
plt.ylabel('Transmission in the near-field channel', size=16)
plt.legend()
plt.show()

# EFFECT OF PHASE COEFFICIENT
f, (ax1, ax2) = plt.subplots(1,2,sharey=False,figsize=(12,6))

system_3Ch.S3x3_update = False
SG_3PD = system_3Ch.global_scattering_matrix(c = 0.1,γ = np.pi/3)
ax1.plot(k,np.real(SG_3PD[2,1,:]),'b', label='γ = π/3')
ax2.plot(k,np.imag(SG_3PD[2,1,:]),'b', label='γ = π/3')

system_3Ch.S3x3_update = False
SG_3PD = system_3Ch.global_scattering_matrix(c = 0.1,γ = 3*np.pi/4)
ax1.plot(k,np.real(SG_3PD[2,1,:]),'r--',dashes=(5, 10), label='γ = 3π/4')
ax2.plot(k,np.imag(SG_3PD[2,1,:]),'r--',dashes=(5, 10), label='γ = 3π/4')

ax1.set_xlabel('Wavenumber, cm⁻¹', size=16)
ax1.set_title('abs{SG₁₁}', size=16)
ax1.legend()

ax2.set_xlabel('Wavenumber, cm⁻¹', size=16)
ax2.set_title('phase{SG₁₁}', size=16)
ax2.legend()

plt.show()

S13 = []
for γ in np.arange(0,2*np.pi,0.1):

	system_3Ch.S3x3_update = False
	system_3Ch.global_scattering_matrix(c = 0.1,γ = γ)

	S13 = np.append(S13,system_3Ch.S3x3[0,2,:])

plt.plot(np.arange(0,2*np.pi,0.1)/np.pi, np.angle(S13)/np.pi)
plt.title('3-port device coupling phase', size=16)
plt.xlabel('γ, π', size=16)
plt.ylabel('Angle{S₁₃}, π', size=16)
plt.show()