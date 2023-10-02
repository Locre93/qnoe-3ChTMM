from qnoe_3ChTMM.main import ScatteringMatrix
import numpy as np

__version__ = "1.0.0"
__author__ = "Lorenzo Orsini"
__contributors__ = ["Matteo Ceccanti"]

# REFERENCES
# For the hBN material constants 	- https://doi.org/10.1038/nmat5047 (Supplementary)
# For graphene plasmons				- https://doi.org/10.1142/9948 (An Introduction to Graphene Plasmonics)

# ----------------------------------------------------------------------------------------------------- #
#                                    Functions and class description                                    #
# ----------------------------------------------------------------------------------------------------- #
# Notes:
# Quasistatic limit - q >> ω⋅ϵ/c
# the variable k is the light wavenumber 1/λ [cm⁻¹]

# def reflecting_interface(k,ϕ):							# returns a ScatteringMatrix class instance 
# def fermi_energy(n):										# returns the graphene Fermi Energy [eV]. Here, n is the charge carrier density [1/cm²]

# class HexagonalBoronNitride:(isotope,thickness,units)
# 	def dielectric_permittivity(k):						# returns the hBN in-plane and out-of-plane permittivities (ϵ_xy,ϵ_xy,ϵ_z)
# 	def magnetic_permeability():						# returns the hBN permeability. 
# 	def mode_coefficients(k,ϵ):							# returns the hBN modes coefficients (ϕ,ρ,R). Here, ϵ is a two-element array of the environment permittivity [Above,Below] - [Reference]
# 	def mode_wavevector(k,n,ϵ):							# returns the hBN hyperbolic phonon polariton wavevector [m⁻¹]. Here, n is the index of the mode
# 	def mode_effective_permittivity(k,n,ϵ):				# returns the hBN hyperbolic phonon polariton effective permittivity ϵ
# 	def mode_effective_refraction_index(k,n,ϵ):			# returns the hBN hyperbolic phonon polariton effective index of refraction
# 	def mode_profile(k,n,ϵ):							# returns the functions of hBN hyperbolic phonon polariton filds: Ex(z), By(z), Ez(z)

# class Graphene:
# 	def mode_wavevector(self,k,Ef,ϵ,Γ=0.0037):					# returns Graphene plasmon wavevector [m⁻¹]
# 	def mode_effective_permittivity(self,k,Ef,ϵ,Γ=0.0037)		# returns Graphene plasmon effective permittivity ϵ

# ----------------------------------------------------------------------------------------------------- #

def reflecting_interface(k,ϕ,r=1):
	S = ScatteringMatrix(k)

	dim = len(k)

	S.S11 = r*np.ones(dim,dtype=complex)*np.exp(1j*ϕ)
	S.S22 = -r*np.ones(dim,dtype=complex)*np.exp(-1j*ϕ)

	S.S12 = np.sqrt(1-r**2)*np.ones(dim,dtype=complex)
	S.S21 = np.sqrt(1-r**2)*np.ones(dim,dtype=complex)

	return S

def fermi_energy(n):
	e = 1.60217663e-19		# Elemntary charge [C]
	ħ = 6.582119569e-16		# Planck constant [eV⋅s] 
	c = 299792458			# Speed of light [m/s]
	vf = c/300				# Fermi velocity [m/s]

	return ħ*vf*100*np.sqrt(np.pi*n)	# Fermi Energy [eV]

class HexagonalBoronNitride:

	def __init__(self,isotope,thickness,units):
		self.isotope = isotope 
		self.thickness = thickness * units

		if isotope == 10:
			self.ϵᴵ = [2.5,5.1]     	# High-frequency Relative permittivity
			self.kᴸᴼ = [845,1650]   	# LO phonon wavenumber cm⁻¹
			self.kᵀᴼ = [785,1394.5]  	# TO phonon wavenumber cm⁻¹
			self.Γ = [1,1.8]         	# Damping constant cm⁻¹
		elif isotope == 11:
			self.ϵᴵ = [3.15,5.32]		# High-frequency Relative permittivity
			self.kᴸᴼ = [814,1608.8] 	# LO phonon wavenumber cm⁻¹
			self.kᵀᴼ = [755,1359.8]		# TO phonon wavenumber cm⁻¹
			self.Γ = [1,2.1]			# Damping constant cm⁻¹ 

		elif isotope == 0:
			self.ϵᴵ = [2.95,4.9]		# High-frequency Relative permittivity
			self.kᴸᴼ = [825,1610] 		# LO phonon wavenumber cm⁻¹
			self.kᵀᴼ = [760,1366.2]		# TO phonon wavenumber cm⁻¹
			self.Γ = [2,7]				# Damping constant cm⁻¹

	def dielectric_permittivity(self,k):

		ϵ_xy = self.ϵᴵ[1]*(1 + (self.kᴸᴼ[1]**2 - self.kᵀᴼ[1]**2)/(self.kᵀᴼ[1]**2 - k**2 - 1j*k*self.Γ[1]))
		ϵ_z = self.ϵᴵ[0]*(1 + (self.kᴸᴼ[0]**2 - self.kᵀᴼ[0]**2)/(self.kᵀᴼ[0]**2 - k**2 - 1j*k*self.Γ[0]))

		return [ϵ_xy,ϵ_xy,ϵ_z]

	def magnetic_permeability(self):
		return [1,1,1]

	def mode_coefficients(self,k,ϵ):

		ϵ_xy = self.dielectric_permittivity(k)[0]
		ϵ_z = self.dielectric_permittivity(k)[2]

		ϕ = np.sqrt(-ϵ_xy/ϵ_z)
		R = [(ϵ_xy - 1j*ϵ[0]*ϕ)/(ϵ_xy + 1j*ϵ[0]*ϕ),(ϵ_xy - 1j*ϵ[1]*ϕ)/(ϵ_xy + 1j*ϵ[1]*ϕ)]
		ρ = (1/np.pi)*(np.angle(R[0]) + np.angle(R[1]))

		return ϕ,ρ,R

	def mode_wavevector(self,k,n,ϵ):

		ϵ_xy = self.dielectric_permittivity(k)[0]
		ϵ_z = self.dielectric_permittivity(k)[2]

		ϕ,ρ,R = self.mode_coefficients(k,ϵ)

		K = (np.pi/(2*self.thickness))*(ρ + 2*n) - (1j/(2*self.thickness))*np.log(np.abs(R[0])*np.abs(R[1]))
		Q = ((np.real(K)*np.real(ϕ) + np.imag(K)*np.imag(ϕ)) + 1j*(-np.real(K)*np.imag(ϕ) + np.imag(K)*np.real(ϕ)))/(np.abs(ϕ)**2)

		return Q

	def mode_effective_permittivity(self,k,n,ϵ):
		return ((0.01/k)*self.mode_wavevector(k,n,ϵ)/(2*np.pi))**2

	def mode_effective_refraction_index(self,k,n,ϵ):

		N = (0.01/k)*self.mode_wavevector(k,n,ϵ)/(2*np.pi)

		return N

	def mode_profile(self, k, n, ϵ):

		ϵ_xy = self.dielectric_permittivity(k)[0]
		ϵ_z = self.dielectric_permittivity(k)[2]

		ϕ,_,R = self.mode_coefficients(k,ϵ)
		Q = self.mode_wavevector(k,n,ϵ)

		r = R[0]*np.exp(-1j*Q*ϕ*self.thickness)
		t = [np.exp(-1j*Q*ϕ*self.thickness/2) + r*np.exp(+1j*Q*ϕ*self.thickness/2), np.exp(+1j*Q*ϕ*self.thickness/2) + r*np.exp(-1j*Q*ϕ*self.thickness/2)]

		def below_hBN_region(z):
			return np.heaviside(-z-self.thickness/2,0.5)

		def hBN_region(z):
			return np.heaviside(z+self.thickness/2,0.5) + np.heaviside(-z+self.thickness/2,0.5) - 1

		def above_hBN_region(z):
			return np.heaviside(z-self.thickness/2,0.5) 

		def Ex(z):
			return below_hBN_region(z)*(t[0]*np.exp(+Q*(z+self.thickness/2))) + hBN_region(z)*(np.exp(+1j*Q*ϕ*z) + r*np.exp(-1j*Q*ϕ*z)) + above_hBN_region(z)*(t[1]*np.exp(-Q*(z-self.thickness/2)))

		def Ez(z):
			return below_hBN_region(z)*(-1j*t[0]*np.exp(+Q*(z+self.thickness/2))) + hBN_region(z)*(-ϕ*(np.exp(+1j*Q*ϕ*z) - r*np.exp(-1j*Q*ϕ*z))) + above_hBN_region(z)*(+1j*t[1]*np.exp(-Q*(z-self.thickness/2)))

		def By(z):
			return below_hBN_region(z)*(100*k*ϵ[0]*(-1j*t[0]*np.exp(+Q*(z+self.thickness/2)))/Q) + hBN_region(z)*(100*k*ϵ_z*(-ϕ*(np.exp(+1j*Q*ϕ*z) - r*np.exp(-1j*Q*ϕ*z)))/Q) + above_hBN_region(z)*(100*k*ϵ[1]*(+1j*t[1]*np.exp(-Q*(z-self.thickness/2)))/Q)

		return Ex, By, Ez

class Graphene:

	def __init__(self):
		pass

	def mode_wavevector(self,k,Ef,ϵ,Γ=0.0037):
		α = 0.0072973525693		# Fine-structure constant
		ħ = 6.582119569e-16		# Planck constant [eV⋅s]
		c = 299792458			# Speed of light [m/s]

		ϵ = (ϵ[0]+ϵ[1])/2 			# Permittivity average
		Eph = (2*np.pi*ħ*c*k*1e2)	# Photon energy [eV]

		Q = ϵ/(2*α*c*ħ)*(Eph/Ef)*(Eph+1j*Γ)

		return Q				# Graphene plasmon wavevector [1/m]

	def mode_effective_permittivity(self,k,Ef,ϵ,Γ=0.0037):
		return ((0.01/k)*self.mode_wavevector(k,Ef,ϵ,Γ)/(2*np.pi))**2

if __name__ == '__main__':

	import matplotlib.pyplot as plt

	# --------------------------- GRAPHENE --------------------------- #

	n = np.arange(1e11,5e13,1e9)
	plt.plot(n,fermi_energy(n))
	plt.xlabel("Charge carrier density, 1/cm²",size=16)
	plt.ylabel("Fermi energy, eV",size=16)
	plt.show()

	Sheet = Graphene()

	# Ef = 0.3		# Fermi energy [eV]
	k = 333			# Wavenumber [cm⁻¹] - 1/λ (10THz)
	ϵ = [4,4]		# Dielectric permittivities of the environment [Above,Below]

	Q = Sheet.mode_wavevector(k,fermi_energy(n),ϵ)	# Plasmon wavevector [m⁻¹]
	λp = 2*np.pi/np.real(Q)							# Plasmon wavelength [m]

	# -------------------- HEXAGONAL BORON NITRIDE -------------------- #

	nm = 1e-9			# units

	thickness = 10     	# hBN thickness [nm]
	isotope = 11 		# hBN isotope
	mode = 0 			# hBN mode

	dz = 0.01     		# Vertical resolution [nm]
	dk = 0.01 			# Wavenumber resolution [cm⁻¹]
	ϵ = [1,1]			# Dielectric permittivities of the environment [Above,Below]

	hBN = HexagonalBoronNitride(isotope,thickness,nm)
	Ex,By,Ez = hBN.mode_profile(1500,mode,ϵ)

	z = np.arange(-10*thickness,10*thickness,dz)*nm
	plt.plot(z/nm, np.abs(Ex(z)))
	plt.vlines(x=+thickness, ymin=0, ymax=2, colors='k',linestyles='dotted')
	plt.vlines(x=-thickness, ymin=0, ymax=2, colors='k',linestyles='dotted')
	plt.title("In-plane electric field profile at 1500cm⁻¹ - Ex(z)",size=16)
	plt.ylabel('Field inteity, a.u.',size=16)
	plt.xlabel('Z, nm',size=16)
	plt.show()

	k = np.arange(1400,1607,dk)

	A0 = hBN.mode_wavevector(k, 0, [1,1])
	A1 = hBN.mode_wavevector(k, 1, [1,1])
	A2 = hBN.mode_wavevector(k, 2, [1,1])
	A3 = hBN.mode_wavevector(k, 3, [1,1])

	plt.plot(np.real(A0),k,label="A0")
	plt.plot(np.real(A1),k,label="A1")
	plt.plot(np.real(A2),k,label="A2")
	plt.plot(np.real(A3),k,label="A3")

	plt.title("Hyperbolic phonon polaritons modes An - dispersion relation",size=16)
	plt.ylabel('Wavenumber, cm⁻¹',size=16)
	plt.xlabel('Mode momentum, m⁻¹',size=16)
	plt.xlim([0,4e9])
	plt.legend()
	plt.show()

	M1 = hBN.mode_wavevector(k, 1, [1,-10000])
	M2 = hBN.mode_wavevector(k, 2, [1,-10000])
	M3 = hBN.mode_wavevector(k, 3, [1,-10000])
	M4 = hBN.mode_wavevector(k, 4, [1,-10000])

	plt.plot(np.real(M1),k,label="M1")
	plt.plot(np.real(M2),k,label="M2")
	plt.plot(np.real(M3),k,label="M3")
	plt.plot(np.real(M4),k,label="M4")

	plt.title("Hyperbolic phonon polaritons modes Mn - dispersion relation",size=16)
	plt.ylabel('Wavenumber, cm⁻¹',size=16)
	plt.xlabel('Mode momentum, m⁻¹',size=16)
	plt.xlim([0,4e9])
	plt.legend()
	plt.show()