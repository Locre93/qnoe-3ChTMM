import numpy as np

__version__ = "1.0.0"
__author__ = "Lorenzo Orsini"
__contributors__ = ["Matteo Ceccanti"]

# Quasi-static limit q >> omega*epsion/c
# In FermiEnergy(n): charge carrier density cm-2 
# Create function table
# Comment and refine test code

# Graphene reference: An introduction to graphene plasmonics

# ----------------------------------------------------------------------------------------------------- #
#                                    Functions and class description                                    #
# ----------------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------------- #

def ReflectingInterface(k,ϕ):
	S = ScatteringMatrix(k)

	dim = len(k)

	S.S12 = np.zeros(dim,dtype=complex)
	S.S21 = np.zeros(dim,dtype=complex)
	S.S11 = np.ones(dim,dtype=complex)*np.exp(1j*ϕ)
	S.S22 = np.ones(dim,dtype=complex)*np.exp(1j*ϕ)

	return S

def FermiEnergy(n):
	e = 1.60217663e-19		# Elemntary charge [C]
	ħ = 6.582119569e-16		# Planck constant [eV⋅s] 
	c = 299792458			# Speed of light [m/s]
	vf = c/300				# Fermi velocity [m/s]

	return ħ*vf*100*np.sqrt(np.pi*n)	# Fermi Energy [eV]

class hexagonalBoronNitride:
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

	def DielectricPermittivity(self, k):

		ϵ_xy = self.ϵᴵ[1]*(1 + (self.kᴸᴼ[1]**2 - self.kᵀᴼ[1]**2)/(self.kᵀᴼ[1]**2 - k**2 - 1j*k*self.Γ[1]))
		ϵ_z = self.ϵᴵ[0]*(1 + (self.kᴸᴼ[0]**2 - self.kᵀᴼ[0]**2)/(self.kᵀᴼ[0]**2 - k**2 - 1j*k*self.Γ[0]))

		return [ϵ_xy,ϵ_xy,ϵ_z]

	def MagneticPermeability(self):
		return [1,1,1]

	def ModeCoefficients(self, k, ϵ):

		ϵ_xy = self.DielectricPermittivity(k)[0]
		ϵ_z = self.DielectricPermittivity(k)[2]

		ϕ = np.sqrt(-ϵ_xy/ϵ_z)
		R = [(ϵ_xy - 1j*ϵ[0]*ϕ)/(ϵ_xy + 1j*ϵ[0]*ϕ),(ϵ_xy - 1j*ϵ[1]*ϕ)/(ϵ_xy + 1j*ϵ[1]*ϕ)]
		ρ = (1/np.pi)*(np.angle(R[0]) + np.angle(R[1]))

		return ϕ,ρ,R

	def ModeWavevector(self, k, n, ϵ):

		ϵ_xy = self.DielectricPermittivity(k)[0]
		ϵ_z = self.DielectricPermittivity(k)[2]

		ϕ,ρ,R = self.ModeCoefficients(k,ϵ)

		K = (np.pi/(2*self.thickness))*(ρ + 2*n) - (1j/(2*self.thickness))*np.log(np.abs(R[0])*np.abs(R[1]))
		Q = ((np.real(K)*np.real(ϕ) + np.imag(K)*np.imag(ϕ)) + 1j*(-np.real(K)*np.imag(ϕ) + np.imag(K)*np.real(ϕ)))/(np.abs(ϕ)**2)

		return Q

	def ModeEffectivePermittivity(self, k, n, ϵ):
		return ((0.01/k)*self.ModeWavevector(k, n, ϵ)/(2*np.pi))**2

	def ModeEffectiveRefractionIndex(self, k, n, ϵ):

		N = (0.01/k)*self.ModeWavevector(k, n, ϵ)/(2*np.pi)

		return N

	def ModeProfile(self, k, n, ϵ):

		ϵ_xy = self.DielectricPermittivity(k)[0]
		ϵ_z = self.DielectricPermittivity(k)[2]

		ϕ,_,R = self.ModeCoefficients(k,ϵ)
		Q = self.ModeWavevector(k,n,ϵ)

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

	def ModeWavevector(self,k,Ef,ϵ,Γ=0.0037):
		α = 0.0072973525693		# Fine-structure constant
		ħ = 6.582119569e-16		# Planck constant [eV⋅s]
		c = 299792458			# Speed of light [m/s]

		ϵ = (ϵ[0]+ϵ[1])/2 			# Permittivity average
		Eph = (2*np.pi*ħ*c*k*1e2)	# Photon energy [eV]

		Q = ϵ/(2*α*c*ħ)*(Eph/Ef)*(Eph+1j*Γ)

		return Q				# Graphene plasmon wavevector [1/m]

	def ModeEffectivePermittivity(self,k,Ef,ϵ,Γ=0.0037):
		return ((0.01/k)*self.ModeWavevector(k,Ef,ϵ,Γ)/(2*np.pi))**2

if __name__ == '__main__':

	import matplotlib.pyplot as plt

	# --------------------------- GRAPHENE --------------------------- #

	Sheet = Graphene()

	n = np.arange(1e11,1e13,1e10)	# Charge carrier density [1/cm²]
	plt.plot(n,FermiEnergy(n))
	plt.show()

	Ef = 0.3		# Fermi energy [eV]
	k = 333			# Wavenumber [1/cm] - 1/λ (10THz)
	ϵ = [4,4]		# Dielectric permittivities of the environment [Above,Below]

	Q = Sheet.ModeWavevector(k,Ef,ϵ)
	ϵ_eff = Sheet.ModeEffectivePermittivity(k,Ef,ϵ)

	λp = 2*np.pi/np.real(Q)
	λ = 0.01/k

	print(λp/λ)
	print(np.sqrt(1/n))

	# -------------------- HEXAGONAL BORON NITRIDE -------------------- #

	nm = 1e-9

	thickness = 10     	# hBN thickness [nm]
	isotope = 11 		# hBN isotope
	mode = 0 			# hBN mode

	dz = 0.01     	# Vertical resolution [nm]
	dk = 1 			# Wavenumber resolution [cm⁻¹]
	ϵ = [1,1]		# Dielectric permittivities of the environment [Above,Below]

	layer = hexagonalBoronNitride(isotope,thickness,nm)
	Ex,By,Ez = layer.ModeProfile(1500, mode, ϵ)

	z = np.arange(-10*thickness,10*thickness,dz)*nm
	plt.plot(z/nm, np.abs(Ex(z)))
	plt.ylabel('Field inteity, a.u.')
	plt.xlabel('Z, nm')
	plt.show()

	k = np.arange(1400,1600,dk)
	q = layer.ModeWavevector(k, 0, [1,1])
	plt.plot(np.real(q),k)
	plt.ylabel('Wavenumber, cm⁻¹')
	plt.xlabel('Mode momentum, m⁻¹')
	# plt.show()

	k = np.arange(1400,1600,dk)
	q = layer.ModeWavevector(k, 1, [1,-1000])
	plt.plot(np.real(q),k)
	plt.ylabel('Wavenumber, cm⁻¹')
	plt.xlabel('Mode momentum, m⁻¹')
	plt.show()

	phi,pho,_ = layer.ModeCoefficients(k, [1,1])
	plt.plot(k,np.abs(pho))
	plt.ylabel('Wavenumber, cm⁻¹')
	plt.xlabel('Mode momentum, m⁻¹')
	plt.show()

	n = layer.ModeEffectiveRefractionIndex(k, mode, ϵ)
	plt.plot(np.real(n),k)
	plt.ylabel('Wavenumber, cm⁻¹')
	plt.xlabel('Mode effective refractive index - real part')
	plt.show()

	plt.plot(np.imag(n),k)
	plt.ylabel('Wavenumber, cm⁻¹')
	plt.xlabel('Mode effective refractive index - imaginary part')
	plt.show()




