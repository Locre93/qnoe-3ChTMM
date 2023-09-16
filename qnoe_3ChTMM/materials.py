import numpy as np

__version__ = "1.0.0"
__author__ = "Lorenzo Orsini"
__contributors__ = ["Matteo Ceccanti"]

class hexagonalBoronNitride:
	def __init__(self, isotope, thickness,units):
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

		ϵ_eff = ((0.01/k)*self.ModeWavevector(k, n, ϵ)/(2*np.pi))**2

		return ϵ_eff

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

if __name__ == '__main__':
	pass

	# from matplotlib import rcParams 
	# import matplotlib.pyplot as plt
	# plt.rcParams['font.family'] = ['sans-serif']

	# nm = 1e-9

	# thickness = 10      # hBN thickness [nm]
	# isotope = 11 		# hBN isotope
	# mode = 0 			# hBN mode

	# dz = 0.01     		# Vertical resolution [nm]
	# dk = 1 				# Wavenumber resolution [cm⁻¹]
	# ϵ = [1,1]		# Dielectric permittivities of the environment [Above,Below]

	# layer = hexagonalBoronNitride(isotope,thickness,nm)
	# Ex,By,Ez = layer.ModeProfile(1500, mode, ϵ)

	# # z = np.arange(-10*thickness,10*thickness,dz)*nm
	# # plt.plot(z/nm, np.abs(Ex(z)))
	# # plt.ylabel('Field inteity, a.u.')
	# # plt.xlabel('Z, nm')
	# # plt.show()

	# k = np.arange(1400,1600,dk)
	# # q = layer.ModeWavevector(k, 0, [1,1])
	# # plt.plot(np.real(q),k)
	# # plt.ylabel('Wavenumber, cm⁻¹')
	# # plt.xlabel('Mode momentum, m⁻¹')
	# # # plt.show()

	# # k = np.arange(1400,1600,dk)
	# # q = layer.ModeWavevector(k, 1, [1,-1000])
	# # plt.plot(np.real(q),k)
	# # plt.ylabel('Wavenumber, cm⁻¹')
	# # plt.xlabel('Mode momentum, m⁻¹')
	# # plt.show()

	# phi,pho,_ = layer.ModeCoefficients(k, [1,1])
	# plt.plot(k,np.abs(pho))
	# plt.ylabel('Wavenumber, cm⁻¹')
	# plt.xlabel('Mode momentum, m⁻¹')
	# plt.show()


	# n = layer.ModeEffectiveRefractionIndex(k, mode, ϵ)
	# plt.plot(np.real(n),k)
	# plt.ylabel('Wavenumber, cm⁻¹')
	# plt.xlabel('Mode effective refractive index - real part')
	# plt.show()

	# plt.plot(np.imag(n),k)
	# plt.ylabel('Wavenumber, cm⁻¹')
	# plt.xlabel('Mode effective refractive index - imaginary part')
	# plt.show()



	# # PolariotnicEffectiveMaterials V1.0.1:

	# k = 1500
	# # q = layer.ModeWavevector(k, mode, ϵ)


	# ϵ_xy = layer.DielectricPermittivity(k)[0]
	# ϵ_z = layer.DielectricPermittivity(k)[2]

	# ϵ = [1,1]			

	# ϕ,_,_ = layer.ModeCoefficients(k,ϵ)

	# qd = np.arange(0.001,10,0.001)

	# R_AM = (ϵ_xy*(1-np.exp(-2*qd))/(1+np.exp(-2*qd)) - 1j*ϵ[0]*ϕ)/(ϵ_xy*(1-np.exp(-2*qd))/(1+np.exp(-2*qd)) + 1j*ϵ[0]*ϕ)
	# R_A = (ϵ_xy - 1j*ϵ[0]*ϕ)/(ϵ_xy + 1j*ϵ[0]*ϕ)*np.ones(np.size(R_AM))

	# plt.plot(qd,np.angle(R_AM))
	# plt.plot(qd,np.angle(R_A))
	# plt.show()






	# def Mode_AM_Coefficients(self, k, ϵ):

	# 	ϵ_xy = self.DielectricPermittivity(k)[0]
	# 	ϵ_z = self.DielectricPermittivity(k)[2]

	# 	dR = (1-np.exp(2*q*d))/(1+np.exp(-2*q*d))

	# 	ϕ = np.sqrt(-ϵ_xy/ϵ_z)
	# 	R = [(ϵ_xy - 1j*ϵ[0]*ϕ)/(ϵ_xy + 1j*ϵ[0]*ϕ),(ϵ_xy - 1j*ϵ[1]*ϕ)/(ϵ_xy + 1j*ϵ[1]*ϕ)]
	# 	ρ = (1/np.pi)*(np.angle(R[0]) + np.angle(R[1]))

	# 	return ϕ,ρ,R

	# def Mode_AM_Wavevector(self, k, n, ϵ):

	# 	ϵ_xy = self.DielectricPermittivity(k)[0]
	# 	ϵ_z = self.DielectricPermittivity(k)[2]

	# 	ϕ,ρ,R = self.ModeCoefficients(k,ϵ)

	# 	K = (np.pi/(2*self.thickness))*(ρ + 2*n) - (1j/(2*self.thickness))*np.log(np.abs(R[0])*np.abs(R[1]))
	# 	Q = ((np.real(K)*np.real(ϕ) + np.imag(K)*np.imag(ϕ)) + 1j*(-np.real(K)*np.imag(ϕ) + np.imag(K)*np.real(ϕ)))/(np.abs(ϕ)**2)

	# 	return Q