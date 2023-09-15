import numpy as np
import itertools

__version__ = "1.0.0"
__author__ = "Lorenzo Orsini"

# Functions for fast narray multiplications
def levi_cevita_tensor(dim):
    arr=np.zeros(tuple([dim for _ in range(dim)]))
    for x in itertools.permutations(tuple(range(dim))):
        mat = np.zeros((dim, dim), dtype=np.int32)
        for i, j in zip(range(dim), x):
            mat[i, j] = 1
        arr[x]=int(np.linalg.det(mat))
    return arr

def matrix_array_inverse_3x3(A):
    shape = A.shape
    dividend = np.einsum('aij,bmn,mikp,njkp->abkp',levi_cevita_tensor(3),levi_cevita_tensor(3),A,A)
    divisor = np.einsum('cde,fgh,fckp,gdkp,hekp->kp',levi_cevita_tensor(3),levi_cevita_tensor(3),A,A,A)
    return 3*np.divide(dividend,divisor)

def SplitStructure(structure,position,site,units):

	k = structure[0].k
	
	if np.isscalar(k):
		k = [k]

	L = TMM(np.concatenate((structure[0:site],[inner(structure[site].name,k,structure[site].ϵ,position,units)],[outer_right(structure[site].name,k,structure[site].ϵ)])))
	R = TMM(np.concatenate(([outer_left(structure[site].name,k,structure[site].ϵ)],[inner(structure[site].name,k,structure[site].ϵ, structure[site].length - position,units)],structure[site+1:len(structure)])))
	
	return L, R

# ------------------- CLASS SCATTERING MATRIX ------------------- #
class ScatteringMatrix:

	def __init__(self,k):

		if np.isscalar(k):
			k = [k]

		dim = len(k)

		self.S11 = np.zeros(dim,dtype=complex)
		self.S12 = np.ones(dim,dtype=complex)
		self.S21 = np.ones(dim,dtype=complex)
		self.S22 = np.zeros(dim,dtype=complex)

	def Redheffer(self,𝐀,𝐁):  # 𝐂 = 𝐀⊗𝐁 - Update 𝐂 - Redheffer star product

		D = 1/(1 - 𝐁.S11*𝐀.S22)
		F = 1/(1 - 𝐀.S22*𝐁.S11)

		self.S11 = 𝐀.S11 + 𝐀.S12*D*𝐁.S11*𝐀.S21
		self.S12 = 𝐀.S12*D*𝐁.S12
		self.S21 = 𝐁.S21*F*𝐀.S21
		self.S22 = 𝐁.S22 + 𝐁.S21*F*𝐀.S22*𝐁.S12

	def Redheffer_left(self,other):  # 𝐀⊗𝐁 - Update 𝐀 - Redheffer star product

		D = 1/(1 - other.S11*self.S22)
		F = 1/(1 - self.S22*other.S11)

		self.S11 = self.S11 + self.S12*D*other.S11*self.S21
		self.S12 = self.S12*D*other.S12
		self.S21 = other.S21*F*self.S21
		self.S22 = other.S22 + other.S21*F*self.S22*other.S12

	def Redheffer_right(self,other):  # 𝐀⊗𝐁 - Update 𝐁 - Redheffer star product

		D = 1/(1 - self.S11*other.S22)
		F = 1/(1 - other.S22*self.S11)
		
		self.S22 = self.S22 + self.S21*F*other.S22*self.S12
		self.S21 = self.S21*F*other.S21
		self.S12 = other.S12*D*self.S12
		self.S11 = other.S11 + other.S12*D*self.S11*other.S21

# ------------------- CLASS LAYER ------------------- #
class layer:

	def __init__(self,name,k,ϵ):
		self.name = name
		self.k = k 										
		self.ϵ = ϵ 

		if np.isscalar(ϵ):
			ϵ = [ϵ]
							
		self.dim = len(ϵ) 								

		self.Λ = np.zeros(self.dim,dtype=complex) 		

		self.A = np.zeros(self.dim,dtype=complex)
		self.B = np.zeros(self.dim,dtype=complex)
		self.X = np.zeros(self.dim,dtype=complex)

		self.S = ScatteringMatrix(self.k)

	def calculate_Λ(self):
		self.Λ = 1j*np.sqrt(self.ε)

	def update(self):
		self.calculate_Λ()
		self.calculate_A()
		self.calculate_B()
		self.calculate_X()
		self.calculate_S()

class inner(layer):

	def __init__(self,name,k,ϵ,length,units):
		super().__init__(name,k,ϵ)
		self.length = length
		self.length_norm = length*units*2*np.pi*k/0.01

	def calculate_A(self):
		self.A = 1 + self.Λ/self.ϵ

	def calculate_B(self):
		self.B = 1 - self.Λ/self.ϵ

	def calculate_X(self):
		self.X = np.exp(self.Λ*self.length_norm)

	def calculate_S(self):
		AUX = 1/(self.A - self.X*self.B*self.X*self.B/self.A)

		self.S.S11 = AUX*(self.X*self.B*self.X - self.B)
		self.S.S12 = AUX*self.X*(self.A - self.B*self.B/self.A)
		self.S.S21 = self.S.S12
		self.S.S22 = self.S.S11

class outer(layer):

	def __init__(self,name,k,ϵ):
		super().__init__(name,k,ϵ)
		self.cp = np.zeros(self.dim,dtype=complex) 		#
		self.ccp = np.zeros(self.dim,dtype=complex) 	#

	def calculate_A(self):
		self.A = 1 + self.ϵ/self.Λ

	def calculate_B(self):
		self.B = 1 - self.ϵ/self.Λ

	def calculate_X(self):
		pass

class outer_left(outer):

	def __init__(self,name,k,ϵ):
		super().__init__(name,k,ϵ)

	def calculate_S(self):
		self.S.S11 = - self.B/self.A
		self.S.S12 = 2/self.A
		self.S.S21 = (self.A - self.B*self.B/self.A)/2
		self.S.S22 = self.B/self.A

class outer_right(outer):

	def __init__(self,name,k,ϵ):
		super().__init__(name,k,ϵ)

	def calculate_S(self):
		self.S.S11 = self.B/self.A
		self.S.S12 = (self.A - self.B*self.B/self.A)/2
		self.S.S21 = 2/self.A
		self.S.S22 = - self.B/self.A

# ------------------- CLASS TMM ------------------- #

class TMM:

	def __init__(self,structure):
		self.structure = structure
		self.k = structure[0].k
		self.layer_update = False
		self.S_update = False

		self.S = ScatteringMatrix(self.k)

	def LayerUpdate(self):
		
		for layer in self.structure:
			layer.update()

		self.layer_update = True

	def GlobalScatteringMatrix(self):
		
		if not self.layer_update:
			self.LayerUpdate()

		for layer in self.structure:
			self.S.Redheffer_left(layer.S)

		self.S_update = True

	def GlobalScatteringCoefficients(self,cp,ccp):

		self.structure[0].cp = cp
		self.structure[-1].ccp = ccp

		if not self.S_update:
			self.GlobalScatteringMatrix()

		self.structure[0].ccp = self.S.S11*self.structure[0].cp + self.S.S12*self.structure[-1].ccp
		self.structure[-1].cp = self.S.S21*self.structure[0].cp + self.S.S22*self.structure[-1].ccp

	def ReflectionImpedance(self):

		if not self.S_update:
			self.GlobalScatteringMatrix()

		Z_left = (1 + self.S.S11)/(1 - self.S.S11)
		Z_right = (1 + self.S.S22)/(1 - self.S.S22)

		return Z_left, Z_right

class TMM_3PD():

	def __init__(self,structure,position,site,units):
		self.k = structure[0].k

		if np.isscalar(self.k):
			self.k = [self.k]

		self.M_update = False
		self.LR_update = False
		self.S3x3_update = False

		self.structure = structure
		self.SplitStructure(position,site,units)

		self.M11 = np.zeros(shape = (3,3,len(self.k)), dtype = complex)
		self.M12 = np.zeros(shape = (3,3,len(self.k)), dtype = complex) 
		self.M21 = np.zeros(shape = (3,3,len(self.k)), dtype = complex)
		self.M22 = np.zeros(shape = (3,3,len(self.k)), dtype = complex) 

	def SplitStructure(self,position,site,units):
	
		self.L = TMM(np.concatenate((self.structure[0:site],[inner(self.structure[site].name,self.k,self.structure[site].ϵ,position,units)],[outer_right(self.structure[site].name,self.k,self.structure[site].ϵ)])))
		self.R = TMM(np.concatenate(([outer_left(self.structure[site].name,self.k,self.structure[site].ϵ)],[inner(self.structure[site].name,self.k,self.structure[site].ϵ, self.structure[site].length - position,units)],self.structure[site+1:len(self.structure)])))

		self.l = (0.01 / self.k)/np.sqrt((np.abs(np.real(self.structure[site].ϵ)) + np.real(self.structure[site].ϵ))/2)

		self.C = ((2*np.pi*(10**2)/self.k)**2)*np.exp(-2*(2*np.pi*(10**2)/self.k)*25e-9)
		self.C = 0.1*self.C/max(self.C)

	def UpdateM(self):

		if not self.LR_update:
			self.UpdateLR()

		I = np.ones(len(self.k),dtype=complex)
		
		self.M11[1,1,:] = self.L.S.S11
		self.M11[2,2,:] = self.R.S.S22

		self.M12[0,0,:] = I
		self.M12[1,1,:] = self.L.S.S12
		self.M12[2,2,:] = self.R.S.S21
		 
		self.M21[0,0,:] = I
		self.M21[1,1,:] = self.L.S.S21
		self.M21[2,2,:] = self.R.S.S12
		
		self.M22[1,1,:] = self.L.S.S22
		self.M22[2,2,:] = self.R.S.S11

		self.M_update = True

	def UpdateLR(self):
		self.L.GlobalScatteringMatrix()
		self.R.GlobalScatteringMatrix()

		self.LR_update = True
		
	def Calculate3PD(self,c,γ):

		if np.isscalar(c):
			c = [c]

		self.S3x3 = np.zeros(shape = (3,3,len(c)), dtype = complex)

		t = (1+np.sqrt(1-2*(np.power(c,2))))/2
		r = -(np.power(c,2))/(2*t)
		b = -r-t
		Co = np.multiply(np.cos(γ),c) + 1j*np.multiply(np.sin(γ),c)
		Ci = np.multiply(np.cos(γ),c) - 1j*np.multiply(np.sin(γ),c)

		self.S3x3[0,0,:] = b
		self.S3x3[0,1,:] = Co
		self.S3x3[0,2,:] = Co
		self.S3x3[1,0,:] = Ci
		self.S3x3[1,1,:] = r
		self.S3x3[1,2,:] = t
		self.S3x3[2,0,:] = Ci
		self.S3x3[2,1,:] = t
		self.S3x3[2,2,:] = r

		self.S3x3_update = True

	def GlobalScatteringMatrix(self,c,γ):

		if np.isscalar(c):
			c = [c]

		if not self.M_update:
			self.UpdateM()

		if not self.S3x3_update:
			self.Calculate3PD(c,γ)

		I = np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.eye(3,dtype=complex), axis=2), repeats=len(self.k), axis=2), axis=3), repeats=len(c), axis=3)
		M = np.einsum('ijk,jlc->ilkc', self.M22, self.S3x3)  # il is 3x3 matrix, k span over wavenumbers, c span over coupling coefficients
		M = (I - M)
		M = matrix_array_inverse_3x3(M)
		M = np.einsum('ijkc,jlk->ilkc', M, self.M21)
		M = np.einsum('ijc,jlkc->ilkc', self.S3x3, M)
		M = np.einsum('ijk,jlkc->ilkc', self.M12, M)
		M = np.repeat(np.expand_dims(self.M11, axis=3), repeats=len(c), axis=3) + M

		return M

	def Scan(self,sites,resolution,units = 1e-9,c = 0.1, γ = 0):

		MAP = np.zeros(shape = (1,len(self.k)), dtype = complex)
		ϵ = np.ones(shape = (1,len(self.k)), dtype = complex)		
		
		x = []
		temp = -resolution
		for site in sites:
			positions = np.arange(resolution,self.structure[site].length,resolution)
			x = np.concatenate((x,positions + temp + resolution))
			temp = np.max(x)
			for position in positions:
				self.SplitStructure(position,site,units)
				self.UpdateLR()
				self.UpdateM()
				self.Calculate3PD(c,γ)
				AUX = self.GlobalScatteringMatrix(c,γ)

				MAP = np.vstack([MAP,AUX[0,0,:,0]])
				ϵ = np.vstack([ϵ,self.structure[site].ϵ])

		MAP = np.delete(MAP, 0, 0)
		ϵ = np.delete(ϵ, 0, 0)

		return x, np.transpose(MAP), np.transpose(ϵ)

class TMM_sSNOM(TMM_3PD):
	def __init__(self,structure,position,site,units):
		super().__init__(structure,position,site,units)

		self.c = 0.1*np.exp(-(np.cos(np.arange(0,2*np.pi,0.2)) + 1))	# Coupling coefficient to model the sSNOM
		self.O = np.zeros(len(self.k),dtype=complex)					# Near-Field optical response

	def Calculate3PD(self):

		self.S3x3 = np.zeros(shape = (3,3,len(self.c)), dtype = complex)

		t = (1+np.sqrt(1-2*(np.power(self.c,2))))/2
		r = -(np.power(self.c,2))/(2*t)
		b = -r-t
		Co = self.c
		Ci = self.c

		self.S3x3[0,0,:] = b
		self.S3x3[0,1,:] = Co
		self.S3x3[0,2,:] = Co
		self.S3x3[1,0,:] = Ci
		self.S3x3[1,1,:] = r
		self.S3x3[1,2,:] = t
		self.S3x3[2,0,:] = Ci
		self.S3x3[2,1,:] = t
		self.S3x3[2,2,:] = r

		self.S3x3_update = True

	def GlobalScatteringMatrix(self):

		if not self.M_update:
			self.UpdateM()

		if not self.S3x3_update:
			self.Calculate3PD()	

		I = np.repeat(np.expand_dims(np.repeat(np.expand_dims(np.eye(3,dtype=complex), axis=2), repeats=len(self.k), axis=2), axis=3), repeats=len(self.c), axis=3)
		M = np.einsum('ijk,jlc->ilkc', self.M22, self.S3x3)  # ij is 3x3 matrix, k span over wavenumbers, c span over coupling coefficients
		M = (I - M)
		M = matrix_array_inverse_3x3(M)
		M = np.einsum('ijkc,jlk->ilkc', M, self.M21)
		M = np.einsum('ijc,jlkc->ilkc', self.S3x3, M)
		M = np.einsum('ijk,jlkc->ilkc', self.M12, M)
		M = np.repeat(np.expand_dims(self.M11, axis=3), repeats=len(self.c), axis=3) + M

		return M

	def NearField(self,Eᴮᴳ,harm):
		N = 500
		B = np.abs(self.GlobalScatteringMatrix()[0,0,:,:] + Eᴮᴳ)**2 

		self.O = np.fft.fft(np.tile(B,N), axis = 1)[:,harm*N]

	def Scan(self,sites,resolution,Eᴮᴳ,harm,units = 1e-9):

		MAP = np.zeros(shape = (1,len(self.k)), dtype = complex)		
		
		x = []
		temp = 0
		for site in sites:
			positions = np.arange(resolution,self.structure[site].length,resolution)
			x = np.concatenate((x,positions + temp))
			temp = temp + self.structure[site].length

			for position in positions:
				self.SplitStructure(position,site,units)
				self.UpdateLR()
				self.UpdateM()
				self.Calculate3PD()
				self.NearField(Eᴮᴳ,harm)

				MAP = np.vstack([MAP,self.O])

		MAP = np.delete(MAP, 0, 0)

		return x, np.transpose(MAP)

	##################################################################################################

	def Calculate_C(self,i):
		self.c = self.C[i]*np.exp(-(np.cos(np.arange(0,2*np.pi,0.2)) + 2)*2*np.pi*60e-9/self.l[i])			# The offset is wrong but we have to take into account the detector frequency cutoff

	def GlobalScatteringMatrix_loop(self):

		I = np.eye(3,dtype=complex)
		M = np.zeros(shape = (3,3,len(self.k),len(self.c)), dtype = complex)

		if not self.M_update:
			self.UpdateM()

		for i in range(len(self.k)):
			self.Calculate_C(i)
			self.Calculate3PD()	

			for j in range(len(self.c)):
				M[:,:,i,j] = np.matmul(np.matmul(self.M12[:,:,i],self.S3x3[:,:,j]),np.matmul(np.linalg.inv(I - np.matmul(self.M22[:,:,i],self.S3x3[:,:,j])),self.M21[:,:,i]))

		M = np.repeat(np.expand_dims(self.M11, axis=3), repeats=len(self.c), axis=3) + M

		return M

	def NearField_loop(self,Eᴮᴳ,harm):

		N = 500
		B = np.abs(self.GlobalScatteringMatrix_loop()[0,0,:,:] + Eᴮᴳ)**2 

		self.O = np.fft.fft(np.tile(B,N), axis = 1)[:,harm*N]

	def Scan_loop(self,sites,resolution,Eᴮᴳ,harm,units = 1e-9):

		MAP = np.zeros(shape = (1,len(self.k)), dtype = complex)		
		
		x = []
		temp = 0
		for site in sites:
			positions = np.arange(resolution,self.structure[site].length,resolution)
			x = np.concatenate((x,positions + temp))
			temp = temp + self.structure[site].length

			for position in positions:
				self.SplitStructure(position,site,units)
				self.UpdateLR()
				self.UpdateM()
				self.Calculate3PD()
				self.NearField_loop(Eᴮᴳ,harm)

				MAP = np.vstack([MAP,self.O])

		MAP = np.delete(MAP, 0, 0)

		return x, np.transpose(MAP)

# ------------------------------------------ #
#                  Test code                 #
# ------------------------------------------ #

# Functions and class description

# def levi_cevita_tensor(dim):							- Levi Cevita Tensor rules
# def matrix_array_inverse_3x3(A):						- Inverse of a 3x3 matrix grouped in a 4-dimensional array: Aij⁻¹ of A[:,:,i,j]
# def SplitStructure(structure,position,site,units):    - Split a 1-dimensional array of layer objects in two ready to be used as input of TMM class

# class ScatteringMatrix: (k)							- Scattering matrix object subdivided in the four quadrant used in the TMM
# 	def Redheffer(self,𝐀,𝐁):  								# 𝐂 = 𝐀⊗𝐁 - Update 𝐂 - Redheffer star product
# 	def Redheffer_left(self,other):  						# 𝐀⊗𝐁 - Update 𝐀 - Redheffer star product
# 	def Redheffer_right(self,other):  						# 𝐀⊗𝐁 - Update 𝐁 - Redheffer star product

# class layer: (name,k,ϵ)								- Layer object which is caracterized by quantities needed to perform TMM
# 	def calculate_Λ(self):									# Calculate the wavelength of the mode propagating through the layer
# 	def update(self):										# Update the pre-defined to zeros or ones components of the layer

# 	class inner(layer):	+(length,units)					- Finite size Layer
# 		def calculate_A(self):								# Method to calculate element A
# 		def calculate_B(self):								# Method to calculate element B
# 		def calculate_X(self):								# Method to calculate element X
# 		def calculate_S(self):								# Method to calculate the scattering matrix of the layer

# 	class outer(layer):	+()								- Semi-infinite Layer
# 		def calculate_A(self):								# Method to calculate element A
# 		def calculate_B(self):								# Method to calculate element B
# 		def calculate_X(self):								# Method to calculate element X

# 		class outer_left(outer): +()					- Left side semi-infinite Layer
# 			def calculate_S(self):							# Method to calculate the scattering matrix of the layer

# 		class outer_right(outer): +()					- Right side semi-infinite Layer
# 			def calculate_S(self):							# Method to calculate the scattering matrix of the layer

# class TMM: (structure)								- Scattering type transfer matrix method object that need only the structure of the device 
# 	def LayerUpdate(self):									# Calculate all the relevant quantities and scattering matrices of each layer
# 	def GlobalScatteringMatrix(self):						# Calculate the scattering matrix of the structure, this method will check if the layers have been updated
# 	def GlobalScatteringCoefficients(self,cp,ccp):			# Calculate the amplitudes of the reflection and transmission fields depending on the input: cp - propagating wave impinging to the structure from left-hand side and ccp - counter propagating wave impinging to the structure from the right-hand side 
# 	def ReflectionImpedance(self):							# Calculate the reflection impedence of the structure of the left- and right-hand side

# class TMM_3PD(): (structure,position,site,units)		- 3-port-device scattering type transfer matrix method object that need only the structure of the device
# 	def SplitStructure(self,position,site,units):			# Split the structure at the point of insertion of the 3-port-device
# 	def UpdateM(self):										# Calculate the M matrix necessary to calculate the global 3x3 scattering matrix
# 	def UpdateLR(self):										# Calculate the left- and right-hand side global 2x2 transfer matrices
# 	def Calculate3PD(self,c,γ):								# Calculate the 3-port-device scattering matrix defined with physical simmetries
# 	def GlobalScatteringMatrix(self,c,γ):					# Calculate the global scattering matrix of the structure with the 3-port-device
# 	def Scan(self,sites,resolution,units,c,γ):				# Calculate the S11 element of the global scattering matrix at different positions of the 3-port-device within the structure

# 	class TMM_sSNOM(TMM_3PD): +()						- subclass of the 3-port-device scattering type transfer matrix method		
# 		def Calculate3PD(self):									# Calculate the 3-port-device scattering matrix which model the sSNOM tip interaction
# 		def GlobalScatteringMatrix(self):						# Calculate the global scattering matrix of the structure with the 3-port-device
# 		def NearField(self,Eᴮᴳ,harm):							# Calculate the near-field optical responce
# 		def Scan(self,sites,resolution,Eᴮᴳ,harm,units):			# Calculate the near-field optical responce at different positions of the 3-port-device within the structure
#		def Calculate_C(self,i):								# Calculate the wavelength-dependent coupling coefficient
# 		def GlobalScatteringMatrix_loop(self):					# Calculate the wavelength-dependent the global scattering matrix of the structure with the 3-port-device
# 		def NearField_loop(self,Eᴮᴳ,harm):						# Calculate the wavelength-dependent the near-field optical responce
# 		def Scan_loop(self,sites,resolution,Eᴮᴳ,harm,units):	# Calculate the wavelength-dependent the near-field optical responce at different positions of the 3-port-device within the structure

if __name__ == '__main__':

	print("Running the test code")

	# import matplotlib.pyplot as plt
	# import plotly.express as px
	# import PolaritonicEffectiveMaterials as Materials

	# # Constants
	# nm = 1e-9

	# # Definition of the parameters
	# dk = 1
	# k = np.arange(1400,1600,dk)

	# # Definition of the polaritonic materials
	# thickness = 30   	# hBN thickness [nm]
	# isotope = 11 		# hBN isotope

	# hBN = Materials.hexagonalBoronNitride(isotope,thickness,nm)
	# A0 = hBN.ModeEffectivePermittivity(k, 0, [1,1])
	# M1 = hBN.ModeEffectivePermittivity(k, 1, [1,-10000])

	# # Definition of the system structure
	# LEFT_BOUNDARY = outer_left("M1",k,M1)
	# SPACING = inner("M1",k,M1,300,nm)
	# CAVITY = inner("A0",k,A0,200,nm)
	# RIGHT_BOUNDARY = outer_right("M1",k,M1)

	# structure = [LEFT_BOUNDARY,SPACING,CAVITY,SPACING,RIGHT_BOUNDARY]

	# # ------------------------- TMM ---------------------- #

	# system = TMM(structure)
	# system.GlobalScatteringMatrix()

	# #  Claculation of the Transmission coefficient of the scattering matrix
	# plt.plot(k,abs(system.S.S12))
	# plt.title('Standard transfer matrix frequency response of the structure')
	# plt.ylabel('Scattering element S12')
	# plt.xlabel('Wavenumber, cm⁻¹')
	# plt.show()

	# # Calculation of the reflection impedance at the boundary of the cavity
	# position = 0
	# site = 2
	# units = nm

	# system_left, system_right = SplitStructure(structure,position,site,units)

	# Z1,_ = system_right.ReflectionImpedance()
	# _,Z2 = system_left.ReflectionImpedance()

	# plt.plot(k,abs(Z1+Z2))
	# plt.ylabel('Impedence sum')
	# plt.xlabel('Wavenumber, cm⁻¹')
	# plt.show()

	# # ------------------------- TMM 3-port-device ---------------------- #

	# c = [0.01,0.1,0.2,0.3,0.5,0.7]
	# site = 2
	# position = 100
	# units = nm

	# system_3PD = TMM_3PD(structure,position,site,units)

	# # Global scattering matrix at different coupling sterngth
	# M = system_3PD.GlobalScatteringMatrix(c,0)

	# plt.plot(k,abs(M[0,0,:,0]))
	# plt.plot(k,abs(M[0,0,:,1]))
	# plt.plot(k,abs(M[0,0,:,2]))
	# plt.plot(k,abs(M[0,0,:,3]))
	# plt.plot(k,abs(M[0,0,:,4]))
	# plt.plot(k,abs(M[0,0,:,5]))
	# plt.title('Response from the center of the cavity')
	# plt.xlabel('Wavenumber, cm⁻¹')
	# plt.ylabel('Reflected electric field, a.u.')
	# plt.show()

	# # 2D Scan of the structure
	# sites = [1,2,3]
	# x, Map, ϵ = system_3PD.Scan(sites,1)

	# X,K = np.meshgrid(x,k)

	# plt.contourf(X,K,abs(Map), 30)
	# plt.title('Spatial and frequency response of the structure')
	# plt.xlabel('Position, nm')
	# plt.ylabel('Wavenumber, cm⁻¹')
	# plt.show()

	# plt.contourf(X,K,abs(ϵ), 30)
	# plt.title('Spatial and frequency response of the structure')
	# plt.xlabel('Position, nm')
	# plt.ylabel('Wavenumber, cm⁻¹')
	# plt.show()

	# # ------------------------- TMM sSNOM ---------------------- #

	# site = 2
	# position = 100
	# units = nm

	# Eᴮᴳ = 0
	# harm = 4

	# system_sSNOM = TMM_sSNOM(structure,position,site,units)
	# system_sSNOM.NearField(Eᴮᴳ,harm)

	# plt.plot(k,np.abs(system_sSNOM.O))
	# plt.title('Near-field response from the center of the cavity')
	# plt.xlabel('Wavenumber, cm⁻¹')
	# plt.ylabel('O4 amplitude, a.u.')
	# plt.show()

	# sites = [1,2,3]
	# x, Map = system_sSNOM.Scan(sites,10,Eᴮᴳ,harm)

	# X,K = np.meshgrid(x,k)

	# plt.contourf(X,K,abs(Map), 30)
	# plt.title('Near-field spatial and frequency response of the structure')
	# plt.xlabel('Position, nm')
	# plt.ylabel('Wavenumber, cm⁻¹')
	# plt.show()

	# # Wavelength-dependent coupling coefficient

	# x, Map = system_sSNOM.Scan_loop(sites,10,Eᴮᴳ,harm)
	# X,K = np.meshgrid(x,k)

	# plt.contourf(X,K,abs(Map), 30)
	# plt.title('Near-field spatial and frequency response of the structure - LOOP')
	# plt.xlabel('Position, nm')
	# plt.ylabel('Wavenumber, cm⁻¹')
	# plt.show()