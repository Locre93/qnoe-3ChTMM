from tqdm import tqdm
import numpy as np
import itertools

__version__ = "1.0.0"
__author__ = "Lorenzo Orsini"
__contributors__ = ["Sergi Battle Porro"]
__references__ = ["https://empossible.net/academics/emp5337/"]

# The variable k is the light wavenumber (1/λ) in cm⁻¹

# TO IMPROVE
# Test code: plots generation and labels
# Add functionality: load and save also the k vector
# Plot of the near field response at the detector (oscilloscope)

# URGENT
# Cross-check with old version of the code
# Add detector cut-off
# Grammar check

# TO DO
# 1. Improve the style of the code - methods starting with capital letters, remove the underscores ext...
# 2. Implement periodic boundary conditions in the TMM class
# 3. Implement multimodal scattering matrices

# ----------------------------------------------------------------------------------------------------- #
#                                    Functions and class description                                    #
# ----------------------------------------------------------------------------------------------------- #

# - Functions for fast narray multiplications:
# def levi_cevita_tensor(dim)							# Levi Cevita Tensor rules
# def matrix_array_inverse_3x3(A)						# Inverse of a 3x3 matrix grouped in a 4-dimensional array: Aij⁻¹ of A[:,:,i,j]

# class ScatteringMatrix:(k)							- Scattering matrix class subdivided used in the TMM
# 	def Redheffer(𝐀,𝐁):  								# 𝐂 = 𝐀⊗𝐁 - Update 𝐂 - Redheffer star product
# 	def Redheffer_left(other):  						# 𝐀⊗𝐁 - Update 𝐀 - Redheffer star product
# 	def Redheffer_right(other):  						# 𝐀⊗𝐁 - Update 𝐁 - Redheffer star product
#	def save(FileName)									# Save the scattering matrix to a numpy file (S11,S12,S21,S22)
#	def load(FileName)									# Load the scattering matrix from a numpy file (S11,S12,S21,S22)

# class Section:(name,k)								- Object representing a section of the one-dimensional structure
# 	def update()										# Method to update the inner variables of the section
# 	def load(FileName)									# Method to load a ScatteringMatrix class instance

# class Chunk(Section):(name,k,ϵ,length,units)			- Subclass of Section: class that propagates the modes across the section 
# 	def calculate_Λ()									# Method to calculate the wavelength of the mode propagating through the section
# 	def calculate_X()									# Method to calculate element X (phase accumulation and absorption upon propagation)
# 	def calculate_S()									# Method to calculate the scattering matrix of the section
# 	def update()										# Method to update the inner variables of the section

# class Interface(Section):(name,k)						- Subclass of Section: class that represent a scattering interface for the modes propagating across the section 
# 	def set(S)											# Method to set the scattering matrix of the section 

# class Effective_Chunk(Chunk):(name,k,ϵ,length,units)	- Subclass of Chunk: class that calculates the effective scattering matrix for an inner section of the one-dimensional structure
# 	def calculate_A()									# Method to calculate element A
# 	def calculate_B()									# Method to calculate element B
# 	def calculate_S()									# Method to calculate the effective scattering matrix of the section
# 	def update()										# Method to update the inner variables of the section

# class Effective_Interface(Interface):(name,k,ϵ)		- Subclass of Interface: class that embeds the outer sections of the one-dimensional structure
# 	def calculate_Λ()									# Method to calculate the wavelength of the mode propagating through the section
# 	def calculate_A()									# Method to calculate element A
# 	def calculate_B()									# Method to calculate element B
# 	def update()										# Method to update the inner variables of the section

# class Effective_Interface_Left(Effective_Interface):(name,k,ϵ)	- Subclass of Effective_Interface: class that calculates the effective scattering matrix for the left outer section of the one-dimensional structure
# 	def calculate_S()												# Method to calculate the scattering matrix of the section

# class Effective_Interface_Right(Effective_Interface):(name,k,ϵ)	- Subclass of Effective_Interface: class that calculates the effective scattering matrix for the right outer section of the one-dimensional structure
# 	def calculate_S()												# Method to calculate the scattering matrix of the section

# class Structure:(array_of_sections)					- class which takes an numpy array of sections
# 	def Target(site)									# Method to calculate the array's index of the section corresponding to the given chunk (site)
# 	def Split(position,site,units)						# Method to split the structure at a given chunk (site) into two segments - it returns two instances of the TMM class
# 	def Sample(site,resolution)							# Method to spatially sample a given chunk (site) at a given resolution
# 	def EffectivePermittivity(site)						# Method to get the effective permittivity of a given chunk (site)

# class TMM:(array_of_sections)							- Scattering type transfer matrix method class
# 	def GlobalScatteringMatrix()						# Method to calculate the global scattering matrix of the structure
# 	def ReflectionImpedance()							# Method to calculate the reflection impedence of the structure of the left- and right-hand side

# class TMM_3PD():(array_of_sections,position,site,units)		- 3-port-device scattering type transfer matrix method class
# 	def SplitStructure(position,site,units)						# Method to split the structure at the point of insertion of the 3-port-device
# 	def UpdateM()												# Method to calculate the M matrix necessary to calculate the global 3x3 scattering matrix
# 	def UpdateLR()												# Method to calculate the left- and right-hand side global 2x2 transfer matrices
# 	def Calculate3PD(c,γ)										# Method to calculate the 3-port-device scattering matrix defined with physical simmetries
# 	def GlobalScatteringMatrix(c,γ)								# Method to calculate the global scattering matrix of the structure with a generic 3-port-device
# 	def Scan(sites,resolution,units = 1e-9,c = 0.1, γ = 0)		# Method to calculate the S11 element of the global scattering matrix at different positions of the 3-port-device within the structure

# class TMM_sSNOM(TMM_3PD):(array_of_sections,position,site,units,coupling)	- sSNOM transfer matrix method class
# 	def Calculate3PD()														# Method to calculate the 3-port-device scattering matrix which model the sSNOM tip interaction
# 	def NearField(Eᴮᴳ,harm)													# Method to calculate the near-field optical responce

# class TMM_sSNOM_Simple(TMM_sSNOM):(array_of_sections,position,site,units,coupling = 0.1)		- Most basic sSNOM transfer matrix method class, only the exponential coupling coefficient taken into account
# 	def GlobalScatteringMatrix()																# Method to calculate the global scattering matrix of the structure with the sSNOM 3-port-device
# 	def Scan(self,sites,resolution,Eᴮᴳ = 0,harm = 4,units = 1e-9)								# Method to calculate the near-field optical responce at different positions of the 3-port-device within the structure

# class TMM_sSNOM_Advanced(TMM_sSNOM):(array_of_sections,position,site,units,coupling = 0.1, tip_radius=25e-9)		- Advanced sSNOM transfer matrix method class, exponential coupling coefficient, polarionic wavelenght, tip radius and detector cut-off
# 	def Calculate_c(i)																								# Method to calculate the wavelength-dependent coupling coefficient
# 	def GlobalScatteringMatrix()																					# Method to calculate the wavelength-dependent the global scattering matrix of the structure with the sSNOM 3-port-device
# 	def Scan(self,sites,resolution,Eᴮᴳ = 0,harm = 4,units = 1e-9)													# Method to calculate the near-field optical responce at different positions of the 3-port-device within the structure

# ----------------------------------------------------------------------------------------------------- #

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

# -------------------------------------- CLASS SCATTERING MATRIX -------------------------------------- #

class ScatteringMatrix:

	def __init__(self,k):
		if np.isscalar(k):
			k = [k]

		dim = len(k)

		self.S11 = np.zeros(dim,dtype=complex)
		self.S12 = np.ones(dim,dtype=complex)
		self.S21 = np.ones(dim,dtype=complex)
		self.S22 = np.zeros(dim,dtype=complex)

	def Redheffer(self,𝐀,𝐁):
		D = 1/(1 - 𝐁.S11*𝐀.S22)
		F = 1/(1 - 𝐀.S22*𝐁.S11)

		self.S11 = 𝐀.S11 + 𝐀.S12*D*𝐁.S11*𝐀.S21
		self.S12 = 𝐀.S12*D*𝐁.S12
		self.S21 = 𝐁.S21*F*𝐀.S21
		self.S22 = 𝐁.S22 + 𝐁.S21*F*𝐀.S22*𝐁.S12

	def Redheffer_left(self,other):
		D = 1/(1 - other.S11*self.S22)
		F = 1/(1 - self.S22*other.S11)

		self.S11 = self.S11 + self.S12*D*other.S11*self.S21
		self.S12 = self.S12*D*other.S12
		self.S21 = other.S21*F*self.S21
		self.S22 = other.S22 + other.S21*F*self.S22*other.S12

	def Redheffer_right(self,other):
		D = 1/(1 - self.S11*other.S22)
		F = 1/(1 - other.S22*self.S11)
		
		self.S22 = self.S22 + self.S21*F*other.S22*self.S12
		self.S21 = self.S21*F*other.S21
		self.S12 = other.S12*D*self.S12
		self.S11 = other.S11 + other.S12*D*self.S11*other.S21

	def save(self,FileName):
		with open(FileName, 'wb') as File:
			np.save(File, self.S11)
			np.save(File, self.S12)
			np.save(File, self.S21)
			np.save(File, self.S22)

	def load(self,FileName):
		with open(FileName, 'rb') as File:
			self.S11 = np.load(File)
			self.S12 = np.load(File)
			self.S21 = np.load(File)
			self.S22 = np.load(File)

# ------------------------------------------- CLASS SECTION ------------------------------------------- #

class Section:

	def __init__(self,name,k):
		self.k = k 	
		self.name = name	

		self.S = ScatteringMatrix(self.k)

	def update(self):
		pass

	def load(self,FileName):
		self.S.load(FileName)

class Chunk(Section):

	def __init__(self,name,k,ϵ,length,units):
		super().__init__(name,k)

		self.ϵ = ϵ 
		self.splittable = True

		self.length = length
		self.length_norm = length*units*2*np.pi*k/0.01

		self.Λ = np.zeros(len(ϵ),dtype=complex)
		self.X = np.zeros(len(ϵ),dtype=complex)

	def calculate_Λ(self):
		self.Λ = 1j*np.sqrt(self.ϵ)

	def calculate_X(self):
		self.X = np.exp(self.Λ*self.length_norm)

	def calculate_S(self):
		self.S.S12 = self.X
		self.S.S21 = self.X

	def update(self):
		self.calculate_Λ()
		self.calculate_X()
		self.calculate_S()

class Interface(Section):

	def __init__(self,name,k):
		super().__init__(name,k)

		self.splittable = False

	def set(self,S):
		self.S = S

class Effective_Chunk(Chunk):

	def __init__(self,name,k,ϵ,length,units):
		super().__init__(name,k,ϵ,length,units)

		if np.isscalar(ϵ):
			ϵ = [ϵ]

		self.A = np.zeros(len(ϵ),dtype=complex)
		self.B = np.zeros(len(ϵ),dtype=complex)

	def calculate_A(self):
		self.A = 1 + self.Λ/self.ϵ

	def calculate_B(self):
		self.B = 1 - self.Λ/self.ϵ

	def calculate_S(self):
		AUX = 1/(self.A - self.X*self.B*self.X*self.B/self.A)

		self.S.S11 = AUX*(self.X*self.B*self.X - self.B)
		self.S.S12 = AUX*self.X*(self.A - self.B*self.B/self.A)
		self.S.S21 = self.S.S12
		self.S.S22 = self.S.S11

	def update(self):
		self.calculate_Λ()
		self.calculate_A()
		self.calculate_B()
		self.calculate_X()
		self.calculate_S()

class Effective_Interface(Interface):

	def __init__(self,name,k,ϵ):
		super().__init__(name,k)

		if np.isscalar(ϵ):
			ϵ = [ϵ]

		self.ϵ = ϵ
		self.Λ = np.zeros(len(ϵ),dtype=complex) 		

		self.A = np.zeros(len(ϵ),dtype=complex)
		self.B = np.zeros(len(ϵ),dtype=complex)

	def calculate_Λ(self):
		self.Λ = 1j*np.sqrt(self.ϵ)

	def calculate_A(self):
		self.A = 1 + self.ϵ/self.Λ

	def calculate_B(self):
		self.B = 1 - self.ϵ/self.Λ

	def update(self):
		self.calculate_Λ()
		self.calculate_A()
		self.calculate_B()
		self.calculate_S()

class Effective_Interface_Left(Effective_Interface):

	def __init__(self,name,k,ϵ):
		super().__init__(name,k,ϵ)

	def calculate_S(self):
		self.S.S11 = - self.B/self.A
		self.S.S12 = 2/self.A
		self.S.S21 = (self.A - self.B*self.B/self.A)/2
		self.S.S22 = self.B/self.A

class Effective_Interface_Right(Effective_Interface):

	def __init__(self,name,k,ϵ):
		super().__init__(name,k,ϵ)

	def calculate_S(self):
		self.S.S11 = self.B/self.A
		self.S.S12 = (self.A - self.B*self.B/self.A)/2
		self.S.S21 = 2/self.A
		self.S.S22 = - self.B/self.A

# ------------------------------------------ CLASS STRUCTURE ------------------------------------------ #

class Structure:

	def __init__(self,array_of_sections):
		self.structure = array_of_sections

	def Target(self,site):
		j = 0
		target = None
		for i in range(len(self.structure)):
			if self.structure[i].splittable:
				j = j + 1
			if site == j:
				target = i
				break

		if site > j:
			raise Exception("In function Structure.Split - the chosen site exceeds the chunks available in the structure.")
		else:
			return target

	def Split(self,position,site,units):
		k = self.structure[0].k
		if np.isscalar(k):
			k = [k]

		target = self.Target(site)

		if self.structure[target].length - position <= 0:
			raise Exception("In function Structure.Split - the chosen position exceeds the chunk lenght.")

		if type(self.structure[target]) == Effective_Chunk:
			L = TMM(np.concatenate((self.structure[0:target],[Effective_Chunk(self.structure[target].name,k,self.structure[target].ϵ,position,units)],[Effective_Interface_Right(self.structure[target].name,k,self.structure[target].ϵ)])))
			R = TMM(np.concatenate(([Effective_Interface_Left(self.structure[target].name,k,self.structure[target].ϵ)],[Effective_Chunk(self.structure[target].name,k,self.structure[target].ϵ, self.structure[target].length - position,units)],self.structure[target+1:len(self.structure)])))

		elif type(self.structure[target]) == Chunk:
			L = TMM(np.concatenate((self.structure[0:target],[Chunk(self.structure[target].name,k,self.structure[target].ϵ,position,units)])))
			R = TMM(np.concatenate(([Chunk(self.structure[target].name,k,self.structure[target].ϵ, self.structure[target].length - position,units)],self.structure[target+1:len(self.structure)])))

		return L,R

	def Sample(self,site,resolution):
		return np.arange(resolution,self.structure[self.Target(site)].length,resolution)

	def EffectivePermittivity(self,site):
		return self.structure[self.Target(site)].ϵ

# ---------------------------------------- CLASS TMM & TMM_3PD ---------------------------------------- #

class TMM:

	def __init__(self,array_of_sections):
		self.array_of_sections = array_of_sections
		self.k = array_of_sections[0].k
		self.S_update = False

		self.S = ScatteringMatrix(self.k)

	def GlobalScatteringMatrix(self):
		for section in self.array_of_sections:
			section.update()
			self.S.Redheffer_left(section.S)

		self.S_update = True

	def ReflectionImpedance(self):
		if not self.S_update:
			self.GlobalScatteringMatrix()

		Z_left = (1 + self.S.S11)/(1 - self.S.S11)
		Z_right = (1 + self.S.S22)/(1 - self.S.S22)

		return Z_left, Z_right

class TMM_3PD():

	def __init__(self,array_of_sections,position,site,units):
		self.k = array_of_sections[0].k

		if np.isscalar(self.k):
			self.k = [self.k]

		self.M_update = False
		self.LR_update = False
		self.S3x3_update = False

		self.structure = Structure(array_of_sections)
		self.L,self.R = self.structure.Split(position,site,units)

		self.M11 = np.zeros(shape = (3,3,len(self.k)), dtype = complex)
		self.M12 = np.zeros(shape = (3,3,len(self.k)), dtype = complex) 
		self.M21 = np.zeros(shape = (3,3,len(self.k)), dtype = complex)
		self.M22 = np.zeros(shape = (3,3,len(self.k)), dtype = complex) 

	def SplitStructure(self,position,site,units):
		self.L,self.R = self.structure.Split(position,site,units)
		self.LR_update = False

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
		for site in tqdm(sites):
			positions = self.structure.Sample(site,resolution)
			x = np.concatenate((x,positions + temp + resolution))
			temp = np.max(x)
			for position in positions:
				self.SplitStructure(position,site,units)
				self.UpdateLR()
				self.UpdateM()
				self.Calculate3PD(c,γ)
				AUX = self.GlobalScatteringMatrix(c,γ)

				MAP = np.vstack([MAP,AUX[0,0,:,0]])
				ϵ = np.vstack([ϵ,self.structure.EffectivePermittivity(site)])

		MAP = np.delete(MAP, 0, 0)
		ϵ = np.delete(ϵ, 0, 0)

		return x, np.transpose(MAP), np.transpose(ϵ)

class TMM_sSNOM(TMM_3PD):

	def __init__(self,array_of_sections,position,site,units,coupling):
		super().__init__(array_of_sections,position,site,units)

		self.coupling = coupling
		self.O = np.zeros(len(self.k),dtype=complex)

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

	def NearField(self,Eᴮᴳ,harm):
		N = 500
		B = np.abs(self.GlobalScatteringMatrix()[0,0,:,:] + Eᴮᴳ)**2 

		self.O = np.fft.fft(np.tile(B,N), axis = 1)[:,harm*N]

class TMM_sSNOM_Simple(TMM_sSNOM):

	def __init__(self,array_of_sections,position,site,units,coupling = 0.1):
		super().__init__(array_of_sections,position,site,units,coupling)
		self.c = coupling*np.exp(-(np.cos(np.arange(0,2*np.pi,0.2)) + 1))

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

	def Scan(self,sites,resolution,Eᴮᴳ = 0,harm = 4,units = 1e-9):
		MAP = np.zeros(shape = (1,len(self.k)), dtype = complex)		
		
		x = []
		temp = 0
		for site in tqdm(sites):
			positions = self.structure.Sample(site,resolution)
			x = np.concatenate((x,positions + temp))
			temp = np.max(x)
			for position in positions:
				self.SplitStructure(position,site,units)
				self.UpdateLR()
				self.UpdateM()
				self.NearField(Eᴮᴳ,harm)

				MAP = np.vstack([MAP,self.O])

		MAP = np.delete(MAP,0,0)

		return x, np.transpose(MAP)

class TMM_sSNOM_Advanced(TMM_sSNOM):

	def __init__(self,array_of_sections,position,site,units,coupling = 0.1, tip_radius=25e-9):
		super().__init__(array_of_sections,position,site,units,coupling)

		self.lp = (0.01 / self.k)/(np.real(np.sqrt(self.structure.EffectivePermittivity(site))))

		self.C = ((2*np.pi*(10**2)/self.k)**2)*np.exp(-2*(2*np.pi*(10**2)/self.k)*tip_radius)				# [Reference Y]
		self.C = self.coupling*self.C/max(self.C)															# Normalization

		self.Calculate_c(0)														

	def Calculate_c(self,i):
		self.c = self.C[i]*np.exp(-(np.cos(np.arange(0,2*np.pi,0.2)) + 2)*2*np.pi*60e-9/self.lp[i])			# The offset is wrong but we have to take into account the detector frequency cutoff

	def GlobalScatteringMatrix(self):
		I = np.eye(3,dtype=complex)
		M = np.zeros(shape = (3,3,len(self.k),len(self.c)), dtype = complex)

		if not self.M_update:
			self.UpdateM()

		for i in range(len(self.k)):
			self.Calculate_c(i)
			self.Calculate3PD()	

			for j in range(len(self.c)):
				M[:,:,i,j] = np.matmul(np.matmul(self.M12[:,:,i],self.S3x3[:,:,j]),np.matmul(np.linalg.inv(I - np.matmul(self.M22[:,:,i],self.S3x3[:,:,j])),self.M21[:,:,i]))

		M = np.repeat(np.expand_dims(self.M11, axis=3), repeats=len(self.c), axis=3) + M

		return M

	def Scan(self,sites,resolution,Eᴮᴳ = 0,harm = 4,units = 1e-9):
		MAP = np.zeros(shape = (1,len(self.k)), dtype = complex)		
		
		x = []
		temp = 0
		for site in tqdm(sites):
			positions = self.structure.Sample(site,resolution)
			x = np.concatenate((x,positions + temp))
			temp = np.max(x)

			self.lp = (0.01 / self.k)/(np.real(np.sqrt(self.structure.EffectivePermittivity(site))))

			for position in positions:
				self.SplitStructure(position,site,units)
				self.UpdateLR()
				self.UpdateM()
				self.NearField(Eᴮᴳ,harm)

				MAP = np.vstack([MAP,self.O])

		MAP = np.delete(MAP,0,0)

		return x, np.transpose(MAP)

# --------------------------------------------- TEST CODE --------------------------------------------- #

if __name__ == '__main__':

	import matplotlib.pyplot as plt
	from materials import *

	Equivalence = False 							# Test the equivalence of Chunks and Interfaces vs Effective_Chunks and Effective_Interfaces
	TransferMatrixMethod = False 					# Test the standard TMM implementation with Chunks, Interfaces, Effective_Chunks and Effective_Interfaces (Equivalence test execution needed at least once)
	TransferMatrixMethod_3PD = True 				# Test the 3PD TMM implementation: (Equivalence test execution needed at least once)
	GlobalScatteringMatrix_3PD = True					# GlobalScatteringMatrix c and γ dependence
	Splitting = True									# Splitting of the structure
	Scan_3PD = True										# Scan for Chunks, Interfaces, Effective_Chunks and Effective_Interfaces
	TransferMatrixMethod_sSNOM_Simple = False 		# Test the sSNOM TMM simple implementation: (Equivalence test execution needed at least once)
	Scan_sSNOM_Simple = False 							# Scan 		- for Chunks, Interfaces, Effective_Chunks and Effective_Interfaces
	Spectrum_sSNOM_Simple = False 						# Spectrum 	- for Chunks, Interfaces, Effective_Chunks and Effective_Interfaces
	TransferMatrixMethod_sSNOM_Advanced = False  	# Test the sSNOM TMM advanced implementation: (Equivalence test execution needed at least once)
	Scan_sSNOM_Advanced = False 						# Scan 		- for Chunks, Interfaces, Effective_Chunks and Effective_Interfaces
	Spectrum_sSNOM_Advanced = False 					# Spectrum 	- for Chunks, Interfaces, Effective_Chunks and Effective_Interfaces

	nm = 1e-9							# Units [m]
	k = np.arange(1400,1580,0.5)		# Wavenumber [1/cm] - 1/λ

	# From material.py:
	# Definition of the polaritonic material (hexagonal boron nitride hBN) [Reference X]
	thickness = 30   	# hBN thickness [nm]
	isotope = 11 		# hBN isotope

	hBN = hexagonalBoronNitride(isotope,thickness,nm)				# Creation of an instance of hexagonalBoronNitride class
	A0 = np.real(hBN.ModeEffectivePermittivity(k, 0, [1,1]))		# Calculation of the effective dielectric permittivity ϵ for the mode A0 (Real part is chosen to simulate a lossless system)
	M1 = np.real(hBN.ModeEffectivePermittivity(k, 1, [1,-10000]))	# Calculation of the effective dielectric permittivity ϵ for the mode M1 (Real part is chosen to simulate a lossless system)

	if Equivalence:

		# A0_M1 Interface calculation
		LEFT_BOUNDARY = Effective_Interface_Left("A0",k,A0)
		RIGHT_BOUNDARY = Effective_Interface_Right("M1",k,M1)

		system_A0_M1 = TMM([LEFT_BOUNDARY,RIGHT_BOUNDARY])
		system_A0_M1.GlobalScatteringMatrix()
		system_A0_M1.S.save("Test 01 - effective A0_M1 interface scattering matrix.npy")

		# M1_A0 Interface calculation
		LEFT_BOUNDARY = Effective_Interface_Left("M1",k,M1)
		RIGHT_BOUNDARY = Effective_Interface_Right("A0",k,A0)

		system_M1_A0 = TMM([LEFT_BOUNDARY,RIGHT_BOUNDARY])
		system_M1_A0.GlobalScatteringMatrix()
		system_M1_A0.S.save("Test 01 - effective M1_A0 interface scattering matrix.npy")

		# A0 Chunk calculation
		LEFT_BOUNDARY = Effective_Interface_Left("A0",k,A0)
		CHUNK_A0 = Effective_Chunk("A0",k,A0,300,nm)
		RIGHT_BOUNDARY = Effective_Interface_Right("A0",k,A0)

		system_A0 = TMM([LEFT_BOUNDARY,CHUNK_A0,RIGHT_BOUNDARY])
		system_A0.GlobalScatteringMatrix()
		system_A0.S.save("Test 01 - effective A0 chunk scattering matrix.npy")

		# M1 Chunk calculation
		LEFT_BOUNDARY = Effective_Interface_Left("M1",k,M1)
		CHUNK_M1 = Effective_Chunk("M1",k,M1,200,nm)
		RIGHT_BOUNDARY = Effective_Interface_Right("M1",k,M1)

		system_M1 = TMM([LEFT_BOUNDARY,CHUNK_M1,RIGHT_BOUNDARY])
		system_M1.GlobalScatteringMatrix()
		system_M1.S.save("Test 01 - effective M1 chunk scattering matrix.npy")

		# Comparison between Chunk and Effective_Chunk
		CHUNK_M1 = Chunk("M1",k,M1,200,nm)
		CHUNK_M1.update()

		# Element S21
		plt.figure(figsize=(12,6))
		plt.subplot(121)
		plt.plot(k,np.real(CHUNK_M1.S.S21),'b')
		plt.plot(k,np.real(system_M1.S.S21),'r--',dashes=(5, 10))
		plt.title('Re{S$_{21}$}', size=16)
		plt.xlabel('Wavenumber, cm⁻¹', size=16)
		plt.subplot(122)
		plt.plot(k,np.imag(CHUNK_M1.S.S21),'b')
		plt.plot(k,np.imag(system_M1.S.S21),'r--',dashes=(5, 10))
		plt.title('Im{S$_{21}$}', size=16)
		plt.xlabel('Wavenumber, cm⁻¹', size=16)
		plt.show()

		# Element S12
		plt.figure(figsize=(12,6))
		plt.subplot(121)
		plt.plot(k,np.real(CHUNK_M1.S.S12),'b')
		plt.plot(k,np.real(system_M1.S.S12),'r--',dashes=(5, 10))
		plt.title('Re{S$_{12}$}', size=16)
		plt.xlabel('Wavenumber, cm⁻¹', size=16)
		plt.subplot(122)
		plt.plot(k,np.imag(CHUNK_M1.S.S12),'b')
		plt.plot(k,np.imag(system_M1.S.S12),'r--',dashes=(5, 10))
		plt.title('Im{S$_{12}$}', size=16)
		plt.xlabel('Wavenumber, cm⁻¹', size=16)
		plt.show()

	if TransferMatrixMethod:

		# Effective system
		LEFT = Effective_Interface_Left("M1",k,M1)
		SPACING = Effective_Chunk("M1",k,M1,200,nm)
		CAVITY = Effective_Chunk("A0",k,A0,400,nm)
		RIGHT = Effective_Interface_Right("M1",k,M1)

		Effective_system = TMM([LEFT,SPACING,CAVITY,SPACING,RIGHT])
		Effective_system.GlobalScatteringMatrix()

		# Sequence system
		A0_M1 = Interface("A0_M1",k)
		M1_A0 = Interface("M1_A0",k)
		SPACING = Chunk("M1",k,M1,200,nm)
		CAVITY = Chunk("A0",k,A0,400,nm)

		# Loading
		A0_M1.load("Test 01 - effective A0_M1 interface scattering matrix.npy")
		M1_A0.load("Test 01 - effective M1_A0 interface scattering matrix.npy")

		Sequence_system = TMM([SPACING,M1_A0,CAVITY,A0_M1,SPACING])
		Sequence_system.GlobalScatteringMatrix()

		# Comparison - Reflection and Trasmission
		plt.figure(figsize=(12,6))
		plt.subplot(121)
		plt.plot(k,np.abs(Sequence_system.S.S11),'b')
		plt.plot(k,np.abs(Effective_system.S.S11),'r--',dashes=(5, 10))
		plt.title('Abs{S$_{11}$}', size=16)
		plt.xlabel('Wavenumber, cm⁻¹', size=16)
		plt.subplot(122)
		plt.plot(k,np.abs(Sequence_system.S.S12),'b')
		plt.plot(k,np.abs(Effective_system.S.S12),'r--',dashes=(5, 10))
		plt.title('Abs{S$_{12}$}', size=16)
		plt.xlabel('Wavenumber, cm⁻¹', size=16)
		plt.show()

	if TransferMatrixMethod_3PD:

		position = 200
		site = 2
		units = nm

		# Effective system
		LEFT = Effective_Interface_Left("M1",k,M1)
		SPACING = Effective_Chunk("M1",k,M1,200,nm)
		CAVITY = Effective_Chunk("A0",k,A0,400,nm)
		RIGHT = Effective_Interface_Right("M1",k,M1)

		Effective_system_TMM_3PD = TMM_3PD([LEFT,SPACING,CAVITY,SPACING,RIGHT],position,site,units)
		Effective_system_TMM = TMM([LEFT,SPACING,CAVITY,SPACING,RIGHT])

		# Sequence system
		A0_M1 = Interface("A0_M1",k)
		M1_A0 = Interface("M1_A0",k)
		SPACING = Chunk("M1",k,M1,200,nm)
		CAVITY = Chunk("A0",k,A0,400,nm)

		# Loading
		A0_M1.load("Test 01 - effective A0_M1 interface scattering matrix.npy")
		M1_A0.load("Test 01 - effective M1_A0 interface scattering matrix.npy")

		Sequence_system_TMM_3PD = TMM_3PD([SPACING,M1_A0,CAVITY,A0_M1,SPACING],position,site,units)
		Sequence_system_TMM = TMM([SPACING,M1_A0,CAVITY,A0_M1,SPACING])

		if Splitting:

			Effective_system_TMM_3PD.UpdateLR()
			Sequence_system_TMM_3PD.UpdateLR()

			# Comparison - L and R
			plt.figure(figsize=(12,6))
			plt.subplot(121)
			plt.plot(k,np.real(Sequence_system_TMM_3PD.L.S.S12),'b')
			plt.plot(k,np.real(Effective_system_TMM_3PD.L.S.S12),'r--',dashes=(5, 10))
			plt.title('Re{L$_{12}$}', size=16)
			plt.xlabel('Wavenumber, cm⁻¹', size=16)
			plt.subplot(122)
			plt.plot(k,np.imag(Sequence_system_TMM_3PD.L.S.S12),'b')
			plt.plot(k,np.imag(Effective_system_TMM_3PD.L.S.S12),'r--',dashes=(5, 10))
			plt.title('Im{L$_{12}$}', size=16)
			plt.xlabel('Wavenumber, cm⁻¹', size=16)
			plt.show()

			plt.figure(figsize=(12,6))
			plt.subplot(121)
			plt.plot(k,np.real(Sequence_system_TMM_3PD.R.S.S12),'b')
			plt.plot(k,np.real(Effective_system_TMM_3PD.R.S.S12),'r--',dashes=(5, 10))
			plt.title('Re{R$_{12}$}', size=16)
			plt.xlabel('Wavenumber, cm⁻¹', size=16)
			plt.subplot(122)
			plt.plot(k,np.imag(Sequence_system_TMM_3PD.R.S.S12),'b')
			plt.plot(k,np.imag(Effective_system_TMM_3PD.R.S.S12),'r--',dashes=(5, 10))
			plt.title('Im{R$_{12}$}', size=16)
			plt.xlabel('Wavenumber, cm⁻¹', size=16)
			plt.show()

		if GlobalScatteringMatrix_3PD:

			# Test at different coupling coefficient
			c = [0.,0.2,0.5]

			Effective_system_TMM.GlobalScatteringMatrix()
			Effective_SG_3PD = Effective_system_TMM_3PD.GlobalScatteringMatrix(c=c,γ=0)

			Sequence_system_TMM.GlobalScatteringMatrix()
			Sequence_SG_3PD = Sequence_system_TMM_3PD.GlobalScatteringMatrix(c=c,γ=0)

			plt.figure(figsize=(12,6))

			plt.subplot(1,2,1)
			plt.plot(k,np.abs(Effective_system_TMM.S.S12),'b')
			plt.plot(k,np.abs(Effective_SG_3PD[2,1,:,0]),'r--',dashes=(5, 10))					# No coupling between the far-field and near-field channels (c = 0)
			plt.plot(k,np.abs(Effective_SG_3PD[2,1,:,1]),'g--',dashes=(5, 2))					# Weak coupling between the far-field and near-field channels (c = 0.2)
			plt.plot(k,np.abs(Effective_SG_3PD[2,1,:,2]),'black')								# Strong coupling between the far-field and near-field channels (c = 0.5)
			plt.title('Effective', size=16)
			plt.xlabel('Wavenumber, cm⁻¹', size=16)

			plt.subplot(1,2,2)
			plt.plot(k,np.abs(Sequence_system_TMM.S.S12),'b')
			plt.plot(k,np.abs(Sequence_SG_3PD[2,1,:,0]),'r--',dashes=(5, 10))					# No coupling between the far-field and near-field channels (c = 0)
			plt.plot(k,np.abs(Sequence_SG_3PD[2,1,:,1]),'g--',dashes=(5, 2))					# Weak coupling between the far-field and near-field channels (c = 0.2)
			plt.plot(k,np.abs(Sequence_SG_3PD[2,1,:,2]),'black')								# Strong coupling between the far-field and near-field channels (c = 0.5)
			plt.title('Sequence', size=16)
			plt.xlabel('Wavenumber, cm⁻¹', size=16)
			plt.show()

			# Test at different phase coefficient
			plt.figure(figsize=(12,6))

			S13 = []
			for γ in np.arange(0,2*np.pi,0.1):

				Effective_system_TMM_3PD.S3x3_update = False
				Effective_SG_3PD = Effective_system_TMM_3PD.GlobalScatteringMatrix(c = 0.1,γ = γ)

				S13 = np.append(S13,Effective_system_TMM_3PD.S3x3[0,2,:])

				plt.subplot(1,2,1)
				plt.plot(k,np.abs(Effective_SG_3PD[2,1,:]))
			
			plt.subplot(1,2,1)
			plt.title('Transmission through the right near-field channel', size=16)

			plt.subplot(1,2,2)
			plt.title('3-port device coupling phase', size=16)
			plt.plot(np.angle(S13))
			plt.show()

		if Scan_3PD:

			sites = [1,2,3]
			resolution = 10

			x,MAP,PERM = Effective_system_TMM_3PD.Scan(sites=sites,resolution=resolution)
			X,K = np.meshgrid(x,k)

			plt.figure(figsize=(12,6))

			plt.subplot(1,2,1)
			plt.contourf(X,K,np.abs(MAP),100)

			plt.subplot(1,2,2)
			plt.contourf(X,K,np.real(PERM),100)
			plt.show()

			x,MAP,PERM = Sequence_system_TMM_3PD.Scan(sites=sites,resolution=resolution)
			X,K = np.meshgrid(x,k)

			plt.figure(figsize=(12,6))

			plt.subplot(1,2,1)
			plt.contourf(X,K,np.abs(MAP),100)

			plt.subplot(1,2,2)
			plt.contourf(X,K,np.real(PERM),100)
			plt.show()

	if TransferMatrixMethod_sSNOM_Simple:

		position = 200
		site = 2
		units = nm

		# Effective system
		LEFT = Effective_Interface_Left("M1",k,M1)
		SPACING = Effective_Chunk("M1",k,M1,200,nm)
		CAVITY = Effective_Chunk("A0",k,A0,400,nm)
		RIGHT = Effective_Interface_Right("M1",k,M1)

		Effective_system_TMM_sSNOM_Simple = TMM_sSNOM_Simple([LEFT,SPACING,CAVITY,SPACING,RIGHT],position,site,units)

		# Sequence system
		A0_M1 = Interface("A0_M1",k)
		M1_A0 = Interface("M1_A0",k)
		SPACING = Chunk("M1",k,M1,200,nm)
		CAVITY = Chunk("A0",k,A0,400,nm)

		# Loading
		A0_M1.load("Test 01 - effective A0_M1 interface scattering matrix.npy")
		M1_A0.load("Test 01 - effective M1_A0 interface scattering matrix.npy")

		Sequence_system_TMM_sSNOM_Simple = TMM_sSNOM_Simple([SPACING,M1_A0,CAVITY,A0_M1,SPACING],position,site,units)

		if Spectrum_sSNOM_Simple:

			Eᴮᴳ = 0
			harm = 4

			Effective_system_TMM_sSNOM_Simple.NearField(Eᴮᴳ=Eᴮᴳ,harm=harm)
			Sequence_system_TMM_sSNOM_Simple.NearField(Eᴮᴳ=Eᴮᴳ,harm=harm)

			plt.plot(k,np.abs(Effective_system_TMM_sSNOM_Simple.O))
			plt.plot(k,np.abs(Sequence_system_TMM_sSNOM_Simple.O),'r--',dashes=(5, 10))
			plt.show()

		if Scan_sSNOM_Simple:

			sites = [1,2,3]
			resolution = 10

			x,MAP = Effective_system_TMM_sSNOM_Simple.Scan(sites=sites,resolution=resolution)
			X,K = np.meshgrid(x,k)

			plt.figure(figsize=(12,6))

			plt.subplot(1,2,1)
			plt.contourf(X,K,np.abs(MAP),100)

			x,MAP = Sequence_system_TMM_sSNOM_Simple.Scan(sites=sites,resolution=resolution)
			X,K = np.meshgrid(x,k)

			plt.subplot(1,2,2)
			plt.contourf(X,K,np.abs(MAP),100)
			plt.show()

	if TransferMatrixMethod_sSNOM_Advanced:

		position = 200
		site = 2
		units = nm

		# Effective system
		LEFT = Effective_Interface_Left("M1",k,M1)
		SPACING = Effective_Chunk("M1",k,M1,200,nm)
		CAVITY = Effective_Chunk("A0",k,A0,400,nm)
		RIGHT = Effective_Interface_Right("M1",k,M1)

		Effective_system_TMM_sSNOM_Advanced = TMM_sSNOM_Advanced([LEFT,SPACING,CAVITY,SPACING,RIGHT],position,site,units)

		# Sequence system
		A0_M1 = Interface("A0_M1",k)
		M1_A0 = Interface("M1_A0",k)
		SPACING = Chunk("M1",k,M1,200,nm)
		CAVITY = Chunk("A0",k,A0,400,nm)

		# Loading
		A0_M1.load("Test 01 - effective A0_M1 interface scattering matrix.npy")
		M1_A0.load("Test 01 - effective M1_A0 interface scattering matrix.npy")

		Sequence_system_TMM_sSNOM_Advanced = TMM_sSNOM_Advanced([SPACING,M1_A0,CAVITY,A0_M1,SPACING],position,site,units)

		if Spectrum_sSNOM_Advanced:

			Eᴮᴳ = 0
			harm = 4

			Effective_system_TMM_sSNOM_Advanced.NearField(Eᴮᴳ=Eᴮᴳ,harm=harm)
			Sequence_system_TMM_sSNOM_Advanced.NearField(Eᴮᴳ=Eᴮᴳ,harm=harm)

			plt.plot(k,np.abs(Effective_system_TMM_sSNOM_Advanced.O))
			plt.plot(k,np.abs(Sequence_system_TMM_sSNOM_Advanced.O),'r--',dashes=(5, 10))
			plt.show()

		if Scan_sSNOM_Advanced:

			sites = [1,2,3]
			resolution = 10

			x,MAP = Effective_system_TMM_sSNOM_Advanced.Scan(sites=sites,resolution=resolution)
			X,K = np.meshgrid(x,k)

			plt.figure(figsize=(12,6))

			plt.subplot(1,2,1)
			plt.contourf(X,K,np.abs(MAP),100)

			x,MAP = Sequence_system_TMM_sSNOM_Advanced.Scan(sites=sites,resolution=resolution)
			X,K = np.meshgrid(x,k)

			plt.subplot(1,2,2)
			plt.contourf(X,K,np.abs(MAP),100)
			plt.show()