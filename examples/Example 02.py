from qnoe_3ChTMM import * 
import matplotlib.pyplot as plt

# Section 4. Simulation of hBN hyperbolic phonon polaritons

nm = 1e-9						# Units [m]
k = np.arange(1400,1560,1)		# Wavenumber [1/cm] - 1/λ

# From material.py:
# Definition of the polaritonic material (hexagonal boron nitride hBN) [Reference X]
thickness = 38   	# hBN thickness [nm]
isotope = 11 		# hBN isotope

hBN = hexagonalBoronNitride(isotope,thickness,nm)			# Creation of an instance of hexagonalBoronNitride class
M1 = hBN.ModeEffectivePermittivity(k, 1, [1,-10000])		# Calculation of the effective dielectric permittivity ϵ for the mode M1 (Real part is chosen to simulate a lossless system)

# Structure initialization
CHUNK_M1 = Chunk("M1",k,M1,1600,nm)
BOUNDARY = Interface(name="Boundary",k=k).set(ReflectingInterface(k=k,ϕ=120*np.pi/180))		# Phese ϕ upon reflection

Flake_Edge = TMM_sSNOM_Advanced([CHUNK_M1,BOUNDARY],position=800,site=1,units=nm)

sites = [1]
resolution = 10

x,MAP = Flake_Edge.Scan(sites=sites,resolution=resolution)
X,K = np.meshgrid(x,k)

plt.contourf(X,K,np.abs(MAP),100)
plt.ylabel('Wavenumber, cm⁻¹', size=16)
plt.xlabel('X, nm', size=16)
plt.title('Near-field fringes with reflection phase (120º)', size=16)
plt.show()