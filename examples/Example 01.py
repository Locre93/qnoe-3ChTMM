from qnoe_3ChTMM import * 
import matplotlib.pyplot as plt

print("Running the example code code")

# Constants
nm = 1e-9

# Definition of the parameters
dk = 1
k = np.arange(1400,1600,dk)

# Definition of the polaritonic materials
thickness = 30   	# hBN thickness [nm]
isotope = 11 		# hBN isotope

hBN = hexagonalBoronNitride(isotope,thickness,nm)
A0 = hBN.ModeEffectivePermittivity(k, 0, [1,1])
M1 = hBN.ModeEffectivePermittivity(k, 1, [1,-10000])

# Definition of the system structure
LEFT_BOUNDARY = outer_left("M1",k,M1)
SPACING = inner("M1",k,M1,300,nm)
CAVITY = inner("A0",k,A0,200,nm)
RIGHT_BOUNDARY = outer_right("M1",k,M1)

structure = [LEFT_BOUNDARY,SPACING,CAVITY,SPACING,RIGHT_BOUNDARY]

# ------------------------- TMM ---------------------- #

system = TMM(structure)
system.GlobalScatteringMatrix()

#  Claculation of the Transmission coefficient of the scattering matrix
plt.plot(k,abs(system.S.S12))
plt.title('Standard transfer matrix frequency response of the structure')
plt.ylabel('Scattering element S12')
plt.xlabel('Wavenumber, cm⁻¹')
plt.show()