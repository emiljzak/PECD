"""Physical constants and unit conversion factors"""

import numpy as np

angstrom_to_au = 1.88973

au_to_ev    = 27.211 
# convert time units to atomic units
time_to_au = {"as" : np.float64(1.0/24.188)}
# 1a.u. (time) = 2.418 e-17s = 24.18 as

ev_to_hz    = 2.41799e14

# convert frequency units to atomic units
freq_to_au  = { "nm" : np.float64(0.057/800.0),
                "Hz": np.float64(1.0/4.13e16)}

# convert electric field from different units to atomic units
field_to_au = { "debye" : np.float64(0.393456),
                "V/cm" : np.float64(1.0/(5.14220652e+9))}

# Planck constants in cm^-1*[time units]
hbar        = { "ps"  : np.float64(5.30883730500664),
                "au"  : np.float64(219474.631588) }

planck     =  6.62606896e-27 # Planck constant
avogno     =  6.0221415E+23  # Avogadro constant
boltz      =  1.380658E-16   # Boltzmann constant
vellgt     =  2.99792458E+8  # m/s Speed of light constant
epsilon0   =  8.85e-12 