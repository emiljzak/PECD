    
import scipy.integrate as integrate
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
flesp = open("/Users/zakemil/Nextcloud/projects/PECD/tests/molecules/h2o/esp_radial",'r')
esp = []
for line in flesp:
    words   = line.split()
    xi       = int(words[0])
    r       = float(words[1])
    v      = float(words[2])
    esp.append([xi,r,v])

esp_int = np.asarray(esp,dtype=float)
esp_int_interp  = interpolate.interp1d(esp_int[:,1], esp_int[:,2],kind='linear')
integral_esp = integrate.quad(lambda x: float(esp_int_interp(x)), 0.011, 5.0)
print(integral_esp)

plt.plot(esp_int[:,1], esp_int[:,2])
plt.show()