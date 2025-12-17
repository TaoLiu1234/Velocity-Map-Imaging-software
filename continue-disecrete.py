import numpy as np
import matplotlib.pyplot as plt

rho = 1000
r = np.pi/2
x = np.linspace(0, np.pi/2, 1000000)
descrete = 2*rho*np.arcsin(x/r)*r

continues = 2*r/np.sqrt(r**2 - x**2)

plt.plot(x, continues, label='Continues')
plt.plot(x, descrete, label='Descrete')
plt.xlabel('Radius (pixels)')
plt.ylabel('Intensity (a.u.)')
plt.title('Continues vs Descrete Intensity vs Radius')
plt.legend()
plt.show()