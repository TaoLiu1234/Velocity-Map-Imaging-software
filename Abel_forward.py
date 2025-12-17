import numpy as np
import matplotlib.pyplot as plt
import abel

rho = 1000
r = np.linspace(0, rho, 500)
f_r = 2*rho/np.sqrt(rho**2 - r**2)

theta = np.arcsin(np.sqrt(rho**2 -r**2)/rho)
f_r2 = 2/np.sin(theta)

# beerlambert law
I0 = 1000
attenuaction_factor = 0.5
x = np.linspace(0,10,1000)
rho = 10
dz = 2 * np.sqrt(rho**2 - x**2)
I_transmitted = I0 * np.exp(-attenuaction_factor * dz)
plt.plot(x,I_transmitted)
plt.xlabel('Radius (pixels)')
plt.ylabel('Transmitted Intensity (a.u.)')
plt.title('Transmitted Intensity vs Radius')
plt.show()

# convolution approach

    
# # integration approach
# dr = np.linspace(0,10,1000)
# # np.sqrt(r^2 + z^2)=10 in this sphere
# rho = 10
# theta = np.linspace(-np.pi/2, np.pi/2, 1000000)
# x = rho * np.cos(theta)
# z = rho * np.sin(theta)
# f = np.ones_like(x)

# I_integrated = np.zeros_like(dr)
# for i in range(len(dr)):
#     if i == 0:
#         mask = (x >= 0) & (x <= dr[i]) 
#     else:
#         mask = (dr[i-1] <= x) & (x <= dr[i])
#     I_integrated[i] = np.sum(f[mask])

# plt.plot(dr, I_integrated)
# plt.xlabel('Radius (pixels)')
# plt.ylabel('Integrated Intensity (a.u.)')
# plt.title('Integrated Intensity vs Radius')
# plt.show()
#plt.plot(r, f_r)
#plt.xlabel('Radius (pixels)')
#plt.ylabel('Intensity (a.u.)')
#plt.title('Abel Forward Transform Input Intensity vs Radius')
#plt.show()
