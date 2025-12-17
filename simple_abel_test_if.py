import numpy as np
import matplotlib.pyplot as plt
import abel
N = 1024
r0 = 400 # ring position
r1 = 30
I0 = 1000

# 定义一维径向函数 f(r)
r = np.linspace(0, r0, N)
f_r1 = np.sin( (r/r0 ) * np.pi / 2)
f_r2 = np.sin( (r/r1 ) * np.pi / 2)
f_r = f_r1+f_r2
# 构造 2D 网格：R 是每个点的半径
center = N // 2
x = np.arange(N) - center
y = np.arange(N) - center
X, Y = np.meshgrid(x, y)
R_grid = np.sqrt(X**2 + Y**2)

# 将 f(r) 插值到每个 (x,y) 位置上的 R_grid
# 使用 np.interp 对每个像素点插值
ring_intensity = np.interp(R_grid, r, f_r, left=0, right=0)
#find out the negative minimum value and making it as offset
min_value = np.min(ring_intensity)
ring_intensity -= min_value

orrecon, distr = abel.rbasex.rbasex_transform(ring_intensity, direction='backward')
r, I, beta = distr.rIbeta()
plt.figure(figsize=(10, 5))
plt.subplot(1, 4, 1)
plt.imshow(ring_intensity, aspect='auto')
plt.axis('equal')
plt.xlabel('Radius (pixels)')
plt.ylabel('Radius (pixels)')
plt.subplot(1, 4, 2)
plt.plot(ring_intensity[560])
plt.xlabel('Radius (pixels)')
plt.ylabel('Intensity (a.u.)')



plt.subplot(1, 4, 3)
plt.imshow(orrecon, aspect='auto')
plt.axis('equal')
plt.xlabel('Radius (pixels)')
plt.ylabel('Radius (pixels)')

plt.subplot(1, 4, 4)
plt.plot(r, I)
plt.axvline(r0, color='r', linestyle='--', label='Ring position r0')
plt.axvline(r1, color='g', linestyle='--', label='Ring position r1')
plt.legend(loc='upper right')
plt.xlabel('Radius (pixels)')
plt.ylabel('Intensity (normalized)')
plt.title('Simple Abel Test Intensity vs Radius')


plt.show()
print('Done')