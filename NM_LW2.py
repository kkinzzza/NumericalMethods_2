import math
import sympy as sp
import numpy as np
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('macosx')

def sin(x):
    return math.sin(x)

def exp(x):
    return math.exp(x)

def cos(x):
    return math.cos(x)

pi = math.pi

x = sp.Symbol('x')
N = 8
a = -1
b = 2
# [-1, 2]
# sin(|x|+2,3^x)
xs = np.array([a+((b-a)*(k-1)/(N-1)) for k in range(1, N+1)])
ys = np.array([sin(abs(i)+2.3**i) for i in xs])

# print(min(xs))
# [print(i) for i in ys]
X1 = np.linspace(xs.min(), xs.max(), 100000)
spline1 = make_interp_spline(xs, ys)
Y1 = spline1(X1)

S = [[0 for i in range(N)] for j in range(N)]
c = [0 for i in range(N)]

h = [0 for i in range(N-1)]
for i in range(N-1):
    h[i] = xs[i+1]-xs[i]

for i in range(1, N-1):
    S[i][i-1] = h[i-1]
    S[i][i] = 2 * (h[i-1]+h[i])
    S[i][i+1] = h[i]
    c[i] = ((ys[i+1]-ys[i])/h[i]) - ((ys[i]-ys[i-1])/h[i-1])

# граничные условия
S[0][0] = 1
S[0][1] = -1
S[N-1][N-2] = -1
S[N-1][N-1] = 1
c[0] = 0
c[N-1] = 0

Sn = np.matrix(S)
Sn_inv = np.linalg.inv(Sn)
cn = np.matrix(c)
cn = cn.transpose()

sigma = np.dot(Sn_inv, cn)

# print(sigma)


k = 0
P = ((x-xs[k])/h[k])*ys[k+1]+((xs[k+1]-x)/h[k])*ys[k]+(h[k])**2*((((x-xs[k])/h[k])**3-(x-xs[k])/h[k])*sigma[k+1]+(((xs[k+1]-x)/h[k])**3-(xs[k+1]-x)/h[k])*sigma[k])

print(sp.simplify(P[0][0]))

colors = ['gold', 'orange', 'darkorange', 'tomato', 'orangered', 'red', 'firebrick']

x1 = [i/100 for i in range(-100, -50, 2)]
x2 = [i/100 for i in range(-66, -18, 2)]
x3 = [i/100 for i in range(-20, 40, 2)]
x4 = [i/100 for i in range(20, 72, 2)]
x5 = [i/100 for i in range(70, 102, 2)]
x6 = [i/100 for i in range(100, 160, 2)]
x7 = [i/100 for i in range(152, 202, 2)]
# print(x1)

plt.plot(x1, [-0.318746346417678*x**2 - 0.644121921825336*x + 0.665388810048865 for x in x1], color=colors[0], label=r'$k=0$')
plt.plot(x2, [1.35606767963305*x**3 + 2.00594110438184*x**2 + 0.684270907202961*x + 0.918416015578064 for x in x2], color=colors[1], label=r'$k=2$')
plt.plot(x3, [-2.5749061273156*x**3 + 0.32123804426099*x**2 + 0.443599041471411*x + 0.906955450543228 for x in x3], color=colors[2], label=r'$k=3$')
plt.plot(x4, [-0.959869192545225*x**3 - 1.06307932839933*x**2 + 0.839118290802932*x + 0.869286950606893 for x in x4], color=colors[3], label=r'$k=4$')
plt.plot(x5, [4.56772264418737*x**3 - 12.9079189785406*x**2 + 9.29971804090384*x - 1.14514156132189 for x in x5], color=colors[4], label=r'$k=5$')
plt.plot(x6, [2.4630673742002*x**3 - 5.6919580528703*x**2 + 1.05290555442349*x + 1.99650129067062 for x in x6], color=colors[5], label=r'$k=6$')
plt.plot(x7, [-8.88178419700125e-16*x**3 + 5.91964528264492*x**2 - 17.1938996871004*x + 11.5543516552784 for x in x7], color=colors[6], label=r'$k=7$')
plt.scatter(xs, ys, color='royalblue', label='Значения')
# plt.plot(X1, Y1, color='midnightblue', label='Аппроксимация')
plt.legend()
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title('Сплайн-интерполяция')
plt.grid(True)
plt.show()
