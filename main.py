from sympy import symbols, Symbol, solve, Eq, lambdify, Matrix
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

x, y, z = symbols('x y z')
k1 = Symbol("k1")
k1m = Symbol("k1m")
k2 = Symbol("k2")
k3 = Symbol("k3")
k3m = Symbol("k3m")


z = 1 - x - y
f1 = k1 * z - k1m * x - k2 * z**2 * x
f2 = k3 * z - k3m * y
system = [Eq(f1, 0), Eq(f2, 0)]
solutions = solve(system, (x, k1))
x_sol0 = solutions[0][0]
k1_sol0 = solutions[0][1]


x_func0 = lambdify((y, k3, k3m), x_sol0)
k1_func0 = lambdify((y, k1m, k2, k3, k3m), k1_sol0)

k1m_vals = [0.001, 0.005, 0.01, 0.015, 0.02]
k2val = 2
k3m_val = 0.003
k3val = 0.0032

y_k = np.linspace(0.001, 0.5, 10000)
plt.figure(figsize=(7, 35))
plt.rc('axes', titlesize= 10)

for i in range(len(k1m_vals)):
    x_val = x_func0(y_k, k3val, k3m_val)
    k1_val = k1_func0(y_k, k1m_vals[i], k2val, k3val, k3m_val)

    x_val, y_permis, k1_val = zip(*[(x, y, z) for x, y, z in zip(x_val, y_k, k1_val) if 0 <= x <= 1])

    plt.subplot(2, 3, i + 1)
    plt.plot(k1_val, x_val, label='$x(k_1)$')
    plt.plot(k1_val, y_permis, label='$y(k_1)$')
    plt.title('$x(k_1)$ и $y(k_1)$ при $k_{-1}$ =' + str(k1m_vals[i]))
    plt.xlabel('$k_1$')
    plt.legend(['$x(k_1)$', '$y(k_1)$'])

plt.show()

k1m_val = 0.03
k3m_vals = [0.0005, 0.001, 0.002, 0.003, 0.004]


plt.figure(figsize=(7, 25))
plt.rc('axes', titlesize= 10)

for i in range(len(k3m_vals)):
    x_val = x_func0(y_k, k3val, k3m_vals[i])
    k1_val = k1_func0(y_k, k1m_val, k2val, k3val, k3m_vals[i])

    x_val, y_permis, k1_val = zip(*[(x, y, z) for x, y, z in zip(x_val, y_k, k1_val) if 0 <= x <= 1])

    plt.subplot(2, 3, i + 1)
    plt.plot(k1_val, x_val, label='$x(k_1)$')
    plt.plot(k1_val, y_permis, label='$y(k_1)$')
    plt.title('$x(k_1)$ и $y(k_1)$ при $k_{-3}$ =' + str(k3m_vals[i]))
    plt.xlabel('$k_1$')
    plt.legend(['$x(k_1)$', '$y(k_1)$'])

plt.show()

F = Matrix([f1, f2])
A = F.jacobian([x, y])
detA = A.det()
traceA = A.trace()


detA_func = lambdify((x, y, k1, k1m, k2, k3, k3m), detA)

traceA_func = lambdify((x, y, k1, k1m, k2, k3, k3m), traceA)

def get_bif_type(x, y, k1_vals, k1m_val, k2val, k3val, k3m_val):
    k1_bif, y_bif, x_bif, markers = [], [], [], []


    detA_vals = [detA_func(x[i], y[i], k1_vals[i], k1m_val, k2val, k3val, k3m_val) for i in range(len(x))]
    trace_vals = [traceA_func(x[i], y[i], k1_vals[i], k1m_val, k2val, k3val, k3m_val) for i in range(len(x))]

    for i in range(len(y) - 1):

        if (detA_vals[i] * detA_vals[i + 1] < 0 or detA_vals[i] == 0):

            tmp_y = y[i] - detA_vals[i] * (y[i + 1] - y[i]) / (detA_vals[i + 1] - detA_vals[i])
            k1_tmp = k1_func0(tmp_y, k1m_val, k2val, k3val, k3m_val)

            if all(abs(k1_tmp - el) >= pow(10, -4) for el in k1_bif):
                x_tmp = x_func0(tmp_y, k3val, k3m_val)
                y_bif.append(tmp_y)
                x_bif.append(x_tmp)
                k1_bif.append(k1_tmp)
                markers.append(1)


        if (trace_vals[i] * trace_vals[i + 1] < 0 or (trace_vals[i] == 0 and detA_vals[i] < 0)):

            tmp_y = y[i] - trace_vals[i] * (y[i + 1] - y[i]) / (trace_vals[i + 1] - trace_vals[i])
            k1_tmp = k1_func0(tmp_y, k1m_val, k2val, k3val, k3m_val)

            if all(abs(k1_tmp - el) >= pow(10, -4) for el in k1_bif):
                x_tmp = x_func0(tmp_y, k3val, k3m_val)
                y_bif.append(tmp_y)
                x_bif.append(x_tmp)
                k1_bif.append(k1_tmp)
                markers.append(2)

    return k1_bif, y_bif, x_bif, markers


plt.figure(figsize=(7, 25))
plt.rc('axes', titlesize= 10)

for j in range(len(k3m_vals)):

    x_val = x_func0(y_k, k3val, k3m_vals[j])
    k1_val = k1_func0(y_k, k1m_val, k2val, k3val, k3m_vals[j])


    x_val, y_permis, k1_val = zip(*[(x, y, z) for x, y, z in zip(x_val, y_k, k1_val) if 0 <= x <= 1])


    k1_bif, y_bif, x_bif, markers = get_bif_type(x_val, y_permis, k1_val, k1m_val, k2val, k3val, k3m_vals[j])


    plt.subplot(2, 3, j + 1)
    plt.plot(k1_val, x_val, label='$x(k_1)$')
    plt.plot(k1_val, y_permis, label='$y(k_1)$')



    k1_s = [k1_bif[i] for i in range(len(k1_bif)) if markers[i] == 1]
    y_s = [y_bif[i] for i in range(len(y_bif)) if markers[i] == 1]
    x_s = [x_bif[i] for i in range(len(x_bif)) if markers[i] == 1]
    plt.plot(k1_s, y_s, 'rs', label='saddle')
    plt.plot(k1_s, x_s, 'rs', label='saddle')



    k1_s = [k1_bif[i] for i in range(len(k1_bif)) if markers[i] == 2]
    y_s = [y_bif[i] for i in range(len(y_bif)) if markers[i] == 2]
    x_s = [x_bif[i] for i in range(len(x_bif)) if markers[i] == 2]
    plt.plot(k1_s, y_s, 'k*', label='hopf')
    plt.plot(k1_s, x_s, 'k*', label='hopf')


    plt.title('$x(k_1)$ и $y(k_1)$ при $k_{-3}$ =' + str(k3m_vals[j]))
    plt.xlabel('$k_1$')
    plt.legend(['$x(k_1)$', '$y(k_1)$', '$saddle$', '$saddle$', '$hopf$', '$hopf$'])

plt.show()


k1val = 0.3


def eval_equation(y, t):
    z = 1 - y[0] - y[1]
    eq1 = k1val * z - k1m_val * y[0] - k2val * (z**2) * y[0]
    eq2 = k3val * z - k3m_val * y[1]
    return [eq1, eq2]


t = np.linspace(0, 2500, 100000)
res = odeint(eval_equation, [0.4, 0.5], t)

resX = res[:, 0]
resY = res[:, 1]

plt.figure(figsize=(7, 10))
plt.rc('axes', titlesize= 10)

plt.subplot(2, 1, 1)
plt.plot(t, resX, label='$x(t)$')
plt.plot(t, resY, label='$y(t)$')
plt.title('Решение системы ОДУ $x(t)$ и $y(t)$')
plt.xlabel('$t$')
plt.legend(['$x(t)$', '$y(t)$'])

plt.subplot(2, 1, 2)
plt.plot(resX, resY, label='$y(x)$')
plt.title('Фазовый портрет системы')
plt.xlabel('$x$')
plt.ylabel('$y$')

plt.show()


F = Matrix([f1, f2])
A = F.jacobian([x, y])
detA = A.det()
traceA = A.trace()

sol_k2 = solve(detA.subs({x: x_sol0, k1: k1_sol0}), k2)
sol_k2_func = lambdify((y, k1m, k3, k3m), sol_k2[0])

sol_k2_tr = solve(traceA.subs({x: x_sol0, k1: k1_sol0}), k2)
sol_k2_func_tr = lambdify((y, k1m, k3, k3m), sol_k2_tr[0])

sol_k1 = k1_sol0.subs({k2: sol_k2[0]})
sol_k1_func = lambdify((y, k1m, k3, k3m), sol_k1)

sol_k1_tr = k1_sol0.subs({k2: sol_k2_tr[0]})
sol_k1_func_tr = lambdify((y, k1m, k3, k3m), sol_k1_tr)

x_val = [x_func0(y_k[i], k3val, k3m_val) for i in range(len(y_k))]

k2_val = [sol_k2_func(y_k[i], k1m_val, k3val, k3m_val) for i in range(len(y_k))]
k1_val = [sol_k1_func(y_k[i], k1m_val, k3val, k3m_val) for i in range(len(y_k))]
x_val, y_permis, k1_val, k2_val = zip(*[(x, y, z, d) for x, y, z, d in zip(x_val, y_k, k1_val, k2_val) if 0 <= x <= 1])
plt.plot(k1_val, k2_val, label='$k_2(k_1)$')


k2_val = [sol_k2_func_tr(y_k[i], k1m_val, k3val, k3m_val) for i in range(len(y_k))]
k1_val = [sol_k1_func_tr(y_k[i], k1m_val, k3val, k3m_val) for i in range(len(y_k))]
x_val, y_permis, k1_val, k2_val = zip(*[(x, y, z, d) for x, y, z, d in zip(x_val, y_k, k1_val, k2_val) if 0 <= x <= 1])
plt.plot(k1_val, k2_val, label='$k_2(k_1)$')

plt.title('Параметрический портрет')
plt.xlabel('$k_1$')
plt.ylabel('$k_2$')
plt.legend(['линии кратности', 'линии нейтральности'])


plt.show()
