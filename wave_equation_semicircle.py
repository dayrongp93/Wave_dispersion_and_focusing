#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 21:58:33 2022

@author: dayron
"""

from fenics import *
from mshr import *
from tqdm import tqdm 
import matplotlib.pyplot as plt

# Geometria del problema
radio = 3
apertura = 0.5
c = 5

T = 2.0
dt = 1e-3
num_steps = int(T/dt)

circle = Circle(Point(0,0), radio)
hoyo1 = Circle(Point(0,3*radio/4), 0.05)
hoyo2 = Circle(Point(1,3*radio/4), 0.05)
hoyo3 = Circle(Point(-1,3*radio/4), 0.05)
rect = Rectangle(Point(-radio,-radio), Point(radio, 0))
dominio = circle - hoyo1 - hoyo2 - hoyo3 - rect
res = 70
mesh = generate_mesh(dominio, res)

# Definicion del espacio de funciones
V = FunctionSpace(mesh, 'Lagrange', 2)

# Definicion del pulso
frec = 1000.0
alfa = 2.5e-2
amp = 1e-3
T0 = 2.3e-3
T = 2e-3

f_t = Expression('amp*sin(2*pi*frec*t)*exp(-alfa*(pow(t-T0,2)/pow(T,2)))',
                 degree=1, amp=amp, frec=frec, alfa=alfa, T0=T0, T=T, t=0)

f_t2 = Expression('amp*sin(2*pi*frec*t)*exp(-alfa*(pow(t-T0,2)/pow(T,2)))',
                 degree=1, amp=amp, frec=frec, alfa=alfa, T0=T0, T=T, t=0)

# Condiciones de frontera
tol = 1e-14
def top_boudary(x, on_boundary):
    return on_boundary and x[0]**2 + x[1]**2 - radio**2 < tol

def left_boundary(x, on_boundary):
    return on_boundary and x[0] > -1*radio and x[0] < -1*apertura and x[1] < tol

def right_boundary(x, on_boundary):
    return on_boundary and x[0] < radio and x[0] > apertura and x[1] < tol

def source1(x, on_boundary):
    return on_boundary and x[0] > -1*apertura and x[0] < apertura and x[1] < tol

def source2(x, on_boundary):
    return on_boundary and x[0] > (-1*apertura-0.1-apertura) and x[0] < (-1*apertura - 0.1) and x[1] < tol

def source3(x, on_boundary):
    return on_boundary and x[0] > (apertura+0.1) and x[0] < (apertura + 0.1 + apertura) and x[1] < tol

# Condiciones de frontera
bc1 = DirichletBC(V, f_t, source1)
bc2 = DirichletBC(V, f_t2, source2)
bc3 = DirichletBC(V, f_t2, source3)

bc = [bc1, bc2, bc3]

# Definimos las condiciones iniciales
u_i1 = Function(V)
u_i2 = Function(V)
u_i3 = Function(V)

g = Expression('0.0', degree=0)

u_i1 = interpolate(g, V)
u_i2 = interpolate(g, V)
u_i3 = interpolate(g, V)

fi = Function(V)
fi = 5*u_i1 - 4*u_i2 + u_i3

# Definicion del problema variacional
u = TrialFunction(V)
v = TestFunction(V)

a = 2*u*v*dx + c**2*dt**2*dot(grad(u), grad(v))*dx
L = fi*v*dx

# Definimos los vectores de presion y tiempo
p = []
time = []

set_log_active(False)
vtkfile = File('wave_equation/wave.pvd')
# Solucion del problema
u = Function(V)
t = 0
t1 = 0
for n in tqdm(range(num_steps), desc="Loading..."):
    # Actualizamos el tiempo
    t1 += dt
    f_t2.t = t1
    if (n<20):
        f_t.t = 0
    else:
        t += dt
        f_t.t = t
    
    # Resolvemos el problema
    problem = LinearVariationalProblem(a, L, u, bc)
    solver = LinearVariationalSolver(problem)
    solver.parameters["linear_solver"] = "gmres"
    solver.parameters["preconditioner"] = "ilu"
    solver.solve()
    
    #solve(a == L, u, bc)
    
    # Actualizamos las soluciones previas
    u_i3.assign(u_i2)
    u_i2.assign(u_i1)
    u_i1.assign(u)
    
    #Recording
    p.append(u(0,0))
    time.append(t1)
    
    # Save solution
    if (n % 5 == 0):
        vtkfile << (u, t)

print("\nComplete.")
plt.plot(time, p)
