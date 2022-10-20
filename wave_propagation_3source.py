#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 12:15:06 2022

@author: dayron
"""

from fenics import *
from mshr import *
from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt

# Geometria del problema
radio = 3
apertura = 0.5
c = 5

T = 1.15
dt = 1e-3
num_steps = int(T/dt)

circle = Circle(Point(0,0), radio)
hoyo1 = Circle(Point(0,1.5), 0.1)
rect = Rectangle(Point(-radio,-radio), Point(radio, 0))
dominio = circle - rect - hoyo1
res = 70
mesh = generate_mesh(dominio, res)
print("Number of cells in mesh: ", mesh.num_cells())

# Definicion del espacio de funciones
V = FunctionSpace(mesh, 'Lagrange', 2)

# Definicion del pulso
frec = 1000.0
alfa = 2.5e-2
amp = 5e-3
T0 = 5e-2
T = 2e-3

# Para el desfasaje del pulso
fi = 10
des = fi*dt

fc_t = Expression('amp*sin(2*pi*frec*(t-des))*exp(-alfa*(pow(t-des-T0,2)/pow(T,2)))',
                 degree=1, amp=amp, frec=frec, alfa=alfa, T0=T0, T=T, des=des, t=0)

fl_t = Expression('amp*sin(2*pi*frec*(t-des))*exp(-alfa*(pow(t-des-T0,2)/pow(T,2)))',
                 degree=1, amp=amp, frec=frec, alfa=alfa, T0=T0, T=T, des=0, t=0)

fr_t = Expression('amp*sin(2*pi*frec*(t-des))*exp(-alfa*(pow(t-des-T0,2)/pow(T,2)))',
                 degree=1, amp=amp, frec=frec, alfa=alfa, T0=T0, T=T, des=0, t=0)

# Condiciones de frontera
tol = 1e-14
def source1(x, on_boundary):
    # Center
    return on_boundary and x[0] > -1*apertura and x[0] < apertura and x[1] < tol

def source2(x, on_boundary):
    # Left
    return on_boundary and x[0] > (-1*apertura-0.1-apertura) and x[0] < (-1*apertura - 0.1) and x[1] < tol

def source3(x, on_boundary):
    # Right
    return on_boundary and x[0] > (apertura+0.1) and x[0] < (apertura + 0.1 + apertura) and x[1] < tol


# Condiciones de frontera
bc1 = DirichletBC(V, fc_t, source1)
bc2 = DirichletBC(V, fl_t, source2)
bc3 = DirichletBC(V, fr_t, source3)

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

# Para no ver el cartel de "Solucion al sistema de ecuaciones lineales"
set_log_active(False)

vtkfile = File('wave_FEM_3source_P2_circ15/wave_FEM_3source_P2_circ15.pvd')
# Solucion del problema
u = Function(V)
t = 0
for n in tqdm(range(num_steps), desc="Loading..."):
    # Actualizamos el tiempo
    t += dt
    fl_t.t = t
    fc_t.t = t
    fr_t.t = t
    
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
    p.append((u(0,0) + u(-apertura/2,0) + u(apertura/2,0) + u(-apertura,0) + u(apertura,0))/5)
    time.append(t)
    
    # Save solution
    if (n % 5 == 0):
        vtkfile << (u, t)

data = np.array([time,p])
data = data.T
np.savetxt('data_circ15.txt',data)
print("\nComplete.")