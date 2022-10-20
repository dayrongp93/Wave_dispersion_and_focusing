#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  2 09:02:54 2022

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

T = 1.0
dt = 1e-3
num_steps = int(T/dt)

circle = Circle(Point(0,0), radio)
rect = Rectangle(Point(-radio,-radio), Point(radio, 0))
dominio = circle - rect
res = 110
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

f_t = Expression('amp*sin(2*pi*frec*t)*exp(-alfa*(pow(t-T0,2)/pow(T,2)))',
                 degree=1, amp=amp, frec=frec, alfa=alfa, T0=T0, T=T, t=0)

# Condiciones de frontera
tol = 1e-14
def source1(x, on_boundary):
    return on_boundary and x[0] > -1*apertura and x[0] < apertura and x[1] < tol

# Condiciones de frontera
bc = DirichletBC(V, f_t, source1)

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
vtkfile = File('wave_equationFEM_1source_P2/wave_propagationFEM_P2.pvd')
# Solucion del problema
u = Function(V)
t = 0
for n in tqdm(range(num_steps), desc="Loading..."):
    # Actualizamos el tiempo
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
    time.append(t)
    
    # Save solution
    if (n % 5 == 0):
        vtkfile << (u, t)

print("\nComplete.")

