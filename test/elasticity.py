#coding: utf-8

from dolfin import *
import numpy as np
import time
from ufl import sign,replace
import sys, os, sympy, shutil, math
parameters['ghost_mode'] = 'shared_facet'
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import matplotlib.pyplot as plt

comm = MPI.comm_world
rank = comm.rank

#Gmsh mesh.
mesh = Mesh()
with XDMFFile("mesh_4.xdmf") as infile:
    infile.read(mesh)
ndim = mesh.topology().dim() # get number of space dimensions
h = mesh.hmax()

##parameters
#mu = 80.77
#lmbda = 121.15
E, nu = Constant(210), Constant(0.3)
mu    = 0.5*E/(1 + nu)
lmbda = E*nu / (1 - nu*nu)

top = CompiledSubDomain("near(x[1], 0.5, 1e-4)")
bottom = CompiledSubDomain("near(x[1], -0.5, 1e-4)")
boundaries = MeshFunction("size_t", mesh,1)
boundaries.set_all(0)
top.mark(boundaries, 1)
bottom.mark(boundaries, 2)
ds = Measure("ds",subdomain_data=boundaries) # left: ds(1), right: ds(2)

def eps(u):
    """Strain tensor as a function of the displacement"""
    return sym(grad(u))

def sigma(u):
    """Stress tensor of the undamaged material as a function of the displacement"""
    return 2.0*mu*(eps(u)) + lmbda*tr(eps(u))*Identity(ndim)

# Create function space for 2D elasticity + Damage
V_u = VectorFunctionSpace(mesh, "CG", 1)
if rank == 0:
    print('nb dof total: %i' % V_u.dim())


# Define the function, test and trial fields
u, du, v = Function(V_u, name='disp'), TrialFunction(V_u), TestFunction(V_u)

#Dirichlet BC
# Displacement
u_D = Constant((-5e-3, 0)) #-5e-3 #-1e-2
bcu_0 = DirichletBC(V_u, u_D, boundaries, 1, method='geometric')
bcu_1 = DirichletBC(V_u, Constant((0.,0.)), boundaries, 2, method='geometric')
bc_u = [bcu_0, bcu_1]

#Setting up solver in disp
solver_u = PETSc.KSP()
solver_u.create(comm)
solver_u.setType('preonly')
solver_u.getPC().setType('lu')
solver_u.setTolerances(rtol=1e-5,atol=1e-8,max_it=5000) #rtol=1e-5,max_it=2000 #rtol=1e-3
solver_u.setFromOptions()

a = inner(sigma(du), grad(v)) * dx
LHS = assemble(a)
for bc in bc_u:
    bc.apply(LHS)
A = as_backend_type(LHS).mat()

l = interpolate(Constant((0,0)), V_u).vector()
for bc in bc_u:
    bc.apply(l)
L = as_backend_type(l).vec()

# solve elastic problem
solver_u.setOperators(A)
XX = u.copy(deepcopy=True)
XV = as_backend_type(XX.vector()).vec()
solver_u.solve(L,XV)
try:
    assert solver_u.getConvergedReason() > 0
except AssertionError:
    if rank == 0:
        print('Error on solver u: %i' % solver_u.getConvergedReason())
        sys.exit()
u.vector()[:] = XV
u.vector().apply('insert')

#img = plot(u[0])
#plt.show()

#post processing
F1 = 1e3 * sigma(u)[0,0] * ds(1)
F1 = abs(assemble(F1))
F2 = 1e3 * sigma(u)[0,0] * ds(2)
F2 = abs(assemble(F2))
if rank == 0:
    print(F1)
    print(F2)
    print('Force: %.5e' % (F1+F2))

savedir='elas'
file_u = File(savedir+"/u.pvd")
file_u << (u,0)

#consistant reaction forces
residual = action(a, u)

v_reac = Function(V_u)
bc = DirichletBC(V_u.sub(0), Constant(1.), boundaries, 2) # DirichletBC(V_u.sub(0), Constant(1.), boundaries, 1)]
#for b in bc:
bc.apply(v_reac.vector())
res = assemble(action(residual, v_reac))
if rank == 0:
    print("Reaction = {}".format(res))
