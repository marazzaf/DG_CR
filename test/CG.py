#coding: utf-8

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time
from ufl import sign,replace
plt.rcParams['image.cmap'] = 'viridis'
import sys, os, sympy, shutil, math
parameters["form_compiler"].update({"optimize": True, "cpp_optimize": True, "representation":"uflacs", "quadrature_degree": 2})
parameters['ghost_mode'] = 'shared_facet'
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import matplotlib.pyplot as plt

comm = MPI.comm_world
rank = comm.rank

#Gmsh mesh. Already cracked
mesh = Mesh()
n_elt = 350
mesh = RectangleMesh(Point(0, -0.5), Point(1,0.5), n_elt, n_elt, "crossed")
cell_size = mesh.hmax()
ndim = mesh.topology().dim() # get number of space dimensions

E, nu = Constant(210), Constant(0.3)
mu    = 0.5*E/(1 + nu)
lmbda = E*nu / (1 - nu*nu)
Gc = Constant(2.7e-3)
ell = Constant(2*cell_size) #0.015) #2*cell_size)
if rank == 0:
    print('\ell: %.3e' % float(ell))
    print(cell_size)
#sys.exit()

boundaries = MeshFunction("size_t", mesh,1)
boundaries.set_all(0)
ds = Measure("ds",subdomain_data=boundaries)

top = CompiledSubDomain("near(x[1], 0.5, 1e-4)")
bottom = CompiledSubDomain("near(x[1], -0.5, 1e-4)")
boundaries = MeshFunction("size_t", mesh,1)
boundaries.set_all(0)
top.mark(boundaries, 1) # mark left as 1
bottom.mark(boundaries, 2) # mark right as 2
ds = Measure("ds",subdomain_data=boundaries)

#To impose alpha=1 on crack
V_alpha = FunctionSpace(mesh, 'CG', 1) #'CR'

def w(alpha):
    """Dissipated energy function as a function of the damage """
    return alpha

def a(alpha):
    """Stiffness modulation as a function of the damage """
    k_ell = Constant(1.e-6) # residual stiffness
    return (1-alpha)**2+k_ell

def eps(u):
    """Strain tensor as a function of the displacement"""
    return sym(grad(u))

def sigma_0(u):
    """Stress tensor of the undamaged material as a function of the displacement"""
    return 2.0*mu*(eps(u)) + lmbda*tr(eps(u))*Identity(ndim)

def sigma(u,alpha):
    """Stress tensor of the damaged material as a function of the displacement and the damage"""
    return (a(alpha))*sigma_0(u)

z = sympy.Symbol("z")
c_w = 4*sympy.integrate(sympy.sqrt(w(z)),(z,0,1))
Gc_eff = Gc * (1 + cell_size/(ell*float(c_w)))

# Create function space for 2D elasticity + Damage
V_u = VectorFunctionSpace(mesh, "CG", 1) #DG
if rank == 0:
    #print('nb dof in disp: %i' % V_u.dim())
    print('nb dof total: %i' % (V_u.dim()+V_alpha.dim()))
#sys.exit()

# Define the function, test and trial fields
u, du, v = Function(V_u, name='disp'), TrialFunction(V_u), TestFunction(V_u)
alpha, dalpha, beta = Function(V_alpha, name='damage'), TrialFunction(V_alpha), TestFunction(V_alpha)

h = CellDiameter(mesh)
h_avg = 0.5 * (h('+') + h('-'))
n = FacetNormal(mesh)

#Dirichlet BC on disp
u_D = Constant((-5e-3,0)) #(-5e-3,0))
bcu_0 = DirichletBC(V_u, u_D, boundaries, 1, method='geometric')
bcu_1 = DirichletBC(V_u, Constant((0.,0.)), boundaries, 2, method='geometric')
bc_u = [bcu_0, bcu_1]

#Energies
pen_value = 2*mu
dissipated_energy = Gc/float(c_w)*(w(alpha)/ell + ell*dot(grad(alpha), grad(alpha)))*dx
elastic_energy = 0.5*inner(sigma(u,alpha), eps(u)) * dx
total_energy =  dissipated_energy + elastic_energy #+ penalty_energy
##Associated bilinear forms
elastic = derivative(elastic_energy,u,v)
elastic = replace(elastic,{u:du})
XX,bb = as_backend_type(assemble(elastic)).mat().getVecs()

# First and second directional derivative wrt alpha
E_alpha = derivative(total_energy,alpha,beta)
E_alpha_alpha = derivative(E_alpha,alpha,dalpha) 

# Damage
bcalpha_0 = DirichletBC(V_alpha, Constant(0), boundaries, 1)
bcalpha_1 = DirichletBC(V_alpha, Constant(0), boundaries, 2) #crack lips
bc_alpha = [bcalpha_0, bcalpha_1]

class DamageProblem():

    def f(self, x):
        """Function to be minimised"""
        alpha.vector()[:] = x
        alpha.vector().apply('insert')
        return assemble(total_energy)

    def F(self, snes, x, F):
        """Gradient (first derivative)"""
        alpha.vector()[:] = x
        alpha.vector().apply('insert')
        F = PETScVector(F)
        return assemble(E_alpha, tensor=F)
    
    def J(self, snes, x, J, P):
        """Hessian (second derivative)"""
        alpha.vector()[:] = x
        alpha.vector().apply('insert')
        J = PETScMatrix(J)
        return assemble(E_alpha_alpha, tensor=J)

pb_alpha = DamageProblem()
solver_alpha = PETSc.SNES().create(comm)
#PETScOptions.set("snes_monitor")
solver_alpha.setTolerances(rtol=1e-5,atol=1e-7) #,max_it=2000)
solver_alpha.setType('vinewtonrsls')
mtf = Function(V_alpha).vector()
solver_alpha.setFunction(pb_alpha.F, mtf.vec())
A = PETScMatrix()
solver_alpha.setJacobian(pb_alpha.J, A.mat())
solver_alpha.getKSP().setType('cg') #cg
solver_alpha.getKSP().getPC().setType('hypre') #hypre
solver_alpha.getKSP().setTolerances(rtol=1e-8) #rtol=1e-6, atol=1e-8)
solver_alpha.getKSP().setFromOptions()
solver_alpha.setFromOptions()
#solver_alpha.view()


lb = interpolate(Constant("0."), V_alpha) # lower bound, initialize to 0
ub = interpolate(Constant("1."), V_alpha) # upper bound, set to 1
for bc in bc_alpha:
    bc.apply(lb.vector())
    bc.apply(ub.vector())

def alternate_minimization(u,alpha,tol=1.e-5,maxiter=100,alpha_0=interpolate(Constant("0.0"), V_alpha)):
    # initialization
    it = 1; err_alpha = 1
    alpha_error = Function(V_alpha)
    solver_alpha.setVariableBounds(lb.vector().vec(),ub.vector().vec())
    en_old = 1e10
    # iteration loop
    while err_alpha>tol and it<maxiter:       
        # solve elastic problem
        solver_u.setOperators(LHS())
        XX = u.copy(deepcopy=True)
        XV = as_backend_type(XX.vector()).vec()
        solver_u.solve(L,XV)
        #solve(PETScMatrix(LHS()), u.vector(), PETScVector(RHS()), "gmres", "ilu")
        try:
            assert solver_u.getConvergedReason() > 0
        except AssertionError:
            if rank == 0:
                print('Error in solver_u: %i' % solver_u.getConvergedReason())
            sys.exit()
        u.vector()[:] = XV
        u.vector().apply('insert')

        #plot(u[0])
        #plt.show()

        #solving damage problem
        xx = alpha.copy(deepcopy=True)
        xv = as_backend_type(xx.vector()).vec()
        solver_alpha.solve(None, xv)
        try:
            assert solver_alpha.getConvergedReason() > 0
        except AssertionError:
            if rank == 0:
                print('Error in solver_alpha: %i' % solver_alpha.getConvergedReason())
            sys.exit()
        alpha.vector()[:] = xv
        alpha.vector().apply('insert')
        alpha_error.vector()[:] = alpha.vector() - alpha_0.vector()
        alpha_error.vector().apply('insert')
        err_alpha = norm(alpha_error.vector(),"linf")

        #plot(alpha)
        #plt.show()

        #monitor the results
        elastic_en = assemble(elastic_energy)
        surface_en = assemble(dissipated_energy)
        en = elastic_en + surface_en
        en_old = en
        if rank == 0:
            print("Iteration:  %2d, Error: %2.8g, Energy: %2.8g" %(it, err_alpha, en))
        # update iteration
        alpha_0.vector()[:] = alpha.vector()
        alpha_0.vector().apply('insert')
        it = it+1
        
    return (err_alpha, it)

#test
solver_u = PETSc.KSP()
solver_u.create(comm)
#PETScOptions.set("ksp_monitor")
solver_u.setType('preonly') #cg
solver_u.getPC().setType('lu') #hypre
solver_u.setTolerances(rtol=1e-5,atol=1e-8) #rtol=1e-8
solver_u.setFromOptions()

#diffusing the phase-field
test = Expression('x[0] > 0.5 && abs(x[1]) < eps ? 1 : 0', eps=0.75*cell_size, degree = 2)
#2*cell_size
#file_u << (interpolate(test, V_alpha), 0)
alpha.vector()[:] = interpolate(test, V_alpha).vector()
alpha.vector().apply('insert')
lb.vector()[:] = alpha.vector() #irreversibility
lb.vector().apply('insert')
#sys.exit()

def LHS():
    LHS = inner(sigma(du,alpha), eps(v)) * dx
    LHS = assemble(LHS)
    for bc in bc_u:
        bc.apply(LHS)
    return as_backend_type(LHS).mat()

RHS = interpolate(Constant((0,0)), V_u).vector()
for bc in bc_u:
    bc.apply(RHS)
L = as_backend_type(RHS).vec()
    
# solve alternate minimization
err,it = alternate_minimization(u,alpha,maxiter=500,tol=1e-4)
    
#postprocessing
F1 = 1e3 * sigma_0(u)[0,0] * ds(1)# +ds(1))
F1 = abs(assemble(F1))
F2 = 1e3 * sigma_0(u)[0,0] * ds(2)# +ds(1))
F2 = abs(assemble(F2))
if rank == 0:
    print(F1)
    print(F2)
    print('Force: %.5e' % (F1+F2))

savedir='CG'
file_alpha = File(savedir+"/alpha.pvd")
file_u = File(savedir+"/u.pvd")

file_alpha << (alpha,0)
file_u << (u,0)

#consistent reaction forces
a = inner(sigma_0(du), eps(v)) * dx

residual = action(a, u)

v_reac = Function(V_u)
bc = DirichletBC(V_u.sub(0), Constant(1.), boundaries, 2, method='geometric')
bc.apply(v_reac.vector())
res = assemble(action(residual, v_reac))
if rank == 0:
    print("Reaction = {}".format(res))
