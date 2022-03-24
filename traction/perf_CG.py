#coding: utf-8

# Commented out IPython magic to ensure Python compatibility.
from dolfin import *
import numpy as np
import time
from ufl import sign,replace
import sys, os, sympy, shutil, math
parameters["form_compiler"].update({"optimize": True, "cpp_optimize": True, "representation":"uflacs", "quadrature_degree": 2})
parameters['ghost_mode'] = 'shared_facet'
#parameters["allow_extrapolation"] = True
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

comm = MPI.comm_world
rank = comm.rank

L = 1; H = 0.1;
n_elt = 45 #45 #100
#Gmsh mesh.
mesh = RectangleMesh(Point(0, -H/2), Point(L,H/2), n_elt*10, n_elt, "crossed")
cell_size = H / n_elt
ndim = mesh.topology().dim() # get number of space dimensions

E, nu = Constant(100), Constant(0)
kappa = (3-nu)/(1+nu)
mu = 0.5*E/(1+nu)
Gc = Constant(1.)
ell = Constant(3*cell_size)
h = mesh.hmax()

left = CompiledSubDomain("near(x[0], 0, 1e-4)")
right = CompiledSubDomain("near(x[0], %s, 1e-4)"%L)
boundaries = MeshFunction("size_t", mesh,1)
boundaries.set_all(0)
left.mark(boundaries, 1) # mark left as 1
right.mark(boundaries, 2) # mark right as 2

#To impose alpha=1 on crack
V_alpha = FunctionSpace(mesh, 'CG', 1)

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
    mu    = 0.5*E/(1 + nu)
    lmbda = E*nu / (1 - nu*nu)
    return 2.0*mu*(eps(u)) + lmbda*tr(eps(u))*Identity(ndim)

def sigma(u,alpha):
    """Stress tensor of the damaged material as a function of the displacement and the damage"""
    return (a(alpha))*sigma_0(u)

z = sympy.Symbol("z")
c_w = 4*sympy.integrate(sympy.sqrt(w(z)),(z,0,1))
tmp = 2*(sympy.diff(w(z),z)/sympy.diff(1/a(z),z)).subs({"z":0})
sigma_c = sympy.sqrt(tmp*Gc*E/(c_w*ell))
eps_c = float(sigma_c/E)

# Create function space for 2D elasticity + Damage
V_u = VectorFunctionSpace(mesh, "CG", 1)

if rank == 0:
    print('nb dof total: %i' % (V_u.dim()+V_alpha.dim()))

# Define the function, test and trial fields
u, du, v = Function(V_u, name='disp'), TrialFunction(V_u), TestFunction(V_u)
alpha, dalpha, beta = Function(V_alpha, name='damage'), TrialFunction(V_alpha), TestFunction(V_alpha)

#Dirichlet BC
# Displacement
u_D = Expression(("t","0"),t = 0.,degree=0)
bcu_0 = DirichletBC(V_u, u_D, boundaries, 2, method='geometric')
bcu_1 = DirichletBC(V_u, Constant((0.,0.)), boundaries, 1, method='geometric')
bc_u = [bcu_0, bcu_1]
# Damage
bcalpha_0 = DirichletBC(V_alpha, 0.0, boundaries, 1)
bcalpha_1 = DirichletBC(V_alpha, 0.0, boundaries, 2)
bc_alpha = [bcalpha_0, bcalpha_1]

#Energies
dissipated_energy = Gc/float(c_w)*(w(alpha)/ell + ell*dot(grad(alpha), grad(alpha)))*dx
elastic_energy = 0.5*inner(sigma(u,alpha), eps(u)) * dx
total_energy = elastic_energy + dissipated_energy
#Associated bilinear form
elastic = derivative(elastic_energy,u,v)
elastic = replace(elastic,{u:du})
XX,bb = as_backend_type(assemble(elastic)).mat().getVecs()

# First and second directional derivative wrt alpha
E_alpha = derivative(total_energy,alpha,beta)
E_alpha_alpha = derivative(E_alpha,alpha,dalpha)

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
solver_alpha.setTolerances(rtol=1e-5,atol=1e-7,max_it=1000) #rtol=1e-5
solver_alpha.setType('vinewtonrsls') #'vinewtonrsls'
bb = Function(V_alpha).vector()
solver_alpha.setFunction(pb_alpha.F, bb.vec())
A = PETScMatrix()
solver_alpha.setJacobian(pb_alpha.J, A.mat())
solver_alpha.getKSP().setType('preonly')
solver_alpha.getKSP().getPC().setType('lu') #'bjacobi' #'lu'
solver_alpha.getKSP().setTolerances(rtol=1e-6, atol=1e-8, max_it=4000) #rtol=1e-8
solver_alpha.getKSP().setFromOptions()
solver_alpha.setFromOptions()
#solver_alpha.view()


lb = interpolate(Constant("0."), V_alpha) # lower bound, initialize to 0
ub = interpolate(Constant("1."), V_alpha) # upper bound, set to 1
for bc in bc_alpha:
    bc.apply(lb.vector())
    bc.apply(ub.vector())
    bc.apply(alpha.vector())

def alternate_minimization(u,alpha,tol=1.e-5,maxiter=100,alpha_0=interpolate(Constant("0.0"), V_alpha)):
    # initialization
    iter = 1; err_alpha = 1
    alpha_error = Function(V_alpha)
    solver_alpha.setVariableBounds(lb.vector().vec(),ub.vector().vec())
    # iteration loop
    while err_alpha>tol and iter<maxiter:
        # solve elastic problem
        solver_u.setOperators(LHS())
        XX = u.copy(deepcopy=True)
        XV = as_backend_type(XX.vector()).vec()
        solver_u.solve(RHS(),XV)
        try:
            assert solver_u.getConvergedReason() > 0
        except AssertionError:
            if rank == 0:
                print('Error on solver u: %i' % solver_u.getConvergedReason())
            sys.exit()
        u.vector()[:] = XV
        u.vector().apply('insert')

        #solving damage problem
        xx = alpha.copy(deepcopy=True)
        xv = as_backend_type(xx.vector()).vec()
        solver_alpha.solve(None, xv)
        try:
            assert solver_alpha.getConvergedReason() > 0
        except AssertionError:
            if rank == 0:
                print('Error on solver alpha: %i' % solver_alpha.getConvergedReason())
            sys.exit()
        alpha.vector()[:] = xv
        alpha.vector().apply('insert')
        alpha_error.vector()[:] = alpha.vector() - alpha_0.vector()
        alpha_error.vector().apply('insert')
        err_alpha = norm(alpha_error.vector(),"linf")
        
        #monitor the results
        elastic_en = assemble(elastic_energy)
        surface_en = assemble(dissipated_energy)
        en = elastic_en + surface_en
        if rank == 0:
            print("Iteration:  %2d, Error: %2.8g, Energy: %2.8g" %(iter, err_alpha, en))
        # update iteration
        alpha_0.vector()[:] = alpha.vector()
        alpha_0.vector().apply('insert')
        iter=iter+1
        assert iter < maxiter
    return (err_alpha, iter)

#Setting up solver in disp
solver_u = PETSc.KSP()
solver_u.create(comm)
#PETScOptions.set("ksp_monitor")
solver_u.setType('preonly')
solver_u.getPC().setType('lu') #try it? #'bjacobi' #'lu'
solver_u.setTolerances(rtol=1e-5,atol=1e-8,max_it=5000) #rtol=1e-5,max_it=2000 #rtol=1e-3
solver_u.setFromOptions()

def LHS():
    LHS = inner(a(alpha)*sigma_0(du), eps(v)) * dx
    LHS = assemble(LHS)
    for bc in bc_u:
        bc.apply(LHS)
    return as_backend_type(LHS).mat()

def RHS():
    RHS = interpolate(Constant((0,0)), V_u).vector()
    for bc in bc_u:
        bc.apply(RHS)
    return as_backend_type(RHS).vec()

#Setting up real bc in alpha
for bc in bc_alpha:
    bc.apply(lb.vector())
    bc.apply(ub.vector())
    bc.apply(alpha.vector())
    
#loop on rest of the problem
load0 = float(eps_c)*L # reference value for the loading (imposed displacement)
epsilon = 1e-1

u_D.t = float(load0*(1+epsilon))
# solve alternate minimization
err,it = alternate_minimization(u,alpha,maxiter=500,tol=1e-4)
#Perf measure
if rank == 0:
    print('num it: %i' % it) 
