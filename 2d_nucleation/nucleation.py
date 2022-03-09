#coding: utf-8

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time
import mshr
from ufl import sign,replace
import sys, os, sympy, shutil, math
parameters["form_compiler"].update({"optimize": True, "cpp_optimize": True, "representation":"uflacs", "quadrature_degree": 2})
parameters['ghost_mode'] = 'shared_facet'
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

comm = MPI.comm_world
rank = comm.rank

resolution = 200 #50 #100 #200
geom = mshr.Circle(Point(0, 0), 1)
mesh = mshr.generate_mesh(geom, resolution)
ndim = mesh.topology().dim()
cell_size = mesh.hmax()
ndim = mesh.topology().dim() # get number of space dimensions

#material parameters
E, nu = Constant(1), Constant(0.3)
kappa = (3-nu)/(1+nu)
mu = 0.5*E/(1+nu)
h = mesh.hmax()
ell = 5*h
#print('ell: %.2e' % ell)
Gc = 1.5

boundaries = MeshFunction("size_t", mesh,1)
boundaries.set_all(0)
ds = Measure("ds",subdomain_data=boundaries)

class Bnd(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
bnd = Bnd()
bnd.mark(boundaries, 1)

#To impose alpha=1 on crack
V_alpha = FunctionSpace(mesh, 'CR', 1) #'CR'
V_beta = FunctionSpace(mesh, 'DG', 0) #to interpolat a(alpha)
W = TensorFunctionSpace(mesh, 'DG', 0)

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

# Create function space for 2D elasticity + Damage
V_u = VectorFunctionSpace(mesh, "DG", 1) #DG

# Define the function, test and trial fields
u, du, v = Function(V_u, name='disp'), TrialFunction(V_u), TestFunction(V_u)
alpha, dalpha, beta = Function(V_alpha, name='damage'), TrialFunction(V_alpha), TestFunction(V_alpha)

h = CellDiameter(mesh)
h_avg = 0.5 * (h('+') + h('-'))
n = FacetNormal(mesh)

#BC
# Displacement
theta = np.arange(0, 2*np.pi, 2*np.pi/8)[7]
#theta = np.arange(np.pi/4, 3*np.pi/4+np.pi/18, np.pi/18)[9]
ut = Expression(("t/E*(cos(theta)-nu*sin(theta))*x[0]", "t/E*(sin(theta)-nu*cos(theta))*x[1]"), t=0., E=E, theta=theta, nu=nu, degree=1)
bc_u =  DirichletBC(V_u, ut, boundaries, 1, method='geometric')
# Damage
bc_alpha = [DirichletBC(V_alpha, 0., boundaries, 1)]

#Writing LHS for disp
def b(alpha):
    test = Expression('pow(1-alpha,2)+k', degree = 2, alpha=alpha, k=Constant(1.e-6))
    return interpolate(test, V_beta)

def w_avg(disp,dam):
    sig = sigma_0(disp)
    lumped = b(dam)
    prod = lumped * sig
    tot = lumped('+')+lumped('-')
    w1 = lumped('+') / tot
    w2 = lumped('-') / tot
    return w1*prod('+') + w2*prod('-')

def pen(alpha):
    lumped = b(alpha)
    return 2*lumped('+')*lumped('-') / (lumped('+')+lumped('-'))
    
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
solver_alpha.getKSP().setType('cg')
solver_alpha.getKSP().getPC().setType('lu') #bjacobi 'lu'
solver_alpha.getKSP().setTolerances(rtol=1e-8) #rtol=1e-6, atol=1e-8)
solver_alpha.getKSP().setFromOptions()
solver_alpha.setFromOptions()
#solver_alpha.view()
#sys.exit()


lb = interpolate(Constant("0."), V_alpha) # lower bound, initialize to 0
ub = interpolate(Constant("1."), V_alpha) # upper bound, set to 1
for bc in bc_alpha:
    bc.apply(lb.vector())
    bc.apply(ub.vector())

def alternate_minimization(u,alpha,tol=1.e-5,maxiter=100,alpha_0=interpolate(Constant("0.0"), V_alpha)):
    # initialization
    iter = 1; err_alpha = 1
    alpha_error = Function(V_alpha)
    solver_alpha.setVariableBounds(lb.vector().vec(),ub.vector().vec())
    en_old = 1e10
    # iteration loop
    while err_alpha>tol and iter<maxiter:
        if iter == 2:
            sig = project(sigma_0(u), W)(0,0)
            file.write('%.3e %.3e %.3e %.3e\n' % (ell, theta, sig[0], sig[3]))
            file.close()
            sys.exit()
        
        #test with fenics
        problem_u = LinearVariationalProblem(LHS(), rhs(elastic), u, bc_u)
        solver_u = LinearVariationalSolver(problem_u)
        solver_u.parameters.update({"linear_solver" : "mumps"})
        solver_u.solve()

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

        #monitor the results
        elastic_en = assemble(elastic_energy)
        surface_en = assemble(dissipated_energy)
        en = elastic_en + surface_en
        en_old = en
        if rank == 0:
            print("Iteration:  %2d, Error: %2.8g, Energy: %2.8g" %(iter, err_alpha, en))
        # update iteration
        #alpha_0.assign(alpha)
        alpha_0.vector()[:] = alpha.vector()
        alpha_0.vector().apply('insert')
        iter=iter+1
        assert iter < maxiter
    return (err_alpha, iter)

savedir = "DG_CR_%i" % resolution
file_alpha = File(savedir+"/alpha.pvd") 
file_u = File(savedir+"/u.pvd")
energies = []

def postprocessing():
    # Dump solution to file
    file_alpha << (alpha,ut.t)
    file_u << (u,ut.t)
    

def LHS():
    LHS = inner(b(alpha)*sigma_0(du), eps(v)) * dx
    LHS += -inner(dot(w_avg(du,alpha),n('+')), jump(v))*dS + inner(dot(w_avg(v,alpha),n('+')), jump(du))*dS
    LHS += pen_value/h_avg * pen(alpha) * inner(jump(du), jump(v))*dS
    return LHS

def RHS():
    RHS = interpolate(Constant((0,0)), V_u).vector()
    bc_u.apply(RHS)
    return as_backend_type(RHS).vec()

#Time-stepping
load0 = float(sqrt(3*Gc*E/(8*ell*(1-nu*sin(2*theta))))) #seems ok
_eps = 1e-3
loads = load0*np.array([0, 1-_eps, 1+_eps])

#Store results
file = open('eigenvalues_%i.txt' % resolution, 'a', 1)

for (i,t) in enumerate(loads):
    ut.t = t
    
    # solve alternate minimization
    alternate_minimization(u,alpha,maxiter=500,tol=1e-4)
    
    # updating the lower bound to account for the irreversibility
    lb.vector()[:] = alpha.vector()
    lb.vector().apply('insert')
    postprocessing()

