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

comm = MPI.comm_world
rank = comm.rank

L = 5; H = 1;
#Gmsh mesh. Already cracked
mesh = Mesh()
with XDMFFile("mesh/mesh_1b.xdmf") as infile: #mesh_surfing_very_fine #coarse #test is finest
    infile.read(mesh)
num_computation = 1
cell_size = mesh.hmax()
ndim = mesh.topology().dim() # get number of space dimensions

E, nu = Constant(1.0), Constant(0.3)
kappa = (3-nu)/(1+nu)
mu = 0.5*E/(1+nu)
Gc = Constant(1.5)
K1 = Constant(1.)
ell = Constant(3*cell_size)
if rank == 0:
    print('\ell: %.3e' % float(ell))
#sys.exit()

boundaries = MeshFunction("size_t", mesh,1)
boundaries.set_all(0)
ds = Measure("ds",subdomain_data=boundaries)
cells_meshfunction = MeshFunction("size_t", mesh, 2)
dxx = dx(subdomain_data=cells_meshfunction)

class NotCrack(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (near(x[0], 0) or near(x[0], L) or near(x[1], -H/2) or near(x[1], H/2))
not_crack = NotCrack()
not_crack.mark(boundaries, 1)

class Crack(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1]) < 1e-2 and x[0] < 1
crack = Crack()
crack.mark(boundaries, 2)

#To impose alpha=1 on crack
V_alpha = FunctionSpace(mesh, 'CG', 1) #'CR'
v = TestFunction(V_alpha)
A = FacetArea(mesh)
vec = assemble(v / A * ds(2)).get_local()
nz = vec.nonzero()[0]

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
Gc_eff = Gc * (1 + cell_size/(ell*float(c_w)))

# Create function space for 2D elasticity + Damage
V_u = VectorFunctionSpace(mesh, "CG", 1)
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
vel = 4 #given velocity
r = Expression('sqrt((x[0]-v*t) * (x[0]-v*t) + x[1] * x[1])', v=vel, t=0, degree = 2)

def BC():
    x = SpatialCoordinate(mesh)
    X = x[1] / (x[0]-vel*r.t)
    c_theta = 1. / sqrt(1 + X**2.) * sign(x[0]-vel*r.t)
    c_theta_2 = sqrt(0.5 * (1+c_theta))
    s_theta_2 = sqrt(0.5 * (1-c_theta)) * sign(x[1])
    return  K1/(2*mu) * sqrt(r/(2*np.pi)) * (kappa - c_theta) * as_vector((c_theta_2,s_theta_2)) #condition de bord de Dirichlet en disp

#Energies
pen_value = 2*mu
dissipated_energy = Gc/float(c_w)*(w(alpha)/ell + ell*dot(grad(alpha), grad(alpha)))*dx
elastic_energy = 0.5*inner(sigma(u,alpha), eps(u)) * dx
total_energy =  dissipated_energy + elastic_energy
##Associated bilinear forms
elastic = derivative(elastic_energy,u,v)
elastic = replace(elastic,{u:du})
XX,bb = as_backend_type(assemble(elastic)).mat().getVecs()

# First and second directional derivative wrt alpha
E_alpha = derivative(total_energy,alpha,beta)
E_alpha_alpha = derivative(E_alpha,alpha,dalpha) 

# Damage
bcalpha_0 = DirichletBC(V_alpha, 0.0, boundaries, 1)
bcalpha_1 = DirichletBC(V_alpha, Constant(1.0), boundaries, 2) #crack lips
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
solver_alpha.getKSP().setType('cg') #preonly #cg
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
        solver_u.solve(RHS(),XV)
        try:
            assert solver_u.getConvergedReason() > 0
        except AssertionError:
            if rank == 0:
                print('Error in solver_u: %i' % solver_u.getConvergedReason())
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
            print("Iteration:  %2d, Error: %2.8g, Energy: %2.8g" %(it, err_alpha, en))
        # update iteration
        alpha_0.vector()[:] = alpha.vector()
        alpha_0.vector().apply('insert')
        it = it +1
        
    return (err_alpha, it)

savedir = "CG_CG_perf_%i" % num_computation    
perf = open(savedir+'/perf.txt', 'w', 1)

def postprocessing(num,it):
    func = project(BC(), V_u)
    err = errornorm(u, func, 'h1') #h1? #h10? #l2?
    err_l2 = errornorm(u, func, 'l2')
    if rank == 0:
        perf.write('%i %i %.3e %.3e\n' % (num, it, err_l2, err))

T = 1 #final simulation time
dt = cell_size / 5 #should be h more ore less

#Starting with crack lips already broken
aux = np.zeros_like(alpha.vector().get_local())
aux[nz] = np.ones_like(nz)
alpha.vector().set_local(aux)
alpha.vector().apply('insert')
lb.vector()[:] = alpha.vector() #irreversibility
lb.vector().apply('insert')
#file_alpha << (alpha,0)

#test
solver_u = PETSc.KSP()
solver_u.create(comm)
#PETScOptions.set("ksp_monitor")
solver_u.setType('preonly') #cg
solver_u.getPC().setType('lu') #lu hypre
solver_u.setTolerances(rtol=1e-5,atol=1e-8, max_it=100) #rtol=1e-8
solver_u.setFromOptions()

def LHS():
    LHS = inner(a(alpha)*sigma_0(du), eps(v)) * dx
    LHS = assemble(LHS)
    bc_u.apply(LHS)
    return as_backend_type(LHS).mat()

def RHS():
    RHS = interpolate(Constant((0,0)), V_u).vector()
    bc_u.apply(RHS)
    return as_backend_type(RHS).vec()

load_steps = np.arange(0.3, T+dt, dt) #normal start: 0.2
N_steps = len(load_steps)

for (i,t) in enumerate(load_steps):
    r.t = t

    #updating BC
    bc_u =  DirichletBC(V_u, BC(), boundaries, 1, method='geometric')
    
    # solve alternate minimization
    err,it = alternate_minimization(u,alpha,maxiter=500,tol=1e-4)
    postprocessing(i,it)
    
    
    # updating the lower bound to account for the irreversibility
    lb.vector()[:] = alpha.vector()
    lb.vector().apply('insert')

sys.exit()
