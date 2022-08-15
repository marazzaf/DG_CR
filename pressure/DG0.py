#coding: utf-8

# Commented out IPython magic to ensure Python compatibility.
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import time
from ufl import sign,replace
#plt.rcParams['image.cmap'] = 'viridis'
import sys, os, sympy, shutil, math
parameters["form_compiler"].update({"optimize": True, "cpp_optimize": True, "representation":"uflacs", "quadrature_degree": 2})
parameters['ghost_mode'] = 'shared_facet'
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

comm = MPI.comm_world
rank = comm.rank

#Gmsh mesh. Already cracked
mesh = Mesh()
with XDMFFile("mesh.xdmf") as infile:
    infile.read(mesh)
cell_size = mesh.hmax()
ndim = mesh.topology().dim() # get number of space dimensions

#material parameters
E = 1
nu = 0
mu = Constant(0.5*E/(1+nu))
lmbda = Constant(nu*E/(1-2*nu)/(1+nu))
Gc = Constant(1)
l0 = Constant(0.114)
ell = Constant(2*cell_size)

boundaries = MeshFunction("size_t", mesh,1)
boundaries.set_all(0)
ds = Measure("ds",subdomain_data=boundaries)

class Bnd(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
bnd = Bnd()
bnd.mark(boundaries, 1)
class LR(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0] > 0.29 or x[0] < -0.29)
lr = LR()
lr.mark(boundaries, 2)

#Space for the phase field
V_alpha = FunctionSpace(mesh, 'CR', 1)
V_beta = FunctionSpace(mesh, 'DG', 0) #for interpolation

metadata={"quadrature_degree": 0}
def local_project(v,V):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv,v_)*dx(metadata=metadata)
    b_proj = inner(v,v_)*dx(metadata=metadata)
    solver = LocalSolver(a_proj,b_proj)
    solver.factorize()
    u = Function(V)
    solver.solve_local_rhs(u)
    return u

def w(alpha):
    return alpha

def a(alpha):
    k_ell = Constant(1.e-6) # residual stiffness
    return (1-alpha)**2+k_ell

def eps(u):
    return sym(grad(u))

def sigma_0(u):
    return 2.0*mu*(eps(u)) + lmbda*tr(eps(u))*Identity(ndim)

def sigma(u,alpha):
    return (a(alpha))*sigma_0(u)

z = sympy.Symbol("z")
c_w = 4*sympy.integrate(sympy.sqrt(w(z)),(z,0,1))
Gc_eff = Gc * (1 + cell_size/(ell*float(c_w)))

# Create function space for 2D elasticity + Damage
V_u = VectorFunctionSpace(mesh, "DG", 1)
if rank == 0:
    print('nb dof total: %i' % (V_u.dim()+V_alpha.dim()))

# Define the function, test and trial fields
u, du, v = Function(V_u, name='disp'), TrialFunction(V_u), TestFunction(V_u)
alpha, dalpha, beta = Function(V_alpha, name='damage'), TrialFunction(V_alpha), TestFunction(V_alpha)

h = CellDiameter(mesh)
h_avg = 0.5 * (h('+') + h('-'))
n = FacetNormal(mesh)
hF = FacetArea(mesh)

#Dirichlet BC on disp
V0 = 2 * sqrt(np.pi*Gc*l0**3/E)
p0 = sqrt(Gc * E / np.pi / l0)
dt = V0 / 10
T = 5 * V0
#p = Expression('2*E*Gc/pi/V', V=0, Gc=Gc, E=E, pi=np.pi, degree=2)
p = Expression('p0*V', V=0, p0=float(p0), degree=1)
u_D = Expression('F * x[1] / abs(x[1])', F = 1e-5, degree = 1)
bcu_1 =  DirichletBC(V_u.sub(1), u_D, boundaries, 1, method='geometric')
bcu_2 =  DirichletBC(V_u.sub(0), Constant(0), boundaries, 2, method='geometric')
bc_u = [bcu_1, bcu_2]

#Writing LHS for disp
def b(alpha):
    test = Expression('pow(1-alpha,2)+k', degree = 2, alpha=alpha, k=Constant(1.e-6))
    return interpolate(test, V_beta) #V_alpha #V_beta

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
total_energy = elastic_energy + dissipated_energy
#Associated bilinear form
elastic = derivative(elastic_energy,u,v)
elastic = replace(elastic,{u:du})
XX,bb = as_backend_type(assemble(elastic)).mat().getVecs()

# First and second directional derivative wrt alpha
E_alpha = derivative(total_energy,alpha,beta)
E_alpha_alpha = derivative(E_alpha,alpha,dalpha)

# Damage
bcalpha_0 = DirichletBC(V_alpha, Constant(0), boundaries, 1, method='geometric')
bc_alpha = [bcalpha_0]

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
solver_alpha.setTolerances(rtol=1e-5,atol=1e-7,max_it=2000) #rtol=1e-5
solver_alpha.setType('vinewtonrsls') #'vinewtonrsls'
bb = Function(V_alpha).vector()
solver_alpha.setFunction(pb_alpha.F, bb.vec())
A = PETScMatrix()
solver_alpha.setJacobian(pb_alpha.J, A.mat())
solver_alpha.getKSP().setType('cg')
solver_alpha.getKSP().getPC().setType('hypre') #'bjacobi' #'lu'
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
        #test
        solve(LHS_bis() == RHS(), u, bcs=bc_u, solver_parameters={"linear_solver": "mumps"},)

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

savedir = "DG"
file_alpha = File(savedir+"/alpha.pvd")
file_u = File(savedir+"/u.pvd")
ld = open(savedir+'/ld.txt', 'w', 1)

v_reac = Function(V_u)
def postprocessing(num,Nsteps):
    ## Dump solution to file
    #if num % 10 == 0:
    file_alpha << (alpha,p.V)
    file_u << (u,p.V)

    test = avg(alpha) * p * jump(u, n) * dS
    print(assemble(test))

#Setting up solver in disp
solver_u = PETSc.KSP()
solver_u.create(comm)
#PETScOptions.set("ksp_monitor")
solver_u.setType('preonly')
solver_u.getPC().setType('lu') #try it? #'lu'
solver_u.setTolerances(rtol=1e-5,atol=1e-8,max_it=1000) #rtol=1e-5,max_it=2000 #rtol=1e-3
solver_u.setFromOptions()

def LHS_bis():
    LHS = inner(b(alpha)*sigma_0(du), eps(v)) * dx
    LHS += -inner(dot(w_avg(du,alpha),n('+')), jump(v))*dS + inner(dot(w_avg(v,alpha),n('+')), jump(du))*dS
    LHS += pen_value/h_avg * pen(alpha) * inner(jump(du), jump(v))*dS
    return LHS

def RHS():
    return avg(alpha) * p * jump(v, n) * dS

#Put the initial crack in the domain
test = Expression('abs(x[0]) < 0.8*l0 && abs(x[1]) < eps ? 1 : 0', l0=l0, eps=0.5*cell_size, degree = 1)
alpha.vector()[:] = interpolate(test, V_alpha).vector()
alpha.vector().apply('insert')
lb.vector()[:] = alpha.vector() #irreversibility
lb.vector().apply('insert')

#Setting up real bc in alpha
for bc in bc_alpha:
    bc.apply(lb.vector())
    bc.apply(ub.vector())
    bc.apply(alpha.vector())

#loop on rest of the problem
t_init = V0/2-dt #3*V0-dt
load_steps = np.arange(t_init, T+dt, dt)
N_steps = len(load_steps)

#Get phase-field right
#solving damage problem
solver_alpha.setVariableBounds(lb.vector().vec(),ub.vector().vec())
xx = alpha.copy(deepcopy=True)
xv = as_backend_type(xx.vector()).vec()
solver_alpha.solve(None, xv)
alpha.vector()[:] = xv
alpha.vector().apply('insert')
file_alpha << (alpha,0)

for (i,t) in enumerate(load_steps):
    p.V = t
    if rank == 0:
        print('Volume: %.2e' % p.V)

    # solve alternate minimization
    alternate_minimization(u,alpha,maxiter=2000,tol=1e-4)
    
    # updating the lower bound to account for the irreversibility
    lb.vector()[:] = alpha.vector()
    lb.vector().apply('insert')
    postprocessing(i,N_steps)

ld.close()
