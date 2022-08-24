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
cell_size = 0.5e-3 #mesh.hmax()
ndim = mesh.topology().dim() # get number of space dimensions

#material parameters
E = 1
nu = 0
mu = Constant(0.5*E/(1+nu))
lmbda = Constant(nu*E/(1-2*nu)/(1+nu))
Gc = Constant(1)
l0 = Constant(0.114)
ell = Constant(3*cell_size)

boundaries = MeshFunction("size_t", mesh,1)
boundaries.set_all(0)
ds = Measure("ds",subdomain_data=boundaries)

class Bnd(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary
bnd = Bnd()
bnd.mark(boundaries, 1)

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
V_u = VectorFunctionSpace(mesh, "CG", 1)
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
bcu_1 = DirichletBC(V_u, Constant((0,0)), boundaries, 1, method='geometric')
bc_u = [bcu_1]

#Energies
pen_value = 2*mu
dissipated_energy = Gc/float(c_w)*(w(alpha)/ell + ell*dot(grad(alpha), grad(alpha)))*dx
elastic_energy = 0.5*inner(sigma(u,alpha), eps(u)) * dx
#aux = conditional(lt(avg(alpha), Constant(0.99)), Constant(0), avg(alpha))
#source = -aux**2 * p * jump(u, n) * dS
#p = Expression('p', p=0, degree=1)
#source = -avg(alpha)**2 * jump(u, n) * dS
source = - inner(u, grad(alpha)) * dx
total_energy = elastic_energy + dissipated_energy - source
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
solver_alpha.getKSP().setType('cg') #cg
solver_alpha.getKSP().getPC().setType('hypre') #'hypre'
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

def alternate_minimization(vol,u,alpha,tol=1.e-5,maxiter=100,alpha_0=interpolate(Constant("0.0"), V_alpha)):
    # initialization
    iter = 1; err_alpha = 1
    alpha_error = Function(V_alpha)
    solver_alpha.setVariableBounds(lb.vector().vec(),ub.vector().vec())
    # iteration loop
    while err_alpha>tol and iter<maxiter:
        #Compute disp à p=1
        solve(LHS() == RHS(), u, bcs=bc_u, solver_parameters={"linear_solver": "mumps"},)
        #break

        #compute pressure
        approx_vol = - 2*inner(u, grad(alpha)) * dx
        print('vol: %.2e' % assemble(approx_vol))
        l = find_crack()
        print(l)
        ref = pi * l * u.vector().get_local().max()
        print('ref vol: %.2e' % float(ref))
        break
        print('disp: %.2e' % u(0,1e-3)[1])
        print('ref: %.2e' % float(2*l0))
        p = vol / assemble(approx_vol)
        print(float(p))
        

        #new disp
        u.vector()[:] = p * u.vector()

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
        try:
            assert iter < maxiter
        except AssertionError:
            print('Max num it reached!')
    return (err_alpha, iter)

savedir = "CG"
file_alpha = File(savedir+"/alpha.pvd")
file_u = File(savedir+"/u.pvd")
ld = open(savedir+'/ld.txt', 'w', 1)
file_ref = File(savedir+"/ref.pvd")

v_reac = Function(V_u)
def postprocessing(V):
    file_alpha << (alpha,V)
    file_u << (u,V)
    #img = plot(u)
    #plt.colorbar(img)
    #plt.show()
    ##sys.exit()

#Setting up solver in disp
solver_u = PETSc.KSP()
solver_u.create(comm)
#PETScOptions.set("ksp_monitor")
solver_u.setType('cg')
solver_u.getPC().setType('hypre') #try it? #'lu'
solver_u.setTolerances(rtol=1e-5,atol=1e-8,max_it=1000) #rtol=1e-5,max_it=2000 #rtol=1e-3
solver_u.setFromOptions()

def LHS():
    LHS = inner(a(alpha)*sigma_0(du), eps(v)) * dx
    return LHS

def RHS():
    return -inner(v, grad(alpha)) * dx

#Put the initial crack in the domain
test = Expression('abs(x[0]) < 0.5*l0 && abs(x[1]) < eps ? 1 : 0', l0=l0, eps=0.5*cell_size, degree = 1)
alpha.vector()[:] = interpolate(test, V_alpha).vector()
alpha.vector().apply('insert')
lb.vector()[:] = alpha.vector() #irreversibility
lb.vector().apply('insert')

#to get crack tip coordinates
xcoor = V_alpha.tabulate_dof_coordinates()
xcoor = xcoor[:,0]

def find_crack():
    aux = alpha.vector().get_local()
    ind = alpha.vector().get_local() > 0.8
    xmax = xcoor[ind].max()
    xmin = xcoor[ind].min()
    return xmax-xmin

    #crack = np.where(alpha.vector().get_local() > 0.8)
    #test = Function(V_alpha)
    #truc = np.zeros_like(aux)
    #truc[crack] = np.ones_like(crack)
    #test.vector()[:] = truc
    #return test

#Setting up real bc in alpha
for bc in bc_alpha:
    bc.apply(lb.vector())
    bc.apply(ub.vector())
    bc.apply(alpha.vector())

#loop on rest of the problem
t_init = V0/10 #3*V0/4-dt #3*V0-dt
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

#find_crack()
#sys.exit()

for (i,V) in enumerate(load_steps):
    if rank == 0:
        print('Volume fraction: %.2e' % (V/V0))

    # solve alternate minimization
    alternate_minimization(V,u,alpha,maxiter=1000,tol=1e-4)
    
    # updating the lower bound to account for the irreversibility
    lb.vector()[:] = alpha.vector()
    lb.vector().apply('insert')
    postprocessing(V)
    break

ld.close()
