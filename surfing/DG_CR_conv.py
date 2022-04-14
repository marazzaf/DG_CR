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
with XDMFFile("mesh/test.xdmf") as infile:
    infile.read(mesh)
num_computation = 0
cell_size = mesh.hmax()
ndim = mesh.topology().dim() # get number of space dimensions

E, nu = Constant(1.0), Constant(0.3)
kappa = (3-nu)/(1+nu)
mu = 0.5*E/(1+nu)
Gc = Constant(1.5)
K1 = Constant(1.)
ell = Constant(3*cell_size) #cell_size
if rank == 0:
    print('\ell: %.3e' % float(ell))
    print(cell_size/5)
#sys.exit()

boundaries = MeshFunction("size_t", mesh,1)
boundaries.set_all(0)
ds = Measure("ds",subdomain_data=boundaries)

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
V_alpha = FunctionSpace(mesh, 'CR', 1) #'CR'
V_beta = FunctionSpace(mesh, 'DG', 0) #test
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
V_u = VectorFunctionSpace(mesh, "DG", 1) #DG
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
total_energy =  dissipated_energy + elastic_energy #+ penalty_energy
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
solver_alpha.getKSP().getPC().setType('hypre') #bjacobi 'lu'
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
        #solve(PETScMatrix(LHS()), u.vector(), PETScVector(RHS()), "gmres", "ilu")
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
        it = it+1
        
    return (err_alpha, it)

#To compute the error
V_theta = FunctionSpace(mesh, "CG", 1)
theta = Function(V_theta, name="Theta")
theta_trial = TrialFunction(V_theta)
theta_test = TestFunction(V_theta)

#to get crack tip coordinates
xcoor = V_alpha.tabulate_dof_coordinates()
xcoor = xcoor[:,0]

def find_crack_tip():
    # Estimate the current crack tip
    ind = alpha.vector().get_local() > 0.9
    
    if ind.any():
        xmax = xcoor[ind].max()
    else:
        xmax = 0.0
    x0 = MPI.max(comm, xmax)

    return [x0, 0]

solver_theta = PETSc.KSP()
solver_theta.create(comm)
solver_theta.setType('cg')
solver_theta.getPC().setType('hypre')
solver_theta.setTolerances(rtol=1e-5)
solver_theta.setFromOptions()

def calc_theta(pos_crack_tip=[0., 0.]):
    #How to determine the crack tip? Barycentre of last facet for which alpha = 1?
    x0 = pos_crack_tip[0]  # x-coordinate of the crack tip
    y0 = pos_crack_tip[1]  # y-coordinate
    r = 2 * float(ell) #from Li article
    R = 5 * float(ell)

    def neartip(x, on_boundary):
        dist = sqrt((x[0]-x0)**2 + (x[1]-y0)**2)
        return dist < r

    def outside(x, on_boundary):
        dist = sqrt((x[0]-x0)**2 + (x[1]-y0)**2)
        return dist > R

    bc1 = DirichletBC(V_theta, Constant(1.0), neartip)
    bc2 = DirichletBC(V_theta, Constant(0.0), outside)
    bcs = [bc1, bc2]
    a = inner(grad(theta_trial), grad(theta_test))*dx
    L = inner(Constant(0.0), theta_test)*dx

    #petsc4py
    a = assemble(a)
    TT,b = as_backend_type(a).mat().getVecs()
    L = assemble(L)
    for bc in bcs:
        bc.apply(a)
        bc.apply(L)
    solver_theta.setOperators(as_backend_type(a).mat())
    solver_theta.solve(as_backend_type(L).vec(),TT)
    assert solver_theta.getConvergedReason() > 0
    theta.vector()[:] = TT
    theta.vector().apply('insert')

#savedir = "/scratch/marazzato/DG_CR_perf_%i" % num_computation
savedir = "conv_%i" % num_computation  
perf = open(savedir+'/conv.txt', 'w', 1)
file_BC = File(savedir+"/bc.pvd")
file_u = File(savedir+"/u.pvd")

def postprocessing(t):
    ##plot(mesh)
    #pos = find_crack_tip()
    #calc_theta(pos)
    #aux = (1-theta)*(BC()-u)
    #img = plot(inner(aux, aux))
    #plt.colorbar(img)
    #plt.show()

    file_u << (u,r.t)
    
    #Perf measure
    func = project(BC(), V_u)
    file_BC << (func,r.t)
    err = errornorm(u, func, 'h1') #h1? #h10? #l2?
    #err_l2 = errornorm(u, func, 'l2')
    pos = find_crack_tip()
    calc_theta(pos)
    aux = (1-theta)*(BC()-u)
    err_l2 = sqrt(assemble(inner(aux,aux) * dx))
    if rank == 0:
        perf.write('%.3e %.3e %.3e\n' % (t, err_l2, err)) 
    

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
solver_u.setType('preonly') #gmres
solver_u.getPC().setType('lu') #gamg lu
solver_u.setTolerances(rtol=1e-5,atol=1e-8) #rtol=1e-8
solver_u.setFromOptions()

def LHS():
    LHS = inner(b(alpha)*sigma_0(du), eps(v)) * dx
    LHS += -inner(dot(w_avg(du,alpha),n('+')), jump(v))*dS + inner(dot(w_avg(v,alpha),n('+')), jump(du))*dS
    LHS += pen_value/h_avg * pen(alpha) * inner(jump(du), jump(v))*dS
    LHS = assemble(LHS)
    bc_u.apply(LHS)
    return as_backend_type(LHS).mat()

def RHS():
    RHS = interpolate(Constant((0,0)), V_u).vector()
    bc_u.apply(RHS)
    return as_backend_type(RHS).vec()

r.t = 0.4
print('t: %.3f' % r.t)

test = Expression('x[0] < pos && abs(x[1]) < eps ? 1 : 0', pos=vel*r.t, eps=1e-2, degree = 2)

file_u << (interpolate(test, V_alpha), 0)
sys.exit()

#updating BC
bc_u =  DirichletBC(V_u, BC(), boundaries, 1, method='geometric')
    
# solve alternate minimization
err,it = alternate_minimization(u,alpha,maxiter=500,tol=1e-4)
postprocessing(r.t)
    
sys.exit()
