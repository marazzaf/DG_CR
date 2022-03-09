#coding: utf-8

from dolfin import *
import numpy as np
import time
from ufl import sign,replace
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
with XDMFFile("../CT_4.xdmf") as infile: #"CT_3.xdmf"
    infile.read(mesh)
cell_size = mesh.hmax()
ndim = mesh.topology().dim() # get number of space dimensions
num_computation = 44

E, nu = Constant(2.98), Constant(0.35)
mu = 0.5*E/(1+nu)
Gc = Constant(2.85e-4)
#ell = Constant(5*cell_size) #5*cell_size
cell_size = 0.05
ell = Constant(2*cell_size)

boundaries = MeshFunction("size_t", mesh,1)
boundaries.set_all(0)
ds = Measure("ds",subdomain_data=boundaries)

class Bnd(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and (x[0] < 1 or 31 < x[1] or x[1] < -31 or x[0] > 60 or (x[0] < 21 and abs(x[1]) < 2.1))
bnd = Bnd()
bnd.mark(boundaries, 4)

class Crack(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1]) < 1 and 30 < x[0] < 37 #10 < x[0] < 37
crack = Crack()
crack.mark(boundaries, 1)

class Upper_hole(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and 2 < x[0] < 22.86 and 10 < x[1] < 25

class Lower_hole(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and 2 < x[0] < 22.86 and -25 < x[1] < -10
upper_hole = Upper_hole()
upper_hole.mark(boundaries, 2)
lower_hole = Lower_hole()
lower_hole.mark(boundaries, 3)

#test to measure COD
class Up(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and 1 < x[1] < 3 and 10 < x[0] < 14

class Down(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and 10 < x[0] < 14 and -3 < x[1] < -1
up = Up()
up.mark(boundaries, 5)
down = Down()
down.mark(boundaries, 6)

#To impose alpha=1 on crack
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
V_u = VectorFunctionSpace(mesh, "DG", 1)

# Define the function, test and trial fields
u, du, v = Function(V_u, name='disp'), TrialFunction(V_u), TestFunction(V_u)
alpha, dalpha, beta = Function(V_alpha, name='damage'), TrialFunction(V_alpha), TestFunction(V_alpha)

h = CellDiameter(mesh)
h_avg = 0.5 * (h('+') + h('-'))
n = FacetNormal(mesh)
hF = FacetArea(mesh)

#Dirichlet BC on disp
t_init = 0.1
u_D = Expression(('0.', 'x[1]/fabs(x[1]) * t'), t=t_init, degree=1)
bcu_1 =  DirichletBC(V_u, u_D, boundaries, 2, method='geometric')
bcu_2 =  DirichletBC(V_u, u_D, boundaries, 3, method='geometric')
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
#problem with this BC!
bcalpha_0 = DirichletBC(V_alpha, Constant(0), boundaries, 4) #rest of the boundary not cracked
bcalpha_1 = DirichletBC(V_alpha, Constant(1), boundaries, 1) #crack lips
bcalpha_2 = DirichletBC(V_alpha, Constant(0), boundaries, 2) #holes not cracked #Constant(0)
bcalpha_3 = DirichletBC(V_alpha, Constant(0), boundaries, 3) #holes not cracked
bcalpha_4 = DirichletBC(V_alpha, Constant(0), boundaries, 5) #holes not cracked
bcalpha_5 = DirichletBC(V_alpha, Constant(0), boundaries, 6) #holes not cracked
bc_alpha = [bcalpha_0, bcalpha_1, bcalpha_2, bcalpha_3, bcalpha_4, bcalpha_5]
#bc_alpha = [bcalpha_1]

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
solver_alpha.getKSP().setType('cg')
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

savedir = "." #"DG_CR_%i" % num_computation  
file_alpha = File(savedir+"/alpha.pvd")
#file_alpha_bis = XDMFFile(savedir+"/alpha.xdmf")
#file_alpha_bis.parameters["flush_output"] = True
#file_alpha_bis.parameters["functions_share_mesh"] = True
#file_alpha_bis.parameters["rewrite_function_mesh"] = False
file_u = File(savedir+"/u.pvd")
#file_u_bis = XDMFFile(savedir+"/u.xdmf")
#file_u_bis.write(mesh)
#file_u_bis.parameters["flush_output"] = True
#file_u_bis.parameters["functions_share_mesh"] = True
#file_u_bis.parameters["rewrite_function_mesh"] = False
#file_sig = File(savedir+"/sigma.pvd")
energies = []
save_energies = open(savedir+'/energies.txt', 'w', 1)
ld = open(savedir+'/ld.txt', 'w', 1)
crack_pos = open(savedir+'/crack_pos.txt', 'w', 1)

W = TensorFunctionSpace(mesh, 'DG', 0)
stress = Function(W, name="stress")

#to get crack tip coordinates
coor = V_alpha.tabulate_dof_coordinates()
xcoor = coor[:,0]
ycoor = coor[:,1]

def find_crack_tip():
    # Estimate the current crack tip
    ind = alpha.vector().get_local() > 0.9
    
    if ind.any():
        xmax = xcoor[ind].max()
        ymax = ycoor[ind].max()
    else:
        xmax,ymax = 0,0
    x0 = MPI.max(comm, xmax)
    y0 = MPI.max(comm, ymax)

    return [x0, y0]

def postprocessing(num,Nsteps):
    ## Dump solution to file
    #if num % (Nsteps//10) == 0:
    file_alpha << (alpha,u_D.t)
    file_u << (u,u_D.t)

    #file_u_bis.write_checkpoint(u,'disp', u_D.t, XDMFFile.Encoding.HDF5, True)        
    #file_alpha_bis.write_checkpoint(alpha,'damage', u_D.t, XDMFFile.Encoding.HDF5, True)
    stress.vector()[:] = local_project(sigma(u,alpha), W).vector()
    stress.vector().apply("insert")
    #file_sig << (stress,u_D.t)

    #Energies
    elastic_energy_value = assemble(elastic_energy)
    surface_energy_value = assemble(dissipated_energy)        
    energies = [elastic_energy_value,surface_energy_value,elastic_energy_value+surface_energy_value]

    #Pos crack tip
    pos = find_crack_tip()

    #Load
    #compute COD
    #COD = u(12.7,2)[1] - u(12.7,-2)[1]
    COD = assemble( u[1]/hF * ds(5) - u[1]/hF * ds(6) )
    load = inner(dot(stress, n), as_vector((0,1))) * ds(2) #on one hole
    load = assemble(load)
    
    if rank == 0:
        #print('%.2f %.3e' % (pos[1],load))
        save_energies.write('%.3e %.5e %.5e %.5e\n' % (u_D.t, energies[0], energies[1], energies[2]))
        ld.write('%.5e %.5e\n' % (COD, load))
        crack_pos.write('%.3e %.5e %.5e\n' % (u_D.t, pos[0], pos[1]))
    

T = 0.25 #final simulation time
#dt = cell_size / 1e4 #should be h more ore less
dt = T/50 #T / 500

#Setting up solver in disp
solver_u = PETSc.KSP()
solver_u.create(comm)
#PETScOptions.set("ksp_monitor")
solver_u.setType('gmres')
solver_u.getPC().setType('lu') #try it? #'bjacobi' #'lu'
solver_u.setTolerances(rtol=1e-5,atol=1e-8,max_it=5000) #rtol=1e-5,max_it=2000 #rtol=1e-3
solver_u.setFromOptions()

def LHS():
    LHS = inner(b(alpha)*sigma_0(du), eps(v)) * dx
    LHS += -inner(dot(w_avg(du,alpha),n('+')), jump(v))*dS + inner(dot(w_avg(v,alpha),n('+')), jump(du))*dS
    LHS += pen_value/h_avg * pen(alpha) * inner(jump(du), jump(v))*dS 
    LHS = assemble(LHS)
    for bc in bc_u:
        bc.apply(LHS)
    return as_backend_type(LHS).mat()

def RHS():
    RHS = interpolate(Constant((0,0)), V_u).vector()
    for bc in bc_u:
        bc.apply(RHS)
    return as_backend_type(RHS).vec()

#test
#solving damage problem to have the nice phase field
solver_alpha.setVariableBounds(lb.vector().vec(),ub.vector().vec())
xx = alpha.copy(deepcopy=True)
xv = as_backend_type(xx.vector()).vec()
solver_alpha.solve(None, xv)
alpha.vector()[:] = xv
alpha.vector().apply('insert')
#file_alpha << (alpha,1)
#sys.exit()
lb.vector()[:] = alpha.vector() #enforcing irreversibility
lb.vector().apply('insert')

#Setting up real bc in alpha
bc_alpha = [bcalpha_0, bcalpha_2, bcalpha_3] #bcalpha_0
for bc in bc_alpha:
    bc.apply(lb.vector())
    bc.apply(ub.vector())
    bc.apply(alpha.vector())

#file_alpha << (alpha,1)
#sys.exit()
    
#loop on rest of the problem
load_steps = np.arange(dt, T+dt, dt)
N_steps = len(load_steps)

for (i,t) in enumerate(load_steps):
    u_D.t += dt
    if rank == 0:
        print('Time: %.2e' % u_D.t)

    # solve alternate minimization
    alternate_minimization(u,alpha,maxiter=500,tol=1e-4)
    
    # updating the lower bound to account for the irreversibility
    lb.vector()[:] = alpha.vector()
    lb.vector().apply('insert')
    postprocessing(i,N_steps)

#file_u_bis.close()
save_energies.close()
ld.close()
crack_pos.close()
