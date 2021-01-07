using Pkg; Pkg.activate(".")

using RobotDynamics
using Altro
using TrajectoryOptimization
using StaticArrays
using LinearAlgebra
using ForwardDiff

const TO = TrajectoryOptimization
const RD = RobotDynamics

using Altro: iLQRSolver

include("backwardpassMC.jl")
include("double_pendulum_mc.jl")

N = 51                      # number of knot points
tf = .5                    # final time

# initial and final conditions
th1 = -pi/4
th2 = -pi/4
d1 = .5*model.l1*[cos(th1);sin(th1)]
d2 = .5*model.l2*[cos(th2);sin(th2)]
x0 = [d1; th1; 2*d1 + d2; th2; zeros(6)]
xf = [d1; th1; 2*d1 + d2; th2; zeros(6)]

# objective
Q = Diagonal(@SVector fill(1e-3, n))
R = Diagonal(@SVector fill(1e-4, m))
Qf = Diagonal(@SVector fill(250.0, n))
obj = LQRObjective(Q,R,Qf,xf,N)

# constraints
cons = ConstraintList(n,m,N)

# problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons);

# compute and verify nominal torques
m1, m2, g = model.m1, model.m2, model.g
uf = SVector((m1*xf[1] + m2*xf[4])*g, (xf[4] - 2*xf[1])*m2*g)
initial_controls!(prob, uf) 

solver = iLQRSolver(prob);
rollout!(solver)
TO.cost!(solver.obj, solver.Z)
TO.dynamics_expansion!(TO.integration(solver), solver.D, solver.model, solver.Z)
TO.cost_expansion!(solver.quad_obj, solver.obj, solver.Z, true, true)
ΔV = backwardpassMC!(solver)