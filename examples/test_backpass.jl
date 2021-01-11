# using Pkg; Pkg.activate(".")

using RobotDynamics
using Altro
using TrajectoryOptimization
using StaticArrays
using LinearAlgebra
using ForwardDiff

const TO = TrajectoryOptimization
const RD = RobotDynamics

using Altro: iLQRSolver

include("double_pendulum_mc.jl")

N = 600   
dt = .01                  # number of knot points
tf = (N-1)*dt           # final time

# initial and final conditions
th1 = -pi/4
th2 = -pi/4
d1 = .5*model.l1*[cos(th1);sin(th1)]
d2 = .5*model.l2*[cos(th2);sin(th2)]
x0 = [d1; th1; 2*d1 + d2; th2; zeros(6)]
xf = [d1; th1; 2*d1 + d2; th2; zeros(6)]

# objective
Q = zeros(n)
Q[3] = Q[6] = Q[9] = Q[12] = 1e-3/dt
Q = Diagonal(SVector{n}(Q))
R = Diagonal(@SVector fill(1e-4/dt, m))
Qf = Diagonal(@SVector fill(250.0, n))
Qf = zeros(n)
Qf[3] = Qf[6] = Qf[9] = Qf[12] = 1e-3
Qf = Diagonal(SVector{n}(Qf))
obj = LQRObjective(Q,R,Qf,xf,N)

# constraints
cons = ConstraintList(n,m,N)

# problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons);

# compute and verify nominal torques
m1, m2, g = model.m1, model.m2, model.g
uf = SVector((m1*xf[1] + m2*xf[4])*g, (xf[4] - 2*xf[1])*m2*g)
initial_controls!(prob, uf) 

# compute gains
solver = iLQRSolver(prob);
rollout!(solver)
TO.cost!(solver.obj, solver.Z)
TO.dynamics_expansion!(TO.integration(solver), solver.D, solver.model, solver.Z)
TO.cost_expansion!(solver.quad_obj, solver.obj, solver.Z, true, true)
ΔV = Altro.backwardpass!(solver)
display(solver.K[end])
display(solver.K[1])
display(solver.d[end])
display(solver.d[1])

# adjust gains
K1, d1 = -solver.K[1], 0*solver.d[1]
for i=1:length(solver.K)
    solver.K[i] .= K1
    solver.d[i] .= d1
end

# perturb x0 and rollout
solver.x0 .= discrete_dynamics(PassThrough, model, KnotPoint(x0,[1., 0],dt))
rollout!(solver, 1.0)

# plot
using Plots
X2 = states(solver.Z̄)
plot([X2[i][3] for i=1:N])
plot!([X2[i][6] for i=1:N])
