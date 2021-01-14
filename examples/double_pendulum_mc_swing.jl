# using Pkg; Pkg.activate(".")

using RobotDynamics
using Altro
using TrajectoryOptimization
using StaticArrays
using LinearAlgebra
using ForwardDiff
# plot
using Plots
# visualize the system 
using MeshCat

const TO = TrajectoryOptimization
const RD = RobotDynamics

using Altro: iLQRSolver

include("double_pendulum_mc.jl")

N = 300   
dt = .01                  # number of knot points
tf = (N-1)*dt           # final time

# initial and final conditions
th1 = -pi/2
th2 = -pi/2
d1 = .5*model.l1*[cos(th1);sin(th1)]
d2 = .5*model.l2*[cos(th2);sin(th2)]
x0 = [d1; th1; 2*d1 + d2; th2; zeros(6)]
th1 = -pi/4
th2 = pi/4
d1 = .5*model.l1*[cos(th1);sin(th1)]
d2 = .5*model.l2*[cos(th2);sin(th2)]
xf = [d1; th1; 2*d1 + d2; th2; zeros(6)]

# objective
Q = zeros(n)
Q[3] = Q[6] = Q[9] = Q[12] = 1e-3/dt
Q = Diagonal(SVector{n}(Q))
R = Diagonal(@SVector fill(1e-4/dt, m))
Qf = zeros(n)
Qf[3] = Qf[6] = Qf[9] = Qf[12] = 2500
Qf = Diagonal(SVector{n}(Qf))
obj = LQRObjective(Q,R,Qf,xf,N)

# constraints
cons = ConstraintList(n,m,N)

# problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons);

# intial rollout with random controls
U0 = [SVector{2}(.01*rand(2)) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);


# solve problem
opts = SolverOptions(verbose=7,static_bp=0)
solver = iLQRSolver(prob, opts);
solve!(solver);

X2 = states(solver.Z)
gr(size = (2750, 2565))
plot([X2[i][3] for i=1:N], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38, label = ["joint 1 angle"],xlabel="time steps",xguidefontsize=38,ylabel="angle (rad)",yguidefontsize=38)
plot!([X2[i][6] for i=1:N], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38, label = ["joint 2 angle"],xlabel="time steps",xguidefontsize=38,ylabel="angle (rad)",yguidefontsize=38)

