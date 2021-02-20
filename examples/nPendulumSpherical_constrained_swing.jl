
include("nPendulumSpherical.jl")
include("nPendulum3D_visualize.jl")

# model and timing
model = nPendulumSpherical()
nq, nv, nc = mc_dims(model)
n, m = size(model)
dt = 0.005
N = 1000
tf = (N-1)*dt     

# initial and final conditions
x0 = [generate_config(model, fill(RotX(.01), model.nb)); zeros(nv)]
xf = [generate_config(model, fill(pi, model.nb)); zeros(nv)]

# objective
Qf = Diagonal(@SVector fill(250., n))
Q = Diagonal(@SVector fill(1e-4/dt, n))
R = Diagonal(@SVector fill(1e-2/dt, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=1e-4) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=250.0)
obj = Objective(costfuns);

# constraints
conSet = ConstraintList(n,m,N)
# force underactuation (acrobot)
bnd = BoundConstraint(n,m, u_min=[0; fill(-Inf,5)], u_max=[0; fill(Inf,5)])
add_constraint!(conSet, bnd, 1:N-1)

# problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet);

using JLD2
@load joinpath(@__DIR__,"acrobot_mc_comparison.jld2") U2D
U0 = [SVector{m}([zeros(3);U2D[k];0;0]) for k = 1:N-1] # warm start with correct answer
U0 = [SVector{m}([zeros(3);5*rand();0;0]) for k = 1:N-1] # random initial controls
initial_controls!(prob, U0)
rollout!(prob);
plot_traj(states(prob), controls(prob))

# ALTRO
opts = SolverOptions(
    cost_tolerance_intermediate=1e-2, constraint_tolerance=1e-4,
    penalty_scaling=10., penalty_initial=1.0,
    verbose=7, static_bp=0, iterations=50)
altro = ALTROSolver(prob, opts)
set_options!(altro, show_summary=true)
solve!(altro);
plot_traj(states(altro), controls(altro))
visualize!(model, states(altro), dt)
