include("QuadVine.jl")
include("Quad_vis.jl")

# model and timing
model = QuadVine(1)
nq, nv, nc = mc_dims(model)
n,m = size(model)
N = 999
dt = 5e-3
tf = (N-1)*dt     

# initial and final conditions
x0 = [generate_config(model, zeros(model.nb)); zeros(nv)]
xf = [generate_config(model, [0,pi]); zeros(nv)]

# objective
Qf = Diagonal(SVector{n}([fill(250., nq); fill(250., nv)]))
Q = Diagonal(SVector{n}([fill(1e-4, nq); fill(1e-4, nv)]))
R = Diagonal(@SVector fill(1e-3/dt, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf), 
                    SVector{m}(trim_controls(model)); w=0.0) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf), 
                    SVector{m}(trim_controls(model)); w=250.0)
obj = Objective(costfuns);

# constraints
conSet = ConstraintList(n,m,N)
bnd = BoundConstraint(n,m, u_min=[fill(0,4);zeros(3)], 
                            u_max=[fill(Inf,4);zeros(3)])
add_constraint!(conSet, bnd, 1:N-1)
add_constraint!(conSet, GoalConstraint(xf), N) 

# problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet);

# initial controls
# using JLD2
# @load "QuadVine_swingup_penalty_allU_noclip.jld2" X U
# initial_controls!(prob, U)
u0 = trim_controls(model)
U0 = [SVector{m}(u0) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);
# plot_traj(states(prob), controls(prob))
# visualize!(model, states(prob), dt)

# altro
opts = SolverOptions(verbose=7, static_bp=0, 
        iterations=10, cost_tolerance=1e-4, 
        cost_tolerance_intermediate=1e-2,
        constraint_tolerance=1.0,
        gradient_tolerance_intermediate=5.,
        projected_newton=false)
altro = ALTROSolver(prob, opts);
set_options!(altro, iterations=50, constraint_tolerance=.075)
solve!(altro);
X,U = states(altro), controls(altro)
plot_traj(states(altro), controls(altro))
visualize!(model, states(altro), dt)

# reset prob
initial_controls!(prob, U0)
rollout!(prob);
# plot_traj(states(prob), controls(prob))
# visualize!(model, states(prob), dt)

# ilqr
opts = SolverOptions(verbose=7, static_bp=0, iterations=1, cost_tolerance=1e-4)
ilqr = Altro.iLQRSolver(prob, opts);
set_options!(ilqr, iterations=50, cost_tolerance=1e-4)
solve!(ilqr);
X,U = states(ilqr), controls(ilqr)
plot_traj(states(ilqr), controls(ilqr))
visualize!(model, states(ilqr), dt)
