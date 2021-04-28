include("QuadVine.jl")
include("Quad_vis.jl")

# model and timing
model = QuadVine(2)
nq, nv, nc = mc_dims(model)
n,m = size(model)
N = 999
dt = 5e-3
tf = (N-1)*dt     

# initial and final conditions
x0 = [generate_config(model, zeros(model.nb)); zeros(nv)]
xf = [generate_config(model, [0,pi,pi]); zeros(nv)]

# objective
gain = 2.5
Qf = Diagonal(SVector{n}([fill(gain, nq); fill(gain, nv)]))
Q = Diagonal(SVector{n}([fill(1e-4, nq); fill(1e-4, nv)]))
R = Diagonal(@SVector fill(1e-2/dt, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf), 
                    SVector{m}(trim_controls(model)); w=1e-4) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf), 
                    SVector{m}(trim_controls(model)); w=gain)
obj = Objective(costfuns);

# problem
cons = ConstraintList(n,m,N)
bnd = BoundConstraint(n, m, u_min=fill(-15, m), 
                            u_max=fill(15, m))
add_constraint!(cons, bnd, 1:N-1)
add_constraint!(cons, GoalConstraint(xf), N)
prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons);

# initial controls
u0 = trim_controls(model)
U0 = [SVector{m}(u0) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);
# plot_traj(states(prob), controls(prob))
# visualize!(model, states(prob), dt)

# ilqr
opts = SolverOptions(verbose=7, static_bp=0, iterations=1, 
                    cost_tolerance=1e-4, penalty_scaling=10.0,
                    projected_newton = false)
ilqr = ALTROSolver(prob, opts);
set_options!(ilqr, iterations=40, cost_tolerance=1e-3, 
            cost_tolerance_intermediate=1e-3,
            constraint_tolerance=1e-3,
            penalty_scaling=2,
            reset_penalties=false)
solve!(ilqr);
X,U = states(ilqr), controls(ilqr)
plot_traj(states(ilqr), controls(ilqr))
visualize!(model, states(ilqr), dt)

# display(plot(hcat(Vector.(Vec.(K[1:end-1]))...)',legend=false))
# display(plot(hcat(Vector.(Vec.(d[990:end]))...)',legend=false))