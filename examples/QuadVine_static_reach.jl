include("QuadVine.jl")
include("Quad_vis.jl")

# model and timing
model = QuadVine(2) 
nq, nv, nc = mc_dims(model)
n,m = size(model)
N = 500
dt = 5e-3
tf = (N-1)*dt     

# initial and final conditions
x0 = [generate_config(model, zeros(model.nb)); zeros(nv)]
shift_pos!(model, x0, [0,0, sum(model.lengths)+.1])
@assert norm(max_constraints(model, x0)) < 1e-6

xf = [generate_config(model, [RotX(0.0), RotX(0), RotX(pi/2)]); zeros(nv)]
shift_pos!(model, xf, [0,0, sum(model.lengths)+.1])
@assert norm(max_constraints(model, xf)) < 1e-6

u = trim_controls(model)
u[2] += .25/.175*9.81
u[4] -= .25/.175*9.81
u[5] = .5*9.81
u[8] = .5*9.81
u_des = u
# x0 = xf

# X = quick_rollout(model, xf, u, dt, 200)
# norm(X[1]-X[end])
# vis = visualize!(model, X, dt)

# objective
Qf = Diagonal(SVector{n}([fill(100., nq); fill(100., nv)]))
Q = Diagonal(SVector{n}([fill(1e1/dt,7);fill(1e-1/dt, nq-7); fill(1e-1/dt, nv)]))
R = Diagonal(@SVector fill(1e-2/dt, m))
# costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf), SVector{m}(trim_controls(model)); w=1e1) for i=1:N]
# costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf), SVector{m}(trim_controls(model)); w=300.0)
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf), SVector{m}(u_des); w=1e-4) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, 0R, SVector{n}(xf), SVector{m}(u_des); w=100.0)
obj = Objective(costfuns);

# edit objective
# [costfuns[i] = TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf), SVector{m}(trim_controls(model)); w=1e1) for i=1:N]
# costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, 0R, SVector{n}(xf), SVector{m}(trim_controls(model)); w=300.0)
# [costfuns[i] = TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf), SVector{m}(u_des); w=1e-4) for i=1:N]
# costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, 0R, SVector{n}(xf), SVector{m}(u_des); w=100.0)

# constraints
conSet = ConstraintList(n,m,N)
bnd = BoundConstraint(n, m, u_min=fill(-15,m), u_max=fill(15,m))
# add_constraint!(conSet, bnd, 1:N-1)
# add_constraint!(conSet, GoalConstraint(xf), N)

# problem
prob = Problem(model, obj, x0, tf, x0=x0, constraints=conSet);

# initial controls
u0 = trim_controls(model)
# U0 = [SVector{m}(u_des) for k = 1:N-1]
U0 = [SVector{m}(u0) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);
# plot_traj(states(prob), controls(prob))
# visualize!(model, states(prob), dt)

# options
opts = SolverOptions(verbose=7, static_bp=0, 
        iterations=5, cost_tolerance=1e-3, 
        cost_tolerance_intermediate=1e-2,
        constraint_tolerance=1e-3,
        projected_newton=false)

# solve with ilqr
ilqr = Altro.iLQRSolver(prob, opts);
set_options!(ilqr, iterations=5, 
            cost_tolerance=1e-4, constraint_tolerance=1e-3)
solve!(ilqr);
# show(ilqr.stats.to)

# plots
X,U = states(ilqr), controls(ilqr)
display(plot(hcat(Vector.(U)...)',xlabel="timestep",ylabel="controls",legend=false))

display(plot(hcat(Vector.(Vec.(ilqr.K[1:end]))...)',legend=false))
# display(plot(hcat(Vector.(Vec.(ilqr.d[1:end]))...)',legend=false))

# animations
vis = visualize!(model, states(ilqr), dt)

# overlay waypoints 
# vis = Visualizer()
# waypoints!(vis, prob, inds=collect(1:60:1000),color=RGBA(.2,.2,.2,.1), color_end=RGBA(.2,.2,.2,1));
# open(vis)

# TrajOptPlots.set_mesh!(vis,model)
# TrajOptPlots.visualize!(vis,model,SVector{n}(xf))
# TrajOptPlots.visualize!(vis,model,SVector{n}(x0))
# render(vis)

# # solve with Altro
# altro = ALTROSolver(prob, opts);
# set_options!(altro, iterations=15, 
#             cost_tolerance=1e-4, constraint_tolerance=1e-3)
# solve!(altro);
