include("QuadVine2.jl")
include("Quad_vis.jl")

# model and timing
model = QuadVine(2) 
nq, nv, nc = mc_dims(model)
n,m = size(model)
N = 1000
dt = 5e-3
tf = (N-1)*dt     

# initial and final conditions
x0 = [generate_config(model, zeros(model.nb)); zeros(nv)]
shift_pos!(model, x0, [0,0, sum(model.lengths)+.1])
@assert norm(max_constraints(model, x0)) < 1e-6

xw = shift_pos(model, x0, [1.5,1.5,0])
@assert norm(max_constraints(model, xw)) < 1e-6

# position constraints
x_min=[fill(-Inf,nq-5); 0.5; fill(-Inf,4+nv)]
@assert all(x_min <= x0)
@assert all(x_min <= xw)

x_max=[.8; .8; fill(Inf,n-2)]
@assert all(x_max >= x0)
@assert all(x_max[3:end] >= xw[3:end])

# objective
Qf = Diagonal(SVector{n}([fill(100., nq); fill(100., nv)]))
Qw = Diagonal(SVector{n}([fill(1e-4/dt, nq-7); fill(1/dt, 7); fill(1/dt, nv)]))
Q = Diagonal(SVector{n}([fill(1e-4/dt, nq); fill(1e-4/dt, nv)]))
R = Diagonal(@SVector fill(1e0/dt, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(x0), SVector{m}(trim_controls(model)); w=1e-4) for i=1:N]
costfuns[round(Int, N/2)] = TO.LieLQRCost(RD.LieState(model), Qw, R, SVector{n}(xw), SVector{m}(trim_controls(model)); w=1e-4)
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, 0R, SVector{n}(x0), SVector{m}(trim_controls(model)); w=100.0)
obj = Objective(costfuns);

# edit obj
[costfuns[i] = TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(x0), SVector{m}(trim_controls(model)); w=1e-4) for i=1:N]
costfuns[round(Int, N/2)] = TO.LieLQRCost(RD.LieState(model), Qw, R, SVector{n}(xw), SVector{m}(trim_controls(model)); w=1e-4)
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, 0R, SVector{n}(x0), SVector{m}(trim_controls(model)); w=100.0)

# constraints
conSet = ConstraintList(n,m,N)
bnd = BoundConstraint(n, m, u_min=fill(-15,m), u_max=fill(15,m), x_min=x_min, x_max=x_max)
add_constraint!(conSet, bnd, 1:N-1)
add_constraint!(conSet, GoalConstraint(xw, SVector{9}([nq-6;nq-5;nq-4;collect(n-6 .+ (1:6))])), round(Int, N/2))

# problem
prob = Problem(model, obj, x0, tf, x0=x0, constraints=conSet);

# initial controls
u0 = trim_controls(model)
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
        iterations_linesearch=10,
        projected_newton=false,
        penalty_initial=1e-3)

# # solve with Altro
altro = ALTROSolver(prob, opts);
set_options!(altro, iterations=30, reset_penalties=false,
        cost_tolerance_intermediate=1e-3,
        cost_tolerance=1e-4, constraint_tolerance=.5)
using TimerOutputs
TimerOutputs.enable_debug_timings(Altro)
solve!(altro);
altro.solver_al.solver_uncon.stats.to

# # plots
X,U = states(altro), controls(altro)
# display(plot([x[1] for x in X],xlabel="timestep",ylabel="drone x position"))
# display(plot([x[nq-4] for x in X],xlabel="timestep",ylabel="ee z position"))
display(plot(hcat(Vector.(U)...)',xlabel="timestep",ylabel="controls"))
# using JLD2
# @save joinpath(@__DIR__, "data/quadvine2_fullrun_constrained.jld2") X U

# # animations
vis = visualize!(model, states(altro), dt)
goal_p = GeometryBasics.Point(SVector{3}(xw[nq-7 .+ (1:3)]))
goal = GeometryBasics.HyperSphere{3,Float64}(goal_p, .3)
setobject!(vis["goal"], goal, MeshPhongMaterial(color=RGBA(1,0,1, .5)))
xlim = GeometryBasics.Rect{3,Float64}(Vec(-3.0, -3.0, 0.0), Vec(3+x_max[1], 3+x_max[2], 3.0))
setobject!(vis["dronespace"], xlim, MeshPhongMaterial(color=RGBA(0,1,0, .3)))

display(plot(hcat(Vector.(Vec.(ilqr.K[1:end]))...)',legend=false))
# display(plot(hcat(Vector.(Vec.(ilqr.d[1:end]))...)',legend=false))

# overlay waypoints 
# vis = Visualizer()
# waypoints!(vis, prob, inds=collect(1:60:500),color=HSL(colorant"green"), color_end=HSL(colorant"red"));
# waypoints!(vis, prob, inds=collect(1:60:500),color=RGBA(.2,.2,.2,.1), color_end=RGBA(.2,.2,.2,1));
# waypoints!(vis, prob, inds=collect(1:60:500),color=RGBA(.2,1,.2,.1), color_end=RGBA(1,.2,.2,1));
# render(vis)

# xlim = GeometryBasics.Rect{3,Float64}(Vec(-1.0, -2.5, 0.0), Vec(2, 5, 1))
# setobject!(vis["awning"], xlim, MeshPhongMaterial(color=RGBA(.5,.5,.5, 1)))
# settransform!(vis["awning"],AffineMap(RotZ(pi/4),[2.1,2.1,1.8]))