include("QuadVine2.jl")
include("Quad_vis.jl")

# model and timing
model = QuadVine(2) 
nq, nv, nc = mc_dims(model)
n,m = size(model)
N = 1000
dt = 5e-3
tf = (N-1)*dt     

# initial condition
shift = [0,0, sum(model.lengths)+.1]
x0 = [generate_config(model, zeros(model.nb)); zeros(nv)]
shift_pos!(model, x0, shift)
@assert norm(max_constraints(model, x0)) < 1e-6

# interpolated trajectory
aa = AngleAxis(pi/4, 1, -1, 0)
dx = [.8,.8,0]
Xhalf = interpolate_config(model, shift, dx, aa, N÷2)
Xtrack = [Xhalf..., reverse(Xhalf)...]
# vis = Visualizer()
# render(vis)
# TrajOptPlots._set_mesh!(vis["robot"], model; color=RGBA(.3,.3,.3,1))
# TrajOptPlots.visualize!(vis, model, SVector{n}(Xtrack[500]))

# position constraints
x_min=[fill(-Inf,nq-5); 0.5; fill(-Inf,4+nv)]
@assert all(x_min <= x0)
@assert all(x_min <= xw)

x_max=[.8; .8; fill(Inf,n-2)]
@assert all(x_max >= x0)
@assert all(x_max[3:end] >= xw[3:end])

# linear interpolate to the goal for the reference traj
# lin interp for quad (xy, no z) rotate vine 
# cost on yaw or attitude

# objective
Qf = Diagonal(SVector{n}([fill(1e0, nq); fill(1e0, nv)]))
Qw = Diagonal(SVector{n}([fill(1e0, nq-7); fill(1e2, 7); fill(1e-4, nv-6); fill(100, 6)]))
Q = Diagonal(SVector{n}([1;1;1;fill(1e-5, nq-3); fill(1e-5, nv)]))
R = Diagonal(SVector{m}([fill(1e0, m-1);1e1]));

costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(x0), SVector{m}(trim_controls(model)); w=1e-4) for i=1:N]
# edit objective
for i = 1:N 
        Qi, Ri, wi = Q, R, 1e-4
        if abs(i-N÷2) < 10
                Qi, wi = Qw, [.01,10.0,100.0]
        elseif i==N
                Qi, Ri, wi = Qf, 0R, 100.0
        end
        costfuns[i] = TO.LieLQRCost(RD.LieState(model), Qi, Ri, SVector{n}(Xtrack[i]), SVector{m}(trim_controls(model)); w=wi)
end

obj = Objective(costfuns);

# constraints
conSet = ConstraintList(n,m,N)
bnd = BoundConstraint(n, m, u_min=fill(-15,m), u_max=fill(15,m), x_min=x_min, x_max=x_max)
# add_constraint!(conSet, bnd, 1:N-1)
# add_constraint!(conSet, GoalConstraint(xw, SVector{9}([nq-6;nq-5;nq-4;collect(n-6 .+ (1:6))])), round(Int, N/2))

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
# altro = ALTROSolver(prob, opts);
# altro = Altro.iLQRSolver(prob, opts);
set_options!(altro, iterations=5, reset_penalties=false,
        cost_tolerance_intermediate=1e-3,
        cost_tolerance=1e-4, constraint_tolerance=.5)
using TimerOutputs
TimerOutputs.enable_debug_timings(Altro)
solve!(altro);
# altro.solver_al.solver_uncon.stats.to

# # plots
X,U = states(altro), controls(altro)
# display(plot([x[1] for x in X],xlabel="timestep",ylabel="drone x position"))
# display(plot([x[nq-4] for x in X],xlabel="timestep",ylabel="ee z position"))
display(plot(hcat(Vector.(U)...)',xlabel="timestep",ylabel="controls"))
# using JLD2
# # @save joinpath(@__DIR__, "data/quadvine2_fullrun_constrained.jld2") X U
# @load joinpath(@__DIR__, "data/quadvine2_fullrun_constrained.jld2") X U

# # animations
# vis = visualize!(model, states(altro), dt)
# goal_p = GeometryBasics.Point(SVector{3}(xw[nq-7 .+ (1:3)]))
# goal = GeometryBasics.HyperSphere{3,Float64}(goal_p, .3)
# setobject!(vis["goal"], goal, MeshPhongMaterial(color=RGBA(1,0,1, .5)))
# xlim = GeometryBasics.Rect{3,Float64}(Vec(-3.0, -3.0, 0.0), Vec(3+x_max[1], 3+x_max[2], 3.0))
# setobject!(vis["dronespace"], xlim, MeshPhongMaterial(color=RGBA(0,1,0, .3)))

ilqr = altro #altro.solver_al.solver_uncon
display(plot(hcat(Vector.(Vec.(ilqr.K[1:end]))...)',legend=false))
display(plot(hcat(Vector.(Vec.(ilqr.d[1:end]))...)',legend=false))

# overlay waypoints 
# vis = Visualizer()
# waypoints!(vis, prob, inds=collect(1:60:500),color=HSL(colorant"green"), color_end=HSL(colorant"red"));
# waypoints!(vis, prob, inds=collect(1:60:500),color=RGBA(.2,.2,.2,.1), color_end=RGBA(.2,.2,.2,1));
# waypoints!(vis, prob, inds=collect(1:60:500),color=RGBA(.2,1,.2,.1), color_end=RGBA(1,.2,.2,1));
# render(vis)

# xlim = GeometryBasics.Rect{3,Float64}(Vec(-1.0, -2.5, 0.0), Vec(2, 5, 1))
# setobject!(vis["awning"], xlim, MeshPhongMaterial(color=RGBA(.5,.5,.5, 1)))
# settransform!(vis["awning"],AffineMap(RotZ(pi/4),[2.1,2.1,1.8]))