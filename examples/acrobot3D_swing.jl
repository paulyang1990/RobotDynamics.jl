include("acrobot3D.jl")

# model and timing
model = Acrobot3D()
n, m = size(model)
dt = 0.005
N = 1000
tf = (N-1)*dt     

# initial and final conditions
x0 = rc_to_mc(model, [.01, 0])
xf = rc_to_mc(model, [pi, 0])

# objective
Qf = Diagonal(@SVector fill(100., n))
Q = Diagonal(@SVector fill(1e-2, n))
R = Diagonal(@SVector fill(1e-1, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=1e-2) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=100.0)
obj = Objective(costfuns);

# constraint
conSet = ConstraintList(n,m,N)
goal = GoalConstraint(xf)
add_constraint!(conSet, goal, N)

# problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet);

u0 = @SVector fill(0.01,m)
U0 = [u0 for k = 1:N-1]
# @load joinpath(@__DIR__,"soln2D.jld2") soln2D
# U0 = soln2D
initial_controls!(prob, U0)
rollout!(prob);
# plot_traj(states(prob), controls(prob))

# solver options
opts = SolverOptions(
    cost_tolerance_intermediate=1e-2,
    penalty_scaling=10.,
    penalty_initial=1.0,
    constraint_tolerance=1e-4,
    verbose=7, static_bp=0, iterations=30
)

# ALTRO
TimerOutputs.enable_debug_timings(Altro)
altro = ALTROSolver(prob, opts)
set_options!(altro, show_summary=true)
solve!(altro);
# @save joinpath(@__DIR__, "acrobot3D_swing.jld2")
plot_traj(states(altro), controls(altro))
show(altro.solver_al.solver_uncon.stats.to)
reset_timer!(altro.solver_al.solver_uncon.stats.to)

# ILQR
# opts = SolverOptions(verbose=7, static_bp=0, iterations=20, cost_tolerance=1e-3)
# ilqr = Altro.iLQRSolver(prob, opts);
# solve!(ilqr);
# plot_traj(states(ilqr), controls(ilqr))

# using StatProfilerHTML
# @profilehtml 
# J = sum(TO.get_J(solver.obj))
# Altro.step!(solver, J, true)

# include("2link_visualize.jl")
# visualize!(model, X, dt)

# @load "C:\\Users\\riann\\Google Drive\\ReposMC\\RobotDynamics.jl\\examples\\acrobot3D_swing.jld2"