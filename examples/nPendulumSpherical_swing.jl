
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
# initially out of plane
x0 = [generate_config(model, fill(.01, model.nb)); zeros(nv)]
xf = [generate_config(model, fill(RotX(.1)*RotY(.1), model.nb)); zeros(nv)]

# objective
Qf = Diagonal(@SVector fill(250., n))
Q = Diagonal(@SVector fill(.01/dt, n))
R = Diagonal(@SVector fill(.01/dt, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=.1) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=250.0)
obj = Objective(costfuns);

# problem
prob = Problem(model, obj, xf, tf, x0=x0);

# initial controls
U0 = [SVector{m}(fill(.1,m)) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);
plot_traj(states(prob), controls(prob))
# visualize!(model, states(prob), dt)

# ILQR
opts = SolverOptions(verbose=7, static_bp=0, iterations=15, cost_tolerance=1e-2)
ilqr = Altro.iLQRSolver(prob, opts);
solve!(ilqr);
plot_diff(states(ilqr), controls(ilqr),RotX(.1)*RotY(.1))
visualize!(model, states(ilqr), dt)

