
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
x0 = [generate_config(model, fill(pi, model.nb)); zeros(nv)]
xf = [generate_config(model, fill(pi, model.nb)); zeros(nv)]

# objective
Qf = Diagonal(@SVector fill(250., n))
Q = Diagonal(@SVector fill(1e-4/dt, n))
R = Diagonal(@SVector fill(1e-2/dt, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=1e-4) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=250.0)
obj = Objective(costfuns);

# problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet);

# initial controls
U0 = [SVector{m}(zeros(m)) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);
# plot_traj(states(prob), controls(prob))
# visualize!(model, states(prob), dt)

# ILQR
opts = SolverOptions(verbose=7, static_bp=0, iterations=50, cost_tolerance=1e-2)
ilqr = Altro.iLQRSolver(prob, opts);
# plot_traj(states(ilqr), controls(ilqr))
# visualize!(model, states(ilqr), dt)

# just one step
Altro.initialize!(ilqr)
Z = ilqr.Z; Z̄ = ilqr.Z̄;
n,m,N = size(ilqr)
_J = TO.get_J(ilqr.obj)
J_prev = sum(_J)
grad_only = false
J = Altro.step!(ilqr, J_prev, grad_only)

display(plot(hcat(Vector.(Vec.(ilqr.K[1:end-1]))...)',legend=false))

# use converged LQR gains
[ilqr.K[i]=ilqr.K[1] for i=2:N-1]
[ilqr.d[i]=ilqr.d[1] for i=2:N-1]

# perturb x0
x0new = [generate_config(model, fill(UnitQuaternion(.1,1,.1,.1), model.nb)); zeros(nv)]
ilqr.x0 .= x0new

# rollout
rollout!(ilqr, 1.0)
plot_diff(states(ilqr.Z̄), controls(ilqr.Z̄))
visualize!(model, states(ilqr.Z̄), dt)
