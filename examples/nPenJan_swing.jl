include("nPenJan.jl")

# model
model = nPenJan()
mech = model.mech
n,m = size(model)
dt = mech.Δt = .005
N = 1000
tf = (N-1)*dt  

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
U0 = [SA[.1,.1,0,.1,.1,0] for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);
plot_traj(states(prob), controls(prob))

# ILQR
opts = SolverOptions(verbose=7, static_bp=0, iterations=1, cost_tolerance=1e-2)
ilqr = Altro.iLQRSolver(prob, opts);
solve!(ilqr);
plot_traj(states(ilqr), controls(ilqr))
