include("nPenJan.jl")

# model
model = nPenJan()
mech = model.mech
n,m = size(model)
dt = mech.Δt = .005
N = 1000
tf = (N-1)*dt  

x0 = [0;0;-.5;zeros(3);1;zeros(6);0;0;-1.5;zeros(3);1;zeros(6)]
xf = [0;0;.5;zeros(3);0;1;zeros(5);0;0;1.5;zeros(3);0;1;zeros(5)]

# objective
Qf = Diagonal(@SVector fill(250., n))
Q = Diagonal(@SVector fill(1e-3/dt, n))
R = Diagonal(@SVector fill(1e-4/dt, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=1e-3) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=250.0)
obj = Objective(costfuns);

# problem
prob = Problem(model, obj, xf, tf, x0=x0);

# initial controls
U0 = [SA[rand(),0.,0,rand(),0,0] for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);
# plot_traj_jan(states(prob), controls(prob))

# ILQR
opts = SolverOptions(verbose=7, static_bp=0, iterations=20, cost_tolerance=1e-4)
ilqr = Altro.iLQRSolver(prob, opts);
solve!(ilqr);
plot_traj_jan(states(ilqr), controls(ilqr))
