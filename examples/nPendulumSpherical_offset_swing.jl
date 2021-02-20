
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
x0 = [generate_config(model, fill(RotY(.1), model.nb)); zeros(nv)]
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

# balancing torques
m1, m2 = model.masses
g = model.g
u_bal = [(m1*x0[1] + m2*x0[8])*g; (x0[8] - 2*x0[1])*m2*g]
u_bal = [0, -u_bal[1], 0, 0, -u_bal[2], 0]

# initial controls
U0 = [SVector{m}(u_bal) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);
plot_traj(states(prob), controls(prob))
# visualize!(model, states(prob), dt)

# ILQR
opts = SolverOptions(verbose=7, static_bp=0, iterations=50, cost_tolerance=1e-2)
ilqr = Altro.iLQRSolver(prob, opts);
solve!(ilqr);
plot_traj(states(ilqr), controls(ilqr))
visualize!(model, states(ilqr), dt)

