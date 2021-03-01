include("nPendulumSpherical.jl")
include("nPenJan.jl")
include("nPen_util.jl")

# model
model = nPenJan()
mech = model.mech
n,m = size(model)
dt = mech.Δt = .005
N = 1000
tf = (N-1)*dt  

x0 = [0;0;.5;zeros(3);0;1;zeros(5);0;0;1.5;zeros(3);0;1;zeros(5)]
xf = [0;0;.5;zeros(3);0;1;zeros(5);0;0;1.5;zeros(3);0;1;zeros(5)]
u0 = zeros(6)

# objective
Q = Diagonal(@SVector fill(10. /dt, n))
R = Diagonal(@SVector fill(1., m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=100.) for i=1:N]
obj = Objective(costfuns);

# problem
prob = Problem(model, obj, xf, tf, x0=x0);

# initial controls
U0 = [u0 for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);
# plot_traj_jan(states(prob), controls(prob))

# ilqr
opts = SolverOptions(verbose=7, static_bp=0, iterations=1, cost_tolerance=1e-2)
ilqr = Altro.iLQRSolver(prob, opts);
solve!(ilqr);

# perturb x0 and rollout
ilqr.x0 .= og_to_jan([generate_config(nPendulumSpherical(), fill(RotX(pi-.1), model.nb)); zeros(12)])
rollout!(ilqr, 1.0)
plot_traj_jan(states(ilqr.Z̄), controls(ilqr.Z̄))

include("nPendulum3D_visualize.jl")
X = [jan_to_og(x) for x in states(ilqr.Z̄)]
visualize!(nPendulumSpherical(), X, dt)