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
# xf = rc_to_mc(model, [.1, -.3])

# problem
Qf = Diagonal(@SVector fill(250., n))
Q = Diagonal(@SVector fill(1e-4/dt, n))
R = Diagonal(@SVector fill(1e-4/dt, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=1e-4) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=300.0)
obj = Objective(costfuns);
prob = Problem(model, obj, xf, tf, x0=x0);

# intial rollout with random controls
U0 = [SVector{m}(2.0*rand(m)) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);

# solve problem
opts = SolverOptions(verbose=7, static_bp=0)
solver = Altro.iLQRSolver(prob, opts);
solve!(solver);

# plot state
X = states(solver)
quats1 = [UnitQuaternion(X[i][4:7]) for i=1:N]
quats2 = [UnitQuaternion(X[i][7 .+ (4:7)]) for i=1:N]
angles1 = [rotation_angle(quats1[i])*rotation_axis(quats1[i])[1] for i=1:N]
angles2 = [rotation_angle(quats2[i])*rotation_axis(quats2[i])[1] for i=1:N]

using Plots
plot(angles1, label = "θ1",xlabel="time step",ylabel="state")
plt = plot!(angles2-angles1,  label = "θ2")
display(plt)

include("2link_visualize.jl")
visualize!(model, X, dt)

# plot control
U = controls(solver)
plot(U)

# @save "acrobot_swing.jld2" X U
