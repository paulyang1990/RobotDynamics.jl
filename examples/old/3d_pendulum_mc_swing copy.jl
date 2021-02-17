# using Pkg; Pkg.activate(".")

using RobotDynamics
using Altro
using TrajectoryOptimization
using StaticArrays
using LinearAlgebra
using ForwardDiff
# plot
using Plots
# visualize the system 
using MeshCat

const TO = TrajectoryOptimization
const RD = RobotDynamics

using Altro: iLQRSolver

include("3DPendulum2x.jl")

model = Pendulum3D2x()

N = 300   
dt = .01                  # number of knot points
tf = (N-1)*dt           # final time

# initial and final conditions
R0 = UnitQuaternion(.9999,.0001,0, 0)
x0 = [R0*[0.; 0.; -.5]; RS.params(R0); zeros(6)]
xf = [0.; 0.;  .5; 0; 1; 0; 0; zeros(6)]
x0 = [x0;x0]
xf = [xf;xf]

# objective
Qf = Diagonal(@SVector fill(100., 26))
Q = Diagonal(@SVector fill(.001/dt, 26))
R = Diagonal(@SVector fill(0.0001/dt, 2))

costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{26}(xf); w=0.1) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{26}(xf); w=100.0)
obj = Objective(costfuns);

prob = Problem(model, obj, xf, tf, x0=x0);

# intial rollout with random controls
# U0 = [SVector{2}(.01*rand(2)) for k = 1:N-1]
U0 = [SVector{2}([.01, .01]) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);

# check dynamics
# X = states(prob)
# quats = [UnitQuaternion(X[i][4:7]) for i=1:N]
# angles = [rotation_angle(quats[i])*rotation_axis(quats[i])[1] for i=1:N]
# plot(angles[1:end-10], label = "θ",xlabel="time step",ylabel="state")
# plot!([X[i][10] for i=1:N],  label = "θ dot")
# quats = [UnitQuaternion(X[i][13 .+ (4:7)]) for i=1:N]
# angles = [rotation_angle(quats[i])*rotation_axis(quats[i])[1] for i=1:N]
# plot!(angles[1:end-10], label = "θ2",xlabel="time step",ylabel="state")
# plt = plot!([X[i][end-3] for i=1:N],  label = "θ dot2")
# display(plt)

# check against single 3DPendulum
# TO.state_diff_jacobian!(solver.G, solver.model, solver.Z)
# TO.dynamics_expansion!(PassThrough, solver.D, solver.model, solver.Z)
# TO.error_expansion!(solver.D, solver.model, solver.G)
# TO.cost_expansion!(solver.quad_obj, solver.obj, solver.Z, false, true)
# TO.error_expansion!(solver.E, solver.quad_obj, solver.model, solver.Z, solver.G)
# G2, D2, Q2, E2 = copy(solver.G), copy(solver.D), copy(solver.quad_obj), copy(solver.E)

# D1[1].A ≈ D2[1].A[1:12,1:12] ≈ D2[1].A[13:end,13:end]
# Q1[1].Q ≈ Q2[1].Q[1:13,1:13] ≈ Q2[1].Q[14:end,14:end]
# Q1[1].q ≈ Q2[1].q[1:13] ≈ Q2[1].q[14:end]
# E1[1].Q ≈ E2[1].Q[1:12,1:12] ≈ E2[1].Q[13:end,13:end]

# solve problem
opts = SolverOptions(verbose=7,static_bp=0)
solver = iLQRSolver(prob, opts);
solve!(solver);

X = states(solver)
quats = [UnitQuaternion(X[i][4:7]) for i=1:N]
angles = [rotation_angle(quats[i])*rotation_axis(quats[i])[1] for i=1:N]
plot(angles[1:end-10], label = "θ",xlabel="time step",ylabel="state")
plot!([X[i][10] for i=1:N],  label = "θ dot")

quats = [UnitQuaternion(X[i][13 .+ (4:7)]) for i=1:N]
angles = [rotation_angle(quats[i])*rotation_axis(quats[i])[1] for i=1:N]
plot!(angles[1:end-10], label = "θ2",xlabel="time step",ylabel="state")
plt = plot!([X[i][end-3] for i=1:N],  label = "θ dot2")
display(plt)

U = controls(solver)
plot(hcat(U...)', legend=false, xlabel="time step",ylabel="torque")
