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

include("3DPendulum.jl")

model = Pendulum3D()

N = 300   
dt = .01                  # number of knot points
tf = (N-1)*dt           # final time

# initial and final conditions
R0 = UnitQuaternion(.9999,.0001,0, 0)
x0 = [R0*[0.; 0.; -.5]; RS.params(R0); zeros(6)]
xf = [0.; 0.;  .5; 0; 1; 0; 0; zeros(6)]

# objective
Qf_diag = RD.fill_state(model, 100., 0., 100., 100.)
Q_diag = RD.fill_state(model, .001/dt, 0., .001/dt, .001/dt)
Qf = Diagonal(Qf_diag)
Q = Diagonal(Q_diag)
R = Diagonal(@SVector fill(0.0001/dt,1))

# costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{13}(xf); w=0.1) for i=1:N]
# costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{13}(xf); w=100.0)
costfuns = [TO.QuatLQRCost(Q, R, SVector{13}(xf); w=0.1) for i=1:N]
costfuns[end] = TO.QuatLQRCost(Qf, R, SVector{13}(xf); w=100.0)
obj = Objective(costfuns);

prob = Problem(model, obj, xf, tf, x0=x0);

# intial rollout with random controls
U0 = [SVector{1}(.01*rand(1)) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);

# solve problem
opts = SolverOptions(verbose=7,static_bp=0)
solver = iLQRSolver(prob, opts);
solve!(solver);

X = states(solver)
quats = [UnitQuaternion(X[i][4:7]) for i=1:N]
angles = [rotation_angle(quats[i])*rotation_axis(quats[i])[1] for i=1:N]
plot(angles[1:end-10], label = "θ",xlabel="time step",ylabel="state")
plt = plot!([X[i][10] for i=1:N],  label = "θ dot")
display(plt)

U = controls(solver)
plot(U, legend=false, xlabel="time step",ylabel="torque")
