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

include("double_pendulum_rc.jl")
include("double_pendulum_costfunctions_rc.jl")

N = 300   
dt = 0.01                  # number of knot points
tf = (N-1)*dt           # final time

# initial and final conditions
th1 = -pi/2
th2 = 0
x0 = [th1; th2; zeros(2)]
th1 = pi/2
th2 = 0
xf = [th1; th2; zeros(2)]

function rc_to_mc(model::DoublePendulumRC, rc_x)
    θ1 = rc_x[1]
    θ2 = rc_x[2]
    l1 = model.l1
    l2 = model.l2
    r1 = l1/2
    r2 = l2/2

    mc_x = [r1*cos(θ1); r1*sin(θ1); θ1; l1*cos(θ1)+r2*cos(θ1+θ2); l1*sin(θ1)+r2*sin(θ1+θ2);θ1+θ2]

    return mc_x
end

# # objective
# Q = zeros(n)
# # Q[1] = Q[2] = 1e-3/dt
# Q[3] = Q[4] = 1e-3/dt
# Q = Diagonal(SVector{n}(Q))
# R = Diagonal(@SVector fill(1e-4/dt, m))
# Qf = zeros(n)
# Qf[1] = Qf[2] = 2500
# Qf[3] = Qf[4] = 2500
# Qf = Diagonal(SVector{n}(Qf))
# obj = LQRObjective(Q,R,Qf,xf,N)

zf = @SVector [0,2,0,0] 
uf = @SVector [0,0]
Q = zeros(4)
Q[1] = Q[2] = 1e-5
Q[3] = Q[4] = 1e-2
Q = Diagonal(SVector{4}(Q))
R = Diagonal(@SVector fill(1e-4, 2))
Qf = zeros(4)
Qf[1] = Qf[2] = 1500
Qf = Diagonal(SVector{4}(Qf))
obj = RCObjective(Q,R,Qf,zf,N;uf = uf)

# constraints
cons = ConstraintList(n,m,N)

# problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons);

# intial rollout with random controls
U0 = [SVector{2}(.01*rand(2)) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);


# solve problem
opts = SolverOptions(verbose=7,static_bp=0)
solver = iLQRSolver(prob, opts);
solve!(solver);

K = solver.K
d = solver.d

X2 = states(solver.Z)

X_mc = [rc_to_mc(model, z.z) for z in solver.Z]

ctrl = controls(solver.Z)


# pyplot()
gr(size = (2250, 1965))
plot([ctrl[i][1] for i=1:N-1], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38, legend=:topright, label = ["joint 1 torque"],xlabel="time steps",xguidefontsize=38,ylabel="torque (Nm)",yguidefontsize=38)
p1 = plot!([ctrl[i][2] for i=1:N-1], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38,legend=:topright,  label = ["joint 2 torque"],xlabel="time steps",xguidefontsize=38,ylabel="torque (Nm)",yguidefontsize=38)


plot([X2[i][1] for i=1:N], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38, legend=:bottomright, label = ["link 1 angle"],xlabel="time steps",xguidefontsize=38,ylabel="angle (rad)",yguidefontsize=38)
p2 = plot!([X2[i][1]+X2[i][2] for i=1:N], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38, legend=:bottomright, label = ["link 1 angle"],xlabel="time steps",xguidefontsize=38,ylabel="angle (rad)",yguidefontsize=38)

plot([X_mc[i][1] for i=1:N], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38, legend=:topleft, label = ["link 1 x"],xlabel="time steps",xguidefontsize=38,ylabel="position (m)",yguidefontsize=38)
plot!([X_mc[i][2] for i=1:N], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38, legend=:topleft, label = ["link 1 y"],xlabel="time steps",xguidefontsize=38,ylabel="position (m)",yguidefontsize=38)
plot!([X_mc[i][4] for i=1:N], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38, legend=:topleft, label = ["link 2 x"],xlabel="time steps",xguidefontsize=38,ylabel="position (m)",yguidefontsize=38)
p3 = plot!([X_mc[i][5] for i=1:N], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38, legend=:topleft, label = ["link 2 y"],xlabel="time steps",xguidefontsize=38,ylabel="position (m)",yguidefontsize=38)


l = @layout [a ; b; c]

plot( p1, p2, p3, layout = l, title=  "Reduced Coordinate Swing Up", titlefontsize=38)


# save the controller so that we can use the controller to
# drive the robot in Matlab 
using CSV, DataFrames
Kout = [reshape(item,1,8) for item in K]
CSV.write("rc_K.csv",  DataFrame(vcat(Kout...)), writeheader=false)
CSV.write("rc_d.csv",  DataFrame(d), writeheader=false)
CSV.write("rc_xref.csv",  DataFrame(X2), writeheader=false)
CSV.write("rc_uref.csv",  DataFrame(ctrl), writeheader=false)
CSV.write("time.csv",  DataFrame([1:N-1]*dt), writeheader=false)