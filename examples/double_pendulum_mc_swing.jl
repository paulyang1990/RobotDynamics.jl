# using Pkg; Pkg.activate(".")

using RobotDynamics
using Altro
using TrajectoryOptimization
using StaticArrays
using LinearAlgebra
using ForwardDiff
# plot
using Plots
import PyPlot
# visualize the system 
using MeshCat

const TO = TrajectoryOptimization
const RD = RobotDynamics

using Altro: iLQRSolver

include("double_pendulum_mc.jl")
include("double_pendulum_costfunctions_mc.jl")
# include("double_pendulum_costfunctions_rc.jl")

N = 300   
dt = .01                  # number of knot points
tf = (N-1)*dt           # final time

# initial and final conditions
th1 = -pi/2
th2 = -pi/2
d1 = .5*model.l1*[cos(th1);sin(th1)]
d2 = .5*model.l2*[cos(th2);sin(th2)]
x0 = [d1; th1; 2*d1 + d2; th2; zeros(6)]
th1 = pi/2
th2 = pi/2
d1 = .5*model.l1*[cos(th1);sin(th1)]
d2 = .5*model.l2*[cos(th2);sin(th2)]
xf = [d1; th1; 2*d1 + d2; th2; zeros(6)]

# objective
# Q = zeros(n)
# # Q[1] = Q[2] = Q[3] = Q[4] = Q[5] = Q[6] = 1e-4/dt
# Q[7] = Q[8] = Q[9] = Q[10] = Q[11] = Q[12] = 1e-3/dt
# Q = Diagonal(SVector{n}(Q))
# R = Diagonal(@SVector fill(1e-4/dt, m))
# Qf = zeros(n)
# Qf[1] = Qf[4] = 2500
# Qf[2] = Qf[5] = 2500
# Qf[3] = Qf[6] = 2500
# # Qf[7] = Qf[8] = Qf[9] = Qf[10] = Qf[11] = Qf[12] = 1250
# Qf = Diagonal(SVector{n}(Qf))
# obj = LQRObjective(Q,R,Qf,xf,N)

# 1,2 - tip position
# 3,4 - minimize link velocity

zf = @SVector [0,2,0,0] 
uf = @SVector [0,0]
Q = zeros(4)
Q[1] = Q[2] = 1e-4
Q[3] = Q[4] = 1e-2
Q = Diagonal(SVector{4}(Q))
R = Diagonal(@SVector fill(1e-4, 2))
Qf = zeros(4)
Qf[1] = Qf[2] = 2000
Qf = Diagonal(SVector{4}(Qf))
obj = MCObjective(Q,R,Qf,zf,N;uf = uf)

# constraints
cons = ConstraintList(n,m,N)

# problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons);

# intial rollout with random controls
U0 = [SVector{2}([0.01,0.01]) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);


# solve problem
opts = SolverOptions(verbose=7,static_bp=0)
solver = iLQRSolver(prob, opts);
solve!(solver);

K = solver.K
d = solver.d

gr(size = (2250, 1965))
ctrl = controls(solver.Z)
X2 = states(solver.Z)

plot([ctrl[i][1] for i=1:N-1], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38, legend=:topright, label = ["joint 1 torque"],xlabel="time steps",xguidefontsize=38,ylabel="torque (Nm)",yguidefontsize=38)
p1 = plot!([ctrl[i][2] for i=1:N-1], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38,legend=:topright,  label = ["joint 2 torque"],xlabel="time steps",xguidefontsize=38,ylabel="torque (Nm)",yguidefontsize=38)

plot([X2[i][3] for i=1:N], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38, legend=:bottomright, label = ["link 1 angle"],xlabel="time steps",xguidefontsize=38,ylabel="angle (rad)",yguidefontsize=38)
p2 = plot!([X2[i][6] for i=1:N], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38, legend=:bottomright, label = ["link 1 angle"],xlabel="time steps",xguidefontsize=38,ylabel="angle (rad)",yguidefontsize=38)

plot([X2[i][1] for i=1:N], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38, legend=:topleft, label = ["link 1 x"],xlabel="time steps",xguidefontsize=38,ylabel="position (m)",yguidefontsize=38)
plot!([X2[i][2] for i=1:N], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38, legend=:topleft, label = ["link 1 y"],xlabel="time steps",xguidefontsize=38,ylabel="position (m)",yguidefontsize=38)
plot!([X2[i][4] for i=1:N], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38, legend=:topleft, label = ["link 2 x"],xlabel="time steps",xguidefontsize=38,ylabel="position (m)",yguidefontsize=38)
p3 = plot!([X2[i][5] for i=1:N], linewidth=4,xtickfontsize=38,ytickfontsize=38,legendfontsize=38, legend=:topleft, label = ["link 2 y"],xlabel="time steps",xguidefontsize=38,ylabel="position (m)",yguidefontsize=38)

l = @layout [a ; b; c]

plot(p1, p2, p3, layout = l, title=  "Maximal Coordinate Swing Up", titlefontsize=38)
# hack a title 
# https://stackoverflow.com/questions/43066957/adding-global-title-to-plots-jl-subplots

# PyPlot.suptitle("Maximal Coordinate Swing Up")

# save the controller so that we can use the controller to
# drive the robot in Matlab 
using CSV, DataFrames
Kout = [reshape(item,1,24) for item in K]
CSV.write("mc_K.csv",  DataFrame(vcat(Kout...)), writeheader=false)
CSV.write("mc_d.csv",  DataFrame(d), writeheader=false)
CSV.write("mc_xref.csv",  DataFrame(X2), writeheader=false)
CSV.write("mc_uref.csv",  DataFrame(ctrl), writeheader=false)
CSV.write("time.csv",  DataFrame([1:N-1]*dt), writeheader=false)