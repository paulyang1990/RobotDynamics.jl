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
# read csv
using CSV, DataFrames

const TO = TrajectoryOptimization
const RD = RobotDynamics

using Altro: iLQRSolver

include("double_pendulum_rc.jl")
include("double_pendulum_costfunctions_rc.jl")


# read reference trajectory and gains 
# CSV.write("mc_K.csv",  DataFrame(vcat(Kout...)), writeheader=false)
# CSV.write("mc_d.csv",  DataFrame(d), writeheader=false)
# CSV.write("mc_xref.csv",  DataFrame(X2), writeheader=false)
# CSV.write("mc_uref.csv",  DataFrame(ctrl), writeheader=false)
# CSV.write("time.csv",  DataFrame([1:N-1]*dt), writeheader=false)

df = DataFrame(CSV.File("rc_K.csv",header=false))
mtx = convert(Matrix,df)
rc_K = [reshape(row,2,4) for row in eachrow(mtx)]
df = DataFrame(CSV.File("rc_d.csv",header=false))
mtx = convert(Matrix,df)
rc_d = [d for d in eachcol(mtx)]
df = DataFrame(CSV.File("rc_xref.csv",header=false))
mtx = convert(Matrix,df)
rc_xref = [d for d in eachcol(mtx)]
df = DataFrame(CSV.File("rc_uref.csv",header=false))
mtx = convert(Matrix,df)
rc_uref = [d for d in eachcol(mtx)]
df = DataFrame(CSV.File("time.csv",header=false))
t_list = convert(Matrix,df)

# simulate
th1 = -pi/2
th2 = 0
x0 = [th1; th2; zeros(2)]
prev_t = 0
xt = x0
x_list = zeros(4, size(t_list,1)+1)
x_list[:,1] = x0
tip_pos = zeros(2, size(t_list,1)+1)
tip_pos[:,1] = [cos(x0[1])+cos(x0[1]+x0[2]), sin(x0[1])+sin(x0[1]+x0[2])]

tip_diff = zeros(size(t_list,1))
for i in 1:size(t_list,1)
    global prev_t
    global xt
    dt = t_list[i] - prev_t
    prev_t = t_list[i]
    curr_x_ref = rc_xref[i]
    ut = rc_uref[i] + rc_K[i]*(xt - rc_xref[i])
    # z = KnotPoint(xt,ut,dt)
    # x_next = RD.discrete_dynamics(RD.Euler, model, z)
    x_next = xt + dt*RD.dynamics(model, xt, ut)
    println(x_next)
    x_list[:,i+1] = xt
    tip_pos[:,i+1] = [cos(xt[1])+cos(xt[1]+xt[2]), sin(xt[1])+sin(xt[1]+xt[2])]
    
    xt = x_next
    ref_tip_pos = [cos(curr_x_ref[1])+cos(curr_x_ref[1]+curr_x_ref[2]), sin(curr_x_ref[1])+sin(curr_x_ref[1]+curr_x_ref[2])]
    tip_diff[i] = norm(tip_pos[:,i+1] - ref_tip_pos)
    
end

gr(size = (2250, 1965))
# Plots.scalefontsizes(0.3)
plt = plot(tip_pos[1,:],tip_pos[2,:], aspect_ratio=:equal)
display(plt)
plot(t_list,tip_diff)
