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

include("double_pendulum_mc.jl")
include("double_pendulum_costfunctions_mc.jl")


# read reference trajectory and gains 
# CSV.write("mc_K.csv",  DataFrame(vcat(Kout...)), writeheader=false)
# CSV.write("mc_d.csv",  DataFrame(d), writeheader=false)
# CSV.write("mc_xref.csv",  DataFrame(X2), writeheader=false)
# CSV.write("mc_uref.csv",  DataFrame(ctrl), writeheader=false)
# CSV.write("time.csv",  DataFrame([1:N-1]*dt), writeheader=false)

df = DataFrame(CSV.File("mc_K.csv",header=false))
mtx = convert(Matrix,df)
mc_K = [reshape(row,2,12) for row in eachrow(mtx)]
df = DataFrame(CSV.File("mc_d.csv",header=false))
mtx = convert(Matrix,df)
mc_d = [d for d in eachcol(mtx)]
df = DataFrame(CSV.File("mc_xref.csv",header=false))
mtx = convert(Matrix,df)
mc_xref = [d for d in eachcol(mtx)]
df = DataFrame(CSV.File("mc_uref.csv",header=false))
mtx = convert(Matrix,df)
mc_uref = [d for d in eachcol(mtx)]
df = DataFrame(CSV.File("time.csv",header=false))
t_list = convert(Matrix,df)

# simulate
th1 = -pi/2
th2 = -pi/2
d1 = .5*model.l1*[cos(th1);sin(th1)]
d2 = .5*model.l2*[cos(th2);sin(th2)]
x0 = [d1; th1; 2*d1 + d2; th2; zeros(6)]
prev_t = 0
xt = x0
x_list = zeros(12, size(t_list,1)+1)
x_list[:,1] = x0
tip_pos = zeros(2, size(t_list,1)+1)
tip_pos[:,1] = [xt[4]+0.5*cos(xt[6]), xt[5]+0.5*sin(xt[6])]

tip_diff = zeros(size(t_list,1))
for i in 1:size(t_list,1)
    global prev_t
    global xt
    dt = t_list[i] - prev_t
    prev_t = t_list[i]
    curr_x_ref = mc_xref[i+1]
    ut = mc_uref[i] + mc_K[i]*(xt - mc_xref[i])
    z = KnotPoint(xt,ut,dt)
    x_next = RD.discrete_dynamics(RD.Euler, model, z)
    println(x_next)
    xt = x_next
    x_list[:,i+1] = xt
    tip_pos[:,i+1] = [xt[4]+0.5*cos(xt[6]), xt[5]+0.5*sin(xt[6])]
    ref_tip_pos = [curr_x_ref[4]+0.5*cos(curr_x_ref[6]), curr_x_ref[5]+0.5*sin(curr_x_ref[6])]
    tip_diff[i] = norm(tip_pos[:,i+1] - ref_tip_pos)
end

gr(size = (2250, 1965))
# Plots.scalefontsizes(0.3)
plt = plot(tip_pos[1,:],tip_pos[2,:], aspect_ratio=:equal)
display(plt)
plot(t_list,tip_diff)
