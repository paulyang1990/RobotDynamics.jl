include("nPenJan.jl")

# model
model = nPenJan()
mech = model.mech
n,m = size(model)
dt = mech.Δt = 1e-5
N = 100
tf = (N-1)*dt  

x0 = [0;0;-.5;zeros(3);1;zeros(6);0;0;-1.5;zeros(3);1;zeros(6)]
xf = [0;0;.5;zeros(3);0;1;zeros(5);0;0;1.5;zeros(3);0;1;zeros(5)]
u0 = [1.,0,0,1.,0,0]

# objective
Qf = Diagonal(@SVector fill(250., n))
Q = Diagonal(@SVector fill(.01/dt, n))
R = Diagonal(@SVector fill(.01/dt, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=.1) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=250.0)
obj = Objective(costfuns);

# problem
prob = Problem(model, obj, xf, tf, x0=x0);

# initial controls
U0 = [u0 for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);
plot_traj_jan(states(prob), controls(prob))

# ilqr
opts = SolverOptions(verbose=7, static_bp=0, iterations=1, cost_tolerance=1e-2)
ilqr = Altro.iLQRSolver(prob, opts);
solve!(ilqr);
plot_traj_jan(states(ilqr), controls(ilqr))


################################ OG ################################
include("nPendulumSpherical.jl")
include("nPendulum3D_visualize.jl")

masses = [body.m for body in mech.bodies]
inertias = [Diagonal([body.J[1,1], body.J[2,2], body.J[3,3]]) for body in mech.bodies]
model2 = nPendulumSpherical{UnitQuaternion{Float64},Float64}(masses, ones(2), .1ones(2), inertias)

nq, nv, nc = mc_dims(model2)
x02 = [generate_config(model2, fill(0, model2.nb)); zeros(nv)]
xf2 = [generate_config(model2, fill(RotX(pi), model2.nb)); zeros(nv)]

costfuns = [TO.LieLQRCost(RD.LieState(model2), Q, R, SVector{n}(xf2); w=.1) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model2), Qf, R, SVector{n}(xf2); w=250.0)
obj2 = Objective(costfuns);
prob2 = Problem(model2, obj2, xf2, tf, x0=x02);

# initial controls
initial_controls!(prob2, U0)
rollout!(prob2);
plot_traj(states(prob2), controls(prob2))

# ilqr
ilqr2 = Altro.iLQRSolver(prob2, opts);
solve!(ilqr2);
plot_traj(states(ilqr2), controls(ilqr2))
# plot_angles(states(ilqr2), controls(ilqr2))
# visualize!(model2,states(ilqr2),dt)

extrema.(og_to_jan.(Vector.(states(prob2))) .- states(prob))

p = [1, 2, 3, 7, 8, 9, 13, 14, 15,19, 20, 21, 4, 5, 6,  10, 11, 12, 16, 17, 18, 22, 23, 24]
# our state x1 q1 x2 q2  v1 w1 v2 w2 
Aj = ilqr.D[1].A 
A = ilqr2.D[1].A
Ape = Aj[:,p]
A2 = Ape[p,:]
println(extrema(A-A2))


# display(plot(hcat(Vector.(vec.(ilqr.K[1:end-1]))...)',legend=false))
# display(plot(hcat(Vector.(vec.(ilqr.K[1:end]))...)',legend=false))
display(plot(hcat(Vector.(vec.(ilqr.d[1:end]))...)',legend=false))

# display(plot(hcat(Vector.(vec.(ilqr2.K[1:end-1]))...)',legend=false))
# display(plot(hcat(Vector.(vec.(ilqr2.K[1:end]))...)',legend=false))
display(plot(hcat(Vector.(vec.(ilqr2.d[1:end]))...)',legend=false))
