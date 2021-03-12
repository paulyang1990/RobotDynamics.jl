include("nPenOrth_mc.jl")

# model
num_links = 2
model = nPenOrthMC(num_links)
n,m = size(model)

# trajectory 
N = 300   
dt = 0.01                  # number of knot points
tf = (N-1)*dt           # final time

# test initial and final state (works)
# x0 = generate_config(model, [pi/2;pi/2;pi/2;pi/2])
# x0 = generate_config(model, [pi/6;0.0;pi/6;pi/6])
# xf = generate_config(model, [0;0.0;0.0;0.0])
x0 = generate_config(model, [pi/6;pi/6])
xf = generate_config(model, [0.0;0.0])

U0 = @SVector fill(0.0, m)
U_list = [U0 for k = 1:N-1]

# simulate the system 
# X_list = [x0 for k = 1:N]
# for i=1:N-1
#     # print("simuluate step: ",i )
#     X_list[i+1] = RD.discrete_dynamics(RD.Euler, model, X_list[i], U_list[i], 1.0, dt)
# end

# objective
Qf = Diagonal(@SVector fill(550., n))
Q = Diagonal(@SVector fill(1e-2, n))
R = Diagonal(@SVector fill(1e-3, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=1e-1) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=550.0)
obj = Objective(costfuns);

# # problem
prob = Problem(model, obj, xf, tf, x0=x0);
# # ILQR
opts = SolverOptions(verbose=7, static_bp=0, iterations=50, cost_tolerance=1e-4)
ilqr = Altro.iLQRSolver(prob, opts);
set_options!(ilqr, iterations=50, cost_tolerance=1e-6)
# # just one step
Altro.initialize!(ilqr)
# Z = ilqr.Z; Z̄ = ilqr.Z̄;
# n,m,N = size(ilqr)
# _J = TO.get_J(ilqr.obj)
# J_prev = sum(_J)
# grad_only = true
# J = Altro.step!(ilqr, J_prev, grad_only)
solve!(ilqr);


X_list = states(ilqr)


using ConstrainedControl
using ConstrainedDynamics
using ConstrainedDynamicsVis

const CC = ConstrainedControl
const CD = ConstrainedDynamics
const CDV = ConstrainedDynamicsVis

include("nPenOrth_Jan.jl")
vismodel = nPenJanOrth(num_links)
mech = vismodel.mech

steps = Base.OneTo(length(X_list))
storage = CD.Storage{Float64}(steps,model.nb)
for step = 1:length(X_list)
    
    setStates!(mech,X_list[step])
    for i=1:model.nb
        storage.x[i][step] = mech.bodies[i].state.xc
        storage.v[i][step] = mech.bodies[i].state.vc
        storage.q[i][step] = mech.bodies[i].state.qc
        storage.ω[i][step] = mech.bodies[i].state.ωc
    end
end
CDV.visualize(mech,storage)