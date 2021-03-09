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

include("nPenOrth_rc.jl")


N = 300   
dt = 0.01                  # number of knot points
tf = (N-1)*dt           # final time

# initial and final conditions
th1 = -pi
th2 = pi/3
th3 = -pi/3
th4 = pi/3
x0 = [th1; th2; th3; th4; zeros(4)]
th1 = 0
th2 = 0
th3 = 0
th4 = 0
xf = [th1; th2; th3; th4; zeros(4)]


# objective
Q = zeros(n)
# Q[1] = Q[2] = 1e-3/dt
Q[1] = Q[2]= Q[3]= Q[4] = 1e-2
Q[5] = Q[6]= Q[7]= Q[8] = 1e-2
Q = Diagonal(SVector{n}(Q))
R = Diagonal(@SVector fill(1e-5, m))
Qf = zeros(n)
Qf[1] = Qf[2] = 4500
Qf[3] = Qf[4] = 4500
Qf = Diagonal(SVector{n}(Qf))
obj = LQRObjective(Q,R,Qf,xf,N)

# constraints
cons = ConstraintList(n,m,N)

# problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=cons);

# intial rollout with random controls
U0 = [@SVector fill(0.0, m) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);


# solve problem
opts = SolverOptions(verbose=7,static_bp=0)
solver = iLQRSolver(prob, opts);
solve!(solver);

K = solver.K
d = solver.d

X2 = states(solver.Z)


using Plots
X = hcat(Vector.(states(solver))...)
display(plot(X[1:4,:]'))


mc_x = rc_to_mc(model,X2)

using ConstrainedControl
using ConstrainedDynamics
using ConstrainedDynamicsVis

const CC = ConstrainedControl
const CD = ConstrainedDynamics
const CDV = ConstrainedDynamicsVis

include("nPenOrth_Jan.jl")
num_links = 4
vismodel = nPenJanOrth(num_links)
mech = vismodel.mech

steps = Base.OneTo(length(mc_x))
storage = CD.Storage{Float64}(steps,model.nb)
for step = 1:length(mc_x)
    
    setStates!(mech,mc_x[step])
    for i=1:model.nb
        storage.x[i][step] = mech.bodies[i].state.xc
        storage.v[i][step] = mech.bodies[i].state.vc
        storage.q[i][step] = mech.bodies[i].state.qc
        storage.ω[i][step] = mech.bodies[i].state.ωc
    end
end
CDV.visualize(mech,storage)