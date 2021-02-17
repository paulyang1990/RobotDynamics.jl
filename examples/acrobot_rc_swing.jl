using RobotDynamics
using Altro
using TrajectoryOptimization
using StaticArrays
using LinearAlgebra
using Plots

const TO = TrajectoryOptimization
const RD = RobotDynamics
import RobotZoo.Acrobot
using Altro: iLQRSolver

model = Acrobot(l = @SVector[.5,.5],J = @SVector[1/12,1/12])
n,m = size(model);

N = 1001
tf = 5.
dt = tf/(N-1)

x0 = @SVector zeros(n)
xf = @SVector [pi, 0, 0, 0];  # i.e. swing up

Q = 1.0e-2*Diagonal(@SVector ones(n))
Qf = 100.0*Diagonal(@SVector ones(n))
R = 1.0e-1*Diagonal(@SVector ones(m))
obj = LQRObjective(Q,R,Qf,xf,N);

conSet = ConstraintList(n,m,N)
goal = GoalConstraint(xf)
add_constraint!(conSet, goal, N)

prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet);

u0 = @SVector fill(0.01,m)
U0 = [u0 for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);

opts = SolverOptions(
    cost_tolerance_intermediate=1e-2,
    penalty_scaling=10.,
    penalty_initial=1.0
)

# ALTRO
altro = ALTROSolver(prob, opts)
set_options!(altro, show_summary=true)
solve!(altro);

X = hcat(Vector.(states(altro))...)
U = controls(altro)

display(plot(X[1:2,:]'))
display(plot(U))

# ILQR
ilqr = Altro.iLQRSolver(prob, opts)
initial_controls!(ilqr, U0)
solve!(ilqr);

X = hcat(Vector.(states(ilqr))...)
U = controls(ilqr)

display(plot(X[1:2,:]'))
display(plot(U))