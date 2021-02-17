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
import RobotZoo.Pendulum

using Altro: iLQRSolver

model = Pendulum()
n,m = size(model);

N = 1001
tf = 5.
dt = tf/(N-1)

x0 = @SVector zeros(n)
xf = @SVector [pi, 0];  # i.e. swing up

Q = 1.0e-2*Diagonal(@SVector ones(n))
Qf = 100.0*Diagonal(@SVector ones(n))
R = 1.0e-1*Diagonal(@SVector ones(m))
obj = LQRObjective(Q,R,Qf,xf,N);

# Create Empty ConstraintList
conSet = ConstraintList(n,m,N)

# Control Bounds
u_bnd = 3.0
bnd = BoundConstraint(n,m, u_min=-u_bnd, u_max=u_bnd)
add_constraint!(conSet, bnd, 1:N-1)

# Goal Constraint
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

altro = ALTROSolver(prob, opts)
set_options!(altro, show_summary=true)
solve!(altro);

X = hcat(Vector.(states(altro))...)
U = controls(altro)

display(plot(X'))
display(plot(U))

ilqr = Altro.iLQRSolver(prob, opts)
initial_controls!(ilqr, U0)
solve!(ilqr);
