include("QuadVine2.jl")
include("Quad_vis.jl")

# model and timing
model = QuadVine(3) 
nq, nv, nc = mc_dims(model)
n,m = size(model)
N = 500
dt = 5e-3
tf = (N-1)*dt     

# initial and final conditions
x0 = [generate_config(model, zeros(model.nb)); zeros(nv)]
@assert norm(max_constraints(model, x0)) < 1e-6

xf = shift_pos(model, x0, [.5,.5,0])
@assert norm(max_constraints(model, xf)) < 1e-6

# objective
Qf = Diagonal(SVector{n}([fill(1e-4, nq-7); fill(250, 7); fill(250., nv)]))
Q = Diagonal(SVector{n}([fill(1e-4/dt, nq); fill(1e-4/dt, nv)]))
R = Diagonal(@SVector fill(1e-2/dt, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf), SVector{m}(trim_controls(model)); w=1e-4) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf), SVector{m}(trim_controls(model)); w=1e-4)
obj = Objective(costfuns);

# constraints
conSet = ConstraintList(n,m,N)
bnd = BoundConstraint(n, m, u_min=zeros(7), 
                            u_max=fill(Inf,7),
                            x_min=[-Inf; -Inf; -.1;fill(-Inf,n-3)],
                            x_max=[.3;.3;fill(Inf,n-2)])
add_constraint!(conSet, bnd, 1:N-1)

# problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet);

# initial controls
u0 = trim_controls(model)
U0 = [SVector{m}(u0) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);
# plot_traj(states(prob), controls(prob))
# visualize!(model, states(prob), dt)

# options
opts = SolverOptions(verbose=7, static_bp=0, 
        iterations=5, cost_tolerance=1e-3, 
        cost_tolerance_intermediate=1e-3,
        constraint_tolerance=1e-3,
        projected_newton=false)

# solve with Altro
altro = ALTROSolver(prob, opts);
set_options!(altro, iterations=20, 
            cost_tolerance=1e-4, constraint_tolerance=1e-3,
            reset_penalties=false)
solve!(altro);
X,U = states(altro), controls(altro)
plot_traj(states(altro), controls(altro))
visualize!(model, states(altro), dt)

# display(plot(hcat(Vector.(Vec.(ilqr.K[1:end]))...)',legend=false))
# display(plot(hcat(Vector.(Vec.(ilqr.d[1:end]))...)',legend=false))
