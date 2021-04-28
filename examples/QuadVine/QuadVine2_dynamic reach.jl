include("QuadVine2.jl")
include("Quad_vis.jl")

# model and timing
model = QuadVine(3) 
nq, nv, nc = mc_dims(model)
n,m = size(model)
N = 900
dt = 5e-3
tf = (N-1)*dt     

# initial and final conditions
x0 = [generate_config(model, zeros(model.nb)); zeros(nv)]
@assert norm(max_constraints(model, x0)) < 1e-6

xf = [generate_config(model, [0;fill(pi/2,model.nb-1)]); zeros(nv)]
@assert norm(max_constraints(model, xf)) < 1e-6

# objective
Qf = Diagonal(SVector{n}([25; 25; 25; fill(1e-4, 4); # drone 
                        fill(1e-4, nq-14); # other links
                        55; 55; 1e-4; fill(1e-4, 4); # final link 
                        fill(10., nv)])) # velocities
Q = Diagonal(SVector{n}([fill(1e-2/dt, 3); fill(1e-4/dt, nq-3); fill(1e-4/dt, nv)]))
R = Diagonal(@SVector fill(1e-3/dt, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf), SVector{m}(trim_controls(model)); w=1e-4) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf), SVector{m}(trim_controls(model)); w=1e-4)
obj = Objective(costfuns);

# constraints
conSet = ConstraintList(n,m,N)
bnd = BoundConstraint(n, m, u_min=zeros(7), 
                            u_max=fill(15,7))#,
                        #     x_min=[-Inf; -Inf; -.1;fill(-Inf,n-3)],
                        #     x_max=[.3;.3;fill(Inf,n-2)])
add_constraint!(conSet, bnd, 1:N-1)
# add_constraint!(conSet, GoalConstraint(xf, SA[1,2,nq-6,nq-5]), N)

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
        constraint_tolerance=1.0,
        projected_newton=false)

# solve with Altro
altro = ALTROSolver(prob, opts);
set_options!(altro, iterations=20, 
            cost_tolerance=1e-4, constraint_tolerance=1e-3,
            reset_penalties=false)
solve!(altro);
X,U = states(altro), controls(altro)
plot_traj(states(altro), controls(altro))
vis = visualize!(model, states(altro), dt)
goal = GeometryBasics.Rect{3,Float64}(Vec(0, 0, -2.0), Vec(.1, .1, 2.0))
setobject!(vis["goal"], goal, MeshPhongMaterial(color=RGB(0,.9,.3)))
settransform!(vis["goal"], AffineMap(one(UnitQuaternion), xf[nq-7 .+ (1:3)]))

K = altro.solver_al.solver_uncon.K
d = altro.solver_al.solver_uncon.d
display(plot(hcat(Vector.(Vec.(K[1:end]))...)',legend=false))
display(plot(hcat(Vector.(Vec.(d[1:end]))...)',legend=false))

Xmat = hcat(Vector.(Vec.(states(altro)))...)
display(plot(Xmat[nq .+ (1:nv), :]',legend=false))
