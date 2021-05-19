include("QuadVine2.jl")
include("Quad_vis.jl")

# model and timing  
nl = 3 # num vine links
model = QuadVine{UnitQuaternion{Float64},Float64}(
        ones(nl+1), # masses
        [.2; ones(nl-1); 1], # lengths
        [1;fill(.1,nl)], # radii
        fill(Diagonal(ones(3)),nl+1)) # inertias
nq, nv, nc = mc_dims(model)
n,m = size(model)
N = 500
dt = 5e-3
tf = (N-1)*dt     

# initial and final conditions
shift = [0,0, sum(model.lengths)+.1]
x0 = [generate_config(model, zeros(model.nb)); zeros(nv)]
shift_pos!(model, x0, shift)
@assert norm(max_constraints(model, x0)) < 1e-6

xf = [generate_config(model, pi/2/model.nb * collect(1:model.nb)); zeros(nv)]
shift_pos!(model, xf, shift)
@assert norm(max_constraints(model, x0)) < 1e-6

# vis = Visualizer()
TrajOptPlots._set_mesh!(vis["robot"], model; color=RGBA(.3,.3,.3,1))
TrajOptPlots.visualize!(vis, model, SVector{n}(xf))
# render(vis)

# objective
Qf = Diagonal(SVector{n}([fill(2.5, nq-7); fill(50, 7); fill(2.5, nv-6); fill(50, 6)]))
Q = Diagonal(SVector{n}([fill(1e-5, nq); fill(1e-5, nv)]))
R = Diagonal(SVector{m}([fill(.01,4);.01;.01;10]))

costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf), SVector{m}(trim_controls(model)); w=1e-4) for i=1:N]
# edit objective
for i = 1:N 
        Qi, Ri, wi = Q, R, 1e-4
        if i==N
                Qi, Ri, wi = Qf, 0R, [.01,fill(1,model.nb-2)...,300.0]
        end
        costfuns[i] = TO.LieLQRCost(RD.LieState(model), Qi, Ri, SVector{n}(xf), SVector{m}(trim_controls(model)); w=wi)
end

obj = Objective(costfuns);

# constraints
conSet = ConstraintList(n,m,N)
bnd = BoundConstraint(n, m, u_min=[fill(0.,4);fill(-6,m-4)], u_max=[fill(Inf,4);fill(6,m-4)])
add_constraint!(conSet, bnd, 1:N-1)
goalcon = GoalConstraint(xf, SVector{13}([collect(nq-7 .+ (1:7));collect(n-6 .+ (1:6))]))
add_constraint!(conSet, goalcon, N)

# problem
prob = Problem(model, obj, x0, tf, x0=x0, constraints=conSet);

# initial controls
u0 = trim_controls(model)
U0 = [SVector{m}(u0) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);

function inspection_w_τ_constraints(vine_u_max)
        # reset bnd
        bnd = BoundConstraint(n, m, u_min=[fill(0.,4);fill(-vine_u_max,m-4)], u_max=[fill(Inf,4);fill(vine_u_max,m-4)])
        conSet.constraints[1]=bnd

        # problem
        prob = Problem(model, obj, x0, tf, x0=x0, constraints=conSet);
        initial_controls!(prob, U0)

        # solve
        opts = SolverOptions(verbose=7, static_bp=0, iterations=10, cost_tolerance=1e-3)
        altro = ALTROSolver(prob, opts);
        set_options!(altro, iterations=70, reset_penalties=true,
                constraint_force_reg = 1e-4, iterations_outer=5,
                cost_tolerance_intermediate=1e-3, penalty_initial=.1,
                cost_tolerance=1e-4, constraint_tolerance=5e-2)
        solve!(altro);
        X = states(altro)
        @show norm((xf-X[end])[(nq-7) .+ (1:3)])

        # plots
        display(plot(hcat(Vector.(controls(altro))...)',xlabel="timestep",ylabel="controls"))
        droney = [x[2] for x in states(altro)]
        dronez = [x[3] for x in states(altro)]
        eey = [x[nq-5] for x in states(altro)]
        eez = [x[nq-4] for x in states(altro)]
        display(plot([droney eey],[dronez eez],aspect_ratio=1,xlabel="drone y",ylabel="drone z"))

        return  states(altro), controls(altro)
end

# run comparisons
X_all = []
U_all = []
for i = [0,2,4,6]
        global X_all, U_all
        X, U = inspection_w_τ_constraints(i)
        push!(X_all, X)
        push!(U_all, U)
end
# plot vine control comparison
display(plot([[u[5] for u in U] for U in U_all]))

# # save results
# using JLD2
# @save joinpath(@__DIR__,"data/inspection/comparisons.jld2") X_all U_all

# # single animation
# vis = visualize!(model, states(altro), dt)
# r = model.radii[end]
# l = model.lengths[end]
# link = GeometryBasics.Rect{3,Float64}(Vec(-r/2, -r/2, -l/2), Vec(r, r, l))
# setobject!(vis["goal"], link, MeshPhongMaterial(color=RGBA(1,0,0,.5)))
# settransform!(vis["goal"], AffineMap(UnitQuaternion(xf[(nq-7) .+ (4:7)]), xf[(nq-7) .+ (1:3)]))
# render(vis)

# # compare animations
# using TrajOptPlots
# vis = Visualizer()
# TrajOptPlots.set_mesh!(vis, prob.model)
# render(vis)
# TrajOptPlots.visualize!(vis, model, tf, X_all[1], X_all[2], X_all[3], colors=[colorant"blue", colorant"purple", colorant"red"])
# # TrajOptPlots.clear_copies!(vis)

# waypoints for specific solve
initial_controls!(prob, U_all[3])
rollout!(prob);
waypoints!(vis, prob, length=10,
        color=RGBA(1,0,0,.1), 
        color_end=RGBA(0,0,1,1.))
