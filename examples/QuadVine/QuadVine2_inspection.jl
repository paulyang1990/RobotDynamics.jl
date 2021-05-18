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
# TrajOptPlots._set_mesh!(vis["robot"], model; color=RGBA(.3,.3,.3,1))
# TrajOptPlots.visualize!(vis, model, SVector{n}(xf))
# render(vis)

# objective
Qf = Diagonal(SVector{n}([fill(2.5, nq-7); fill(50, 7); fill(2.5, nv-6); fill(50, 6)]))
Q = Diagonal(SVector{n}([fill(1e-5, nq); fill(1e-5, nv)]))
R = Diagonal(SVector{m}([fill(.01,4);.7;.7;10]))

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

# problem
prob = Problem(model, obj, x0, tf, x0=x0);

# initial controls
u0 = trim_controls(model)
U0 = [SVector{m}(u0) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);

# solve
opts = SolverOptions(verbose=7, static_bp=0, iterations=10, cost_tolerance=1e-3)
ilqr = Altro.iLQRSolver(prob, opts);
set_options!(ilqr, iterations=20)
solve!(ilqr);
X = states(ilqr)
@show norm((xf-X[end])[(nq-7) .+ (1:3)])

# plots
display(plot(hcat(Vector.(Vec.(ilqr.K[1:end]))...)',legend=false))
display(plot(hcat(Vector.(Vec.(ilqr.d[1:end]))...)',legend=false))
display(plot(hcat(Vector.(controls(ilqr))...)',xlabel="timestep",ylabel="controls"))

# animations
vis = visualize!(model, states(ilqr), dt)
r = model.radii[end]
l = model.lengths[end]
link = GeometryBasics.Rect{3,Float64}(Vec(-r/2, -r/2, -l/2), Vec(r, r, l))
setobject!(vis["goal"], link, MeshPhongMaterial(color=RGBA(1,0,0,.5)))
settransform!(vis["goal"], AffineMap(UnitQuaternion(xf[(nq-7) .+ (4:7)]), xf[(nq-7) .+ (1:3)]))
render(vis)
