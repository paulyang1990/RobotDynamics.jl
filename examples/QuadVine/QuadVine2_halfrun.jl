include("QuadVine2.jl")
include("Quad_vis.jl")

# model and timing
model = QuadVine(2) 
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

# interpolated trajectory
aa = AngleAxis(pi/4, 1, -1, 0)
dx = [.8,.8,0]
Xtrack = interpolate_config(model, shift, dx, aa, N)

# objective
Qw = Diagonal(SVector{n}([fill(25, nq-7); fill(25, 7); fill(25, nv-6); fill(25, 6)]))
Q = Diagonal(SVector{n}([fill(1e-5, nq); fill(1e-5, nv)]))
R = Diagonal(SVector{m}([fill(1e0, m-1);100]))

costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(x0), SVector{m}(trim_controls(model)); w=1e-4) for i=1:N]
# edit objective
for i = 1:N 
        Qi, Ri, wi = Q, R, 1e-4
        if i==N
                Qi, Ri, wi = Qw, 0R, [.01,10.0,100.0]
        end
        costfuns[i] = TO.LieLQRCost(RD.LieState(model), Qi, Ri, SVector{n}(Xtrack[i]), SVector{m}(trim_controls(model)); w=wi)
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
solve!(ilqr);

# plots
display(plot(hcat(Vector.(Vec.(ilqr.K[1:end]))...)',legend=false))
display(plot(hcat(Vector.(Vec.(ilqr.d[1:end]))...)',legend=false))
display(plot(hcat(Vector.(controls(ilqr))...)',xlabel="timestep",ylabel="controls"))

# resolve without remaking ilqr
initial_controls!(prob, U0)
rollout!(prob);
solve!(ilqr);

# plots
display(plot(hcat(Vector.(Vec.(ilqr.K[1:end]))...)',legend=false))
display(plot(hcat(Vector.(Vec.(ilqr.d[1:end]))...)',legend=false))
display(plot(hcat(Vector.(controls(ilqr))...)',xlabel="timestep",ylabel="controls"))

# # animations
# vis = visualize!(model, states(altro), dt)
# goal_p = GeometryBasics.Point(SVector{3}(xw[nq-7 .+ (1:3)]))
# goal = GeometryBasics.HyperSphere{3,Float64}(goal_p, .3)
# setobject!(vis["goal"], goal, MeshPhongMaterial(color=RGBA(1,0,1, .5)))
# xlim = GeometryBasics.Rect{3,Float64}(Vec(-3.0, -3.0, 0.0), Vec(3+x_max[1], 3+x_max[2], 3.0))
# setobject!(vis["dronespace"], xlim, MeshPhongMaterial(color=RGBA(0,1,0, .3)))
