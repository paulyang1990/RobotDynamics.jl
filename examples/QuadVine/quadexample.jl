using RobotDynamics, Rotations, RobotZoo
using TrajectoryOptimization
using StaticArrays, LinearAlgebra
using TrajOptPlots
using MeshCat
using Plots
using FileIO, MeshIO

# Set up model and discretization
model = RobotZoo.Quadrotor();
n,m = size(model)
N = 101                # number of knot points
tf = 5.0               # total time (sec)
dt = tf/(N-1)          # time step (sec)

x0_pos = SA[0, -10, 1.]
xf_pos = SA[0, +10, 1.]
x0 = RobotDynamics.build_state(model, x0_pos, UnitQuaternion(I), zeros(3), zeros(3))
xf = RobotDynamics.build_state(model, xf_pos, UnitQuaternion(I), zeros(3), zeros(3));

# Set up waypoints
wpts = [SA[+10, 0, 1.],
        SA[-10, 0, 1.],
        xf_pos]
times = [33, 66, 101]   # in knot points

# Set up nominal costs
Q = Diagonal(RobotDynamics.fill_state(model, 1e-5, 1e-5, 1e-3, 1e-3))
R = Diagonal(@SVector fill(1e-4, 4))
q_nom = UnitQuaternion(I)
v_nom = zeros(3)
ω_nom = zeros(3)
x_nom = RobotDynamics.build_state(model, zeros(3), q_nom, v_nom, ω_nom)
cost_nom = LQRCost(Q, R, x_nom)

# Set up waypoint costs
Qw_diag = RobotDynamics.fill_state(model, 1e3,1,1,1)
Qf_diag = RobotDynamics.fill_state(model, 10., 100, 10, 10)
costs = map(1:length(wpts)) do i
    r = wpts[i]
    xg = RobotDynamics.build_state(model, r, q_nom, v_nom, ω_nom)
    if times[i] == N
        Q = Diagonal(Qf_diag)
    else
        Q = Diagonal(1e-3*Qw_diag)
    end

    LQRCost(Q, R, xg)
end

# Build Objective
costs_all = map(1:N) do k
    i = findfirst(x->(x ≥ k), times)
    if k ∈ times
        costs[i]
    else
        cost_nom
    end
end
obj = Objective(costs_all);

u0 = @SVector fill(0.5*model.mass/m, m)
U_hover = [copy(u0) for k = 1:N-1]; # initial hovering control trajectory

conSet = ConstraintList(n,m,N)
bnd = BoundConstraint(n,m, u_min=0.0, u_max=12.0)
add_constraint!(conSet, bnd, 1:N-1)

prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet)
initial_controls!(prob, U_hover)
rollout!(prob);

using Altro
opts = SolverOptions(
    penalty_scaling=100.,
    penalty_initial=0.1,
)

solver = ALTROSolver(prob, opts);
set_options!(solver,verbose=7,iterations=200)
solve!(solver);
println("Cost: ", cost(solver))
println("Constraint violation: ", max_violation(solver))
println("Iterations: ", iterations(solver))

display(plot(hcat(Vector.(controls(solver))...)',xlabel="timestep",ylabel="controls"))
ilqr=solver.solver_al.solver_uncon
display(plot(hcat(Vector.(vec.(ilqr.K[1:end]))...)',legend=false))
display(plot(hcat(Vector.(vec.(ilqr.d[1:end]))...)',legend=false))

function ModifiedMeshFileObject(obj_path::String, material_path::String; scale::T=0.1) where {T}
    obj = MeshFileObject(obj_path)
    rescaled_contents = rescale_contents(obj_path, scale=scale)
    # material = select_material(material_path)
    material = MeshPhongMaterial(color=RGBA(1,1,1, 0.5))

    mod_obj = MeshFileObject(
        rescaled_contents,
        obj.format,
        material,
        obj.resources,
        )
    return mod_obj
end

function rescale_contents(obj_path::String; scale::T=0.1) where T
    lines = readlines(obj_path)
    rescaled_lines = copy(lines)
    for (k,line) in enumerate(lines)
        if length(line) >= 2
            if line[1] == 'v'
                stringvec = split(line, " ")
                vals = map(x->parse(Float64,x),stringvec[end-2:end])
                rescaled_vals = vals .* scale
                rescaled_lines[k] = join([stringvec[1]; string.(rescaled_vals)], " ")
            end
        end
    end
    rescaled_contents = join(rescaled_lines, "\r\n")
    return rescaled_contents
end

function TrajOptPlots._set_mesh!(vis, m::RobotZoo.Quadrotor)
    quad_scaling = 0.3
    obj_path =  joinpath(@__DIR__, "quadrotor.obj")
    rescaled_contents = rescale_contents(obj_path, scale=quad_scaling)

    scaled_obj = MeshFileGeometry(rescaled_contents, "obj")
    setobject!(vis, scaled_obj, MeshPhongMaterial(color=RGBA(.3,.3,.3,1)))
end

vis = Visualizer()
render(vis)
TrajOptPlots.set_mesh!(vis, model)
TrajOptPlots.visualize!(vis, solver);
