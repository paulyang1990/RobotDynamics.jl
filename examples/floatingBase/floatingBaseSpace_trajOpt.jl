include("floatingBaseSpace.jl")
using Profile
using TimerOutputs
model = FloatingSpaceOrth(3)
n,m = size(model)
x0 = generate_config(model, [0.01;0.01;0.01;0.01], fill.(0.01,model.nb))

xf = generate_config(model, [0.3;0.3;1.0;pi/6], fill.(pi/6,model.nb))

# TODO run these functions to use them to trigger compile 
n,m = size(model)
n̄ = state_diff_size(model)
@show n
@show n̄
x0 = generate_config_with_rand_vel(model, [2.0;2.0;1.0;pi/4], fill.(pi/4,model.nb))

U = 0.01*rand(6+model.nb)
dt = 0.001;
λ_init = 10*ones(5*model.nb)
λ = λ_init
x = x0
@time x1, λ = discrete_dynamics(model,x, U, λ, dt)
@time fdyn(model,x1, x, U, λ, dt)
# println(norm(fdyn(model,x1, x, u, λ, dt)))
x = x0;
for i=1:5
    println("step: ",i)
    @time x1, λ = discrete_dynamics(model,x, U, λ, dt)
    println(norm(fdyn(model,x1, x, U, λ, dt)))
    println(norm(g(model,x1)))
    x = x1
end

# trajectory 
N = 100   
dt = 0.005                  # number of knot points
tf = (N-1)*dt           # final time

U0 = @SVector fill(0.00001, m)
U_list = [U0 for k = 1:N-1]

x0 = generate_config(model, [0.01;0.01;0.01;0.01], fill.(0.01,model.nb))

xf = generate_config(model, [0.3;0.3;1.0;pi/6], fill.(pi/6,model.nb))

# objective
Qf = Diagonal(@SVector fill(550., n))
Q = Diagonal(@SVector fill(1e-2, n))
R = Diagonal(@SVector fill(1e-3, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=1e-1) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=550.0)
obj = Objective(costfuns);

# constraints
# Create Empty ConstraintList
conSet = ConstraintList(n,m,N)
# constraint 1, Control Bounds
# bnd = BoundConstraint(n,m, u_min=fill(-150,m), u_max=fill(150,m))
# add_constraint!(conSet, bnd, 1:N-1)

# constraint 2, limit the velocity of the tip 
# linear constraint on link nb+1
# bnd = BoundConstraint(n,m, u_min=fill(-150,m), u_max=fill(150,m))
# add_constraint!(conSet, bnd, 1:N-1)

const to = TimerOutput()
# # problem
prob = Problem(model, obj, xf, tf, x0=x0, constraints=conSet);

initial_controls!(prob, U_list)
opts = SolverOptions(verbose=7, static_bp=0, iterations=150, cost_tolerance=1e-4, constraint_tolerance=1e-4)
altro = ALTROSolver(prob, opts)
set_options!(altro, show_summary=true)
solve!(altro);

# n,m,N = size(ilqr)
# J = Inf
# _J = TO.get_J(ilqr.obj)
# J_prev = sum(_J)
# grad_only = false
# J = Altro.step!(ilqr, J_prev, grad_only)
# to = ilqr.stats.to
# init = ilqr.opts.reuse_jacobians  # force recalculation if not reusing
# @timeit_debug to "diff jac"     TO.state_diff_jacobian!(ilqr.G, ilqr.model, ilqr.Z)

# @timeit_debug to "dynamics jac" TO.dynamics_expansion!(Altro.integration(ilqr), ilqr.D, ilqr.model, ilqr.Z)

# @timeit_debug to "err jac"      TO.error_expansion!(ilqr.D, ilqr.model, ilqr.G)
# @timeit_debug to "cost exp"     TO.cost_expansion!(ilqr.quad_obj, ilqr.obj, ilqr.Z, init, true)
# @timeit_debug to "cost err"     TO.error_expansion!(ilqr.E, ilqr.quad_obj, ilqr.model, ilqr.Z, ilqr.G)
# @timeit_debug to "backward pass" ΔV = Altro.backwardpass!(ilqr)
# @timeit_debug to "forward pass" Altro.forwardpass!(ilqr, ΔV, J_prev)

"""
ilqr.stats.to for FloatingSpace(2)
────────────────────────────────────────────────────────────────────────
Time                   Allocations      
──────────────────────   ───────────────────────
Tot / % measured:         59.0s / 68.9%           9.49GiB / 98.3%    

Section         ncalls     time   %tot     avg     alloc   %tot      avg
────────────────────────────────────────────────────────────────────────
forward pass        31    36.7s  90.3%   1.18s   8.70GiB  93.3%   287MiB
dynamics jac        31    2.81s  6.91%  90.7ms    337MiB  3.52%  10.9MiB
backward pass       31    819ms  2.01%  26.4ms    302MiB  3.16%  9.75MiB
calc gains     3.07k    638ms  1.57%   208μs    301MiB  3.15%   100KiB
calc ctg       3.07k    177ms  0.43%  57.6μs   1.22MiB  0.01%     416B
cost err            31    145ms  0.36%  4.68ms   1.89MiB  0.02%  62.5KiB
err jac             31    125ms  0.31%  4.02ms     0.00B  0.00%    0.00B
diff jac            31   35.0ms  0.09%  1.13ms   1.47MiB  0.02%  48.4KiB
cost exp            31   9.71ms  0.02%   313μs     0.00B  0.00%    0.00B
────────────────────────────────────────────────────────────────────────
"""

"""
ilqr.stats.to for FloatingSpace(6)
────────────────────────────────────────────────────────────────────────
Time                   Allocations      
──────────────────────   ───────────────────────
Tot / % measured:         6429s / 13.6%            239GiB / 71.7%    

Section         ncalls     time   %tot     avg     alloc   %tot      avg
────────────────────────────────────────────────────────────────────────
forward pass        20     841s  96.3%   42.0s    168GiB  98.1%  8.40GiB
backward pass       20    22.4s  2.56%   1.12s   1.87GiB  1.09%  96.0MiB
calc ctg       1.98k    19.6s  2.24%  9.90ms   0.96GiB  0.56%   510KiB
calc gains     1.98k    2.77s  0.32%  1.40ms    934MiB  0.53%   483KiB
dynamics jac        20    7.64s  0.87%   382ms    920MiB  0.52%  46.0MiB
diff jac            20    1.06s  0.12%  53.1ms    268MiB  0.15%  13.4MiB
cost err            20    987ms  0.11%  49.3ms    226MiB  0.13%  11.3MiB
err jac             20    456ms  0.05%  22.8ms     0.00B  0.00%    0.00B
cost exp            20   22.0ms  0.00%  1.10ms     0.00B  0.00%    0.00B
────────────────────────────────────────────────────────────────────────
"""



# k = 1
# if ilqr.Z[k].dt == 0
#     z = copy(ilqr.Z[k])
#     z.dt = ilqr.Z[1].dt
#     Altro.discrete_jacobian_MC!(Altro.integration(ilqr), ilqr.D[k].∇f, ilqr.D[k].G, ilqr.model, z)
# else
#     Altro.discrete_jacobian_MC!(Altro.integration(ilqr), ilqr.D[k].∇f, ilqr.D[k].G, ilqr.model, ilqr.Z[k])
# end

X_list = states(altro)

mech = vis_mech_generation(model)
steps = Base.OneTo(Int(N))
storage = CD.Storage{Float64}(steps,length(mech.bodies))
for idx = 1:N
    setStates!(model,mech,X_list[idx])
    for i=1:model.nb+1
        storage.x[i][idx] = mech.bodies[i].state.xc
        storage.v[i][idx] = mech.bodies[i].state.vc
        storage.q[i][idx] = mech.bodies[i].state.qc
        storage.ω[i][idx] = mech.bodies[i].state.ωc
    end
end
visualize(mech,storage, env = "editor")

U_list = controls(altro)

using Plots
# plot all controls
plot(1:N-1, U_list)

# a final simulation pass to get "real" state trajectory
λ_init = zeros(5*model.nb)
λ = λ_init
Xfinal_list = copy(X_list)
Xfinal_list[1] = SVector{n}(x0)
for idx = 1:N-1
    println("step: ",idx)
    x1, λ1 = discrete_dynamics(model,Xfinal_list[idx], U_list[idx], λ, dt)
    println(norm(fdyn(model,x1, Xfinal_list[idx], U_list[idx], λ1, dt)))
    println(norm(g(model,x1)))
    setStates!(model,mech,x1)
    Xfinal_list[idx+1] = SVector{n}(x1)
    λ = λ1
end

# plot velocity of the last link
statea_inds!(model, model.nb+1)
# p = [Xfinal_list[i][model.v_ainds[1]] for i=1:N]
# plot(1:N, p)
# p = [Xfinal_list[i][model.v_ainds[2]] for i=1:N]
# plot!(1:N, p)
p = zeros(N,3)
for dim=1:3
p[:,dim] .= [Xfinal_list[i][model.v_ainds[dim]] for i=1:N]
end
plot(1:N, p)

