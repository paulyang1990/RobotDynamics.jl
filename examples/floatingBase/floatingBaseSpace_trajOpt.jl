include("floatingBaseSpace.jl")
using Profile
using TimerOutputs
const to = TimerOutput()
model = FloatingSpace()
n,m = size(model)
x0 = generate_config(model, [0.01;0.01;0.01;0.01], fill.(0.01,model.nb))

xf = generate_config(model, [0.3;0.3;1.0;pi/6], fill.(pi/6,model.nb))

# trajectory 
N = 100   
dt = 0.005                  # number of knot points
tf = (N-1)*dt           # final time

U0 = @SVector fill(0.00001, m)
U_list = [U0 for k = 1:N-1]


# objective
Qf = Diagonal(@SVector fill(550., n))
Q = Diagonal(@SVector fill(1e-2, n))
R = Diagonal(@SVector fill(1e-3, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=1e-1) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=550.0)
obj = Objective(costfuns);

# # problem
prob = Problem(model, obj, xf, tf, x0=x0);
opts = SolverOptions(verbose=7, static_bp=0, iterations=150, cost_tolerance=1e-4)
ilqr = Altro.iLQRSolver(prob, opts);
TimerOutputs.enable_debug_timings(Altro)
Altro.timeit_debug_enabled()
Altro.initialize!(ilqr)
solve!(ilqr);

"""
ilqr.stats.to for FloatingSpace()
 ────────────────────────────────────────────────────────────────────────
                                 Time                   Allocations      
                         ──────────────────────   ───────────────────────
    Tot / % measured:         88.3s / 13.7%           8.49GiB / 37.9%    

 Section         ncalls     time   %tot     avg     alloc   %tot      avg
 ────────────────────────────────────────────────────────────────────────
 forward pass        18    5.44s  45.0%   302ms   2.57GiB  79.8%   146MiB
 dynamics jac        18    2.92s  24.2%   162ms    109MiB  3.31%  6.06MiB
 backward pass       18    2.82s  23.3%   157ms    362MiB  11.0%  20.1MiB
   calc ctg       1.78k    2.02s  16.7%  1.13ms    192MiB  5.82%   110KiB
   calc gains     1.78k    798ms  6.61%   448μs    171MiB  5.18%  98.1KiB
 diff jac            18    466ms  3.86%  25.9ms    104MiB  3.17%  5.80MiB
 cost err            18    405ms  3.35%  22.5ms   89.8MiB  2.72%  4.99MiB
 err jac             18   29.6ms  0.25%  1.65ms     0.00B  0.00%    0.00B
 cost exp            18   2.93ms  0.02%   163μs     0.00B  0.00%    0.00B
 ────────────────────────────────────────────────────────────────────────
"""

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


# k = 1
# if ilqr.Z[k].dt == 0
#     z = copy(ilqr.Z[k])
#     z.dt = ilqr.Z[1].dt
#     Altro.discrete_jacobian_MC!(Altro.integration(ilqr), ilqr.D[k].∇f, ilqr.D[k].G, ilqr.model, z)
# else
#     Altro.discrete_jacobian_MC!(Altro.integration(ilqr), ilqr.D[k].∇f, ilqr.D[k].G, ilqr.model, ilqr.Z[k])
# end



X_list = states(ilqr)

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