include("floatingBaseSpace.jl")
using Profile
using TimerOutputs
const to = TimerOutput()
model = FloatingSpaceOrth(2)
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