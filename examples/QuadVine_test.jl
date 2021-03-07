include("QuadVine.jl")
include("Quad_vis.jl")

model = QuadVine(2)
nq, nv, nc = mc_dims(model)
n,m = size(model)
N = 2
dt = 5e-3
x0 = [generate_config(model, zeros(model.nb)); zeros(nv)]
u0 = trim_controls(model)

X = quick_rollout(model, x0, u0, dt, N)
# visualize!(model, X, dt)

# quats1 = [UnitQuaternion(X[i][4:7]) for i=1:N]
# quats2 = [UnitQuaternion(X[i][7 .+ (4:7)]) for i=1:N]
# angles1 = [rotation_angle(quats1[i])*rotation_axis(quats1[i])[1] for i=1:N]
# angles2 = [rotation_angle(quats2[i])*rotation_axis(quats2[i])[1] for i=1:N]

# using Plots
# plot(angles1, label = "θ1",xlabel="time step",ylabel="state")
# plt = plot!(angles2-angles1,  label = "θ2")
# display(plt)


## DYANMICS
# nb = 3
# model = QuadVine(nb)
# nq, nv, nc = mc_dims(model)
# dt = 0.001
# θ = [.3, .5, .7]
# x0 = [generate_config(model, θ); zeros(nv)]
# u0 = fill(.3, nb)
# z = KnotPoint(x0, u0, dt)
# @show norm(max_constraints(model, x0)) 
# x1 = RD.discrete_dynamics(PassThrough, model, z)
# x1, λ = discrete_dynamics_MC(PassThrough, model, x0, u0, 0., dt)

## JACOBIAN
n,m = size(model)
n̄ = RD.state_diff_size(model)

DExp = TO.DynamicsExpansionMC(model)
diff1 = SizedMatrix{n,n̄}(zeros(n,n̄))
RD.state_diff_jacobian!(diff1, RD.LieState(model), SVector{n}(x0))
diff2 = SizedMatrix{n,n̄}(zeros(n,n̄))
RD.state_diff_jacobian!(diff2, RD.LieState(model), SVector{n}(x0))
z = KnotPoint(x0,u0,dt)
Altro.discrete_jacobian_MC!(PassThrough, DExp.∇f, DExp.G, model, z)
TO.save_tmp!(DExp)
TO.error_expansion!(DExp, diff1, diff2)
A,B,C,G = TO.error_expansion(DExp, model)

# using SparseArrays
# display(spy(sparse(A), marker=2, legend=nothing, c=palette([:black], 2)))
# display(spy(sparse(B), marker=2, legend=nothing, c=palette([:black], 2)))
# display(spy(sparse(C), marker=2, legend=nothing, c=palette([:black], 2)))
# display(spy(sparse(G), marker=2, legend=nothing, c=palette([:black], 2)))
