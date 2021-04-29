include("QuadVine2.jl")
include("Quad_vis.jl")

using SparseArrays

model = QuadVine(2)
nq, nv, nc = mc_dims(model)
n,m = size(model)
N = 2
dt = 5e-3
x0 = [generate_config(model, rand(model.nb)); .1*rand(nv)]
u0 = .1*rand(m)
x⁺, λ = Altro.discrete_dynamics_MC(PassThrough, model, x0, u0, 0., dt)

## JACOBIAN
n,m = size(model)
n̄ = RD.state_diff_size(model)

DExp = TO.DynamicsExpansionMC(model)
diff1 = SizedMatrix{n,n̄}(zeros(n,n̄))
RD.state_diff_jacobian!(diff1, RD.LieState(model), SVector{n}(x0))
diff2 = SizedMatrix{n,n̄}(zeros(n,n̄))
RD.state_diff_jacobian!(diff2, RD.LieState(model), SVector{n}(x0))
z = KnotPoint(x0,u0,dt)
Altro.discrete_jacobian_MC!(PassThrough, DExp, model, z, x⁺, λ)
# TO.save_tmp!(DExp)
# TO.error_expansion!(DExp, diff1, diff2)
# A,B,C,G = TO.error_expansion(DExp, model)

function f_imp(z)
    # Unpack
    _x⁺ = z[1:(nq+nv)]
    _x = z[(nq+nv) .+ (1:nq+nv)]
    _u = z[2*(nq+nv) .+ (1:m)]
    _λ = z[2*(nq+nv)+m .+ (1:nc)]
    return [f_pos(model, _x⁺, _x, _u, _λ, dt); f_vel(model,  _x⁺, _x, _u, _λ, dt)]
end
auto_partials = ForwardDiff.jacobian(f_imp, [x⁺;z.z;λ])
DExp.all_partials ≈ auto_partials
display(spy(sparse(round.(DExp.all_partials - auto_partials,digits=5)), marker=2, legend=nothing, c=palette([:black], 2)))

#-----------------------------------------------
using SparseArrays

function jl_aug(s)
    # Unpack
    _x⁺ = convert(Array{eltype(s)}, x⁺)
    _x⁺[nq .+ (1:nv)] = s[1:nv]

    propagate_config!(model, _x⁺, x0, dt)
    -max_constraints_jacobian(model, _x⁺)'λ
end
jl_jac= ForwardDiff.jacobian(jl_aug, x⁺[nq .+ (1:nv)])

function fc_aug(s)
    # Unpack
    _x⁺ = convert(Array{eltype(s)}, x⁺)
    _x⁺[nq .+ (1:nv)] = s[1:nv]
    _λ = s[nv .+ (1:nc)]

    propagate_config!(model, _x⁺, x0, dt)
    fc(model, _x⁺, x0, u0, _λ, dt)
end
fc_jac = ForwardDiff.jacobian(fc_aug, [x⁺[nq .+ (1:nv)];λ])

F = zeros(nc+nv,nc+nv)
fc_jacobian!(F, model, x⁺, x0, u0, λ, dt)
extrema(F-fc_jac)
display(spy(sparse(round.(F-fc_jac,digits=5)), marker=2, legend=nothing, c=palette([:black], 2)))
