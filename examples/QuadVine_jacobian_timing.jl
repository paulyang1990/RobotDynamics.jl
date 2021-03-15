include("QuadVine.jl")

model = QuadVine(2)
nq, nv, nc = mc_dims(model)
n,m = size(model)
n̄ = RD.state_diff_size(model)

N = 2
dt = 5e-3
x0 = [generate_config(model, zeros(model.nb)); zeros(nv)]
xf = copy(x0)
x⁺ = copy(x0)
u0 = trim_controls(model)
z = KnotPoint(x0,u0,dt)
@time Altro.discrete_dynamics_MC(PassThrough, model, x0, u0, 0, dt)
x⁺, λ = Altro.discrete_dynamics_MC(PassThrough, model, x0, u0, 0, dt)

#################################################################
# NEW ABC JACOBIAN

DExp = TO.DynamicsExpansionMC(model)
Altro.discrete_jacobian_MC!(PassThrough, DExp, model, z, x⁺, λ)

DExp2 = TO.DynamicsExpansionMC(model)
old_discrete_jacobian_MC!(PassThrough, $DExp2, $model, $z)
DExp.∇f ≈ DExp2.∇f

@benchmark Altro.discrete_dynamics_MC(PassThrough, $model, $x0, $u0, 0, $dt)
@benchmark Altro.discrete_jacobian_MC!(PassThrough, $DExp, $model, $z, $x⁺, $λ)

using StatProfilerHTML
@profilehtml begin
    for i = 1:20
        # Altro.discrete_dynamics_MC(PassThrough, model, x0, u0, 0, dt)
        Altro.discrete_jacobian_MC!(PassThrough, DExp, model, z, x⁺, λ)
    end
end
@profilehtml TO.dynamics_expansion!(PassThrough, ilqr.D, model, ilqr.Z, ilqr.Λ)

#################################################################
# ALL_PARTIALS
# x = copy(x0)
# u = zeros(m)
# x⁺, λ = discrete_dynamics_MC(PassThrough, model, x, u, 0, dt)

# function f_imp(z)
#     # Unpack
#     _x⁺ = z[1:(nq+nv)]
#     _x = z[(nq+nv) .+ (1:nq+nv)]
#     _u = z[2*(nq+nv) .+ (1:m)]
#     _λ = z[2*(nq+nv)+m .+ (1:nc)]
#     return [f_pos(model, _x⁺, _x, _u, _λ, dt); f_vel(model,  _x⁺, _x, _u, _λ, dt)]
# end

# input =  [x⁺;x;u;λ]
# output = ForwardDiff.jacobian(f_imp, input)
# using SparseArrays
# display(spy(sparse(output), marker=2, legend=nothing, c=palette([:black], 2)))

#################################################################
# NEW G JACOBIAN
function max_constraints_jacobian2(model::QuadVine{R}, x) where R
    nb = model.nb
    nq, nv, _ = mc_dims(model)
    P = Lie_P(model)
    l = model.lengths
    d = zeros(3)
    rot = RD.rot_states(RD.LieState(UnitQuaternion{eltype(x)}, Lie_P(model)), x)
    J = zeros(nc, nv)
    for i=1:nb-1
        # shift vals
        row = 3*(i-1)
        col = 6*(i-1)

        # ∂c∂qa
        qa = rot[i]
        d[3] = -l[i]/2
        J[row .+ (1:3), col .+ (4:6)] = RS.∇rotate(qa, d)*RS.∇differential(qa)

        # ∂c∂qb
        qb = rot[i+1]
        d[3] = -l[i+1]/2
        J[row .+ (1:3), col .+ (10:12)] = RS.∇rotate(qb, d)*RS.∇differential(qa)
        
        for j=1:3
            J[row+j, col+j] = 1 # ∂c∂xa = I
            J[row+j, col+6+j] = -1 # ∂c∂xb = -I
        end
    end
    return J
end

J = max_constraints_jacobian(model, x⁺)
J2 = max_constraints_jacobian2(model, x⁺)

@benchmark max_constraints_jacobian($model, $x⁺)
@benchmark max_constraints_jacobian2($model, $x⁺)