using RobotDynamics
using TrajectoryOptimization
using Rotations
using ForwardDiff
using StaticArrays, LinearAlgebra
using BenchmarkTools
using Altro

const TO = TrajectoryOptimization
const RD = RobotDynamics
const RS = Rotations

struct nPendulum3D{R,T} <: LieGroupModelMC{R}
    masses::Array{T,1}
    lengths::Array{T,1}
    radii::Array{T,1} 
    inertias::Array{Diagonal{T,Array{T,1}},1}

    g::T # gravity
    nb::Int # number of rigid bodies
    p::Int # constraint force dimension
 
    function nPendulum3D{R,T}(masses, lengths, radii) where {R<:Rotation, T<:Real} 
        @assert length(masses) == length(lengths) == length(radii)
        nb = length(masses)
        inertias = []
        for (m, l, r) in zip(masses, lengths, radii)
            Ixx = 1/4*m*r^2 + 1/12*m*l^2
            push!(inertias, Diagonal([Ixx, Ixx, .5*m*r^2]))
        end

        new(masses, lengths, radii, inertias, 9.81, nb, 5*nb)
    end 
end
nPendulum3D() = nPendulum3D{UnitQuaternion{Float64},Float64}(ones(2), ones(2), .1*ones(2))
nPendulum3D(n) = nPendulum3D{UnitQuaternion{Float64},Float64}(ones(n), ones(n), .1*ones(n))

Altro.config_size(model::nPendulum3D) = 7*model.nb
Lie_P(model::nPendulum3D) = (fill(3, model.nb)..., Int(6*model.nb))
RD.LieState(model::nPendulum3D{R}) where R = RD.LieState(R,Lie_P(model))
RD.control_dim(model::nPendulum3D) = model.nb

function max_constraints(model::nPendulum3D{R}, x) where R
    nq = config_size(model)
    P = Lie_P(model)
    lie = RD.LieState(UnitQuaternion{eltype(x)}, (P[1:end-1]..., 0))

    pos = RD.vec_states(lie, x) 
    rot = RD.rot_states(lie, x) 

    l = model.lengths
    c = zeros(eltype(x), model.p)

    # first link endpoint
    d1 = rot[1] * [0;0;-l[1]/2]
    c[1:3] = pos[1] - d1

    # other links endpoint
    for i=2:model.nb
        d1 = rot[i-1] * [0;0;-l[i-1]/2]
        d2 = rot[i] * [0;0;-l[i]/2]
        c[3*(i-1) .+ (1:3)] = (pos[i-1] + d1) - (pos[i] - d2)
    end
    
    # all link orientation constraint
    shift = 3*model.nb
    for i=1:model.nb
        c[(shift + 2*(i-1)) .+ (1:2)] = RS.params(rot[i])[3:4]
    end
    
    return c
end

function Altro.is_converged(model::nPendulum3D, x)
    c = max_constraints(model, x)
    return norm(c) < 1e-6
end

function max_constraints_jacobian(model::nPendulum3D, x⁺::Vector{T}) where T
    nq, nv, _ = mc_dims(model)
    c_aug(x) = max_constraints(model, x)
    J_big = ForwardDiff.jacobian(c_aug, x⁺[1:nq])

    links = length(model.masses)
    n,m = size(model)
    n̄ = state_diff_size(model)
    G = SizedMatrix{n,n̄}(zeros(T,n,n̄))
    RD.state_diff_jacobian!(G, RD.LieState(UnitQuaternion{T}, Lie_P(model)) , SVector{n}(x⁺))
    return J_big*G[1:nq,1:nv]
end

function forces(model::nPendulum3D, x⁺, x, u)
    nb = model.nb
    g = model.g
    [[0;0;-model.masses[i]*g] for i=1:nb]
end

function torques(model::nPendulum3D, x⁺, x, u)
    nb = model.nb
    torques = [[u[1];0;0]]
    for i=2:nb
        push!(torques, [u[i]-u[i-1];0;0])
    end
    return torques
end

function get_vels(model::nPendulum3D, x)
    nb = model.nb
    vec = RD.vec_states(model, x)[end]    
    vs = [vec[6*(i-1) .+ (1:3)] for i=1:nb]
    ωs = [vec[6*(i-1) .+ (4:6)] for i=1:nb]
    return vs, ωs
end

function propagate_config!(model::nPendulum3D{R}, x⁺::Vector{T}, x, dt) where {R, T}
    nb = model.nb
    nq, nv, _ = mc_dims(model)
    P = Lie_P(model)

    vec = RD.vec_states(model, x)
    rot = RD.rot_states(RD.LieState(UnitQuaternion{T}, P), x)

    vs⁺, ωs⁺ = get_vels(model, x⁺)
    for i=1:nb
        vind = RD.vec_inds(R, P, i)
        x⁺[vind] = vec[i] + vs⁺[i]*dt
        rind = RD.rot_inds(R, P, i)
        x⁺[rind] = RS.params(RS.expm(ωs⁺[i]*dt) * rot[i])
    end

    return 
end

function propagate_config(model::nPendulum3D, x⁺, x, dt)
    x⁺ = copy(x⁺)
    propagate_config!(model, x⁺, x, dt)
    nq = config_size(model)
    return x⁺[1:nq]
end

function f_pos(model::nPendulum3D, x⁺, x, u, λ, dt)
    nq = config_size(model)
    return x⁺[1:nq] - propagate_config(model, x⁺, x, dt)
end

function f_vel(model::nPendulum3D, x⁺, x, u, λ, dt)
    J = max_constraints_jacobian(model, x⁺)
    
    nb = model.nb
    nq, nv, nc = mc_dims(model)

    ms = model.masses
    Is = model.inertias

    vs, ωs = get_vels(model, x)
    vs⁺, ωs⁺ = get_vels(model, x⁺)

    fs = forces(model,x⁺,x,u) 
    τs = torques(model,x⁺,x,u)

    f_vels = zeros(eltype(x⁺), 6*nb)
    for i=1:nb
        # translational
        f_vels[6*(i-1) .+ (1:3)] = ms[i]*(vs⁺[i]-vs[i])/dt - fs[i]

        # rotational
        ω⁺ = ωs⁺[i]
        ω = ωs[i]
        sq⁺ = sqrt(4/dt^2-ω⁺'ω⁺)
        sq = sqrt(4/dt^2-ω'ω)
        f_vels[6*(i-1) .+ (4:6)] = sq⁺*Is[i]*ω⁺ - sq*Is[i]*ω + cross(ω⁺,Is[i]*ω⁺) + cross(ω,Is[i]*ω) - 2*τs[i]
    end

    return f_vels - J'λ
end

function fc(model::nPendulum3D, x⁺, x, u, λ, dt)
    f = f_vel(model, x⁺, x, u, λ, dt)
    c = max_constraints(model, x⁺)
    return [f;c]
end

function fc_jacobian(model::nPendulum3D, x⁺, x, u, λ, dt)
    nq, nv, nc = mc_dims(model)

    function fc_aug(s)
        # Unpack
        _x⁺ = convert(Array{eltype(s)}, x⁺)
        _x⁺[nq .+ (1:nv)] = s[1:nv]
        _λ = s[nv .+ (1:nc)]

        propagate_config!(model, _x⁺, x, dt)
        fc(model, _x⁺, x, u, _λ, dt)
    end
    ForwardDiff.jacobian(fc_aug, [x⁺[nq .+ (1:nv)];λ])
end

function line_step!(model::nPendulum3D, x⁺_new, λ_new, x⁺, λ, Δs, x, dt)
    nq, nv, nc = mc_dims(model)
    
    # update lambda
    Δλ = Δs[nv .+ (1:nc)]
    λ_new .= λ - Δλ

    # update v⁺
    Δv⁺ = Δs[1:nv]
    x⁺_new[nq .+ (1:nv)] .= x⁺[nq .+ (1:nv)] - Δv⁺    

    # compute configuration from v⁺
    propagate_config!(model, x⁺_new, x, dt)
    return    
end

# function adjust_step_size!(model::nPendulum3D, x⁺, Δs)
#     return
#     Δs ./= maximum(abs.(Δs[1:6]))/100
#     _, ωs = get_vels(model, x⁺)
#     Δω1, Δω2 = Δs[1:3], Δs[4:6]
#     γ = 1.0
#     while (ω1-γ*Δω1)'*(ω1-γ*Δω1) > (1/dt^2)
#         # print("%")
#         γ /= 2
#     end
    
#     while (ω2-γ*Δω2)'*(ω2-γ*Δω2) > (1/dt^2)
#         # print("%")
#         γ /= 2
#     end
#     Δs .= γ*Δs
# end

function discrete_dynamics_MC(::Type{Q}, model::nPendulum3D, 
    z::AbstractKnotPoint) where {Q<:RobotDynamics.Explicit}
  
    x = state(z) 
    u = control(z)
    t = z.t 
    dt = z.dt
    
    nq, nv, nc = mc_dims(model)

    # initial guess
    λ = zeros(nc)
    x⁺ = Vector(x)

    x⁺_new, λ_new = copy(x⁺), copy(λ)

    max_iters, line_iters, ϵ = 100, 20, 1e-6
    for i=1:max_iters  
        # print("iter ", i, ": ")

        # Newton step    
        err_vec = fc(model, x⁺, x, u, λ, dt)
        err = norm(err_vec)
        F = fc_jacobian(model, x⁺, x, u, λ, dt)
        Δs = F\err_vec
       
        # line search
        j=0
        err_new = err + 1        
        while (err_new > err) && (j < line_iters)
            # print("-")
            # adjust_step_size!(model, x⁺, Δs)
            line_step!(model, x⁺_new, λ_new, x⁺, λ, Δs, x, dt)
            _, ωs⁺ = get_vels(model, x⁺_new)
            if all(1/dt^2 .>= dot(ωs⁺,ωs⁺))
                # print("!")
                err_new = norm(fc(model, x⁺_new, x, u, λ_new, dt))
            end
            Δs /= 2
            j += 1
        end
        # println(" steps: ", j)
        x⁺ .= x⁺_new
        λ .= λ_new

        # convergence check
        if err_new < ϵ
            return x⁺, λ
        end
    end

    # throw("Newton did not converge. ")
    println("Newton did not converge. ")
    return x⁺, λ
end

function RD.discrete_dynamics(::Type{Q}, model::nPendulum3D, 
    z::AbstractKnotPoint) where {Q<:RobotDynamics.Explicit}
    x, λ = discrete_dynamics_MC(Q, model, z)
    return x
end

function Altro.discrete_jacobian_MC(::Type{Q}, model::nPendulum3D,
    z::AbstractKnotPoint{T,N,M′}) where {T,N,M′,Q<:RobotDynamics.Explicit}

    nq, nv, nc = mc_dims(model)

    z_copy = copy(z)
    if z_copy.dt == 0
        z_copy.dt = 1e-4
    end

    x = state(z_copy) 
    u = control(z_copy)
    t = z_copy.t 
    dt = z_copy.dt

    # compute next state and lagrange multiplier
    x⁺, λ = discrete_dynamics_MC(Q, model, z_copy)

    function f_imp(z)
        # Unpack
        _x⁺ = z[1:(nq+nv)]
        _x = z[(nq+nv) .+ (1:nq+nv)]
        _u = z[2*(nq+nv) .+ (1:m)]
        _λ = z[2*(nq+nv)+m .+ (1:nc)]
        return [f_pos(model, _x⁺, _x, _u, _λ, dt); f_vel(model,  _x⁺, _x, _u, _λ, dt)]
    end

    n = length(x)
    m = length(u)
    
    all_partials = ForwardDiff.jacobian(f_imp, [x⁺;x;u;λ])
    ABC = -all_partials[:,1:n]\all_partials[:,n+1:end]

    att_jac = RS.∇differential(UnitQuaternion(x[4:7]...))
    att_jac⁺ = RS.∇differential(UnitQuaternion(x⁺[4:7]...))

    ABC′ = zeros(2*nv,n+m+nc)
    ABC′[1:3, :] = ABC[1:3, :]
    ABC′[4:6, :] = att_jac⁺'*ABC[4:7, :]
    ABC′[nv .+ (1:nv), :] = ABC[nq .+ (1:nv), :]

    A_big = ABC′[:, 1:(nq+nv)]
    B = ABC′[:, nq+nv .+ (1:m)]
    C = ABC′[:, nq+nv+m .+ (1:nc)]

    A = zeros(2*nv,2*nv)
    A[:, 1:3] =  A_big[:, 1:3]
    A[:, 4:6] = A_big[:, 4:7]*att_jac
    A[:, nv .+ (1:nv)] = A_big[:, nq .+ (1:nv)]

    J = max_constraints_jacobian(model, x⁺)
    G = [J zeros(size(J))]
    return A, B, C, G
end

# compute maximal coordinate configuration given body rotations
function generate_config(model, rotations)
    @assert model.nb == length(rotations)
    pin = zeros(3)
    q = zeros(0)   
    l = model.lengths
    for i = 1:model.nb
        r = UnitQuaternion(rotations[i])
        delta = r * [0,0,-l[i]/2]
        q = [q; pin+delta; Rotations.params(r)]
        pin += 2*delta
    end
    return q
end

function generate_config(model, θ::Vector{<:Number})
    rotations = UnitQuaternion.(RotX.(θ))
    return generate_config(model, rotations)
end

## DYANMICS
# nb = 3
# model = nPendulum3D(nb)
# nq, nv, nc = mc_dims(model)
# dt = 0.001
# θ = [.3, .5, .7]
# x0 = [generate_config(model, θ); zeros(nv)]
# u0 = fill(.3, nb)
# z = KnotPoint(x0, u0, dt)
# @show norm(max_constraints(model, x0)) 
# x1, λ = discrete_dynamics_MC(PassThrough, model, z)

## ROLLOUT
function quick_rollout(model, x0, u, dt, N)
    X = [x0]
    xnext = x0
    for i=1:N
        z = KnotPoint(xnext, u, dt)
        xnext = RD.discrete_dynamics(PassThrough, model, z)
        push!(X, xnext)
    end
    return X
end

# nb = 6
# model = nPendulum3D(nb)
# nq, nv, nc = mc_dims(model)
# N = 1000
# dt = 1e-3
# θ = rand(nb)
# x0 = [generate_config(model, θ); zeros(nv)]
# u0 = fill(.1, nb)

# X = quick_rollout(model, x0, u0, dt, N)
# quats1 = [UnitQuaternion(X[i][4:7]) for i=1:N]
# quats2 = [UnitQuaternion(X[i][7 .+ (4:7)]) for i=1:N]
# angles1 = [rotation_angle(quats1[i])*rotation_axis(quats1[i])[1] for i=1:N]
# angles2 = [rotation_angle(quats2[i])*rotation_axis(quats2[i])[1] for i=1:N]

# using Plots
# plot(angles1, label = "θ1",xlabel="time step",ylabel="state")
# plt = plot!(angles2-angles1,  label = "θ2")
# display(plt)

# include("nPendulum3D_visualize.jl")
# visualize!(model, X, dt)
