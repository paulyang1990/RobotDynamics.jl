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

struct Pendulum3D{R,T} <: RD.RigidBodyMC{R}
    l::T
    r::T
    
    M::Array{T} # mass matrix
    b::T # damping
    
    g::T

    # constraint force dimension
    p::Int

    function Pendulum3D{R,T}(m::T, l::T, r::T, b::T) where {R<:Rotation, T<:Real} 
        M = Diagonal([m,m,m,1.0/12.0*m*l^2,1.0/12.0*m*l^2,.5*m*r^2])
        new(l, r, M, b, 9.81, 3)
    end 
end
Pendulum3D() = Pendulum3D{UnitQuaternion{Float64},Float64}(1.0, 1.0, .1, .1)


Altro.config_size(model::Pendulum3D) = 7

# Specify the state and control dimensions
RD.control_dim(::Pendulum3D) = 1

function max_constraints(model::Pendulum3D, x)
    return x[1:3] - UnitQuaternion(x[4:7]...) * [0;0;-model.l/2] # spherical
    return [x[1:3] - UnitQuaternion(x[4:7]...) * [0;0;-model.l/2];x[6:7]] # pin
end

function Altro.is_converged(model::Pendulum3D, x)
    c = max_constraints(model, x)
    return norm(c) < 1e-6
end

function max_constraints_jacobian(model::Pendulum3D, x⁺)
    c!(x) = max_constraints(model, x)
    J_big = ForwardDiff.jacobian(c!, x⁺)
    R⁺ = UnitQuaternion(x⁺[4:7]...)
    att_jac⁺ = RS.∇differential(R⁺)
    return [J_big[:,1:3] J_big[:,4:7]*att_jac⁺]
end

function forces(model::Pendulum3D, x⁺, x, u)
    [0;0;-model.M[1,1]*9.81]
end

function torques(model::Pendulum3D, x⁺, x, u)
    [u[1]-model.b*x⁺[11];0;0] 
end

function wrenches(model::Pendulum3D, x⁺, x, u)
    f = forces(model, x⁺, x, u)
    τ = torques(model, x⁺, x, u)
    return [f; τ]
end

function sqrt_term(ω,dt)
    sqrt(4/dt^2-ω'ω)
end

function propagate_config!(model::Pendulum3D, x⁺, x, dt)
    v⁺ = x⁺[8:10]
    x⁺[1:3] = x[1:3] + v⁺*dt
    
    ω⁺ = x⁺[11:13]
    x⁺[4:7] = RS.params(RS.expm(ω⁺*dt) * UnitQuaternion(x[4:7]...))
    return 
end

function propagate_config(model::Pendulum3D, x⁺, x, dt)
    x⁺ = copy(x⁺)
    propagate_config!(model, x⁺, x, dt)
    return x⁺[1:7]
end

function f_pos(model::Pendulum3D, x⁺, x, u, λ, dt)
    return x⁺[1:7] - propagate_config(model, x⁺, x, dt)
end

function f_vel(model::Pendulum3D, x⁺, x, u, λ, dt)
    J = max_constraints_jacobian(model, x⁺)
    
    mass = model.M[1:3,1:3]
    iner = model.M[4:6,4:6]

    v⁺ = x⁺[8:10]
    ω⁺ = x⁺[11:13]
    v = x[8:10]
    ω = x[11:13]

    f_t = mass*(v⁺-v)/dt - forces(model,x⁺,x,u)  
    f_r = sqrt(1/dt^2-ω⁺'ω⁺)*iner*ω⁺ - sqrt(1/dt^2-ω'ω)*iner*ω + cross(ω⁺,iner*ω⁺) + cross(ω,iner*ω) - .5*torques(model,x⁺,x,u)
    
    return [f_t;f_r]*dt-J'λ
end

function fc(model::Pendulum3D, x⁺, x, u, λ, dt)
    f = f_vel(model, x⁺, x, u, λ, dt)
    c = max_constraints(model, x⁺)
    return [f;c]
end

function fc_jacobian(model::Pendulum3D, x⁺, x, u, λ, dt)
    nv, nc = 6, length(λ)
    function fc_aug(s)
        # Unpack
        _x⁺ = convert(Array{eltype(s)}, x⁺)
        _x⁺[7 .+ (1:nv)] = s[1:nv]
        _λ = s[nv .+ (1:nc)]

        propagate_config!(model, _x⁺, x, dt)
        fc(model, _x⁺, x, u, _λ, dt)
    end
    ForwardDiff.jacobian(fc_aug, [x⁺[8:end];λ])
end

function line_step!(model::Pendulum3D, x⁺_new, λ_new, x⁺, λ, Δs, x, dt)
    # update lambda
    Δλ = Δs[7:end]
    λ_new .= λ - Δλ

    # update v⁺
    Δv⁺ = Δs[1:6]
    x⁺_new[7 .+ (1:6)] .= x⁺[7 .+ (1:6)] - Δv⁺    

    # compute configuration from v⁺
    propagate_config!(model, x⁺_new, x, dt)
    return    
end

function discrete_dynamics_MC(::Type{Q}, model::Pendulum3D, 
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

    max_iters, line_iters, ϵ = 100, 10, 1e-6
    for i=1:max_iters  
        # Newton step    
        err_vec = fc(model, x⁺, x, u, λ, dt)
        err = norm(err_vec)
        F = fc_jacobian(model, x⁺, x, u, λ, dt)
        Δs = F\err_vec
       
        # line search
        j=0
        err_new = err + 1        
        while (err_new > err) && (j < line_iters)
            line_step!(model, x⁺_new, λ_new, x⁺, λ, Δs, x, dt)
            err_new = norm(fc(model, x⁺_new, x, u, λ_new, dt))
            Δs /= 2
            j += 1
        end
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

function RD.discrete_dynamics(::Type{Q}, model::Pendulum3D, 
    z::AbstractKnotPoint) where {Q<:RobotDynamics.Explicit}
    x, λ = discrete_dynamics_MC(Q, model, z)
    return x
end

function discrete_jacobian_MC(::Type{Q}, model::Pendulum3D,
    z::AbstractKnotPoint{T,N,M′}) where {T,N,M′,Q<:RobotDynamics.Explicit}
    
    if z.dt == 0
        z.dt = 1e-4
    end

    nq, nv, nc = mc_dims(model)

    x = state(z) 
    u = control(z)
    t = z.t 
    dt = z.dt

    # compute next state and lagrange multiplier
    x⁺, λ = discrete_dynamics_MC(Q, model, z)

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
    return A,B,C,G,ABC
end

# model = Pendulum3D()
# dt = 0.01
# R0 = UnitQuaternion(.9999,.0001,0, 0)
# x0 = [R0*[0.; 0.; -.5]; RS.params(R0); zeros(6)]
# z = KnotPoint(x0,[1.],dt)

# # Evaluate the discrete dynamics and Jacobian
# x′ = RD.discrete_dynamics(PassThrough, model, z)
# println(x′[4:7])

# A,B,C,G,ABC = Altro.discrete_jacobian_MC(PassThrough, model, z)
# println(A)

# n,m = size(model)
# n̄ = state_diff_size(model)
# G1 = SizedMatrix{n,n̄}(zeros(n,n̄))
# RD.state_diff_jacobian!(G1, RD.LieState(model), SVector{13}(x0))
# G2 = SizedMatrix{n,n̄}(zeros(n,n̄))
# RD.state_diff_jacobian!(G2, RD.LieState(model), SVector{13}(x′))

# tmpA = ABC[:,1:13]
# tmpB = ABC[:,14:14]
# tmpC = ABC[:,15:end]

# DA = G2'tmpA*G1
# DB = G2'tmpB
# DC = G2'tmpC