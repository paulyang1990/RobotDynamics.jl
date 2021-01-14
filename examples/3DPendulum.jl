using RobotDynamics
using TrajectoryOptimization
using Rotations
using ForwardDiff
using StaticArrays, LinearAlgebra
using BenchmarkTools

const TO = TrajectoryOptimization
const RD = RobotDynamics

# Define the model struct to inherit from `RigidBody{R}`
struct Pendulum3D{R,T} <: RigidBody{R}
    M::Array{T} # mass matrix
    b::T # damping

    g::T

    # constraint force dimension
    p::Int

    function Pendulum3D{R,T}(m::T, l::T, r::T, b::T) where {R<:Rotation, T<:Real} 
        M = Diagonal([m,m,m,1.0/12.0*m*l^2,1.0/12.0*m*l^2,.5*m*r^2])
        new(M, b, 9.81, 3)
    end 
end
Pendulum3D() = Pendulum3D{UnitQuaternion{Float64},Float64}(1.0, 1.0, .1, .1)

# Specify the state and control dimensions
RD.control_dim(::Pendulum3D) = 1

# Define some simple "getter" methods that are required to evaluate the dynamics
RobotDynamics.mass(model::Pendulum3D) = model.M

function max_constraints(model::Pendulum3D, x)
    return [x[1:3] - UnitQuaternion(x[4:7]...) * [0;0;-.5];x[6:7]]
end

function max_constraints_jacobian(model::Pendulum3D, x⁺)
    c!(x) = max_constraints(model, x)
    J_big = ForwardDiff.jacobian(c!, x⁺)
    R⁺ = UnitQuaternion(x⁺[4:7]...)
    att_jac⁺ = Rotations.∇differential(R⁺)
    return [J_big[:,1:3] J_big[:,4:7]*att_jac⁺]
end

function wrenches(model::Pendulum3D, x⁺, x, u)
    [0;0;-9.81; u[1]-.1*x⁺[11];0;0] 
end

function fc(model::Pendulum3D, x⁺, x, u, λ, dt)
    c = max_constraints(model, x⁺)
    J = max_constraints_jacobian(model, x⁺)
    F = wrenches(model, x⁺, x, u) * dt

    v⁺ = x⁺[8:13]
    v = x[8:13]

    [model.M*(v⁺-v) - J'*λ - F; c]
end

function fc_jacobian(model::Pendulum3D, x⁺, x, u, λ, dt)
    nv, nc = 6, length(λ)
    function fc_aug(s)
        r⁺ = x[1:3] + s[1:3]*dt
        q⁺ = Rotations.params(Rotations.expm(s[4:6]*dt) * UnitQuaternion(x[4:7]...))
        fc(model, [r⁺;q⁺;s[1:6]], x, u, s[6 .+ (1:nc)], dt)
    end
    ForwardDiff.jacobian(fc_aug, [x⁺[8:end];λ])
end

function line_step!(model::Pendulum3D, x⁺_new, λ_new, x⁺, λ, Δs, x)
    # update lambda and v
    Δλ = Δs[7:end]
    λ_new .= λ - Δλ

    Δv⁺ = Δs[1:6]
    x⁺_new[7 .+ (1:6)] .= x⁺[7 .+ (1:6)] - Δv⁺    

    # compute configuration from v⁺
    x⁺_new[1:3] = x[1:3] + x⁺_new[7 .+ (1:3)]*dt
    x⁺_new[4:7] = Rotations.params(Rotations.expm(x⁺_new[7 .+ (4:6)]*dt) * UnitQuaternion(x[4:7]...))
    return    
end

function discrete_dynamics_MC(::Type{Q}, model::Pendulum3D, 
    z::AbstractKnotPoint) where {Q<:RobotDynamics.Explicit}
  
    x = state(z) 
    u = control(z)
    t = z.t 
    dt = z.dt

    nq = 7
    nv = 6
    nc = 5

    # initial guess
    λ = zeros(nc)
    x⁺ = Vector(x)

    x⁺_new, λ_new = copy(x⁺), copy(λ)

    max_iters, line_iters, ϵ = 100, 10, 1e-12
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
            line_step!(model, x⁺_new, λ_new, x⁺, λ, Δs, x)
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

    throw("Newton did not converge. ")
end

function RD.discrete_dynamics(::Type{Q}, model::Pendulum3D, 
    z::AbstractKnotPoint) where {Q<:RobotDynamics.Explicit}
    x, λ = discrete_dynamics_MC(Q, model, z)
    return x
end

function discrete_jacobian_MC(::Type{Q}, model::Pendulum3D,
    z::AbstractKnotPoint{T,N,M′}) where {T,N,M′,Q<:RobotDynamics.Explicit}
    
    nq = 7
    nv = 6
    nc = 5

    x = state(z) 
    u = control(z)
    t = z.t 
    dt = z.dt

    # compute next state and lagrange multiplier
    x⁺, λ = discrete_dynamics_MC(Q, model, z)

    function f_imp(z)
        # Unpack
        q⁺ = z[1:nq]
        v⁺ = z[nq .+ (1:nv)]
        q = z[nq+nv .+ (1:nq)]
        v = z[2*nq+nv .+ (1:nv)]
        u = z[2*(nq+nv) .+ (1:m)]
        λ = z[2*(nq+nv)+m .+ (1:nc)]
    
        M = 1.0*Matrix(I,6,6)
        b = 0.1
        
        J = max_constraints_jacobian(model, q⁺)
        F = [0;0;-9.81; u[1]-b*v⁺[4];0;0] * dt

        ω⁺ = v⁺[4:6]
        quat⁺ = Rotations.params(Rotations.expm(ω⁺*dt) * UnitQuaternion(q[4:7]...))

        return [M*(v⁺-v) - (J'*λ + F); q⁺ - [q[1:3]+v⁺[1:3]*dt; quat⁺]]
    end

    n = length(x)
    m = length(u)
    
    all_partials = ForwardDiff.jacobian(f_imp, [x⁺;x;u;λ])
    ABC = -all_partials[:,1:n]\all_partials[:,n+1:end]

    att_jac = Rotations.∇differential(UnitQuaternion(x[4:7]...))
    att_jac⁺ = Rotations.∇differential(UnitQuaternion(x⁺[4:7]...))

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
    return A,B,C,G
end

model = Pendulum3D()
dt = 0.01
R0 = UnitQuaternion(.9999,.0001,0, 0)
x0 = [R0*[0.; 0.; -.5]; Rotations.params(R0); zeros(6)]
z = KnotPoint(x0,[.0],dt)

# Evaluate the discrete dynamics and Jacobian
x′ = RD.discrete_dynamics(PassThrough, model, z)
println(x′)

A,B,C,G = discrete_jacobian_MC(PassThrough, model, z)
println(A)

# G = zeros(state_dim(model), RobotDynamics.state_diff_size(model))
# x,u = rand(model)
# z = KnotPoint(x,u,0.01)
# RobotDynamics.state_diff_jacobian!(G, model, x)
