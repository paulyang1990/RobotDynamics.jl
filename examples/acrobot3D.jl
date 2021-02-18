using RobotDynamics
using TrajectoryOptimization
using Rotations
using ForwardDiff
using StaticArrays, LinearAlgebra
using BenchmarkTools
using Altro
using Plots

const TO = TrajectoryOptimization
const RD = RobotDynamics
const RS = Rotations

using TimerOutputs
const to = TimerOutput()

struct Acrobot3D{R,T} <: LieGroupModelMC{R}
    masses::Array{T,1}
    lengths::Array{T,1}
    radii::Array{T,1} 

    mass::Diagonal{T,Array{T,1}} # system mass
    iner::Diagonal{T,Array{T,1}} # system inertia    

    g::T # gravity
    p::Int # constraint force dimension
 
    function Acrobot3D{R,T}(masses, lengths, radii) where {R<:Rotation, T<:Real} 
        @assert length(masses) == length(lengths) == length(radii) == 2
        m1,m2 = masses
        l1,l2 = lengths
        r1,r2 = radii

        mass = Diagonal([m1,m1,m1,m2,m2,m2])
        I1 = 1/4*m1*r1^2 + 1/12*m1*l1^2
        I2 = 1/4*m2*r2^2 + 1/12*m2*l2^2
        iner = Diagonal([I1, I1, .5*m1*r1^2,
                         I2, I2, .5*m2*r2^2])
        iner = Diagonal(ones(6))
        new(masses, lengths, radii, mass, iner, 9.81, 10)
    end 
end
Acrobot3D() = Acrobot3D{UnitQuaternion{Float64},Float64}(ones(2), ones(2), .1*ones(2))

Altro.config_size(model::Acrobot3D) = 14
Lie_P(::Acrobot3D) = (3,3,12)
RD.LieState(::Acrobot3D{R}) where R = RD.LieState(R,(3,3,12))

RD.control_dim(::Acrobot3D) = 1

function rc_to_mc(model::Acrobot3D, rc_x)
    θ1 = rc_x[1]
    θ2 = rc_x[2]
    l1, l2 = model.lengths

    R1 = UnitQuaternion(RotX(θ1))
    R2 = UnitQuaternion(RotX(θ1+θ2))
    mc_x = [R1*[0.; 0.; -l1/2]; 
            RS.params(R1);
            R1*[0.; 0.; -l1] + R2*[0.; 0.; -l2/2]; 
            RS.params(R2); 
            zeros(12)]

    return mc_x
end

function max_constraints(model::Acrobot3D, x)
    l = model.lengths
    d1 = UnitQuaternion(x[4:7]...) * [0;0;-l[1]/2]
    d2 = UnitQuaternion(x[7 .+ (4:7)]...) * [0;0;-l[2]/2]
    return [x[1:3] - d1;            
            (x[1:3] + d1) - (x[7 .+ (1:3)] - d2);
            x[6:7];
            x[7 .+ (6:7)]]
end

function Altro.is_converged(model::Acrobot3D, x)
    c = max_constraints(model, x)
    return norm(c) < 1e-6
end

function max_constraints_jacobian(model::Acrobot3D, x⁺::Vector{T}) where T
    nq, nv, _ = mc_dims(model)
    c_aug(x) = max_constraints(model, x)
    J_big = ForwardDiff.jacobian(c_aug, x⁺[1:nq])

    links = length(model.masses)
    n,m = size(model)
    n̄ = state_diff_size(model)
    G = SizedMatrix{n,n̄}(zeros(T,n,n̄))
    RD.state_diff_jacobian!(G, RD.LieState(UnitQuaternion{T}, (3,3,12)) , SVector{n}(x⁺))
    return J_big*G[1:nq,1:nv]
end

function forces(model::Acrobot3D, x⁺, x, u)
    links = length(model.masses)
    g = model.g
    [[0;0;-model.masses[i]*g] for i=1:links]
end

function torques(model::Acrobot3D, x⁺, x, u)
    [[-u[1];0;0], [u[1];0;0]]
end

function get_vels(model::Acrobot3D, x)
    vec = RD.vec_states(model, x)    
    v1 = vec[3][1:3]
    ω1 = vec[3][4:6]
    v2 = vec[3][6 .+ (1:3)]
    ω2 = vec[3][6 .+ (4:6)]
    return v1, ω1, v2, ω2
end

function propagate_config!(model::Acrobot3D, x⁺, x, dt)
    nq, nv, _ = mc_dims(model)

    vec = RD.vec_states(model, x)
    rot = RD.rot_states(RD.LieState(UnitQuaternion{eltype(x)}, Lie_P(model)), x)

    v1⁺, ω1⁺, v2⁺, ω2⁺ = get_vels(model, x⁺)

    pos1⁺ = vec[1] + v1⁺*dt
    pos2⁺ = vec[2] + v2⁺*dt
    quat1⁺ = RS.params(RS.expm(ω1⁺*dt) * rot[1])
    quat2⁺ = RS.params(RS.expm(ω2⁺*dt) * rot[2])

    x⁺[1:nq] = [pos1⁺; quat1⁺; pos2⁺; quat2⁺]
    return 
end

function propagate_config(model::Acrobot3D, x⁺, x, dt)
    x⁺ = copy(x⁺)
    propagate_config!(model, x⁺, x, dt)
    nq = config_size(model)
    return x⁺[1:nq]
end

function f_pos(model::Acrobot3D, x⁺, x, u, λ, dt)
    nq = config_size(model)
    return x⁺[1:nq] - propagate_config(model, x⁺, x, dt)
end

function f_vel(model::Acrobot3D, x⁺, x, u, λ, dt)
    J = max_constraints_jacobian(model, x⁺)
    
    M = model.masses
    I1 = model.iner[1:3,1:3]
    I2 = model.iner[4:6,4:6]

    v1, ω1, v2, ω2 = get_vels(model, x)
    v1⁺, ω1⁺, v2⁺, ω2⁺ = get_vels(model, x⁺)

    fs = forces(model,x⁺,x,u) 
    τs = torques(model,x⁺,x,u)

    ft1 = M[1]*(v1⁺-v1)/dt - fs[1]
    ft2 = M[2]*(v2⁺-v2)/dt - fs[2]

    fr1 = sqrt(4/dt^2-ω1⁺'ω1⁺)*I1*ω1⁺ - sqrt(4/dt^2-ω1'ω1)*I1*ω1 + cross(ω1⁺,I1*ω1⁺) + cross(ω1,I1*ω1) - 2*τs[1]
    fr2 = sqrt(4/dt^2-ω2⁺'ω2⁺)*I2*ω2⁺ - sqrt(4/dt^2-ω2'ω2)*I2*ω2 + cross(ω2⁺,I2*ω2⁺) + cross(ω2,I2*ω2) - 2*τs[2]

    return [ft1;fr1;ft2;fr2]-J'λ
end

function fc(model::Acrobot3D, x⁺, x, u, λ, dt)
    f = @timeit to "f_vel" f_vel(model, x⁺, x, u, λ, dt)
    c = @timeit to "max_con" max_constraints(model, x⁺)
    return [f;c]
end

function fc_jacobian(model::Acrobot3D, x⁺, x, u, λ, dt)
    nq, nv, nc = mc_dims(model)

    function fc_aug(s)
        # Unpack
        _x⁺ = convert(Array{eltype(s)}, x⁺)
        _x⁺[nq .+ (1:nv)] = s[1:nv]
        _λ = s[nv .+ (1:nc)]

        propagate_config!(model, _x⁺, x, dt)
        fc(model, _x⁺, x, u, _λ, dt)
    end
    @timeit to "fd_jac" ForwardDiff.jacobian(fc_aug, [x⁺[nq .+ (1:nv)];λ])
end

function line_step!(model::Acrobot3D, x⁺_new, λ_new, x⁺, λ, Δs, x, dt)
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

function adjust_step_size!(model::Acrobot3D, x⁺, Δs)
    return
    Δs ./= maximum(abs.(Δs[1:6]))/100
    _, ω1, _, ω2 = get_vels(model, x⁺)
    Δω1, Δω2 = Δs[1:3], Δs[4:6]
    γ = 1.0
    while (ω1'-γ*Δω1')*(ω1-γ*Δω1) > (1/dt^2)
        # print("%")
        γ /= 2
    end
    
    while (ω2'-γ*Δω2')*(ω2-γ*Δω2) > (1/dt^2)
        # print("%")
        γ /= 2
    end
    Δs .= γ*Δs
end

function discrete_dynamics_MC(::Type{Q}, model::Acrobot3D, 
    x, u, t, dt) where {Q<:RobotDynamics.Explicit}
    
    nq, nv, nc = mc_dims(model)

    # initial guess
    λ = zeros(nc)
    x⁺ = Vector(x)

    x⁺_new, λ_new = copy(x⁺), copy(λ)

    max_iters, line_iters, ϵ = 100, 20, 1e-6
    @timeit to "for loop" for i=1:max_iters  
        # print("iter ", i, ": ")

        # Newton step    
        err_vec =  @timeit to "fc" fc(model, x⁺, x, u, λ, dt)
        err = norm(err_vec)
        F = @timeit to "fc_jac" fc_jacobian(model, x⁺, x, u, λ, dt)
        Δs = @timeit to "backslash" F\err_vec
       
        # line search
        j=0
        err_new = err + 1        
        while (err_new > err) && (j < line_iters)
            # print("-")
            adjust_step_size!(model, x⁺, Δs)
            @timeit to "linestep" line_step!(model, x⁺_new, λ_new, x⁺, λ, Δs, x, dt)
            _, ω1⁺, _, ω2⁺ = get_vels(model, x⁺_new)
            if (1/dt^2>=ω1⁺'ω1⁺) && (1/dt^2>=ω2⁺'ω2⁺)
                # print("!")
                err_new =  @timeit to "normfc" norm(fc(model, x⁺_new, x, u, λ_new, dt))
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

# function RD.discrete_dynamics(::Type{Q}, model::Acrobot3D, 
#     z::AbstractKnotPoint) where {Q<:RobotDynamics.Explicit}
#     x, λ = discrete_dynamics_MC(Q, model, state(z), control(z), z.t, z.dt)
#     return x
# end

function RD.discrete_dynamics(::Type{Q}, model::Acrobot3D, x, u, t, dt) where Q
    x, λ = @timeit to "dd_MC" discrete_dynamics_MC(Q, model,  x, u, t, dt)
    return x
end

function Altro.discrete_jacobian_MC!(::Type{Q}, ∇f, G, model::Acrobot3D,
    z::AbstractKnotPoint{T,N,M′}) where {T,N,M′,Q<:RobotDynamics.Explicit}
    
    n,m = size(model)
    n̄ = state_diff_size(model)
    nq, nv, nc = mc_dims(model)

    x = state(z) 
    u = control(z)

    # compute next state and lagrange multiplier
    x⁺, λ = @timeit to "dd_MC" discrete_dynamics_MC(Q, model, x, u, z.t, max(1e-4, z.dt))

    function f_imp(z)
        # Unpack
        _x⁺ = z[1:(nq+nv)]
        _x = z[(nq+nv) .+ (1:nq+nv)]
        _u = z[2*(nq+nv) .+ (1:m)]
        _λ = z[2*(nq+nv)+m .+ (1:nc)]
        return [f_pos(model, _x⁺, _x, _u, _λ, dt); f_vel(model,  _x⁺, _x, _u, _λ, dt)]
    end
    
    all_partials =  @timeit to "fd jac" ForwardDiff.jacobian(f_imp, [x⁺;x;u;λ])
    ∇f .=  @timeit to "backslash" -all_partials[:,1:n]\all_partials[:,n+1:end]

    G[:,1:n̄-nv] .= @timeit to "mc jac" max_constraints_jacobian(model, x⁺)
end

function discrete_jacobian_MC(::Type{Q}, model::Acrobot3D,
    z::AbstractKnotPoint{T,N,M′}) where {T,N,M′,Q<:RobotDynamics.Explicit}

    nq, nv, nc = mc_dims(model)

    x = state(z) 
    u = control(z)

    # compute next state and lagrange multiplier
    x⁺, λ = discrete_dynamics_MC(Q, model, x, u, z.t, max(1e-4, z.dt))

    function f_imp(z)
        # Unpack
        _x⁺ = z[1:(nq+nv)]
        _x = z[(nq+nv) .+ (1:nq+nv)]
        _u = z[2*(nq+nv) .+ (1:m)]
        _λ = z[2*(nq+nv)+m .+ (1:nc)]
        return [f_pos(model, _x⁺, _x, _u, _λ, dt); f_vel(model,  _x⁺, _x, _u, _λ, dt)]
    end

    n,m = size(model)
    n̄ = state_diff_size(model)
    
    all_partials = ForwardDiff.jacobian(f_imp, [x⁺;x;u;λ])
    ABC = -all_partials[:,1:n]\all_partials[:,n+1:end]
    
    G1 = SizedMatrix{n,n̄}(zeros(n,n̄))
    RD.state_diff_jacobian!(G1, RD.LieState(model), SVector{n}(x))
    G2 = SizedMatrix{n,n̄}(zeros(n,n̄))
    RD.state_diff_jacobian!(G2, RD.LieState(model), SVector{n}(x⁺))

    tmpA = ABC[:, 1:(nq+nv)]
    tmpB = ABC[:, nq+nv .+ (1:m)]
    tmpC = ABC[:, nq+nv+m .+ (1:nc)]

    A = G2'tmpA*G1
    B = G2'tmpB
    C = G2'tmpC

    J = max_constraints_jacobian(model, x⁺)
    G = [J zeros(size(J))]
    return A, B, C, G
end

## DYNAMICS
# model = Acrobot3D()
# dt = 0.001
# R01 = UnitQuaternion(RotX(.3))
# R02 = UnitQuaternion(RotX(.7))
# x0 = [R01*[0.; 0.; -.5]; 
#         RS.params(R01);
#         R01*[0.; 0.; -1] + R02*[0.; 0.; -.5]; 
#         RS.params(R02); 
#         zeros(12)]
# u0 = [0.]
# z = KnotPoint(x0,u0, dt)
# @show norm(max_constraints(model, x0)) 
# x1, λ = discrete_dynamics_MC(PassThrough, model, state(z), control(z), z.t, z.dt)

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

function plot_traj(X, U)
    N = length(X)
    quats1 = [UnitQuaternion(X[i][4:7]) for i=1:N]
    quats2 = [UnitQuaternion(X[i][7 .+ (4:7)]) for i=1:N]
    angles1 = [rotation_angle(quats1[i])*rotation_axis(quats1[i])[1] for i=1:N]
    angles2 = [rotation_angle(quats2[i])*rotation_axis(quats2[i])[1] for i=1:N]
    del=.1
    for i=2:N
        if 2*pi - del < angles1[i-1]-angles1[i] < 2*pi + del
            angles1[i]+=2*pi
        elseif 2*pi - del < -angles1[i-1]+angles1[i] < 2*pi + del
            angles1[i]-=2*pi
        end
        if 2*pi - del < angles2[i-1]-angles2[i] < 2*pi + del
            angles2[i]+=2*pi
        elseif 2*pi - del < -angles2[i-1]+angles2[i] < 2*pi + del
            angles2[i]-=2*pi
        end
    end
    plot(angles1, label = "θ1",xlabel="time step",ylabel="state")
    plt = plot!(angles2,  label = "θ2")
    display(plt)
    display(plot(U))
    return angles1, angles2
end

# N = 1000
# dt = 1e-3
# R01 = UnitQuaternion(RotX(.3))
# R02 = UnitQuaternion(RotX(.7))
# x0 = [R01*[0.; 0.; -.5]; 
#         RS.params(R01);
#         R01*[0.; 0.; -1] + R02*[0.; 0.; -.5]; 
#         RS.params(R02); 
#         zeros(12)]

# reset_timer!(to)
# X = quick_rollout(model, x0, [-.5], dt, N)
# show(to)
# quats1 = [UnitQuaternion(X[i][4:7]) for i=1:N]
# quats2 = [UnitQuaternion(X[i][7 .+ (4:7)]) for i=1:N]
# angles1 = [rotation_angle(quats1[i])*rotation_axis(quats1[i])[1] for i=1:N]
# angles2 = [rotation_angle(quats2[i])*rotation_axis(quats2[i])[1] for i=1:N]

# using Plots
# plot(angles1, label = "θ1",xlabel="time step",ylabel="state")
# plt = plot!(angles2-angles1,  label = "θ2")
# display(plt)

# include("2link_visualize.jl")
# visualize!(model, X, dt)

## JACOBIAN
# z = KnotPoint(x0, u0, dt)
# reset_timer!(to)
# A1, B1, C1, G1 = discrete_jacobian_MC(PassThrough, model, z)

# n,m = size(model)
# n̄ = RD.state_diff_size(model)

# DExp = TO.DynamicsExpansionMC(model)
# diff1 = SizedMatrix{n,n̄}(zeros(n,n̄))
# RD.state_diff_jacobian!(diff1, RD.LieState(model), SVector{n}(x0))
# diff2 = SizedMatrix{n,n̄}(zeros(n,n̄))
# RD.state_diff_jacobian!(diff2, RD.LieState(model), SVector{n}(x1))

# @timeit to "dj_MC!" Altro.discrete_jacobian_MC!(PassThrough, DExp.∇f, DExp.G, model, z)
# show(to)
# TO.save_tmp!(DExp)
# TO.error_expansion!(DExp, diff1, diff2)
# A2, B2, C2, G2 = TO.error_expansion(DExp, model)
# @show extrema(A1 - A2)
# @show extrema(B1 - B2)
# @show extrema(C1 - C2)
# @show extrema(G1 - G2)
