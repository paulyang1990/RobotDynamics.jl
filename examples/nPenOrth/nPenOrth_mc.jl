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

struct nPenOrthMC{R,T} <: LieGroupModelMC{R}
    # link
    m::T 
    l::T
    inertias::Array{T,2}
      
    g::T

    # number of links
    nb::Integer
    p::Integer # constraint force dimension

    function nPenOrthMC{R,T}(m, l, nb) where {R<:Rotation, T<:Real} 
        x = 0.1
        y = 0.1
        z = l 
        new(m, l, 1 / 12 * m * diagm([y^2 + z^2;x^2 + z^2;x^2 + y^2]), 9.81, nb, 5*nb)
        # constraint: 3 for pin 3 for rotation
    end
end

nPenOrthMC() = nPenOrthMC{UnitQuaternion{Float64},Float64}(1.0, 1.0,2)
nPenOrthMC(nb::Integer) = nPenOrthMC{UnitQuaternion{Float64},Float64}(1.0, 1.0,nb)

# arrange state as Jan
# x v q w, x v q w,...
#1,2,3, 4,5,6, 7,8,9,10, 11,12,13
Altro.config_size(model::nPenOrthMC) = 7*model.nb
Lie_P(model::nPenOrthMC) = (6,fill(9, model.nb-1)..., 3)
RD.LieState(model::nPenOrthMC{R}) where R = RD.LieState(R,Lie_P(model))
RD.control_dim(model::nPenOrthMC) = model.nb

function generate_config(model::nPenOrthMC, rotations)
    @assert model.nb == length(rotations)
    pin = zeros(3)
    prev_q = UnitQuaternion(1.,0.0,0.0,0.0)
    state = zeros(0)
    for i = 1:model.nb
        r = UnitQuaternion(rotations[i])
        link_q = prev_q * r
        delta = link_q * [0,0,model.l/2]
        link_x = pin+delta
        state = [state; link_x;zeros(3);Rotations.params(link_q);zeros(3)]

        prev_q = link_q
        pin += 2*delta
    end


    return state
end

function generate_config(model::nPenOrthMC, θ::Vector{<:Number})
    @assert model.nb == length(θ)
    rotations = []
    # joints are arranged orthogonally
    for i=1:length(θ)
        if mod(i,2) == 0
            push!(rotations, UnitQuaternion(RotY(θ[i])))
        else
            push!(rotations, UnitQuaternion(RotX(θ[i])))
        end
    end
    return generate_config(model, rotations)
end


function max_constraints(model::nPenOrthMC{R}, x) where R
    nq = config_size(model)
    P = Lie_P(model)
    lie = RD.LieState(UnitQuaternion{eltype(x)}, (P[1:end-1]..., 0))

    pos = RD.vec_states(lie, x) 
    rot = RD.rot_states(lie, x) 


    l = model.l
    c = zeros(eltype(x), model.p)

    # first link endpoint
    d1 = rot[1] * [0;0;l/2]
    c[1:3] = pos[1][1:3] - d1    
    c[4] = rot[1][1,2]
    c[5] = rot[1][1,3]
    # other links endpoint
    for i=2:model.nb
        d1 = rot[i-1] * [0;0;l/2]
        d2 = rot[i] * [0;0;l/2]
        if i-1 == 1
            c[5*(i-1) .+ (1:3)] = (pos[i-1][1:3] + d1) - (pos[i][4:6] - d2)
        else
            c[5*(i-1) .+ (1:3)] = (pos[i-1][4:6] + d1) - (pos[i][4:6] - d2)
        end

        R_joint = rot[i]'*rot[i-1]
        if i%2 == 0
            c[5*(i-1) + 4] = R_joint[2,1]
            c[5*(i-1) + 5] = R_joint[2,3]
        else
            c[5*(i-1) + 4] = R_joint[1,2]
            c[5*(i-1) + 5] = R_joint[1,3]
        end
        
    end
    
    return c
end

# only return the jacobian with respect to position
function max_constraints_jacobian(model::nPenOrthMC, x⁺::Vector{T}) where T
    nq, nv, np = mc_dims(model)
    n,m = size(model)
    n̄ = state_diff_size(model)
    # n = 52
    # n̄ = 48
    # nq = 28
    # nv = 24
    # np = 12
    c_aug(x) = max_constraints(model, x)
    J_big = ForwardDiff.jacobian(c_aug, x⁺) #np x nq

    G = SizedMatrix{n,n̄}(zeros(T,n,n̄))
    RD.state_diff_jacobian!(G, RD.LieState(UnitQuaternion{T}, Lie_P(model)) , SVector{n}(x⁺))
    
    # index of q in n̄
    ind = BitArray(undef, n̄)
    for i=1:model.nb
        ind[(i-1)*12 .+ (1:3)] .= 1
        ind[(i-1)*12 .+ (7:9)] .= 1
    end
    
    subG = G[get_configs_ind(model,x⁺),ind]
    return J_big[:,get_configs_ind(model,x⁺)]*subG  #np x n̄q  (n̄q = nv)
end

function Altro.is_converged(model::nPenOrthMC, x)
    c = max_constraints(model, x)
    return norm(c) < 1e-6
end

function forces(model::nPenOrthMC, x⁺, x, u)
    nb = model.nb
    g = model.g
    [[0;0;-model.m*g] for i=1:nb]
end

function torques(model::nPenOrthMC, x⁺, x, u)
    nb = model.nb
    # each link should be affected by two torques. But for nPenOrth 
    # one torque is resisted by the structure 
    return [ i%2==0 ? [0;u[i];0] : [u[i];0;0] for i=1:nb]
end

# get x[], q[] from state x v q w, x v q w....
function get_configs(model::nPenOrthMC, x)
    # nb = model.nb
    # vec = RD.vec_states(model, x) # size nb+1   
    # # rot = RD.rot_states(model, x) # size nb 
    # xs = [if i > 1 vec[i][4:6] else vec[i][1:3] end for i=1:nb]
    # qs = [rot[i] for i=1:nb]
    xs = [x[(i-1)*13 .+ (1:3)] for i=1:nb]
    qs = [x[(i-1)*13 .+ (7:10)] for i=1:nb]
    return xs, qs
end
# lie state is not good for getting config and vel...
function get_configs_ind(model::nPenOrthMC, x)
    n,m = size(model)
    ind = BitArray(undef, n)
    for i=1:model.nb
        ind[(i-1)*13 .+ (1:3)] .= 1
        ind[(i-1)*13 .+ (7:10)] .= 1
    end
    return ind
end

# get v[], w[] from state x v q w, x v q w....
function get_vels(model::nPenOrthMC, x)
    nb = model.nb
    # vec = RD.vec_states(model, x) # size nb+1   
    # vs = [if i > 1 vec[i][7:9] else vec[i][4:6] end for i=1:nb]
    # ωs = [vec[i+1][1:3] for i=1:nb]
    vs = [x[(i-1)*13 .+ (4:6)] for i=1:nb]
    ωs = [x[(i-1)*13 .+ (11:13)] for i=1:nb]
    return vs, ωs
end
function get_vels_ind(model::nPenOrthMC, x)
    n,m = size(model)
    ind = BitArray(undef, n)
    for i=1:model.nb
        ind[(i-1)*13 .+ (4:6)] .= 1
        ind[(i-1)*13 .+ (11:13)] .= 1
    end
    return ind
end

function propagate_config!(model::nPenOrthMC{R}, x⁺::Vector{T}, x, dt) where {R, T}
    nb = model.nb
    nq, nv, np = mc_dims(model)
    P = Lie_P(model)
    lie = RD.LieState(UnitQuaternion{eltype(x)}, P)

    # vec = RD.vec_states(model, x)
    # rot = RD.rot_states(model, x)
    vec = RD.vec_states(lie, x) 
    rot = RD.rot_states(lie, x) 

    vs⁺, ωs⁺ = get_vels(model, x⁺)
    for i=1:nb
        pos_ind =(i-1)*13 .+  (1:3)
        # due to the irregularity in vec state
        if i == 1
            x⁺[pos_ind] = vec[i][1:3] + vs⁺[i]*dt
        else
            x⁺[pos_ind] = vec[i][4:6] + vs⁺[i]*dt
        end
        
        rot_ind = (i-1)*13 .+ (7:10)
        x⁺[rot_ind] = dt/2 * RS.lmult(rot[i]) * [sqrt(4/dt^2 - ωs⁺[i]'ωs⁺[i]); ωs⁺[i]]
    end

    return 
end

# propagate and extract config from the state 
function propagate_config(model::nPenOrthMC, x⁺, x, dt)
    x⁺ = copy(x⁺)
    propagate_config!(model, x⁺, x, dt)
    return x⁺[get_configs_ind(model,x⁺)]
end

function f_pos(model::nPenOrthMC, x⁺, x, u, λ, dt)
    return x⁺[get_configs_ind(model,x⁺)] - propagate_config(model, x⁺, x, dt)
end

function f_vel(model::nPenOrthMC, x⁺, x, u, λ, dt)
    J = max_constraints_jacobian(model, x⁺) # np x nv

    nb = model.nb
    nq, nv, nc = mc_dims(model)

    ms = model.m
    Is = model.inertias

    vs, ωs = get_vels(model, x)
    vs⁺, ωs⁺ = get_vels(model, x⁺)

    fs = forces(model,x⁺,x,u) 
    τs = torques(model,x⁺,x,u)
    f_vels = zeros(eltype(x⁺), 6*nb) # nv
    for i=1:nb
        # translational
        f_vels[6*(i-1) .+ (1:3)] = ms*(vs⁺[i]-vs[i])/dt - fs[i]

        # rotational
        ω⁺ = ωs⁺[i]
        ω = ωs[i]
        sq⁺ = sqrt(4/dt^2-ω⁺'ω⁺)
        sq = sqrt(4/dt^2-ω'ω)
        f_vels[6*(i-1) .+ (4:6)] = sq⁺*Is*ω⁺ - sq*Is*ω + cross(ω⁺,Is*ω⁺) + cross(ω,Is*ω) - 2*τs[i]
    end
    return f_vels - J'λ
end

function fc(model::nPenOrthMC, x⁺, x, u, λ, dt)
    f = f_vel(model, x⁺, x, u, λ, dt)
    c = max_constraints(model, x⁺)
    return [f;c]
end

function fc_jacobian(model::nPenOrthMC, x⁺, x, u, λ, dt)
    nq, nv, nc = mc_dims(model)

    function fc_aug(s)
        # Unpack
        _x⁺ = convert(Array{eltype(s)}, x⁺)
        _x⁺[get_vels_ind(model, x⁺)] = s[1:nv]
        _λ = s[nv .+ (1:nc)]

        propagate_config!(model, _x⁺, x, dt)
        fc(model, _x⁺, x, u, _λ, dt)
    end
    ForwardDiff.jacobian(fc_aug, [x⁺[get_vels_ind(model, x⁺)];λ])
end

function line_step!(model::nPenOrthMC, x⁺_new, λ_new, x⁺, λ, Δs, x, dt)
    nq, nv, nc = mc_dims(model)
    
    # update lambda
    Δλ = Δs[nv .+ (1:nc)]
    λ_new .= λ - Δλ

    # update v⁺
    Δv⁺ = Δs[1:nv]
    x⁺_new[get_vels_ind(model, x⁺_new)] .= x⁺[get_vels_ind(model, x⁺)] - Δv⁺    

    # compute configuration from v⁺
    propagate_config!(model, x⁺_new, x, dt)
    return    
end

function discrete_dynamics_MC(::Type{Q}, model::nPenOrthMC, 
    x, u, t, dt) where {Q<:RobotDynamics.Explicit}
  
    nq, nv, nc = mc_dims(model)

    # initial guess
    λ = zeros(eltype(x),nc)
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

function RD.discrete_dynamics(::Type{Q}, model::nPenOrthMC, x, u, t, dt) where Q
    x, λ = discrete_dynamics_MC(Q, model,  x, u, t, dt)
    return x
end

function Altro.discrete_jacobian_MC!(::Type{Q}, ∇f, G, model::nPenOrthMC,
    z::AbstractKnotPoint{T,N,M′}) where {T,N,M′,Q<:RobotDynamics.Explicit}

    n,m = size(model)
    n̄ = state_diff_size(model)
    nq, nv, nc = mc_dims(model)

    x = state(z) 
    u = control(z)
    dt = z.dt
    @assert dt != 0

    # compute next state and lagrange multiplier
    x⁺, λ = discrete_dynamics_MC(Q, model, x, u, z.t, dt)

    function f_imp(z)
        # Unpack
        _x⁺ = z[1:(nq+nv)]
        _x = z[(nq+nv) .+ (1:nq+nv)]
        _u = z[2*(nq+nv) .+ (1:m)]
        _λ = z[2*(nq+nv)+m .+ (1:nc)]
        out_x = zeros(eltype(z),n)
        out_x[get_configs_ind(model, _x⁺)] = f_pos(model, _x⁺, _x, _u, _λ, dt)
        out_x[get_vels_ind(model, _x⁺)] = f_vel(model,  _x⁺, _x, _u, _λ, dt)
        return out_x
    end

    all_partials = ForwardDiff.jacobian(f_imp, [x⁺;x;u;λ])
    ∇f .= -all_partials[:,1:n]\all_partials[:,n+1:end]

    # index of q in n̄
    ind = BitArray(undef, n̄)
    for i=1:model.nb
        ind[(i-1)*12 .+ (1:3)] .= 1
        ind[(i-1)*12 .+ (7:9)] .= 1
    end
    G[:,ind] .= max_constraints_jacobian(model, x⁺)
end