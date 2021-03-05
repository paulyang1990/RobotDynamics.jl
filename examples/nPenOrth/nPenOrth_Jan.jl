using RobotDynamics
using TrajectoryOptimization
using Rotations
using ForwardDiff
using StaticArrays, LinearAlgebra
using BenchmarkTools
using Altro
using Plots
using ConstrainedControl
using ConstrainedDynamics
using ConstrainedDynamicsVis

const TO = TrajectoryOptimization
const RD = RobotDynamics
const RS = Rotations
const CC = ConstrainedControl
const CD = ConstrainedDynamics
const CDV = ConstrainedDynamicsVis

import TrajectoryOptimization: dynamics_expansion!

struct nPenJanOrth{R,T,Nn,Nb,Ne,Ni} <: LieGroupModelMC{R}
    mech::Mechanism{T,Nn,Nb,Ne,Ni}
    nb::Int # number of rigid bodies
    p::Int # constraint force dimension
    m::Int # control dim
end

function nPenJanOrth(mech::Mechanism{T,Nn,Nb,Ne,Ni}) where {T,Nn,Nb,Ne,Ni}
    p = 0
    m = 0
    for eqc in mech.eqconstraints
        m += 6-length(eqc)
        CD.isinactive(eqc) && continue
        p += length(eqc)
    end
    return nPenJanOrth{UnitQuaternion{T},T,Nn,Nb,Ne,Ni}(mech, Nb, p, m)
end

function nPenJanOrth(N::Integer)
    # Parameters
    length1 = 1.0
    width, depth = 0.1, 0.1
    joint_axis = [1.0;0.0;0.0]
    R = UnitQuaternion(RotZ(pi/2))
    # x=y=3.0 # for higher inertias

    p2 = [0.0;0.0;length1/2] # joint connection point


    # Links
    origin = Origin{Float64}()
    links = [Box(width, depth, length1, length1,color = RGBA(0.1*i, 0.2*i, 1.0/i)) for i = 1:N]

    # Constraints
    joint_between_origin_and_link1 = EqualityConstraint(Revolute(origin, links[1], joint_axis; p2=-p2))
    # joint_between_origin_and_link1 = EqualityConstraint(ConstrainedDynamics.Spherical(origin, links[1]; p2=-p2))

    constraints = [joint_between_origin_and_link1]
    if N > 1
        # constraints = [constraints; [EqualityConstraint(ConstrainedDynamics.Spherical(links[i], links[i+1]; p1 = p2, p2=-p2)) for i=1:N-1]]
        constraints = [constraints; [EqualityConstraint(Revolute(links[i], links[i+1], R^i*joint_axis; p1 = p2, p2=-p2)) for i=1:N-1]]
    end

    mech = Mechanism(origin, links, constraints, g=-9.81)
    setPosition!(origin,links[1],p2 = -p2,Δq = UnitQuaternion(RotX(0.02*randn())))
    for i=1:N-1
        if mod(i,2) == 0
            setPosition!(links[i],links[i+1],p1=p2,p2=-p2,Δq=UnitQuaternion(RotY(0.1*randn())))
        else
            setPosition!(links[i],links[i+1],p1=p2,p2=-p2,Δq=UnitQuaternion(RotX(0.1*randn())))
        end
    end

    return nPenJanOrth(mech)
end

Altro.config_size(model::nPenJanOrth) = 7*model.nb
Lie_P(model::nPenJanOrth) = (6, fill(9, model.nb-1)..., 3)
RD.LieState(model::nPenJanOrth{R}) where R = RD.LieState(R, Lie_P(model))
RD.control_dim(model::nPenJanOrth) = model.m

function Altro.is_converged(model::nPenJanOrth, x)
    mech = model.mech
    setStates!(mech, x)
    CD.currentasknot!(mech)
    for eqc in mech.eqconstraints
        if norm(CD.g(mech, eqc)) > 1e-3
            @info string("Bad constraint satisfaction at constraint: ", eqc.id, ", |g| = ", norm(CD.g(mech, eqc)))
            return false
        end
    end
    return true
end

function fullargsinds(i)
    # x, v, q, ω
    return 13*(i-1) .+ (1:3), 
            13*(i-1) .+ (4:6), 
            13*(i-1) .+ (7:10), 
            13*(i-1) .+ (11:13)
end

controlinds(i) = i


# compute maximal coordinate configuration given body rotations
# state config x v q w 
function generate_config(model::nPenJanOrth, rotations)
    @assert model.nb == length(rotations)
    pin = zeros(3)
    prev_q = UnitQuaternion(1.,0.0,0.0,0.0)
    state = zeros(0)
    for i = 1:model.nb
        r = UnitQuaternion(rotations[i])
        link_q = prev_q * r
        delta = link_q * [0,0,mech.bodies[i].shape.xyz[3]/2]
        link_x = pin+delta
        state = [state; link_x;zeros(3);Rotations.params(link_q);zeros(3)]

        prev_q = link_q
        pin += 2*delta
    end


    return state
end

function generate_config(model::nPenJanOrth, θ::Vector{<:Number})
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

function setStates!(mech, z)
    for (i, body) in enumerate(mech.bodies)   
        xinds, vinds, qinds, ωinds = fullargsinds(i)   
        setPosition!(body; x = SVector{3}(z[xinds]), q = UnitQuaternion(z[qinds]...))
        setVelocity!(body; v = SVector{3}(z[vinds]), ω = SVector{3}(z[ωinds]))
    end
end

function setControls!(mech, u)
    # for (i, body) in enumerate(mech.bodies)   
    #     inds = controlinds(i)

    #     # if (i<length(mech.bodies))
    #     #     tau =
    #     #     if mod(i,2) == 0
    #     #         CD.setForce!(body, τ = SVector{3}(0,u[inds],0))
    #     #     else
    #     #         CD.setForce!(body, τ = SVector{3}(u[inds],0,0))
    #     #     end
    #     # else
    #     # end

    #     if (i <length(mech.bodies))
    #         a = -u[controlinds(i+1)]
    #         b = u[inds]
    #         if mod(i,2) == 0
    #             tau = SVector{3}(a,b,0)
    #         else
    #             tau = SVector{3}(b,a,0)
    #         end
    #         CD.setForce!(body, τ = tau )

    #     else      
    #         if mod(i,2) == 0
    #             CD.setForce!(body, τ = SVector{3}(0,u[inds],0))
    #         else
    #             CD.setForce!(body, τ = SVector{3}(u[inds],0,0))
    #         end
    #     end
    # end
    for (i,id) in enumerate(getid.(mech.eqconstraints))
        (mech, geteqconstraint(mech, id), u[i])
    end    
end

# Janstate  x1    v1      q1      w1        x2       v2        q2        w2 
function getStates(mech, sol=true)
    nb = length(mech.bodies)
    z = zeros(13*nb)
    for (i, body) in enumerate(mech.bodies)        
        xinds, vinds, qinds, ωinds = fullargsinds(i)
        f = sol ? CD.fullargssol : CD.fullargsc
        z[xinds],z[vinds],q,z[ωinds] = f(body.state)
        z[qinds] = RS.params(q)
    end
    return z
end

function RD.discrete_dynamics(::Type{Q}, model::nPenJanOrth, x, u, t, dt) where Q
    mech = model.mech
    mech.Δt = dt
    
    # initialize
    setStates!(mech, x)
    CD.discretizestate!(mech)
    foreach(CD.setsolution!, mech.bodies)

    # compute next state
    setControls!(mech, u)
    foreach(CD.applyFτ!, mech.eqconstraints, mech, false)
    CD.newton!(mech)
    
    return getStates(mech)
end

function discrete_jacobian_MC!(::Type{Q}, A,B,C, G, model::nPenJanOrth,
    z::AbstractKnotPoint{T,N,M′}) where {T,N,M′,Q<:RobotDynamics.Explicit}
    mech = model.mech
    bodyids = getid.(mech.bodies)
    eqcids = getid.(mech.eqconstraints)

    n,m = size(model)
    n̄ = state_diff_size(model)
    nq, nv, nc = mc_dims(model)
    x = state(z)
    u = control(z)
    dt = z.dt
    @assert dt != 0
    
    xd = SArray{Tuple{3},Float64,1,3}[]
    vd = SArray{Tuple{3},Float64,1,3}[]
    qd = UnitQuaternion{Float64}[]
    ωd = SArray{Tuple{3},Float64,1,3}[]
    Fτd = SArray{Tuple{1},Float64,1,1}[]
    for i=1:model.nb
        xinds, vinds, qinds, ωinds = fullargsinds(i)
        push!(xd, x[xinds])
        push!(vd, x[vinds])
        push!(qd, UnitQuaternion(x[qinds]...))
        push!(ωd, x[ωinds])

        uinds = controlinds(i)
        push!(Fτd, SA[u[uinds]])
    end
    
    LA, LB, LC, LG = CC.linearsystem(mech, xd, vd, qd, ωd, Fτd, bodyids, eqcids) 


    A .= LA
    B .= LB
    C .= LC
    G .= LG
end

function TO.dynamics_expansion!(Q, D::Vector{<:TO.DynamicsExpansionMC}, model::nPenJanOrth, Z::Traj)
    for k in eachindex(D)
        if Z[k].dt == 0
            z = copy(Z[k])
            z.dt = Z[1].dt
            discrete_jacobian_MC!(Q, D[k].A, D[k].B, D[k].C, D[k].G, model, z)
        else
            discrete_jacobian_MC!(Q, D[k].A, D[k].B, D[k].C, D[k].G, model, Z[k])
        end
      end
end

function TO.error_expansion!(::Vector{<:TO.DynamicsExpansionMC}, ::nPenJanOrth, G)
	return
end

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

