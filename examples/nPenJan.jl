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

import TrajectoryOptimization: dynamics_expansion!

struct nPenJan{R,T,Nn,Nb,Ne,Ni} <: LieGroupModelMC{R}
    mech::Mechanism{T,Nn,Nb,Ne,Ni}
    nb::Int # number of rigid bodies
    p::Int # constraint force dimension
    m::Int # control dim
end

function nPenJan(mech::Mechanism{T,Nn,Nb,Ne,Ni}) where {T,Nn,Nb,Ne,Ni}
    p = 0
    m = 0
    for eqc in mech.eqconstraints
        m += 6-length(eqc)
        CD.isinactive(eqc) && continue
        p += length(eqc)
    end
    return nPenJan{UnitQuaternion{T},T,Nn,Nb,Ne,Ni}(mech, Nb, p, m)
end

function nPenJan()
    # Parameters
    l1 = 1.0
    l2 = 1.0
    x, y = .1, .1
    x=y=3.0 # for higher inertias

    vert11 = [0.;0.;l1 / 2]
    vert12 = -vert11

    vert21 = [0.;0.;l2 / 2]

    # initial orientation
    phi1 = 0
    q1 = UnitQuaternion(RotX(phi1))

    # Links
    origin = Origin{Float64}()
    link1 = Box(x, y, l1, l1, color = RGBA(1., 1., 0.))
    link2 = Box(x, y, l2, l2, color = RGBA(1., 1., 0.))

    # Constraints
    socket0to1 = EqualityConstraint(ConstrainedDynamics.Spherical(origin, link1; p2=vert11))
    socket1to2 = EqualityConstraint(ConstrainedDynamics.Spherical(link1, link2; p1=vert12, p2=vert21))

    links = [link1;link2]
    constraints = [socket0to1;socket1to2]

    mech = Mechanism(origin, links, constraints)
    mech.Δt = .005

    setPosition!(origin,link1,p2 = vert11,Δq = q1)
    setPosition!(link1,link2,p1 = vert12,p2 = vert21,Δq = one(UnitQuaternion))

    return nPenJan(mech)
end

Altro.config_size(model::nPenJan) = 7*model.nb
Lie_P(model::nPenJan) = (6, fill(9, model.nb-1)..., 3)
RD.LieState(model::nPenJan{R}) where R = RD.LieState(R, Lie_P(model))
RD.control_dim(model::nPenJan) = model.m

function Altro.is_converged(model::nPenJan, x)
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

controlinds(i) = 3*(i-1) .+ (1:3)

function setStates!(mech, z)
    for (i, body) in enumerate(mech.bodies)   
        xinds, vinds, qinds, ωinds = fullargsinds(i)   
        setPosition!(body; x = SVector{3}(z[xinds]), q = UnitQuaternion(z[qinds]...))
        setVelocity!(body; v = SVector{3}(z[vinds]), ω = SVector{3}(z[ωinds]))
    end
end

function setControls!(mech, u)
    for (i, body) in enumerate(mech.bodies)   
        inds = controlinds(i)
        CD.setForce!(body, τ = SVector{3}(u[inds]))
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

function RD.discrete_dynamics(::Type{Q}, model::nPenJan, x, u, t, dt) where Q
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

function discrete_jacobian_MC!(::Type{Q}, D, model, z) where Q
    mech = model.mech
    bodyids = getid.(mech.bodies)
    eqcids = getid.(mech.eqconstraints)

    x = state(z)
    u = control(z)
    
    xd = SArray{Tuple{3},Float64,1,3}[]
    vd = SArray{Tuple{3},Float64,1,3}[]
    qd = UnitQuaternion{Float64}[]
    ωd = SArray{Tuple{3},Float64,1,3}[]
    Fτd = SArray{Tuple{3},Float64,1,3}[]
    for (i, body) in enumerate(mech.bodies)
        xinds, vinds, qinds, ωinds = fullargsinds(i)
        push!(xd, x[xinds])
        push!(vd, x[vinds])
        push!(qd, UnitQuaternion(x[qinds]...))
        push!(ωd, x[ωinds])

        uinds = controlinds(i)
        push!(Fτd, u[uinds])
    end
    
    A, B, C, G = ConstrainedControl.linearsystem(mech, xd, vd, qd, ωd, Fτd, bodyids, eqcids) 

    D.A .= A
    D.B .= B
    D.C .= C
    D.G .= G
end

function TO.dynamics_expansion!(Q, D::Vector{<:TO.DynamicsExpansionMC}, model::nPenJan, Z::Traj)
    for k in eachindex(D)
        if Z[k].dt == 0
            z = copy(Z[k])
            z.dt = Z[1].dt
            discrete_jacobian_MC!(Q, D[k], model, Z[k])            
        else
            discrete_jacobian_MC!(Q, D[k], model, Z[k])
        end
    end
end

function TO.error_expansion!(::Vector{<:TO.DynamicsExpansionMC}, ::nPenJan, G)
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

function plot_traj_jan(X, U)
    N = length(X)
    nb = Int(length(X[1])/13)
    qmats = []
    for i=1:nb
        xinds, vinds, qinds, ωinds = fullargsinds(i)  
        quats = [X[i][qinds] for i=1:N]
        qmat = hcat(quats...)'
        display(plot(qmat))
        push!(qmats, qmat)
    end

    Umat = hcat(Vector.(U)...)'
    display(plot(Umat))

    return qmats, Umat
end
