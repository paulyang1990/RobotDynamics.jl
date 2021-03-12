using Rotations
using ForwardDiff
using StaticArrays, LinearAlgebra
using BenchmarkTools
using Plots
using ConstrainedControl
using ConstrainedDynamics
using ConstrainedDynamicsVis

const RS = Rotations
const CC = ConstrainedControl
const CD = ConstrainedDynamics
import ConstrainedDynamics: lineardynamics, linearsystem

function CD.lineardynamics(mechanism::Mechanism{T,Nn,Nb}, eqcids) where {T,Nn,Nb}
    Δt = mechanism.Δt
    bodies = mechanism.bodies
    eqcs = mechanism.eqconstraints
    graph = mechanism.graph

    nc = 0
    for eqc in eqcs
        CD.isinactive(eqc) && continue
        nc += length(eqc)
    end

    nu = 0
    for id in eqcids
        eqc = geteqconstraint(mechanism, id)
        nu += 6-length(eqc)
    end


    # calculate next state
    CD.discretizestate!(mechanism)
    foreach(CD.setsolution!, mechanism.bodies)
    foreach(CD.applyFτ!, eqcs, mechanism, false)
    CD.newton!(mechanism)

    # get state linearization 
    Fz = zeros(T,Nb*13,Nb*12)
    Fu = zeros(T,Nb*13,Nb*6)
    Fλ = zeros(T,Nb*13,nc)
    Ffz = zeros(T,Nb*13,Nb*13)
    invFfzquat = zeros(T,Nb*12,Nb*13)

    Bcontrol = zeros(T,Nb*6,nu)

    for (id,body) in CD.pairs(bodies)
        col6 = CD.offsetrange(id,6)
        col12 = CD.offsetrange(id,12)
        col13 = CD.offsetrange(id,13)

        Fzi = CD.∂F∂z(body, Δt)
        Fui = CD.∂F∂u(body, Δt)
        Ffzi, invFfzquati = CD.∂F∂fz(body, Δt)

        Fz[col13,col12] = Fzi
        Fu[col13,col6] = Fui
        Ffz[col13,col13] = Ffzi
        invFfzquat[col12,col13] = invFfzquati
    end

    Ffz += CD.linearconstraintmapping(mechanism)
    Fz += CD.linearforcemapping(mechanism)

    n1 = 1
    n2 = 0

    for id in eqcids
        eqc = CD.geteqconstraint(mechanism, id)
        n2 += 6-length(eqc)

        parentid = eqc.parentid
        if parentid !== nothing
            col6 = CD.offsetrange(parentid,6)
            # in nPendulumSpherical we dont apply u to parent body
            Bcontrol[col6,n1:n2] = CD.∂Fτ∂ua(mechanism, eqc, parentid)
        end
        for childid in eqc.childids
            col6 = CD.offsetrange(childid,6)
            Bcontrol[col6,n1:n2] = CD.∂Fτ∂ub(mechanism, eqc, childid)
            # should this be I
            # Bcontrol[col6[4:6],n1:n2] = Matrix(I,3,3) 
        end

        n1 = n2+1
    end

    G, Fλ = CD.linearconstraints(mechanism)

    return invFfzquat, Ffz, Fz, Fu, Bcontrol, Fλ, G

    # invFfz = invFfzquat * inv(Ffz)
    # A = -invFfz * Fz
    # Bu = -invFfz * Fu * Bcontrol
    # Bλ = -invFfz * Fλ

    # return A, Bu, Bλ, G
end

function CD.linearsystem(mechanism::Mechanism{T,Nn,Nb}, xd, vd, qd, ωd, Fτd, bodyids, eqcids) where {T,Nn,Nb}
    statesold = [CD.State{T}() for i=1:Nb]

    # store old state and set new initial state
    for (i,id) in enumerate(bodyids)
        stateold = CD.settempvars!(CD.getbody(mechanism, id), xd[i], vd[i], zeros(T,3), qd[i], ωd[i], zeros(T,3), zeros(T,6))
        statesold[i] = stateold
    end
    for (i,id) in enumerate(eqcids)
        CD.setForce!(mechanism, CD.geteqconstraint(mechanism, id), Fτd[i])
    end

    return CD.lineardynamics(mechanism, eqcids)
    # A, Bu, Bλ, G = CD.lineardynamics(mechanism, eqcids)

    # # restore old state
    # for (i,id) in enumerate(bodyids)
    #     body = getbody(mechanism, id)
    #     body.state = statesold[i]
    # end

    # return A, Bu, Bλ, G
end


# # Parameters
# l1 = 1.0
# l2 = 1.0
# x, y = .1, .1

# vert11 = [0.;0.;l1 / 2]
# vert12 = -vert11

# vert21 = [0.;0.;l2 / 2]

# # initial orientation
# phi1 = 0
# q1 = UnitQuaternion(RotX(phi1))

# # Links
# origin = Origin{Float64}()
# link1 = Box(x, y, l1, l1, color = RGBA(1., 1., 0.))
# link2 = Box(x, y, l2, l2, color = RGBA(1., 1., 0.))

# # Constraints
# socket0to1 = EqualityConstraint(ConstrainedDynamics.Spherical(origin, link1; p2=vert11))
# socket1to2 = EqualityConstraint(ConstrainedDynamics.Spherical(link1, link2; p1=vert12, p2=vert21))

# links = [link1;link2]
# constraints = [socket0to1;socket1to2]


# mech = Mechanism(origin, links, constraints)
# mech.Δt = 1e-7
# setPosition!(origin,link1,p2 = vert11,Δq = q1)
# setPosition!(link1,link2,p1 = vert12,p2 = vert21,Δq = one(UnitQuaternion))

# # Jacobians
# bodyids = getid.(links)
# eqcids = getid.(constraints)
# Nb = 2
# xd = [[[0;0;0.5]];[[0;0;1.5]]]
# qd=[[UnitQuaternion(RotX(pi))];[UnitQuaternion(RotX(pi))]]
# vd = [SA[0; 0; 0] for i=1:Nb]
# ωd = [SA[0; 0; 0] for i=1:Nb]
# Fτd = [SA[0; 0; 0] for i=1:length(eqcids)]
# invFfzquat, Ffz, Fz, Fu, Bcontrol, Fλ, G = ConstrainedControl.linearsystem(mech, xd, vd, qd, ωd, Fτd, bodyids, eqcids) 
# # Janstate  x1    v1      q1      w1        x2       v2        q2        w2 


# using SparseArrays
# display(spy(sparse(Ffz), marker=2, legend=nothing, c=palette([:black], 2)))
