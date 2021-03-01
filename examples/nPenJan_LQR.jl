using ConstrainedControl, ConstrainedDynamics, ConstrainedDynamicsVis

import ConstrainedControl: LQR, control_lqr!

mutable struct LQR2{T,N,NK} <: Controller
    K::Vector{Vector{SMatrix{1,NK,T,NK}}} # for each time step and each eqc

    xd::Vector{SVector{3,Float64}}
    vd::Vector{SVector{3,Float64}}
    qd::Vector{UnitQuaternion{T}}
    ωd::Vector{SVector{3,Float64}}

    eqcids::Vector{Integer}
    Fτd::Vector{SVector{3,Float64}}

    control!::Function
end

function LQR(A, Bu, Bλ, G, Q, R, horizon, eqcids, xd, vd, qd, ωd, Fτd, Δt, ::Type{T}; controlfunction::Function = control_lqr!) where {T}
    Q = cat(Q...,dims=(1,2))*Δt
    R = cat(R...,dims=(1,2))*Δt

    N = horizon/Δt
    if N<Inf
        N = Integer(ceil(horizon/Δt))
        Ntemp = N
    else
        Ntemp = Integer(ceil(10/Δt)) # 10 second time horizon as maximal horizon for convergence for Inf
    end

    # calculate K
    if size(G)[1] == 0
        @assert size(Bλ)[2] ==0
        if N==Inf
            K = CC.dlqr(A, Bu, Q, R, N)
            Ku = [[K[i:i,:] for i=1:size(K)[1]]]
        else
            Ku = CC.dlqr(A,Bu,zeros(size(A)[1],0),zeros(0,size(A)[1]),Q,R,N)
        end
    else
        Ku = CC.dlqr(A, Bu, Bλ, G, Q, R, Ntemp)
        if N == Inf
            Ku[1] != Ku[2] && @info "Riccati recursion did not converge."
            Ku = [Ku[1]]
        end
    end
    
    return (Ku, xd, vd, qd, ωd, eqcids, Fτd, controlfunction)
end

function CC.control_lqr!(mechanism::Mechanism{T,Nn,Nb}, lqr::LQR2{T,N}, k) where {T,Nn,Nb,N}
    Δz = zeros(T,Nb*12)
    qvm = QuatVecMap()
    for (id,body) in pairs(mechanism.bodies)
        colx = (id-1)*12+1:(id-1)*12+3
        colv = (id-1)*12+4:(id-1)*12+6
        colq = (id-1)*12+7:(id-1)*12+9
        colω = (id-1)*12+10:(id-1)*12+12

        state = body.state
        Δz[colx] = state.xsol[2]-lqr.xd[id]
        Δz[colv] = state.vsol[2]-lqr.vd[id]
        Δz[colq] = RS.rotation_error(state.qsol[2],lqr.qd[id],qvm)
        Δz[colω] = state.ωsol[2]-lqr.ωd[id]
    end

    if k<N
        for (i,id) in enumerate(lqr.eqcids)
            val1 = lqr.K[k][3*(i-1)+1]*Δz
            val2 = lqr.K[k][3*(i-1)+2]*Δz
            val3 = lqr.K[k][3*(i-1)+3]*Δz
            u = lqr.Fτd[i] - [val1; val2; val3]
            setForce!(mechanism, geteqconstraint(mechanism, id), u)
        end
    end

    return
end


l1 = 1.0
l2 = sqrt(2) / 2
x, y = .1, .1

vert11 = [0.;0.;l1 / 2]
vert12 = -vert11

vert21 = [0.;0.;l2 / 2]

# Initial orientation
phi1 = pi-.01
q1 = UnitQuaternion(RotX(phi1))

# Links
origin = Origin{Float64}()
link1 = Box(x, y, l1, l1, color = RGBA(1., 1., 0.))
link2 = Box(x, y, l2, l2, color = RGBA(1., 1., 0.))

# Constraints
socket0to1 = EqualityConstraint(Spherical(origin, link1; p2=vert11))
socket1to2 = EqualityConstraint(Spherical(link1, link2; p1=vert12, p2=vert21))

links = [link1;link2]
constraints = [socket0to1;socket1to2]

mech = Mechanism(origin, links, constraints)
setPosition!(origin,link1,p2 = vert11,Δq = q1)
setPosition!(link1,link2,p1 = vert12,p2 = vert21,Δq = one(UnitQuaternion))

xd = [[0,0,.5],[0,0,1.5]]
qd = [UnitQuaternion(RotX(pi)),UnitQuaternion(RotX(pi))]
Fτd = [zeros(3) for i=1:length(constraints)]

Q = [diagm(ones(12))*0.0 for i=1:2]
Q[1][7,7]=4.0
Q[1][10,10]=4.0
Q[2][7,7]=1.0
Q[2][10,10]=1.0
R = [diagm(ones(3)) for i=1:2]

Ku, xd, vd, qd, ωd, eqcids, Fτd, controlfunction = LQR(mech, getid.(links), getid.(constraints), Q, R, 10., xd=xd, qd=qd, Fτd=Fτd)
N = 10/mech.Δt
NK = size(Ku[1][1])[2]
lqr = LQR2{Float64,N,NK}(Ku, xd, vd, qd, ωd, eqcids, Fτd, controlfunction)
storage = simulate!(mech,10,lqr,record = true)
visualize(mech,storage)
