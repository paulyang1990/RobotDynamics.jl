"""
WIP: ordering of states in jans and ours is not the same
"""

using ConstrainedDynamics
using ConstrainedDynamicsVis
using ConstrainedControl
using LinearAlgebra
using Rotations

# Parameters
l1 = 1.0
l2 = 1.0
x, y = .1, .1

vert11 = [0.;0.;l1 / 2]
vert12 = -vert11

vert21 = [0.;0.;l2 / 2]

# Desired orientation
phi1 = pi
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
mech.Δt = .001
setPosition!(origin,link1,p2 = vert11,Δq = q1)
setPosition!(link1,link2,p1 = vert12,p2 = vert21,Δq = one(UnitQuaternion))

# Jacobians
bodyids = getid.(links)
eqcids = getid.(constraints)
Nb = 2
xd = [[[0;0;0.5]];[[0;0;1.5]]]
qd=[[UnitQuaternion(RotX(ϕ1))];[UnitQuaternion(RotX(ϕ2))]]
vd = [SA[0; 0; 0] for i=1:Nb]
ωd = [SA[0; 0; 0] for i=1:Nb]
Fτd = [SA[0; 0; 0] for i=1:length(eqcids)]
Aj, Bu, Bλ, Gj = ConstrainedControl.linearsystem(mech, xd, vd, qd, ωd, Fτd, bodyids, eqcids) 

"""
======================== REPEAT FOR nPendulumSpherical =================
"""

include("nPendulumSpherical.jl")

# Model
nb = length(mech.bodies)
masses = [mech.bodies[i].m for i=1:nb]
inertias = [Diagonal([mech.bodies[i].J[1,1], mech.bodies[i].J[2,2] , mech.bodies[i].J[3,3] ]) for i=1:nb]
model = nPendulumSpherical{UnitQuaternion{Float64},Float64}(masses, ones(2), .1ones(2), inertias)

nq, nv, nc = mc_dims(model)
n, m = size(model)
n̄ = RD.state_diff_size(model)

dt = mech.Δt
xf = [generate_config(model, fill(pi, model.nb)); zeros(nv)]

# Jacobian
DExp = TO.DynamicsExpansionMC(model)
diff1 = SizedMatrix{n,n̄}(zeros(n,n̄))
RD.state_diff_jacobian!(diff1, RD.LieState(model), SVector{n}(xf))
diff2 = SizedMatrix{n,n̄}(zeros(n,n̄))
RD.state_diff_jacobian!(diff2, RD.LieState(model), SVector{n}(xf))

z = KnotPoint(xf, zeros(6), dt)
Altro.discrete_jacobian_MC!(PassThrough, DExp.∇f, DExp.G, model, z)
TO.save_tmp!(DExp)
TO.error_expansion!(DExp, diff1, diff2)
A,B,C,G = TO.error_expansion(DExp, model)

A ≈ Aj