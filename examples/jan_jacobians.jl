using Rotations
using ForwardDiff, FiniteDiff
using StaticArrays, LinearAlgebra
using BenchmarkTools
using Plots
using ConstrainedControl
using ConstrainedDynamics
using ConstrainedDynamicsVis

const RS = Rotations
const CC = ConstrainedControl
const CD = ConstrainedDynamics

## Mechanism
# Parameters
l1 = 1.0
l2 = 1.0
x, y = .1, .1

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
mech.Δt = 1e-7

CD.settempvars!(getbody(mech, 1), [0,0,.5], zeros(3), zeros(3), UnitQuaternion(RotX(pi)), zeros(3), zeros(3), zeros(6))
CD.settempvars!(getbody(mech, 2), [0,0,1.5], zeros(3), zeros(3), UnitQuaternion(RotX(pi)), zeros(3), zeros(3), zeros(6))

## B Jacobians
# spherical 1
# id = 3
# eqc = CD.geteqconstraint(mech, id)
# childid=1
# CD.∂Fτ∂ub(mech, eqc, childid) # this one seems weird

# joint = eqc.constraints[1] # translational
# vertices = joint.vertices
# xb = [0,0,.5]
# qb = UnitQuaternion(RotX(pi))
# Bτb = CD.VLᵀmat(qb) * CD.RVᵀmat(qb) * CD.skew(vertices[2])

# # spherical 2
# id = 4
# eqc = CD.geteqconstraint(mech, id)
# parentid=1
# childid=2
# CD.∂Fτ∂ua(mech, eqc, parentid)
# CD.∂Fτ∂ub(mech, eqc, childid)

## G Jacobians

# spherical 1
id = 3
childid = 1
eqc = CD.geteqconstraint(mech, id)
joint = eqc.constraints[1] # translational
xb = [0,0,.5]
qb = UnitQuaternion(RotX(pi))
max_cons = CD.g(joint, xb, qb) # constraint equation

g_aug(x) = CD.g(joint, x[1:3], UnitQuaternion(x[4:7]...))
G = FiniteDiff.finite_difference_jacobian(g_aug, [xb...;RS.params(qb)]) # automated jacobian

part1, part2 = CD.∂g∂posb(joint, xb, qb) # handmade jacobian
handmade = [part1 part2]

extrema(G-handmade) # one element is off?

# extra1 = CD.Rmat(CD.ωbar(cstate.ωsol[2],mech.Δt)*mech.Δt/2)*CD.LVᵀmat(cstate.qsol[2])
# extra2 = CD.Lmat(cstate.qsol[2])*CD.derivωbar(cstate.ωsol[2],mech.Δt)*mech.Δt/2
# ccol3c12 = CD.offsetrange(childid,3,12,3)
# ccol3d12 = CD.offsetrange(childid,3,12,4)

# spherical 2
# for the second spherical joint, need to look into dgdx and dgdq
# id = 4
# eqc = CD.geteqconstraint(mech, id)
# joint = eqc.constraints[1] # translational
# xa = [0,0,.5]
# qa = UnitQuaternion(RotX(pi))
# xb = [0,0,1.5]
# qb = UnitQuaternion(RotX(pi))
# max_cons = CD.g(joint, xa, qa, xb, qb) # constraint equation

# pXl, pQl = CD.∂g∂posa(joint, xa, qa, xb, qb) # handmade jacobian
# cXl, cQl = CD.∂g∂posb(joint, xa, qa, xb, qb) # handmade jacobian
# handmade = [pXl pQl cXl cQl]

# pstate = mech.bodies[1].state
# cstate = mech.bodies[2].state
# extra1 = CD.Rmat(CD.ωbar(cstate.ωsol[2],mech.Δt)*mech.Δt/2)*CD.LVᵀmat(qa)
# extra2 = CD.Rmat(CD.ωbar(pstate.ωsol[2],mech.Δt)*mech.Δt/2)*CD.LVᵀmat(qb)
# cQl*extra2

# RS.∇differential(UnitQuaternion(RotX(pi))) ≈ extra1
# pQl*extra1 ≈ Gj[4:6,7:9]

# G2[4:6,7:9]

# g_aug(x) = CD.g(joint, x[1:3], UnitQuaternion(x[4:7]...), x[8:10], UnitQuaternion(x[11:14]...))
# G = ForwardDiff.jacobian(g_aug, [xa...;RS.params(qa);xb...;RS.params(qb)]) # automated jacobian

# extrema(G-handmade) # two elements are off?
