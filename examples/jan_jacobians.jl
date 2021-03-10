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

## Jacobians

# ∂Fτ∂ub
id = 3
eqc = CD.geteqconstraint(mech, id)
chunk = CD.∂Fτ∂ub(mech, eqc, 1)

# ∂g dconfig
joint = eqc.constraints[1]
xb = [0,0,.5]
qb = UnitQuaternion(RotX(pi))
max_cons = CD.g(joint, xb, qb) # constraint equation

g_aug(x) = CD.g(joint, x[1:3], UnitQuaternion(x[4:7]...))
G = ForwardDiff.jacobian(g_aug, [xb...;RS.params(qb)]) # automated jacobian

part1, part2 = CD.∂g∂posb(joint, xb, qb) # handmade jacobian
handmade = [part1 part2]

extrema(G-handmade) # one element is off?