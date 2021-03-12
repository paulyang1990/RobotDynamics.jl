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

include("nPenOrth_Jan.jl")

# model
num_links = 2
model = nPenJanOrth(num_links)
mech = model.mech
n,m = size(model)
dt = mech.Δt = .005

# test initial and final state (works)
# x0 = generate_config(model, [pi/2;pi/2;pi/2;pi/2])
x0 = generate_config(model, [pi/6;pi/6])
xf = generate_config(model, [0.0;0.0])
# x0 = generate_config(model, [pi/6;pi/6.0])
# xf = generate_config(model, [0;0.0])
# visualize state 
setStates!(mech,x0)

# initial controls
U0 = [SVector{m}(fill(0.0,m)) for k = 1:N]

# test ABCG jacobian
# D = TO.DynamicsExpansionMC(model)
xd, vd, qd, ωd, Fτd = state_parts(model, x0,U0[1])
bodyids = getid.(mech.bodies)
eqcids = getid.(mech.eqconstraints)
A1, B1, C1, G1 = ConstrainedControl.linearsystem(mech, xd, vd, qd, ωd, Fτd, bodyids, eqcids) 

include("nPenOrth_mc.jl")

# model
num_links = 2
model2 = nPenOrthMC(num_links)
n,m = size(model2)
n̄ = RD.state_diff_size(model2)
# Jacobian
DExp = TO.DynamicsExpansionMC(model2)
diff = SizedMatrix{n,n̄}(zeros(n,n̄))
RD.state_diff_jacobian!(diff1, RD.LieState(model2), SVector{n}(x0))


z = KnotPoint(x0, U0[1], dt)
Altro.discrete_jacobian_MC!(PassThrough, DExp.∇f, DExp.G, model2, z)
TO.save_tmp!(DExp)
# TO.error_expansion!(DExp, diff1, diff2)
# A,B,C,G = TO.error_expansion(DExp, model)
A2 = DExp.tmpA
B2 = DExp.tmpB
C2 = DExp.tmpC
G2 = DExp.G

# ## Jacobians of one constraint

# # ∂Fτ∂ub
# id = 3
# eqc = CD.geteqconstraint(mech, id)
# chunk = CD.∂Fτ∂ub(mech, eqc, 1)

# # ∂g dconfig
# joint = eqc.constraints[1]
# xb = [0,0,.5]
# qb = UnitQuaternion(RotX(pi))
# max_cons = CD.g(joint, xb, qb) # constraint equation

# g_aug(x) = CD.g(joint, x[1:3], UnitQuaternion(x[4:7]...))
# G = ForwardDiff.jacobian(g_aug, [xb...;RS.params(qb)]) # automated jacobian

# part1, part2 = CD.∂g∂posb(joint, xb, qb) # handmade jacobian
# handmade = [part1 part2]

# extrema(G-handmade) # one element is off