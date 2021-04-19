include("nPenJan_forwarddiff_test.jl")

using ConstrainedDynamics
using ConstrainedDynamicsVis
using ConstrainedControl
using LinearAlgebra
using Rotations
using StaticArrays

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

# Constraints
socket0to1 = EqualityConstraint(ConstrainedDynamics.Spherical(origin, link1; p2=vert11))

links = [link1]
constraints = [socket0to1]

mech = Mechanism(origin, links, constraints)
mech.Δt = 1e-7
setPosition!(origin,link1,p2 = vert11,Δq = q1)

# Jacobians
bodyids = getid.(links)
eqcids = getid.(constraints)
Nb = 1
xd = [[0;0;0.5]]
qd=[UnitQuaternion(RotX(pi))]
vd = [SA[0; 0; 0] for i=1:Nb]
ωd = [SA[0; 0; 0] for i=1:Nb]
Fτd = [SA[0; 0; 0] for i=1:length(eqcids)]
invFfzquat, Ffz, Fz, Fu, Bcontrol, Fλ, Gj = ConstrainedControl.linearsystem(mech, xd, vd, qd, ωd, Fτd, bodyids, eqcids) 
Aj = -Ffz\Fz
Bu = -Ffz\Fu * Bcontrol
Bλ = -Ffz\Fλ

"""
======================== REPEAT FOR nPendulumSpherical =================
"""

include("nPendulumSpherical.jl")

# Model
nb = length(mech.bodies)
masses = [mech.bodies[i].m for i=1:nb]
inertias = [Diagonal([mech.bodies[i].J[1,1], mech.bodies[i].J[2,2] , mech.bodies[i].J[3,3] ]) for i=1:nb]
model = nPendulumSpherical{UnitQuaternion{Float64},Float64}(masses, ones(nb), .1ones(nb), inertias)

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

z = KnotPoint(xf, zeros(m), dt)
Altro.discrete_jacobian_MC!(PassThrough, DExp.∇f, DExp.G, model, z)
TO.save_tmp!(DExp)
# TO.error_expansion!(DExp, diff1, diff2)
# A,B,C,G = TO.error_expansion(DExp, model)
A = DExp.tmpA
B = DExp.tmpB
C = DExp.tmpC

"""
======================== COMPARISONS =================
"""

# ourstate x1      q1      v1      w1         
# p =    [1 2 3. 4 5 6 7 . 8 9 10. 11 12 13] 
# Janstate  x1    v1      q1      w1       
p_big = [1, 2, 3, 8, 9, 10, 4, 5, 6, 7, 11, 12, 13]
# ourstate x1      q1      v1      w1         
# p =    [1 2 3. 4 5 6. 7 8 9. 10 11 12 ] 
# Janstate  x1    v1      q1       w1        
p = [1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12]


# using SparseArrays
# display(spy(sparse(C2), marker=2, legend=nothing, c=palette([:black], 2)))
# display(spy(sparse(Bλ), marker=2, legend=nothing, c=palette([:black], 2)))

# compare error map
diff_temp = diff1[:,p]
diff1T = diff_temp[p_big,:]'
diff1T ≈ invFfzquat

# compare A
Ape = A[:,p_big]
A2 = Ape[p_big,:]
extrema(A2*invFfzquat'-Aj)

# compare B
B2 = B[p_big,:]
extrema(B2-Bu)

# compare C
C2 = C[p_big,:]
extrema(C2-Bλ)

# compare G
G2 = DExp.G[:,p]
extrema(G2-Gj)