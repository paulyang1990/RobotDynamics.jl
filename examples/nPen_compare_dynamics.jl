using ConstrainedDynamics
using ConstrainedDynamicsVis


# Parameters
ex = [1.;0.;0.]

l1 = 1.0
l2 = 1.0
x, y = .1, .1

vert11 = [0.;0.;l1 / 2]
vert12 = -vert11

vert21 = [0.;0.;l2 / 2]

# Initial orientation
phi1 = pi / 4
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
setPosition!(link1,link2,p1 = vert12,p2 = vert21,Δq = inv(q1)*UnitQuaternion(RotY(0.2)))

N = 2000
tf = N*mech.Δt
storage = simulate!(mech, tf, record = true)
# visualize(mech, storage)

# convert storage to states()
function convert_storage(storage, mech)
    nb = length(mech.bodies)
    N = length(storage.x[1])
    states = []
    for (x1,x2,q1,q2,v1,v2,ω1,ω2) in zip(storage.x..., storage.q..., storage.v..., storage.ω...)
        push!(states, [x1...,RS.params(q1)...,x2...,RS.params(q2)...,v1...,ω1...,v2...,ω2...])
        # state = []
        # append!(state,x1)
        # ,RS.params(q1),x2,RS.params(q2),v1,ω1,v2,ω2)
        # push!(states, state)
    end
    return states
end

Xjan = convert_storage(storage, mech)

"""
======================= REPEAT WITH nPendulumSpherical ==============================
"""

include("nPendulumSpherical.jl")
include("nPendulum3D_visualize.jl")

nb = length(mech.bodies)
masses = [mech.bodies[i].m for i=1:nb]
inertias = [Diagonal([mech.bodies[i].J[1,1], mech.bodies[i].J[2,2] , mech.bodies[i].J[3,3] ]) for i=1:nb]
model = nPendulumSpherical{UnitQuaternion{Float64},Float64}(masses, ones(2), .1ones(2), inertias)
nq, nv, nc = mc_dims(model)
n, m = size(model)
dt = mech.Δt
x0 = Xjan[1]

X = quick_rollout(model, x0, zeros(6), dt, N)
# visualize!(model, X, dt)

for k = 1:length(x0)
    plot([X[i][k] for i=1:N])
    display(plot!([Xjan[i][k] for i=1:N]))
end