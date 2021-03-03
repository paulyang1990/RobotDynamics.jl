using ConstrainedDynamics
using ConstrainedDynamicsVis
using ConstrainedControl
using LinearAlgebra


# Parameters
N=8
joint_axis = [1.0;0.0;0.0]
R = UnitQuaternion(RotZ(pi/2))

length1 = 1.0
width, depth = 0.1, 0.1

p2 = [0.0;0.0;length1/2] # joint connection point

# Desired orientation
ϕ = 0.

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


xd=[[0;0.;0.5+(i-1)*length1] for i=1:N]
qd=[UnitQuaternion(RotX(0.0))  for i=1:N]

Q = [diagm(ones(12))*1.0 for i=1:N]
R = [diagm(ones(1)) for i=1:N]

# lqr = LQR(mech, getid.(links), getid.(constraints), Q, R, 10, xd=xd, qd=qd,Fτd = [SA{Float64}[0;0;0] for i=1:length(getid.(constraints))])

lqr = LQR(mech, getid.(links), getid.(constraints), Q, R, 10, xd=xd, qd=qd)


storage = simulate!(mech,10,lqr, record = true)
visualize(mech,storage)