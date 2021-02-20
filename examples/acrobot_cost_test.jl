include("Acrobot3D.jl")

model = Acrobot3D()
n̄ = state_diff_size(model)
n,m = size(model)
N = 1  
dt = .01                  # number of knot points
tf = (N-1)*dt           # final time

# initial and final conditions
x0 = rc_to_mc(model, [.01, 0])
xf = rc_to_mc(model, [pi, 0])

# objective
Qf = Diagonal(@SVector fill(100., n))
Q = Diagonal(@SVector fill(1e-2, n))
R = Diagonal(@SVector fill(1e-1, m))

# error_expansion
z = KnotPoint(x0,[1.],dt)
Z = RD.Traj([z])
E = TO.QuadraticObjective(n̄,m,1)
quad_obj = TO.QuadraticObjective(E, model)
liecf = TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=0.1);
lieobj = Objective([liecf]);

G = [SizedMatrix{n,n̄}(zeros(n,n̄)) for i=1:2]
TO.state_diff_jacobian!(G, model, Z)
G1 = copy(G[1])
TO.cost_expansion!(quad_obj, lieobj, Z, true, true)
TO.error_expansion!(E, quad_obj, model, Z, G)
lieQ = copy(E[1].Q)
