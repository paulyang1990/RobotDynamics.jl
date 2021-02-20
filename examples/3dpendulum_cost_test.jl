include("3DPendulum.jl")

model = Pendulum3D()
n̄ = state_diff_size(model)
n,m = size(model)
N = 300   
dt = .01                  # number of knot points
tf = (N-1)*dt           # final time

# initial and final conditions
R0 = UnitQuaternion(.9999,-.0001,0, 0)
x0 = [R0*[0.; 0.; -.5]; RS.params(R0); zeros(6)]
xf = [0.; 0.;  .5; 0; 1; 0; 0; zeros(6)]

# objective
Q_diag = RD.fill_state(model, .001/dt, 0., .001/dt, .001/dt)
Q = Diagonal(Q_diag)
R = Diagonal(@SVector fill(0.001/dt,1))

# compare stage costs
quatcf = TO.QuatLQRCost(Q, R, SVector{n}(xf); w=0.1);
liecf = TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=0.1);
u0 = [.1]
z = KnotPoint(x0, u0, dt)
@show TO.stage_cost(quatcf, z)
@show TO.stage_cost(liecf, z)

@show TO.stage_cost(quatcf, SVector{n}(x0))
@show TO.stage_cost(liecf, SVector{n}(x0))

z1 = KnotPoint(x0, u0, 0.)
@show TO.stage_cost(quatcf, z1)
@show TO.stage_cost(liecf, z1)

# compare gradients
quatE = QuadraticCost{Float64}(n, m)
TO.gradient!(quatE, quatcf, x0)
TO.hessian!(quatE, quatcf, x0)

lieE = QuadraticCost{Float64}(n, m)
TO.gradient!(lieE, liecf, x0)
TO.hessian!(lieE, liecf, x0)

@show extrema(quatE.Q-lieE.Q)
@show extrema(quatE.q-lieE.q)
@show extrema(quatE.H-lieE.H)
@show extrema(quatE.R-lieE.R)
@show extrema(quatE.r-lieE.r)
@show extrema(quatE.c-lieE.c)

# error_expansion
Z = RD.Traj([z])
E = TO.QuadraticObjective(n̄,m,1)
quad_obj = TO.QuadraticObjective(E, model)
lieobj = Objective([liecf]);
quatobj = Objective([quatcf]);
G = [SizedMatrix{n,n̄}(zeros(n,n̄))]

TO.state_diff_jacobian!(G, model, Z)
TO.cost_expansion!(quad_obj, lieobj, Z, true, true)
TO.error_expansion!(E, quad_obj, model, Z, G)
lieQ = copy(E[1].Q)

TO.cost_expansion!(quad_obj, quatobj, Z, true, true)
TO.error_expansion!(E, quad_obj, model, Z, G)
quatQ = copy(E[1].Q)

# traj error_expansion
z = KnotPoint(x0,[1.],dt)
Z = RD.Traj([z])
E = TO.QuadraticObjective(n̄,m,1)
quad_obj = TO.QuadraticObjective(E, model)
lieobj = Objective([liecf]);
quatobj = Objective([quatcf]);
G = [SizedMatrix{n,n̄}(zeros(n,n̄))]

TO.state_diff_jacobian!(G, model, Z)
TO.cost_expansion!(quad_obj, lieobj, Z, true, true)
TO.error_expansion!(E, quad_obj, model, Z, G)
lieQ = copy(E[1].Q)

using SparseArrays
display(spy(sparse(lieQ), marker=2, legend=nothing, c=palette([:black], 2)))

# # intial rollout with random controls
# U0 = [SVector{1}(.01*rand(1)) for k = 1:N-1]
# initial_controls!(prob, U0)
# rollout!(prob);

# # test costs
# X = states(prob)
# x1 = copy(x0)
# u1 = control(prob.Z[1])
# cf = TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=0.1);
# cf = costfuns[end]
# @show cf.Q
# @show cf.R
# @show cf.q
# @show cf.r
# @show cf.c
# @show cf.w
# @show cf.vinds
# @show cf.qinds
# @show cf.qrefs

# @show TO.stage_cost(cf, x1)
# @show TO.stage_cost(cf, prob.Z[1])

# # by hand
# x1_v = x1[cf.vinds]
# xf_v = xf[cf.vinds]
# sum = .5*(x1_v-xf_v)'*Diagonal(cf.Q)*(x1_v-xf_v)
# sum2 = TO.veccost(cf.Q, cf.q, x1, cf.vinds)+cf.c
# @show sum-sum2

# control_cost = .5u1'*Diagonal(cf.R)*u1 + cf.r'u1


