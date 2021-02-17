include("acrobot3D.jl")

# model and timing
model = Acrobot3D()
n, m = size(model)
dt = 0.005
N = 500
tf = (N-1)*dt     

# initial and final conditions
function rc_to_mc(model::Acrobot3D, rc_x)
    θ1 = rc_x[1]
    θ2 = rc_x[2]
    l1, l2 = model.lengths

    R1 = UnitQuaternion(RotX(θ1))
    R2 = UnitQuaternion(RotX(θ1+θ2))
    mc_x = [R1*[0.; 0.; -l1/2]; 
            RS.params(R1);
            R1*[0.; 0.; -l1] + R2*[0.; 0.; -l2/2]; 
            RS.params(R2); 
            zeros(12)]

    return mc_x
end

x0 = rc_to_mc(model, [.01, 0])
xf = rc_to_mc(model, [.3, .4])

# objective
Qf = Diagonal(@SVector fill(100., n))
Q = Diagonal(@SVector fill(.01/dt, n))
R = Diagonal(@SVector fill(0.001/dt, m))

costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=0.1) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=100.0)
obj = Objective(costfuns);

prob = Problem(model, obj, xf, tf, x0=x0);

# intial rollout with random controls
U0 = [SVector(.5) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);

# test costs
X = states(prob)
x1 = copy(x0)
u1 = control(prob.Z[1])
cf = TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=0.1);
cf = costfuns[end]
@show cf.Q
@show cf.R
@show cf.q
@show cf.r
@show cf.c
@show cf.w
@show cf.vinds
@show cf.qinds
@show cf.qrefs

@show TO.stage_cost(cf, x1)
@show TO.stage_cost(cf, prob.Z[1])

# by hand
x1_v = x1[cf.vinds]
xf_v = xf[cf.vinds]
sum = .5*(x1_v-xf_v)'*Diagonal(cf.Q)*(x1_v-xf_v)
sum2 = TO.veccost(cf.Q, cf.q, x1, cf.vinds)+cf.c
@show sum-sum2

control_cost = .5u1'*Diagonal(cf.R)*u1 + cf.r'u1


