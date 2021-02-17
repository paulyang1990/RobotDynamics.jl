include("acrobot3D.jl")

model = Acrobot3D()
n, m = size(model)
dt = 0.005   
N = 100

x0 = rc_to_mc(model, [.01, 0])
X = @timeit to "rollout" quick_rollout(model, x0, [-.5], dt, N)
show(to)
reset_timer!(to)

z = KnotPoint(x0,[.4],dt)
A, B, C, G = @timeit to "discrete_dynamics_MC" Altro.discrete_jacobian_MC(PassThrough, model, z)
@timeit to "step" Altro.step!(solver, J, true)