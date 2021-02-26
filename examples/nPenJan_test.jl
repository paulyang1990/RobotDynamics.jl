include("nPenJan.jl")

model = nPenJan()
n,m = size(model)
x0 = getStates(mech, false)
u0 = [1,.1,0,1,.1,0]
dt = mech.Δt

# dynamics
z = KnotPoint(x0, u0, dt)
@show Altro.is_converged(model, x0)
x1 = RD.discrete_dynamics(PassThrough, model, z)

# rollout
N = 1000
X = quick_rollout(model, x0, u0, dt, N)
plot_traj(X, [X[i][1:2] for i=1:N])

# jacobian
DExp = TO.DynamicsExpansionMC(model)
discrete_jacobian_MC!(PassThrough, DExp, model, z)
A,B,C,G = TO.error_expansion(DExp, model)
