include("nPenJan.jl")
include("nPendulumSpherical.jl")
include("nPendulum3D_visualize.jl")
include("nPen_util.jl")
############################# JAN ##########################################

model_j = nPenJan()
mech = model_j.mech
x0_j = getStates(mech, false)
u0 = [1,.1,0,1,.1,0]
dt = mech.Δt = 1e-5
N = 100
Xjan = quick_rollout(model_j, x0_j, u0, dt, N)
Xjan2 = jan_to_og.(Xjan)

############################# OG ##########################################

masses = [body.m for body in mech.bodies]
inertias = [Diagonal([body.J[1,1], body.J[2,2], body.J[3,3]]) for body in mech.bodies]
model = nPendulumSpherical{UnitQuaternion{Float64},Float64}(masses, ones(2), .1ones(2), inertias)
x0 = jan_to_og(x0_j)
X = quick_rollout(model, x0, u0, dt, N)
# visualize!(model, X, dt)

for k = 1:length(x0)
    plot([X[i][k] for i=1:N])
    display(plot!([Xjan2[i][k] for i=1:N]))
end

all(Xjan2 .≈ X)