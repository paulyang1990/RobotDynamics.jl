include("floatingBaseSpace.jl")
model = FloatingSpaceOrth(23)
x0 = generate_config(model, [0.0;0.0;1.0;pi/2], fill.(pi/13,model.nb));

Tf = 12
dt = 0.005
N = Int(Tf/dt)

mech = vis_mech_generation(model)
x = x0
λ_init = zeros(5*model.nb)
λ = λ_init
U = 0.03*rand(6+model.nb)
U[1] = 4
U[4] = 4
# U[7] = 0.0001
steps = Base.OneTo(Int(N))
storage = CD.Storage{Float64}(steps,length(mech.bodies))
for idx = 1:N
    println("step: ",idx)
    x1, λ1 = discrete_dynamics(model,x, U, λ, dt)
    println(norm(fdyn(model,x1, x, U, λ1, dt)))
    println(norm(g(model,x1)))
    setStates!(model,mech,x1)
    for i=1:model.nb+1
        storage.x[i][idx] = mech.bodies[i].state.xc
        storage.v[i][idx] = mech.bodies[i].state.vc
        storage.q[i][idx] = mech.bodies[i].state.qc
        storage.ω[i][idx] = mech.bodies[i].state.ωc
    end
    x = x1
    λ = λ1
end
visualize(mech,storage, env = "editor")