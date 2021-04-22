include("floatingBaseSpace.jl")
using Profile
using TimerOutputs
const to = TimerOutput()
model = FloatingSpace()
n,m = size(model)
x0 = generate_config(model, [0.01;0.01;0.01;0.01], fill.(0.01,model.nb))

xf = generate_config(model, [1.0;1.0;1.0;pi/4], fill.(pi/4,model.nb))

# trajectory 
N = 100   
dt = 0.005                  # number of knot points
tf = (N-1)*dt           # final time

U0 = @SVector fill(0.00001, m)
U_list = [U0 for k = 1:N-1]


# objective
Qf = Diagonal(@SVector fill(550., n))
Q = Diagonal(@SVector fill(1e-2, n))
R = Diagonal(@SVector fill(1e-3, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=1e-1) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=550.0)
obj = Objective(costfuns);

# # problem
prob = Problem(model, obj, xf, tf, x0=x0);
opts = SolverOptions(verbose=7, static_bp=0, iterations=150, cost_tolerance=1e-4)
ilqr = Altro.iLQRSolver(prob, opts);
TimerOutputs.enable_debug_timings(Altro)
Altro.timeit_debug_enabled()
solve!(ilqr);


X_list = states(ilqr)

mech = vis_mech_generation(model)
steps = Base.OneTo(Int(N))
storage = CD.Storage{Float64}(steps,length(mech.bodies))
for idx = 1:N
    setStates!(model,mech,X_list[idx])
    for i=1:model.nb+1
        storage.x[i][idx] = mech.bodies[i].state.xc
        storage.v[i][idx] = mech.bodies[i].state.vc
        storage.q[i][idx] = mech.bodies[i].state.qc
        storage.ω[i][idx] = mech.bodies[i].state.ωc
    end
end
visualize(mech,storage, env = "editor")