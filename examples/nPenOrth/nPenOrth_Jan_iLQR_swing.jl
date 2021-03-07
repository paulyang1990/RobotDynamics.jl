include("nPenOrth_Jan.jl")

# model
num_links = 4
model = nPenJanOrth(num_links)
mech = model.mech
n,m = size(model)
dt = mech.Δt = .005
N = 1000
tf = (N-1)*dt  

# test initial and final state (works)
# x0 = generate_config(model, [pi/2;pi/2;pi/2;pi/2])
x0 = generate_config(model, [pi/6;0.0;pi/6;pi/6])
xf = generate_config(model, [0;0.0;0.0;0.0])
# x0 = generate_config(model, [pi/6;pi/6.0])
# xf = generate_config(model, [0;0.0])
# visualize state 
setStates!(mech,x0)
# steps = Base.OneTo(1)
# storage = CD.Storage{Float64}(steps,length(mech.bodies))
# for i=1:model.nb
#     storage.x[i][1] = mech.bodies[i].state.xc
#     storage.v[i][1] = mech.bodies[i].state.vc
#     storage.q[i][1] = mech.bodies[i].state.qc
#     storage.ω[i][1] = mech.bodies[i].state.ωc
# end
# CDV.visualize(mech,storage)

#LQR (works)
U0 = [SVector{m}(fill(0.0,m)) for k = 1:N]
xd, vd, qd, ωd, Fτd = state_parts(model, xf,U0[1])

Q = [diagm(ones(12))*1.0 for i=1:num_links]
R = [diagm(ones(1)) for i=1:num_links]
lqr = LQR(mech, getid.(mech.bodies), getid.(mech.eqconstraints), Q, R, 10, xd=xd, qd=qd)
# storage = simulate!(mech,10,lqr, record = true)
# visualize(mech,storage)
setStates!(mech,x0)
Δz = zeros(num_links*12)
qvm = QuatVecMap()
for (id,body) in pairs(mech.bodies)
    colx = (id-1)*12+1:(id-1)*12+3
    colv = (id-1)*12+4:(id-1)*12+6
    colq = (id-1)*12+7:(id-1)*12+9
    colω = (id-1)*12+10:(id-1)*12+12

    state = body.state
    Δz[colx] = state.xc-lqr.xd[id]
    Δz[colv] = state.vc-lqr.vd[id]
    Δz[colq] = Rotations.rotation_error(state.qc,lqr.qd[id],qvm)
    Δz[colω] = state.ωc-lqr.ωd[id]
end

uu = zeros(m)
for i=1:m
    tmp = lqr.Fτd[i] - lqr.K[1][i]*Δz
    println(lqr.K[1][i]*Δz)
    uu[i] = tmp[1]
end
uinf = SVector{m}(uu)

# # objective
Qf = Diagonal(@SVector fill(1250., n))
Q = Diagonal(@SVector fill(1e-4/dt, n))
R = Diagonal(@SVector fill(1e-3/dt, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=1e-4) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=1250.0)
obj = Objective(costfuns);

# problem
prob = Problem(model, obj, xf, tf, x0=x0);


# initial controls
U0 = [uinf for k = 1:N]
initial_controls!(prob, U0)
rollout!(prob);

# test jacobian
# D = TO.DynamicsExpansionMC(model)
# xd, vd, qd, ωd, Fτd = state_parts(model, x0,U0[1])
# bodyids = getid.(mech.bodies)
# eqcids = getid.(mech.eqconstraints)
# LA, LB, LC, LG = CC.linearsystem(mech, xd, vd, qd, ωd, Fτd, bodyids, eqcids)
# x02 = RD.discrete_dynamics(RD.Euler, model, x0, U0[1], 1.0, 0.05)
# λ = vcat(mech.eqconstraints[5].λsol[1],
#          mech.eqconstraints[6].λsol[1],
#          mech.eqconstraints[7].λsol[1],
#          mech.eqconstraints[8].λsol[1])
# x02_lin = LA*x0 + LB*U0[1] + LC * λ
# ILQR
opts = SolverOptions(verbose=7, static_bp=0, iterations=50, cost_tolerance=1e-5)
ilqr = Altro.iLQRSolver(prob, opts);
solve!(ilqr);

# just one step
# Altro.initialize!(ilqr)
# Z = ilqr.Z; Z̄ = ilqr.Z̄;
# n,m,N = size(ilqr)
# _J = TO.get_J(ilqr.obj)
# J_prev = sum(_J)
# grad_only = false
# J = Altro.step!(ilqr, J_prev, grad_only)


steps = Base.OneTo(length(states(ilqr.Z̄)))
storage = CD.Storage{Float64}(steps,length(mech.bodies))
for step = 1:length(states(ilqr.Z̄))
    
    setStates!(mech,states(ilqr.Z̄)[step])
    for i=1:model.nb
        storage.x[i][step] = mech.bodies[i].state.xc
        storage.v[i][step] = mech.bodies[i].state.vc
        storage.q[i][step] = mech.bodies[i].state.qc
        storage.ω[i][step] = mech.bodies[i].state.ωc
    end
end
CDV.visualize(mech,storage)