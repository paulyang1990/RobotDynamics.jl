include("3DPendulum.jl")

using ForwardDiff
using LinearAlgebra
using Plots
using Rotations
using Random

# Build model
model = Pendulum3D()
n = 12 # number of states 
m = 1 # number of controls

#initial and goal conditions
R0 = UnitQuaternion(.9999,.0001,0, 0)
x0 = [R0*[0.; 0.; -.5]; RS.params(R0); zeros(6)]
xf = [0.; 0.;  .5; 0; 1; 0; 0; zeros(6)]

#costs
Q = .001*Diagonal([ones(3); zeros(4); ones(6)])
Qf = 100.0*Diagonal([ones(3); zeros(4); ones(6)])
R = 0.0001*Matrix(I,m,m)
cost_w = .1
cost_wf = 100.

#simulation
dt = 0.01
tf = 3.0

function state_error(x,x0)
    nq = 7
    nv = 6
    nc = 5

    err = zeros(2*nv)
    dx = x-x0
    for i=1:1
        err[6*(i-1) .+ (1:3)] = dx[7*(i-1) .+ (1:3)]
        dq = UnitQuaternion(x[7*(i-1) .+ (4:7)]...) ⊖ UnitQuaternion(x0[7*(i-1) .+ (4:7)]...)
        err[6*(i-1) .+ (4:6)] = dq[:]
    end
    err[nv .+ (1:nv)] = dx[nq .+ (1:nv)]
    return err
end

#iLQR
function rollout(x0,U,f,dt,tf)
    nc = 5
    N = convert(Int64,floor(tf/dt))
    X = zeros(size(x0,1),N)
    Lam = zeros(nc,N-1)
    X[:,1] = x0
    for k = 1:N-1
        # print("k = $k ")
        X[:,k+1], Lam[:,k] = f(X[:,k],U[:,k],dt)
    end
    return X, Lam
end

function cost(X,U,Q,R,Qf,xf)
    N = size(X,2)
    J = 0.0
    for k = 1:N-1
        # dx = state_error(X[:,k], xf)
        dx = X[:,k] - xf
        J += 0.5*dx'*Q*dx + 0.5*U[:,k]'*R*U[:,k]
        q = X[4:7,k]
        dq = xf[4:7]'q
        J += cost_w*min(1+dq, 1-dq)
    end
    # dx = state_error(X[:,N], xf)
    dx = X[:,N] - xf
    J += 0.5*dx'*Qf*dx
    q = X[4:7,N]
    dq = xf[4:7]'q
    J += cost_wf*min(1+dq, 1-dq)
    return J
end

function compute_Qq(Q, w, x, xf)
    n = 12
    Q_ = Q[1,1]*Matrix(I,n,n)
    Q_[4:6,4:6] = abs(xf[4:7]'x[4:7])*Matrix(I,3,3)
    
    q_ = Q*(x - xf)
    deleteat!(q_,4)
    att_jac = RS.∇differential(UnitQuaternion(x[4:7]))
    q_[4:6] = w*att_jac'*xf[4:7]
    return Q_, q_
end

function backwardpass(X,Lam,U,F,Q,R,Qf,xf)
    nq = 7
    nv = 6
    nc = 5

    Q_og = Q
    _, N = size(X)
    n = 12
    m = size(U,1)

    S = zeros(n,n,N)
    s = zeros(n,N)    
    K = zeros(m,n,N-1)
    l = zeros(m,N-1)
    
    S[:,:,N], s[:,N] = compute_Qq(Qf, cost_wf, X[:,N], xf)
    
    mu = 0.0
    k = N-1
    
    while k >= 1
        Q, q = compute_Qq(Q_og, cost_w, X[:,k], xf)
        r = R*U[:,k]
        S⁺ = S[:,:,k+1]
        s⁺ = s[:,k+1]
        
        A,B,C,G = F(X[:,k+1],X[:,k],U[:,k],Lam[:,k],dt)
        
        D = B - C/(G*C)*G*B
        M11 = R + D'*S⁺*B
        M12 = D'*S⁺*C
        M21 = G*B
        M22 = G*C

        M = [M11 M12;M21 M22]
        b = [D'*S⁺;G]*A

        K_all = M\b
        Ku = K_all[1:m,:]
        Kλ = K_all[m+1:m+nc,:]
        K[:,:,k] = Ku

        l_all = M\[r + D'*s⁺; zeros(nc)]
        lu = l_all[1:m,:]
        lλ = l_all[m+1:m+nc,:]
        l[:,k] = lu

        Abar = A-B*Ku-C*Kλ
        bbar = -B*lu - C*lλ
        S[:,:,k] = Q + Ku'*R*Ku + Abar'*S⁺*Abar
        s[:,k] = q - Ku'*r + Ku'*R*lu + Abar'*S⁺*bbar + Abar'*s⁺

        k = k - 1;
    end
    return K, l
end

function forwardpass(X,U,f,J,K,l)
    nq = 7
    nv = 6
    nc = 5

    N = size(X,2)
    m = size(U,1)
    Lam = zeros(nc,N-1)
    X_prev = copy(X)
    J_prev = copy(J)
    U_ = zeros(m,N-1)
    J = Inf
    dJ = 0.0
    
    alpha = 1.0
    while J > J_prev
        for k = 1:N-1
            dx = state_error(X[:,k], X_prev[:,k])
            U_[:,k] = U[:,k] - K[:,:,k]*dx - alpha*l[:,k]
            try
                X[:,k+1], Lam[:,k] = f(X[:,k],U_[:,k],dt);
            catch e
                println(e)
                k = 1
                alpha /= 2.
            end            
        end

        J = cost(X,U_,Q,R,Qf,xf)
        dJ = J_prev - J
        alpha = alpha/2.0;
    end

    println("New cost: $J")
    println("- Line search iters: ", abs(log(.5,alpha)))
    println("- Actual improvement: $(dJ)")
    return X, U_, J, Lam
end


function solve(x0,m,f,F,Q,R,Qf,xf,dt,tf,iterations=100,eps=1e-5;control_init="random")
    N = convert(Int64,floor(tf/dt))
    X = zeros(size(x0,1),N)
    
    if control_init == "random"
        Random.seed!(0)
        U = 5.0*rand(m,N-1)
    else
        U = zeros(m,N-1)
    end
    U0 = copy(U)
        
    X, Lam = rollout(x0,U,f,dt,tf)
    X0 = copy(X)
    Lam0 = copy(Lam)
    J_prev = cost(X,U,Q,R,Qf,xf)
    println("Initial Cost: $J_prev\n")
    
    K = zeros(2,2,2)
    l = zeros(2,2)
    for i = 1:iterations
        println("*** Iteration: $i ***")
        K, l = backwardpass(X,Lam,U,F,Q,R,Qf,xf)
        X, U, J, Lam = forwardpass(X,U,f,J_prev,K,l)

        if abs(J-J_prev) < eps
          println("-----SOLVED-----")
          println("eps criteria met at iteration: $i")
          break
        end
        J_prev = copy(J)
    end
    
    return X, U, K, l, X0, U0, Lam0
end

function stable_rollout(Ku,x0,u0,f,dt,tf)
    N = convert(Int64,floor(tf/dt))
    X = zeros(size(x0,1),N)
    U = zeros(m,N-1)
    Lam = zeros(nc,N-1)
    X[:,1] = x0
    for k = 1:N-1
        dx = state_error(X[:,k], xf)
        U[:,k] = u0-Ku*dx
        X[:,k+1], Lam[:,k] = f(X[:,k],U[:,k],dt)
    end
    return X, Lam, U
end

function f(x,u,dt)
    z = KnotPoint(x,u,dt)
    discrete_dynamics_MC(PassThrough,model,z)
end

function getABCG(x⁺,x,u,λ,dt)
    z = KnotPoint(x,u,dt)
    discrete_jacobian_MC(PassThrough, model, z)
end

# ROLLOUT
# X, Lam = rollout(x0,rand(1,floor(Int, tf/dt)),f,dt,tf)
# X, Lam = rollout(x0,.1*ones(1,floor(Int, tf/dt)),f,dt,tf)

# SWINGUP
X, U, K, l, X0, U0, Lam0 = solve(x0,m,f,getABCG,Q,R,Qf,xf,dt,tf,20,control_init="random");

# PLOTS
_,N = size(X)

# Kth = [K[1,4,i] for i=1:N-1]
# Kthd = [K[1,10,i] for i=1:N-1]
# plot([Kth Kthd])
# plot(Kthd)

quats = [UnitQuaternion(X[4:7,i]) for i=1:N]
angles = [rotation_angle(quats[i])*rotation_axis(quats[i])[1] for i=1:N]
plot(-angles[1:end-10])
plot!(X[10,:])
# plot(U[:])
