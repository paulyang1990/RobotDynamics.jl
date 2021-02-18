"""
============================ 3D ACROBOT =====================================================
"""

include("acrobot3D.jl")

# model and timing
model = Acrobot3D()
n, m = size(model)
dt = 0.005
N = 1000
tf = (N-1)*dt     

# initial and final conditions
x0 = rc_to_mc(model, [.01, 0])
xf = rc_to_mc(model, [pi, 0])

# objective
Qf = Diagonal(@SVector fill(250., n))
Q = Diagonal(@SVector fill(1e-4/dt, n))
R = Diagonal(@SVector fill(1e-4/dt, m))
costfuns = [TO.LieLQRCost(RD.LieState(model), Q, R, SVector{n}(xf); w=0.) for i=1:N]
costfuns[end] = TO.LieLQRCost(RD.LieState(model), Qf, R, SVector{n}(xf); w=0.0)
obj = Objective(costfuns);

# problem
prob = Problem(model, obj, xf, tf, x0=x0);
U0 = [SVector{m}(5*rand(m)) for k = 1:N-1]
initial_controls!(prob, U0)
rollout!(prob);

# ILQR
opts = SolverOptions(verbose=7, static_bp=0, iterations=50, cost_tolerance=1e-4)
ilqr = Altro.iLQRSolver(prob, opts);
solve!(ilqr);
plot_traj(states(ilqr), controls(ilqr))

"""
============================ 2D ACROBOT =====================================================
"""

using ForwardDiff
using LinearAlgebra
using Plots
using Random

## Simple Pendulum
n = 12 # number of states 
m = 1 # number of controls

#initial and goal conditions
θ0 = .01
d = [.5sin(θ0); -.5cos(θ0)]
x0 = [d; θ0; 3d; θ0; zeros(6)]
xf = [0.; .5; pi; 0.; 1.5; pi; zeros(6)]

# costs
Q = 1e-4*Matrix(I,n,n)
Q[3,3]=Q[6,6]=0
Qf = 250.0*Matrix(I,n,n)
Qf[3,3]=Qf[6,6]=0
R = 1e-4*Matrix(I,m,m)

#simulation
dt = 0.005
tf = 5.0

# Maximal dynamics
function c!(x) 
    d1 = .5*[cos(x[3]-pi/2);sin(x[3]-pi/2)]
    d2 = .5*[cos(x[6]-pi/2);sin(x[6]-pi/2)]
    return [x[1:2] - d1;
            (x[1:2]+d1) - (x[4:5]-d2)]
end

function f(x,u,dt)
    m = 1.
    g = 9.81
    M = 1.0*Matrix(I,6,6)

    q = x[1:6]
    v = x[7:12]
    λ = zeros(4)

    q⁺ = copy(q)
    v⁺ = copy(v)

    max_iters = 1000
    for i=1:max_iters      
        c = c!(q⁺)
        J = ForwardDiff.jacobian(c!, q⁺)
        F = [0; -m*g; -u[1];0; -m*g; u[1]] * dt

        # Check break condition
        f = M*(v⁺-v) - J'*λ - F
        # println(norm([f;c]))
        if norm([f;c]) < 1e-9
            # println("breaking at iter: $i")
            break
        end
        i == max_iters && print("!")#throw("Max iters reached")

        # Newton solve
        A = [M -J';
            -J zeros(4,4)]
        d = [M*v + F; 
            (c + J*(q-q⁺))/dt]
        sol = A\d
        
        # Update        
        v⁺ = sol[1:6]
        λ = sol[7:end]
        q⁺ = q + v⁺*dt
    end
    return [q⁺; v⁺], λ 
end

function getABCG(x⁺,x,u,λ,dt)
    function f_imp(z)
        # Unpack
        q⁺ = z[1:6]
        v⁺ = z[7:12]
        q = z[13:18]
        v = z[19:24]
        u = z[25]
        λ = z[26:end]
    
        M = 1.0*Matrix(I,6,6)
        J = ForwardDiff.jacobian(c!, q⁺)
        F = [0; -9.81; -u;0; -9.81; u] * dt
        return [M*(v⁺-v) - (J'*λ + F); q⁺ - (q + v⁺*dt)]
    end

    n = length(x)
    m = length(u)
    
    all_partials = ForwardDiff.jacobian(f_imp, [x⁺;x;u;λ])
    ABC = -all_partials[:,1:12]\all_partials[:,13:end]
    A = ABC[:, 1:n]
    B = ABC[:, n .+ (1:m)]
    C = ABC[:, n+m+1:end]

    J = ForwardDiff.jacobian(c!, x⁺[1:6])
    G = [J zeros(4,6)]
    return A,B,C,G
end

#iLQR
function rollout(x0,U,f,dt,tf)
    N = convert(Int64,floor(tf/dt))
    X = zeros(size(x0,1),N)
    Lam = zeros(4,N-1)
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
      J += 0.5*(X[:,k]' - xf')*Q*(X[:,k] - xf) + 0.5*U[:,k]'*R*U[:,k]
    end
    J += 0.5*(X[:,N]' - xf')*Qf*(X[:,N] - xf)
    return J
end

function backwardpass(X,Lam,U,F,Q,R,Qf,xf)
    n, N = size(X)
    m = size(U,1)
    nc = 4
    
    S = zeros(n,n,N)
    s = zeros(n,N)    
    K = zeros(m,n,N-1)
    l = zeros(m,N-1)
    
    S[:,:,N] = Qf
    s[:,N] = Qf*(X[:,N] - xf)
    v1 = 0.0
    v2 = 0.0

    mu = 0.0
    k = N-1
    
    while k >= 1
        q = Q*(X[:,k] - xf)
        r = R*(U[:,k])
        S⁺ = S[:,:,k+1]
        s⁺ = s[:,k+1]
        
        A,B,C,G = F(X[:,k+1],X[:,k],U[:,k],Lam[:,k],dt)
        
        D = B - C/(G*C)*G*B
        M11 = R + D'*S⁺*B
        M12 = D'*S⁺*C
        M21 = G*B
        M22 = G*C

        M = [M11 M12;M21 M22] + mu*I
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
    return K, l, v1, v2
end

function forwardpass(X,U,f,J,K,l,v1,v2,c1=0.0,c2=1.0)
    N = size(X,2)
    m = size(U,1)
    Lam = zeros(4,N-1)
    X_prev = copy(X)
    J_prev = copy(J)
    U_ = zeros(m,N-1)
    J = Inf
    dV = 0.0
    dJ = 0.0
    z = 0.0
    
    alpha = 1.0
    count = 0
    while J > J_prev# && count < 1#|| z < c1 || z > c2 
        for k = 1:N-1
          U_[:,k] = U[:,k] - K[:,:,k]*(X[:,k] - X_prev[:,k]) - alpha*l[:,k]
          X[:,k+1], Lam[:,k] = f(X[:,k],U_[:,k],dt);
        end

        J = cost(X,U_,Q,R,Qf,xf)
        
        # dV = alpha*v1 + (alpha^2)*v2/2.0
        dJ = J_prev - J
        # z = dJ/dV[1]

        alpha = alpha/2.0;
        count += 1
    end

    println("New cost: $J")
    println("- Line search iters: ", abs(log(.5,alpha)))
    # println("- Expected improvement: $(dV[1])")
    println("- Actual improvement: $(dJ)")
    # println("- (z = $z)\n")
    return X, U_, J, Lam
end

function solve(x0,m,f,F,Q,R,Qf,xf,dt,tf,iterations=100,eps=1e-4;control_init="random")
    N = convert(Int64,floor(tf/dt))
    X = zeros(size(x0,1),N)
    
    if control_init == "random"
        Random.seed!(0)
        U = 5.0*rand(m,N-1)        
    else
        U = control_init
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
        K, l, v1, v2 = backwardpass(X,Lam,U,F,Q,R,Qf,xf)
        X, U, J, Lam = forwardpass(X,U,f,J_prev,K,l,v1,v2)

        if abs(J-J_prev) < eps
          println("-----SOLVED-----")
          println("eps criteria met at iteration: $i")
          break
        end
        J_prev = copy(J)
    end
    
    return X, U, K, l, X0, U0, Lam0
end

X, U, K, l, X0, U0, Lam0 = solve(x0,m,f,getABCG,Q,R,Qf,xf,dt,tf,101,1e-4,control_init="random");

display(plot(range(0,stop=tf,length=size(X,2)),X[3:3:end,:]'))
plot(U')

"""
============================ SAVE RESULTS =====================================================
"""

X3D = states(ilqr)
U3D = controls(ilqr)
X2D = X
U2D = U
using JLD2
@save joinpath(@__DIR__, "acrobot_mc_comparison.jld2") X3D U3D X2D U2D