# NOTE: run double_pendulum_mc.jl first

#goal 
th1 = -pi/4
th2 = -pi/4
d1 = .5*model.l1*[cos(th1);sin(th1)]
d2 = .5*model.l2*[cos(th2);sin(th2)]
xf = [d1; th1; 2*d1 + d2; th2; zeros(6)]

#costs
Q = zeros(n,n)
Q[3,3] = 1e-3
Q[6,6] = 1e-3
Q[9,9] = 1e-3
Q[12,12] = 1e-3
Qf = zeros(n,n)
Qf[3,3] = 250
Qf[6,6] = 250
Qf[9,9] = 250
Qf[12,12] = 250
R = 1e-4*Matrix(I,m,m)

function f(x,u,dt)
    z = KnotPoint(x,u,dt)
    return discrete_dynamics_MC(PassThrough, model, z)
end

function getABCG(x⁺,x,u,λ,dt)
    z = KnotPoint(x,u,dt)
    return discrete_jacobian_MC(PassThrough, model, z)
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
    return K, l, v1, v2
end

function stable_rollout(Ku,x0,u0,f,dt,tf)
    N = convert(Int64,floor(tf/dt))
    X = zeros(size(x0,1),N)
    U = zeros(m,N-1)
    Lam = zeros(4,N-1)
    X[:,1] = x0
    for k = 1:N-1
        U[:,k] = u0-Ku*(X[:,k]-xf)
        X[:,k+1], Lam[:,k] = f(X[:,k],U[:,k],dt)
    end
    return X, Lam, U
end

#simulation
dt = 0.01
tf = 6.0

# compute and verify nominal torques
m1, m2, g = model.m1, model.m2, model.g
uf = [(m1*xf[1] + m2*xf[4])*g; (xf[4] - 2*xf[1])*m2*g]
xf′, λf = f(xf,uf,dt) # check xf′ = xf

# compute stabilizing gains
timesteps = 599
X = repeat(xf,outer=(1,timesteps+1))
Lam = repeat(λf,outer=(1,timesteps))
U = repeat(uf,outer=(1,timesteps))
K, l, v1, v2 = backwardpass(X,Lam,U,getABCG,Q,R,Q,xf)
display(K[:,:,end])
display(K[:,:,1])
display(l[:,end])
display(l[:,1])

# K[:,:,1] = 
# 2×12 Array{Float64,2}:
#   24.4454   24.4454  -7.52077  -6.90328  -6.90328    4.52472   1.30973   1.30973   0.308706  1.30109  1.30109  -0.310742
#  -79.4631  -79.4631  11.3462   32.3121   32.3121   -21.9933   -2.62809  -2.62809  -0.619448  1.51489  1.51489   1.59596

K6 = [K[1,6,i] for i=1:timesteps]
K3 = [K[1,3,i] for i=1:timesteps]
plot([K3 K6])

# run stablizing controller
Ku = K[:,:,1]
x1, _ = f(xf,[1., 0],dt) # perturbance
X, Lam, U=stable_rollout(Ku,x1,U[:,1],f,dt,tf)
plot(X[3,:])
plot!(X[6,:])
println(norm(X[:,end]-xf))