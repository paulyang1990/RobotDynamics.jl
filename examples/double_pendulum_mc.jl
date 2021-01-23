using RobotDynamics
using TrajectoryOptimization
using StaticArrays
using LinearAlgebra
using ForwardDiff
using Plots
using Altro

const TO = TrajectoryOptimization
const RD = RobotDynamics

# Define the model struct with parameters
struct DoublePendulumMC{T} <: AbstractModelMC
  # the first link 
  m1::T 
  l1::T
    
  # the second link
  m2::T 
  l2::T
  
  g::T

  # constraint force dimension
  p::Int 

  function DoublePendulumMC{T}(m::T, l::T) where {T<:Real} 
    new(m,l,m,l,9.81, 4)
  end 
end
DoublePendulumMC() = DoublePendulumMC{Float64}(1.0, 1.0)

# define implicit dynamics propogation
function dynamics_con(model::DoublePendulumMC, x)
  l1 = model.l1
  l2 = model.l2
  d1 = .5*l1*[cos(x[3]);sin(x[3])]
  d2 = .5*l2*[cos(x[6]);sin(x[6])]
  return [x[1:2] - d1;
          (x[1:2]+d1) - (x[4:5]-d2)]
end

function Altro.is_converged(model::DoublePendulumMC, x)
  c = dynamics_con(model, x)
  return norm(c) < 1e-6
end

function angles_to_mc(model::DoublePendulumMC, th1, th2)
  d1 = .5*model.l1*[cos(th1);sin(th1)]
  d2 = .5*model.l2*[cos(th2);sin(th2)]
  return [d1; th1; 2*d1 + d2; th2; zeros(6)]
end

function discrete_dynamics_MC(::Type{Q}, 
  model::DoublePendulumMC, z::AbstractKnotPoint) where {Q<:RobotDynamics.Explicit}

  x = state(z) 
  u = control(z)
  t = z.t 
  dt = z.dt

  m1 = model.m1
  m2 = model.m2
  l1 = model.l1
  l2 = model.l2
  g = model.g
  M = Diagonal([m1,m1,1.0/12.0*m1*l1^2,m2,m2,1.0/12.0*m2*l2^2])

  q = x[1:6]
  v = x[7:12]
  λ = zeros(4)

  q⁺ = copy(q)
  v⁺ = copy(v)

  max_iters = 1000
  for i=1:max_iters      
      c = dynamics_con(model, q⁺)
      curry(xs) = dynamics_con(model, xs)
      J = ForwardDiff.jacobian(curry, q⁺)   # curry function 
      F = [0; -m1*g; u[1]-u[2];0; -m2*g; u[2]] * dt

      # Check break condition
      f = M*(v⁺-v) - J'*λ - F
      # println(norm([f;c]))
      if norm([f;c]) < 1e-12
          # println("breaking at iter: $i")
          break
      end
      i == max_iters && @warn "Max iters reached"

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
  # return [q⁺; v⁺] 
  
end

function Altro.discrete_jacobian_MC(::Type{Q}, model::DoublePendulumMC,
  z::AbstractKnotPoint{T,N,M′}) where {T,N,M′,Q<:RobotDynamics.Explicit}

  n, m = size(model)
  x = state(z) 
  u = control(z)
  t = z.t 
  dt = z.dt

  m1 = model.m1
  m2 = model.m2
  l1 = model.l1
  l2 = model.l2
  g = model.g
  M = Diagonal([m1,m1,1.0/12.0*m1*l1^2,m2,m2,1.0/12.0*m2*l2^2])

  # compute next state and lagrange multiplier
  x⁺, λ = discrete_dynamics_MC(Q, model, z)

  # constraint function
  dc(x) = dynamics_con(model, x)

  # implicit function
  function f_imp(z)
    # Unpack
    q⁺ = z[1:6]
    v⁺ = z[7:12]
    q = z[13:18]
    v = z[19:24]
    u = z[25:26]
    λ = z[27:end]
    
    J = ForwardDiff.jacobian(dc, q⁺)
    F = [0; -m1*g; u[1]-u[2];0; -m2*g; u[2]] * dt
    return [q⁺ - (q + v⁺*dt); M*(v⁺-v) - (J'*λ + F)]
  end

  # compute jacobians
  x = state(z) 
  u = control(z)
  all_partials = ForwardDiff.jacobian(f_imp, [x⁺;x;u;λ])
  ABC = -all_partials[:,1:12]\all_partials[:,13:end]
  A = ABC[:, 1:n]
  B = ABC[:, n .+ (1:m)]
  C = ABC[:, n+m+1:end]

  J = ForwardDiff.jacobian(dc, x⁺[1:6])
  G = [J zeros(4,6)]
  return A,B,C,G
end

function RD.discrete_dynamics(::Type{Q}, model::DoublePendulumMC, 
  z::AbstractKnotPoint) where {Q<:RobotDynamics.Explicit}
  x, λ = discrete_dynamics_MC(Q, model, z)
  return x
end

function RD.discrete_jacobian!(::Type{Q}, ∇f, model::DoublePendulumMC,
  z::AbstractKnotPoint{T,N,M}) where {T,N,M,Q<:RobotDynamics.Explicit}


end

# Specify the state and control dimensions
RD.state_dim(::DoublePendulumMC) = 12
RD.control_dim(::DoublePendulumMC) = 2

# Create the model
model = DoublePendulumMC()
n,m = size(model)
p = 4 # constraint force size

# Generate random state and control vector
x,u = rand(model)
dt = 0.01
z = KnotPoint(x,u,dt)

# # Evaluate the discrete dynamics and Jacobian
# x′ = discrete_dynamics(PassThrough, model, z)
# println(x)
# println(x′)

# A,B,C,G = discrete_jacobian_MC(PassThrough, model, z)
# println(A)

# # rollout dynamics
# th1 = -pi/4
# th2 = -pi/4
# d1 = .5*model.l1*[cos(th1);sin(th1)]
# d2 = .5*model.l2*[cos(th2);sin(th2)]
# x0 = [d1; th1; 2*d1 + d2; th2; zeros(6)]
# N = 200
# Z = Traj(n, m, dt, N)
# RD.rollout!(Euler, model, Z, x0)

# # plot
# using Plots
# X = states(Z)
# th1 = [X[t][3] for t = 1:N]
# th2 = [X[t][6] for t = 1:N]
# plot([th1 th2]*180/pi)

# @inline TO.DynamicsExpansionMC(model::DoublePendulumMC) = TO.DynamicsExpansionMC{Float64}(model)
# @inline function TO.DynamicsExpansionMC{T}(model::DoublePendulumMC) where T
# 	n,m = size(model)
# 	n̄ = state_diff_size(model)
# 	TO.DynamicsExpansionMC{T}(n,n̄,m,model.p)
# end
# function TO.dynamics_expansion!(Q, D::Vector{<:TO.DynamicsExpansionMC}, model::DoublePendulumMC,
#   Z::Traj)
#   for k in eachindex(D)
#     D[k].A,D[k].B,D[k].C,D[k].G = discrete_jacobian_MC(Q, model, Z[k])

#   end
# end
# @inline TO.error_expansion(D::TO.DynamicsExpansionMC, model::DoublePendulumMC) = D.A, D.B, D.C, D.G

# function TO.error_expansion!(D::Vector{<:TO.DynamicsExpansionMC}, model::AbstractModel, G)
# 	return
# end

# dynamic expansion data structure 
# D = [TO.DynamicsExpansionMC{Float64}(n,m,p) for k = 1:N]
# TO.dynamics_expansion!(Euler, D, model, Z)

# A,B,C,G = TO.error_expansion(D[10],model)

