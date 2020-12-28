using RobotDynamics
using StaticArrays
using LinearAlgebra
using ForwardDiff

# Define the model struct with parameters
struct DoublePendulumMC{T} <: AbstractModel
  # the first link 
  m1::T 
  l1::T
    
  # the second link
  m2::T 
  l2::T
  
  g::T

  function DoublePendulumMC{T}(m::T, l::T) where {T<:Real} 
    new(m,l,m,l,9.81)
  end 
end
DoublePendulumMC() = DoublePendulumMC{Float64}(1.0, 1.0)

# define implicit dynamics propogation
function dynamicsCon(model::DoublePendulumMC, x)
  l1 = model.l1
  l2 = model.l2
  d1 = .5*l1*[cos(x[3]);sin(x[3])]
  d2 = .5*l2*[cos(x[6]);sin(x[6])]
  return [x[1:2] - d1;
          (x[1:2]+d1) - (x[4:5]-d2)]
end

function RobotDynamics.integrate(::Type{Euler}, model::DoublePendulumMC, x::StaticVector, u::StaticVector, t, dt)

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
      c = dynamicsCon(model, q⁺)
      curry(xs) = dynamicsCon(model, xs)
      J = ForwardDiff.jacobian(curry, q⁺)   # curry function 
      F = [0; -m1*g; u[1]-u[2];0; -m2*g; u[2]] * dt

      # Check break condition
      f = M*(v⁺-v) - J'*λ - F
      # println(norm([f;c]))
      if norm([f;c]) < 1e-12
          # println("breaking at iter: $i")
          break
      end
      i == max_iters && throw("Max iters reached")

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



# Specify the state and control dimensions
RobotDynamics.state_dim(::DoublePendulumMC) = 12
RobotDynamics.control_dim(::DoublePendulumMC) = 2

# Create the model
model = DoublePendulumMC()
n,m = size(model)

# Generate random state and control vector
x,u = rand(model)
dt = 0.1
z = KnotPoint(x,u,dt)


# Evaluate the discrete dynamics and Jacobian
x′ = discrete_dynamics(Euler, model, z)
println(x)
println(x′)

