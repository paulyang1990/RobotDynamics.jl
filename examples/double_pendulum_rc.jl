using RobotDynamics
using StaticArrays

# Define the model struct with parameters
struct DoublePendulumRC{T} <: AbstractModel
  # the first link 
  m1::T 
  l1::T
    
  # the second link
  m2::T 
  l2::T
  
  g::T
  function DoublePendulumRC{T}(m::T, l::T) where {T<:Real} 
    new(m,l,m,l,9.81)
  end   
end

DoublePendulumRC() = DoublePendulumRC{Float64}(1.0, 1.0)

# Define the continuous dynamics
function RobotDynamics.dynamics(model::DoublePendulumRC, x, u)
    m1 = model.m1
    m2 = model.m2
    l1 = model.l1
    r1 = l1/2
    l2 = model.l2
    r2 = l2/2
    g = model.g 
    I1 = 1.0/12.0*m1*l1^2
    I2 = 1.0/12.0*m2*l2^2

    θ1  = x[1]
    θ2  = x[2]
    dθ1 = x[3]
    dθ2 = x[4]
    qd = x[ @SVector [3,4] ]
    
    α = I1 + I2 + m1*r1^2 + m2*(l1^2+r2^2)
    β = m2*l1*r2
    δ = I2 + m2*r2^2

    s2 = sin(θ2)
    c2 = cos(θ2)

    H = @SMatrix [α+2*β*c2 δ+β*c2; β*c2 δ]
    C = @SMatrix [-β*s2*dθ2 -β*s2*(dθ1+dθ2); β*s2*dθ1 0]
    # H = @SMatrix [mc+mp mp*l*c; mp*l*c mp*l^2]
    # C = @SMatrix [0 -mp*qd[2]*l*s; 0 0]
    # G = @SVector [0, mp*g*l*s]
    # B = @SVector [1, 0]

    qdd = H\(-C*qd + u)
    return [qd; qdd]
end

# Specify the state and control dimensions
RobotDynamics.state_dim(::DoublePendulumRC) = 4
RobotDynamics.control_dim(::DoublePendulumRC) = 2

# Create the model
model = DoublePendulumRC()
n,m = size(model)

# Generate random state and control vector
x,u = rand(model)
dt = 0.1
z = KnotPoint(x,u,dt)

# Evaluate the continuous dynamics and Jacobian
ẋ = dynamics(model, x, u)
∇f = RobotDynamics.DynamicsJacobian(model)   # only allocate memory
jacobian!(∇f, model, z)   # calls jacobian in integration.jl

# Evaluate the discrete dynamics and Jacobian
x′ = discrete_dynamics(RK3, model, z)
discrete_jacobian!(RK3, ∇f, model, z)
println(x′)