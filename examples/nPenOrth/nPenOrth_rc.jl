using RobotDynamics
using StaticArrays
using ModernRoboticsBook
using Rotations
using StaticArrays, LinearAlgebra

# Define the model struct with parameters
struct nPenOrthRC{T} <: AbstractModel
    # link
    m::T 
    l::T
      
    g::T

    # number of links
    nb::Integer

    function nPenOrthRC{T}(m::T, l::T, n::Integer) where {T<:Real} 
      new(m,l,9.81,n)
    end   
end
nPenOrthRC() = nPenOrthRC{Float64}(1.0, 1.0,2)
nPenOrthRC(nb::Integer) = nPenOrthRC{Float64}(1.0, 1.0,nb)

# following functions follow notations in ModernRobotics
function getTwistList(model::nPenOrthRC)
    Slist = zeros(6, model.nb)
    w = [1.0;0.0;0.0]
    p = [0.0;0.0;0.0]
    R = UnitQuaternion(RotZ(pi/2))
    Slist[:,1] = [w;-cross(w,p)]
    for i=1:model.nb-1
        w = normalize(R^i*[1.0;0.0;0.0])
        p = [0.0;0.0;model.l*i]
        Slist[:,i+1] = [w;-cross(w,p)]
    end
    return Slist
end

function getMList(model::nPenOrthRC)
    M_list = Array{Float64,2}[]
    M1 = [I [0.0,0.0,model.l/2]; [0 0 0 1]];
    push!(M_list, M1)
    for i=1:model.nb-1
        Mi = [I [0.0,0.0,model.l/2+model.l*i]; [0 0 0 1]];
        push!(M_list, Mi)
    end

    Mend = [I [0.0,0.0,model.l*model.nb]; [0 0 0 1]];   
    push!(M_list, Mend) 
    
    M2_list=Array{Float64,2}[]
    push!(M2_list, M1)
    for i=2:model.nb+1
        push!(M2_list, inv(M_list[i-1])*M_list[i])
    end
    
    return M2_list
end

function getGList(model::nPenOrthRC)
    x = 0.1
    y = 0.1
    z = model.l
    Inertia = 1 / 12 * model.m * diagm([y^2 + z^2;x^2 + z^2;x^2 + y^2]);
    G = [Inertia zeros(3,3);
        zeros(3,3) diagm([model.m, model.m, model.m])];
    return [G for i=1:model.nb]
end
# Define the continuous dynamics
function RobotDynamics.dynamics(model::nPenOrthRC, x, u)
    θ = x[1:model.nb]
    dθ = x[model.nb+1:model.nb*2]
    Slist = getTwistList(model)
    Mlist = getMList(model)
    Glist = getGList(model)
    ddθ = ForwardDynamics(θ, dθ, Array(u), [0.0;0.0;-model.g], zeros(6), Mlist, Glist, Slist)

    return [dθ; ddθ]
end

# Specify the state and control dimensions
RobotDynamics.state_dim(model::nPenOrthRC) = model.nb*2
RobotDynamics.control_dim(model::nPenOrthRC) = model.nb

# state is the same as Jan state
function rc_to_mc(model::nPenOrthRC, rc_x::AbstractArray{T}) where T
    out_x = [zeros(0) for i=1:length(rc_x)]

    for i= 1:length(rc_x)
        rotations = []
        # joints are arranged orthogonally
        for j=1:model.nb
            if mod(j,2) == 0
                push!(rotations, UnitQuaternion(RotY(rc_x[i][j])))
            else
                push!(rotations, UnitQuaternion(RotX(rc_x[i][j])))
            end
        end
        pin = zeros(3)
        prev_q = UnitQuaternion(1.,0.0,0.0,0.0)
        for j = 1:model.nb
            r = UnitQuaternion(rotations[j])
            link_q = prev_q * r
            delta = link_q * [0,0,model.l/2]
            link_x = pin+delta
            out_x[i] = [out_x[i]; link_x;zeros(3);Rotations.params(link_q);zeros(3)]
            prev_q = link_q
            pin += 2*delta
        end
    end
    return out_x
end


# test
model = nPenOrthRC(4)
n,m = size(model)

# Generate random state and control vector
x,u = rand(model)
dt = 0.01
z = KnotPoint(x,u,dt)

# Evaluate the continuous dynamics and Jacobian
ẋ = dynamics(model, x, u)
∇f = RobotDynamics.DynamicsJacobian(model)   # only allocate memory
jacobian!(∇f, model, z)   # calls jacobian in integration.jl

# Evaluate the discrete dynamics and Jacobian
x′ = discrete_dynamics(RK3, model, z)
discrete_jacobian!(RK3, ∇f, model, z)
println(x′)