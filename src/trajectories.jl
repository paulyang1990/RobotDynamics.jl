abstract type AbstractTrajectory{n,m,T} <: AbstractVector{T} end

terminal_control(Z::AbstractTrajectory) = !RobotDynamics.is_terminal(Z[end])
traj_size(Z::AbstractTrajectory{n,m}) where {n,m} = n,m,length(Z)
num_vars(Z::AbstractTrajectory) = num_vars(traj_size(Z)..., terminal_control(Z))
eachcontrol(Z::AbstractTrajectory) = terminal_control(Z) ? Base.OneTo(length(Z)) : Base.OneTo(length(Z)-1)

function num_vars(n::Int, m::Int, N::Int, equal::Bool=false)
    Nu = equal ? N : N-1
    return N*n + Nu*m
end

function Base.copyto!(Z1::AbstractTrajectory, Z2::AbstractTrajectory)
	@assert traj_size(Z1) == traj_size(Z2)
	for k = 1:length(Z1)
		Z1[k] = Z2[k]
	end
	return Z1
end

@inline states(Z::AbstractTrajectory) = state.(Z)
function controls(Z::AbstractTrajectory)
	return [control(Z[k]) for k in eachcontrol(Z) ]
end

function Base.isapprox(Z1::AbstractTrajectory, Z2::AbstractTrajectory)
    all(zs->zs[1] ≈ zs[2], zip(Z1,Z2))
end

function set_dt!(Z::AbstractTrajectory, dt::Real)
    t = Z[1].t
    for z in Z
        z.dt = dt
        z.t = t
        t += dt
    end
    return t 
end

"""
    Traj{n,m,T,KP}

A vector of `AbstractKnotPoint`s of type `KP` with state dimension `n`,
control dimension `m`, and value type `T`

Supports iteration and indexing.

# Constructors
    Traj(n, m, dt, N, equal=false)
    Traj(x, u, dt, N, equal=false)
    Traj(X, U, dt, t)
    Traj(X, U, dt)
"""
struct Traj{n,m,T,KP} <: AbstractTrajectory{n,m,T}
	data::Vector{KP}
	function Traj(Z::Vector{<:AbstractKnotPoint{T,n,m}}) where {n,m,T}
		new{n,m,T,eltype(Z)}(Z)
	end
end

# AbstractArray interface
@inline Base.iterate(Z::Traj, k::Int) = iterate(Z.data, k)
Base.IteratorSize(Z::Traj) = Base.HasLength()
Base.IteratorEltype(Z::Traj) = Base.IteratorEltype(Z.data)
@inline Base.eltype(Z::Traj) = eltype(Z.data)
@inline Base.length(Z::Traj) = length(Z.data)
@inline Base.size(Z::Traj) = size(Z.data)
@inline Base.getindex(Z::Traj, i) = Z.data[i]
@inline Base.setindex!(Z::Traj, v, i) = Z.data[i] = v
@inline Base.firstindex(Z::Traj) = 1
@inline Base.lastindex(Z::Traj) = lastindex(Z.data)
Base.IndexStyle(::Traj) = IndexLinear()

Traj(Z::Traj) = Z

function Base.copy(Z::AbstractTrajectory) where {T,N,M}
    Traj([KnotPoint(copy(z.z), copy(z._x), copy(z._u), z.dt, z.t) for z in Z])
end

function Traj(n::Int, m::Int, dt::AbstractFloat, N::Int; equal=false)
    x = NaN*@SVector ones(n)
    u = @SVector zeros(m)
    Traj(x,u,dt,N; equal=equal)
end

function Traj(x::SVector, u::SVector, dt::AbstractFloat, N::Int; equal=false)
    equal ? uN = N : uN = N-1
    Z = [KnotPoint(x,u,dt,(k-1)*dt) for k = 1:uN]
    if !equal
        m = length(u)
        push!(Z, KnotPoint(x,m,(N-1)*dt))
    end
    return Traj(Z)
end

function Traj(X::Vector, U::Vector, dt::Vector, t=cumsum(dt) .- dt[1])
    Z = [KnotPoint(X[k], U[k], dt[k], t[k]) for k = 1:length(U)]
    if length(U) == length(X)-1
        push!(Z, KnotPoint(X[end],length(U[1]),t[end]))
    end
    return Traj(Z)
end


states(Z::Traj, i) = [state(z)[i] for z in Z]

function set_states!(Z::Traj, X)
    for k in eachindex(Z)
		RobotDynamics.set_state!(Z[k], X[k])
    end
end

function set_states!(Z::Traj, X::AbstractMatrix)
    for k in eachindex(Z)
		RobotDynamics.set_state!(Z[k], X[:,k])
    end
end

function set_controls!(Z::AbstractTrajectory, U)
    for k in 1:length(Z)-1
		RobotDynamics.set_control!(Z[k], U[k])
    end
end

function set_controls!(Z::AbstractTrajectory, U::AbstractMatrix)
    for k in 1:length(Z)-1
		RobotDynamics.set_control!(Z[k], U[:,k])
    end
end

function set_controls!(Z::AbstractTrajectory, u::SVector)
    for k in 1:length(Z)-1
		RobotDynamics.set_control!(Z[k], u)
    end
end

function set_times!(Z::AbstractTrajectory, ts)
    for k in eachindex(ts)
        Z[k].t = ts[k]
        k < length(ts) && (Z[k].dt = ts[k+1] - ts[k])
    end
end

function get_times(Z::Traj)
    [z.t for z in Z]
end

function shift_fill!(Z::Traj, n=1)
    N = length(Z)
    isterm = is_terminal(Z[end])
    for k = 1+n:N 
        Z[k-n] = copy(Z[k])
    end
    xf = state(Z[N-n]) 
    uf = control(Z[N-n])
    dt = Z[N-n-1].dt
    for k = N-n:N
        set_state!(Z[k], xf) 
        Z[k].t = Z[k-1].t + dt
        if k == N && is_terminal(Z[k])
            Z[k].dt = 0
        else
            set_control!(Z[k], uf) 
            Z[k].dt = dt
        end
    end
end

function Base.copyto!(Z::Traj, Z0::Traj)
	@assert length(Z) == length(Z0)
	for k in eachindex(Z)
		copyto!(Z[k].z, Z0[k].z)
	end
end

function Base.copyto!(Z::Union{Vector{<:KnotPoint},Traj{<:Any,<:Any,<:Any,<:KnotPoint}}, Z0::Traj)
	@assert length(Z) == length(Z0)
	for k in eachindex(Z)
		Z[k].z = Z0[k].z
	end
end

#~~~~~~~~~~~~~~~~~~~~~~~~~~ FUNCTIONS ON TRAJECTORIES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

"Evaluate the discrete dynamics for all knot points"
function discrete_dynamics!(::Type{Q}, f, model, Z::Traj) where Q
    for k in eachindex(Z)
        f[k] = RobotDynamics.discrete_dynamics(Q, model, Z[k])
    end
end


@inline state_diff_jacobian!(G, model::AbstractModel, Z::Traj) = nothing
function state_diff_jacobian!(G, model::LieGroupModel, Z::Traj)
	for k in eachindex(Z)
		G[k] .= 0
		state_diff_jacobian!(G[k], model, Z[k])
	end
end

function rollout!(::Type{Q}, model::AbstractModel, Z::AbstractTrajectory, x0=state(Z[1])) where Q <: QuadratureRule
    set_state!(Z[1], x0)
    for k = 2:length(Z)
        RobotDynamics.propagate_dynamics(Q, model, Z[k], Z[k-1])
    end
end
