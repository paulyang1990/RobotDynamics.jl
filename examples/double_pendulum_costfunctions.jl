using RobotDynamics
using TrajectoryOptimization
using StaticArrays
using LinearAlgebra
using ForwardDiff
using Plots
using Altro

const TO = TrajectoryOptimization
const RD = RobotDynamics


"""
This is a lot like the QuadraticCostFunction, however we need
to first transform the MC state x into z by z = f(x), and then the cost is 
```math
\\frac{1}{2} z^T Q z + \\frac{1}{2} u^T R u + q^T z + r^T u + c
```
"""
abstract type MCCostFunction{n,m,T} <: TO.CostFunction end

# following functions are mostly the same as that in TO.CostFunctions.jl
# we just remove H and r here 
TO.is_diag(cost::MCCostFunction) = TO.is_blockdiag(cost) && cost.Q isa Diagonal && cost.R isa Diagonal
TO.is_blockdiag(::MCCostFunction) = true
TO.state_dim(::MCCostFunction{n}) where n = 12
TO.control_dim(::MCCostFunction{<:Any,m}) where m = m

function (::Type{MC})(Q::AbstractArray, R::AbstractArray;
    zref::AbstractVector=(@SVector zeros(eltype(Q), size(Q,1))),
    uref::AbstractVector=(@SVector zeros(eltype(R), size(R,1))),
    kwargs...) where MC <: MCCostFunction
    MC(Q, R, zref, uref; kwargs...)
end

function MCCostFunction(Q::AbstractArray, R::AbstractArray,
    zref::AbstractVector, uref::AbstractVector;
    kwargs...)
    if LinearAlgebra.isdiag(Q) && LinearAlgebra.isdiag(R) 
        MCDiagonalCost(diag(Q), diag(R), zref, uref)
    else
        MCQuadraticCost(Q, R, zref, uref; kwargs...)
    end
end

# stage and terminal cost are different from QuadraticCostFunction
function TO.stage_cost(cost::MCCostFunction, x::AbstractVector, u::AbstractVector)
    J = 0.5*(u-cost.uref)'cost.R*(u-cost.uref)  + TO.stage_cost(cost, x)

    return J
end
# stage and terminal cost are different from QuadraticCostFunction
function TO.stage_cost(cost::MCCostFunction, x::AbstractVector{T}) where T
    # first calculate tip position z from MC state
    z = @SVector [x[4]+0.5*cos(x[6]), x[5]+0.5*sin(x[6]),x[9],x[12]]
    J = 0.5*(z-cost.zref)'cost.Q*(z-cost.zref)
    return J
end
function TO.gradient!(E::TO.QuadraticCostFunction, cost::MCCostFunction, x)
    
    #E.q .= cost.Q*x .+ cost.q
    # this is different from that of QuadraticCostFunction
    z = @SVector [x[4]+0.5*cos(x[6]), x[5]+0.5*sin(x[6]),x[9],x[12]]
    Jx = @SMatrix [0 0 0 1 0 -0.5*sin(x[6]) 0 0 0 0 0 0; 0 0 0 0 1 0.5*cos(x[6]) 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 1 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 1]
    E.q .= Jx'cost.Q*z - Jx'cost.Q*cost.zref
    return false
end

function TO.gradient!(E::TO.QuadraticCostFunction, cost::MCCostFunction, x, u)
    TO.gradient!(E, cost, x)
    E.r .= cost.R*u - cost.R*cost.uref

    return false
end
# I know Q is 2D
function TO.hessian!(E::TO.QuadraticCostFunction, cost::MCCostFunction, x)
    Jx = @SMatrix [0 0 0 1 0 -0.5*sin(x[6]) 0 0 0 0 0 0; 0 0 0 0 1 0.5*cos(x[6]) 0 0 0 0 0 0; 0 0 0 0 0 0 0 0 1 0 0 0; 0 0 0 0 0 0 0 0 0 0 0 1]
    E.Q = Jx'*cost.Q*Jx;

    dgfHz = - (cos(x[6])*(cost.Q[3,1]*(x[9] - cost.zref[3]) + cost.Q[4,1]*(x[12] - cost.zref[4]) + cost.Q[1,1]*(x[4] - cost.zref[1] + cos(x[6])/2) + cost.Q[2,1]*(x[5] - cost.zref[2] + sin(x[6])/2)))/2 - (sin(x[6])*(cost.Q[3,2]*(x[9] - cost.zref[3]) + cost.Q[4,2]*(x[12] - cost.zref[4]) + cost.Q[2,1]*(x[4] - cost.zref[1] + cos(x[6])/2) + cost.Q[2,2]*(x[5] - cost.zref[2] + sin(x[6])/2)))/2
    E.Q[6,6] += dgfHz 
    return true
end

function TO.hessian!(E::TO.QuadraticCostFunction, cost::MCCostFunction, x, u)
    TO.hessian!(E, cost, x)
    if TO.is_diag(cost)
        for i = 1:length(u); E.R[i,i] = cost.R[i,i]; end
    else
        E.R .= cost.R
    end
    
    return true
end

function Base.copy(c::QC) where QC<:MCCostFunction
    QC(copy(c.Q), copy(c.R),  zref=copy(c.zref), uref=copy(c.uref),
        terminal=c.terminal, checks=false)
end

#######################################################
#              COST FUNCTION INTERFACE                #
#######################################################
struct MCDiagonalCost{n,m,T} <: MCCostFunction{n,m,T}
    Q::Diagonal{T,SVector{4,T}}
    R::Diagonal{T,SVector{m,T}}
    zref::MVector{4,T}
    uref::MVector{m,T}
    terminal::Bool
    function MCDiagonalCost(Qd::StaticVector{4}, Rd::StaticVector{m},
                          zref::StaticVector{4},  uref::StaticVector{m}; 
                          terminal::Bool=false, checks::Bool=true, kwargs...) where {m}
        T = promote_type(typeof(c), eltype(Qd), eltype(Rd), eltype(zref), eltype(uref))
        if checks
            if any(x->x<0, Qd)
                @warn "Q needs to be positive semi-definite."
            elseif any(x->x<=0, Rd) && !terminal
                @warn "R needs to be positive definite."
            end
        end
        n = 12
        new{n,m,T}(Diagonal(SVector(Qd)), Diagonal(SVector(Rd)), zref, uref, terminal)
    end
end

Base.copy(c::MCDiagonalCost) = 
    MCDiagonalCost(c.Q.diag, c.R.diag, copy(c.zref), copy(c.uref), 
        checks=false, terminal=c.terminal)

# convert to form for inner constructor
function MCDiagonalCost(Q::AbstractArray, R::AbstractArray,
    zref::AbstractVector, uref::AbstractVector; kwargs...)
    n = 12
    m = length(r)
    Qd = SVector{4}(diag(Q))
    Rd = SVector{m}(diag(R))
    MCDiagonalCost(Qd, Rd, convert(MVector{4},zref), convert(MVector{m},uref); kwargs...)
end
mutable struct MCQuadraticCost{n,m,T,TQ,TR} <: MCCostFunction{n,m,T}
    Q::TQ                     # Quadratic stage cost for states (4,4)
    R::TR                     # Quadratic stage cost for controls (m,m)
    zref::MVector{4,T}           # Linear term on states (n,)
    uref::MVector{m,T}           # Linear term on controls (m,)
    terminal::Bool
    function (::Type{QC})(Q::TQ, R::TR, 
            zref::Tq, uref::Tr; checks=true, terminal=false, kwargs...) where {TQ,TR,Tq,Tr,QC<:MCQuadraticCost}
        @assert size(Q,1) == length(zref)
        @assert size(R,1) == length(uref)
        if checks
            if !isposdef(Array(R)) && !terminal
                @warn "R is not positive definite"
            end
            if !TO.ispossemidef(Array(Q))
                @warn "Q is not positive semidefinite"
            end
        end
        n = 12
        m = size(R,1)
        T = promote_type(eltype(Q), eltype(R), eltype(zref), eltype(uref))
        new{n,m,T,TQ,TR}(Q,R,zref,uref,terminal)
    end
    function MCQuadraticCost{n,m,T,TQ,TR}(qcost::MCQuadraticCost) where {n,m,T,TQ,TR}
        new{n,m,T,TQ,TR}(qcost.Q, qcost.R, qcost.zref, qcost.uref, qcost.terminal)
    end
end
TO.state_dim(cost::MCQuadraticCost) = 12
TO.control_dim(cost::MCQuadraticCost) = length(cost.uref)
TO.is_blockdiag(cost::MCQuadraticCost) = true

function MCQuadraticCost{T}(n::Int,m::Int; terminal=false) where T
    Q = SizedMatrix{4,4}(Matrix(one(T)*I,4,4))
    R = SizedMatrix{m,m}(Matrix(one(T)*I,m,m))
    zref = SizedVector{4}(zeros(T,4))
    uref = SizedVector{m}(zeros(T,m))
    MCQuadraticCost(Q,R,zref,uref, checks=false, terminal=terminal)
end

MCQuadraticCost(cost::MCQuadraticCost) = cost

@inline function (::Type{<:MCQuadraticCost})(dcost::MCDiagonalCost)
    MCQuadraticCost(dcost,Q, dcost.R, zref=dcost.zref, uref=dcost.uref, terminal=dcost.terminal)
end

function Base.convert(::Type{<:MCQuadraticCost}, dcost::MCDiagonalCost)
    MCQuadraticCost(dcost.Q, dcost.R, zref=dcost.zref, uref=dcost.uref, terminal=dcost.terminal)
end
Base.promote_rule(::Type{<:MCCostFunction}, ::Type{<:MCCostFunction}) = MCQuadraticCost

function Base.promote_rule(::Type{<:MCQuadraticCost{n,m,T1,Q1,R1}},
    ::Type{<:MCQuadraticCost{n,m,T2,Q2,R2}}) where {n,m,T1,T2,Q1,Q2,R1,R2}
    T = promote_type(T1,T2)
    function diag_type(T1,T2,n)
        elT = promote_type(eltype(T1), eltype(T2))
        if T1 == T2
            return T1
        elseif (T1 <: Diagonal) && (T2 <: Diagonal)
            return Diagonal{elT,MVector{n,elT}}
        else
            return SizedMatrix{n,n,elT,2}
        end
    end
    Q = diag_type(Q1,Q2,4)
    R = diag_type(R1,R2,m)
    MCQuadraticCost{n,m,T, Q, R}
end

@inline Base.convert(::Type{QC}, cost::MCQuadraticCost) where QC <: MCQuadraticCost = QC(cost)


function MCObjective(Q::AbstractArray, R::AbstractArray, Qf::AbstractArray,
    zf::AbstractVector, N::Int; checks=true, uf=@SVector zeros(size(R,1)))

    n = 12
    m = size(R,1)

    ℓ = MCQuadraticCost(Q, R, zf, uf, checks=checks)
    ℓN = MCQuadraticCost(Qf, R, zf, uf, checks=false, terminal=true)

    TO.Objective(ℓ, ℓN, N)
end

# test 
# objective,tip pose
zf = @SVector [0,2,0,0]
uf = @SVector [0,0]
Q = zeros(4)
Q[1] = Q[2] = 1e-3
Q = Diagonal(SVector{4}(Q))
R = Diagonal(@SVector fill(1e-4, 2))
Qf = zeros(4)
Qf[1] = Qf[2] = 2500
Qf = Diagonal(SVector{4}(Qf))
N = 300
obj = MCObjective(Q,R,Qf,zf,N;uf = uf)

