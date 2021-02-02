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
to first transform the RC state x into z by z = f(x), and then the cost is 
```math
\\frac{1}{2} z^T Q z + \\frac{1}{2} u^T R u + q^T z + r^T u + c
```
"""
abstract type RCCostFunction{n,m,T} <: TO.CostFunction end

# following functions are mostly the same as that in TO.CostFunctions.jl
# we just remove H and r here 
TO.is_diag(cost::RCCostFunction) = TO.is_blockdiag(cost) && cost.Q isa Diagonal && cost.R isa Diagonal
TO.is_blockdiag(::RCCostFunction) = true
TO.state_dim(::RCCostFunction{n}) where n = 4
TO.control_dim(::RCCostFunction{<:Any,m}) where m = m

function (::Type{RC})(Q::AbstractArray, R::AbstractArray;
    zref::AbstractVector=(@SVector zeros(eltype(Q), size(Q,1))),
    uref::AbstractVector=(@SVector zeros(eltype(R), size(R,1))),
    kwargs...) where RC <: RCCostFunction
    RC(Q, R, zref, uref; kwargs...)
end

function RCCostFunction(Q::AbstractArray, R::AbstractArray,
    zref::AbstractVector, uref::AbstractVector;
    kwargs...)
    if LinearAlgebra.isdiag(Q) && LinearAlgebra.isdiag(R) 
        RCDiagonalCost(diag(Q), diag(R), zref, uref)
    else
        RCQuadraticCost(Q, R, zref, uref; kwargs...)
    end
end

# stage and terminal cost are different from QuadraticCostFunction
function TO.stage_cost(cost::RCCostFunction, x::AbstractVector, u::AbstractVector)
    J = 0.5*(u-cost.uref)'cost.R*(u-cost.uref)  + TO.stage_cost(cost, x)

    return J
end
# stage and terminal cost are different from QuadraticCostFunction
function TO.stage_cost(cost::RCCostFunction, x::AbstractVector{T}) where T
    # first calculate tip position z from RC state
    z = @SVector [cos(x[1])+cos(x[1]+x[2]), sin(x[1])+sin(x[1]+x[2]),x[3],x[4]]
    J = 0.5*(z-cost.zref)'cost.Q*(z-cost.zref)
    return J
end
function TO.gradient!(E::TO.QuadraticCostFunction, cost::RCCostFunction, x)
    
    #E.q .= cost.Q*x .+ cost.q
    # this is different from that of QuadraticCostFunction
    z = @SVector [cos(x[1])+cos(x[1]+x[2]), sin(x[1])+sin(x[1]+x[2]),x[3],x[4]]
    Jx = @SMatrix [-sin(x[1]+x[2])-sin(x[1]) -sin(x[1]+x[2]) 0 0; cos(x[1]+x[2])+cos(x[1])  cos(x[1]+x[2])  0 0; 0 0 1 0; 0 0 0 1]
    E.q .= Jx'cost.Q*z - Jx'cost.Q*cost.zref
    return false
end

function TO.gradient!(E::TO.QuadraticCostFunction, cost::RCCostFunction, x, u)
    TO.gradient!(E, cost, x)
    E.r .= cost.R*u - cost.R*cost.uref

    return false
end
# I know Q is 2D
function TO.hessian!(E::TO.QuadraticCostFunction, cost::RCCostFunction, x)
    z = @SVector [cos(x[1])+cos(x[1]+x[2]), sin(x[1])+sin(x[1]+x[2]),x[3],x[4]]
    Jx = @SMatrix [-sin(x[1]+x[2])-sin(x[1]) -sin(x[1]+x[2]) 0 0; cos(x[1]+x[2])+cos(x[1])  cos(x[1]+x[2])  0 0; 0 0 1 0; 0 0 0 1]
    dgf = cost.Q*(z-cost.zref)
    E.Q = Jx'*cost.Q*Jx;

    # I got this from matlab 
    sparse_dJz1 = @SMatrix [-cos(x[1]+x[2])-cos(x[1]) -cos(x[1]+x[2]) 0 0; -cos(x[1]+x[2]) -cos(x[1]+x[2])  0  0; 0 0 0 0; 0 0 0 0]
    sparse_dJz2 = @SMatrix [-sin(x[1]+x[2])-sin(x[1]) -sin(x[1]+x[2]) 0 0; -sin(x[1]+x[2]) -sin(x[1]+x[2])  0  0; 0 0 0 0; 0 0 0 0]
    
    E.Q = E.Q + dgf[1]*sparse_dJz1
    E.Q = E.Q + dgf[2]*sparse_dJz2
    return true
end

function TO.hessian!(E::TO.QuadraticCostFunction, cost::RCCostFunction, x, u)
    TO.hessian!(E, cost, x)
    if TO.is_diag(cost)
        for i = 1:length(u); E.R[i,i] = cost.R[i,i]; end
    else
        E.R .= cost.R
    end
    
    return true
end

function Base.copy(c::QC) where QC<:RCCostFunction
    QC(copy(c.Q), copy(c.R),  zref=copy(c.zref), uref=copy(c.uref),
        terminal=c.terminal, checks=false)
end

#######################################################
#              COST FUNCTION INTERFACE                #
#######################################################
struct RCDiagonalCost{n,m,T} <: RCCostFunction{n,m,T}
    Q::Diagonal{T,SVector{4,T}}
    R::Diagonal{T,SVector{m,T}}
    zref::MVector{4,T}
    uref::MVector{m,T}
    terminal::Bool
    function RCDiagonalCost(Qd::StaticVector{4}, Rd::StaticVector{m},
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
        n = 4
        new{n,m,T}(Diagonal(SVector(Qd)), Diagonal(SVector(Rd)), zref, uref, terminal)
    end
end

Base.copy(c::RCDiagonalCost) = 
    RCDiagonalCost(c.Q.diag, c.R.diag, copy(c.zref), copy(c.uref), 
        checks=false, terminal=c.terminal)

# convert to form for inner constructor
function RCDiagonalCost(Q::AbstractArray, R::AbstractArray,
    zref::AbstractVector, uref::AbstractVector; kwargs...)
    n = 4
    m = length(r)
    Qd = SVector{4}(diag(Q))
    Rd = SVector{m}(diag(R))
    RCDiagonalCost(Qd, Rd, convert(MVector{4},zref), convert(MVector{m},uref); kwargs...)
end
mutable struct RCQuadraticCost{n,m,T,TQ,TR} <: RCCostFunction{n,m,T}
    Q::TQ                     # Quadratic stage cost for states (4,4)
    R::TR                     # Quadratic stage cost for controls (m,m)
    zref::MVector{4,T}           # Linear term on states (n,)
    uref::MVector{m,T}           # Linear term on controls (m,)
    terminal::Bool
    function (::Type{QC})(Q::TQ, R::TR, 
            zref::Tq, uref::Tr; checks=true, terminal=false, kwargs...) where {TQ,TR,Tq,Tr,QC<:RCQuadraticCost}
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
    function RCQuadraticCost{n,m,T,TQ,TR}(qcost::RCQuadraticCost) where {n,m,T,TQ,TR}
        new{n,m,T,TQ,TR}(qcost.Q, qcost.R, qcost.zref, qcost.uref, qcost.terminal)
    end
end
TO.state_dim(cost::RCQuadraticCost) = 4
TO.control_dim(cost::RCQuadraticCost) = length(cost.uref)
TO.is_blockdiag(cost::RCQuadraticCost) = true

function RCQuadraticCost{T}(n::Int,m::Int; terminal=false) where T
    Q = SizedMatrix{4,4}(Matrix(one(T)*I,4,4))
    R = SizedMatrix{m,m}(Matrix(one(T)*I,m,m))
    zref = SizedVector{4}(zeros(T,4))
    uref = SizedVector{m}(zeros(T,m))
    RCQuadraticCost(Q,R,zref,uref, checks=false, terminal=terminal)
end

RCQuadraticCost(cost::RCQuadraticCost) = cost

@inline function (::Type{<:RCQuadraticCost})(dcost::RCDiagonalCost)
    RCQuadraticCost(dcost,Q, dcost.R, zref=dcost.zref, uref=dcost.uref, terminal=dcost.terminal)
end

function Base.convert(::Type{<:RCQuadraticCost}, dcost::RCDiagonalCost)
    RCQuadraticCost(dcost.Q, dcost.R, zref=dcost.zref, uref=dcost.uref, terminal=dcost.terminal)
end
Base.promote_rule(::Type{<:RCCostFunction}, ::Type{<:RCCostFunction}) = RCQuadraticCost

function Base.promote_rule(::Type{<:RCQuadraticCost{n,m,T1,Q1,R1}},
    ::Type{<:RCQuadraticCost{n,m,T2,Q2,R2}}) where {n,m,T1,T2,Q1,Q2,R1,R2}
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
    RCQuadraticCost{n,m,T, Q, R}
end

@inline Base.convert(::Type{QC}, cost::RCQuadraticCost) where QC <: RCQuadraticCost = QC(cost)


function RCObjective(Q::AbstractArray, R::AbstractArray, Qf::AbstractArray,
    zf::AbstractVector, N::Int; checks=true, uf=@SVector zeros(size(R,1)))

    n = 4
    m = size(R,1)

    ℓ = RCQuadraticCost(Q, R, zf, uf, checks=checks)
    ℓN = RCQuadraticCost(Qf, R, zf, uf, checks=false, terminal=true)

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




