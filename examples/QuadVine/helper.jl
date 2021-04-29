function ∇rotate(q::AbstractVector, r::AbstractVector)
    rhat = SA[0, r[1], r[2], r[3]]
    2RS.vmat()*RS.rmult(SVector{4}(q))'RS.rmult(rhat)
end

function ∇differential(q::AbstractVector)
    SA[
        -q[2] -q[3] -q[4];
         q[1] -q[4]  q[3];
         q[4]  q[1] -q[2];
        -q[3]  q[2]  q[1];
    ]
end

function dq⁺dw⁺(ω⁺, rot, d, dt) 
    sq⁺ = sqrt(4/dt^2 - ω⁺'ω⁺)
    rot⁺ = dt/2 * RS.lmult(rot) * SA[sq⁺, ω⁺[1], ω⁺[2], ω⁺[3]]
    return rot⁺, ∇rotate(rot⁺, d), dt/2*RS.lmult(rot)*[-ω⁺'/sq⁺;I]
end

function ∂Lᵀ∂q() 
    return SA{Float64}[
        1 0 0 0
        0 -1 0 0
        0 0 -1 0
        0 0 0 -1
        
        0 1 0 0
        1 0 0 0
        0 0 0 -1
        0 0 1 0
        
        0 0 1 0
        0 0 0 1
        1 0 0 0
        0 -1 0 0
        
        0 0 0 1
        0 0 -1 0
        0 1 0 0
        1 0 0 0
    ]
end

function ∂R∂q() 
    return SA{Float64}[
        1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 1
        
        0 -1 0 0
        1 0 0 0
        0 0 0 -1
        0 0 1 0
        
        0 0 -1 0
        0 0 0 1
        1 0 0 0
        0 -1 0 0
        
        0 0 0 -1
        0 0 -1 0
        0 1 0 0
        1 0 0 0
    ]
end

function ∂Gqbᵀλ∂qb(q_b, λt, d)
    vertex2 = SA[0.0,d[1],d[2],d[3]]                      
    a = 2*RS.vmat()*kron((RS.rmult(vertex2)'*RS.rmult(q_b)*RS.hmat()*λt[1:3])',I(4))*∂Lᵀ∂q()
    b = 2*RS.vmat()*RS.lmult(q_b)'*RS.rmult(vertex2)'*kron((RS.hmat()*λt[1:3])',I(4))*∂R∂q()
    return a+b
end