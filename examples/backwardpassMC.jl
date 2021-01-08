
"""
Calculates the optimal feedback gains K,d as well as the 2nd Order approximation of the
Cost-to-Go, using a backward Riccati-style recursion. (non-allocating)
"""
function backwardpassMC!(solver::iLQRSolver{T,QUAD,L,O,n,n̄,m}) where {T,QUAD<:QuadratureRule,L,O,n,n̄,m}
	N = solver.N

    # Objective
    obj = solver.obj
    model = solver.model

    # Extract variables
    Z = solver.Z; K = solver.K; d = solver.d;
    G = solver.G
    S = solver.S
	Quu_reg = solver.Quu_reg
	Qux_reg = solver.Qux_reg

    # Terminal cost-to-go
	Q = solver.E[N]
    S[N].Q .= Q.Q
    S[N].q .= Q.q

    # Initialize expecte change in cost-to-go
    ΔV = @SVector zeros(2)

	k = N-1
    while k > 0
        cost_exp = solver.E[k]
        dyn_exp = solver.D[k]

        # Compute gains
		Kλ, lλ = _calc_gains!(K[k], d[k], S[k+1], cost_exp, dyn_exp)

		# Calculate cost-to-go (using unregularized Quu and Qux)
        ΔV += _calc_ctg!(S[k], S[k+1], cost_exp, dyn_exp, K[k], d[k], Kλ, lλ)
        
        k -= 1
    end

    return ΔV

end

function _calc_ctg!(S, S⁺, cost_exp, dyn_exp, Ku, d, Kλ, dλ)
    A,B,C = dyn_exp.A, dyn_exp.B, dyn_exp.C
    Q,q,R,r,c = cost_exp.Q,cost_exp.q,cost_exp.R,cost_exp.r,cost_exp.c
    
    Abar = A -B*Ku -C*Kλ
    bbar = -B*d -C*dλ
    S.Q .= Q + Ku'*R*Ku + Abar'*S⁺.Q*Abar
    S.q .= q - Ku'*r + Ku'*R*d + Abar'*S⁺.Q*bbar + Abar'*S⁺.q

    # return ΔV
    t1 = 0
	t2 = 0
    return  @SVector [t1, t2]
end

function _calc_gains!(K, d, S, cost_exp, dyn_exp)
    S⁺, s⁺ = S.Q, S.q
    Q,q,R,r,c = cost_exp.Q,cost_exp.q,cost_exp.R,cost_exp.r,cost_exp.c 
    A,B,C,G = dyn_exp.A, dyn_exp.B, dyn_exp.C, dyn_exp.G

    n,m = size(B)
    _,p = size(C)

    D = B - C/(G*C)*G*B
    M11 = R + D'*S⁺*B
    M12 = D'*S⁺*C
    M21 = G*B
    M22 = G*C

    M = [M11 M12;M21 M22]
    b = [D'*S⁺;G]*A

    K_all = M\b
    Ku = K_all[1:m,:]
    Kλ = K_all[m .+ (1:p),:]

    l_all = M\[r + D'*s⁺; zeros(p)]
    lu = l_all[1:m]
    lλ = l_all[m .+ (1:p)]

    K .= Ku
    d .= lu

    return Kλ, lλ
end
