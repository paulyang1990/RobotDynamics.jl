# Algo 1 from https://arxiv.org/pdf/1506.05481.pdf
function bend_twist_decomp(qa, qb)
    q = qa*qb
    
    # twist
    q_t = ((q.w!=0) || (q.z!=0)) ? 
        UnitQuaternion(q.a,0,0,q.z) : one(UnitQuaternion)
    θ_t = rotation_angle(q_t)

    # swing
    q_b = q*conj(q_t)
    θ_b = 2*acos(q_b.w)

    return q_t, θ_t, q_b, θ_b

end