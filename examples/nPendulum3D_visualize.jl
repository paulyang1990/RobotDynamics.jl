using Rotations
using MeshCat, GeometryTypes, GeometryBasics, CoordinateTransformations, Colors
using GeometryTypes: HyperRectangle, Vec

function visualize!(m, Z, Δt)
    l, r, nb = m.lengths, m.radii, m.nb
    P = Lie_P(m)
    N = length(Z)

    vis = Visualizer()
    open(vis)

    for i=1:nb
        link = GeometryBasics.Rect{3,Float64}(Vec(-r[i]/2, -r[i]/2, -l[i]/2), Vec(r[i], r[i], l[i]))
        setobject!(vis["link$i"], link, MeshPhongMaterial(color=RGB(0, 1, 0)))
    end

    # Generate animation
    anim = MeshCat.Animation(round(Int,1/Δt))
    for k = 1:N
        atframe(anim, k-1) do
            pos = RD.vec_states(m, Z[k]) 
            rot = RD.rot_states(m, Z[k]) 
            for i=1:nb
                settransform!(vis["link$i"], AffineMap(rot[i], pos[i]))
            end            
        end
    end

    setanimation!(vis, anim)
end