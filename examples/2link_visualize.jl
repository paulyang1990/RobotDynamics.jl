using MeshCat, GeometryTypes, GeometryBasics, CoordinateTransformations, Colors
using GeometryTypes: HyperRectangle, Vec

get_lengths(m::Acrobot3D) = m.lengths
get_radii(m::Acrobot3D) = m.radii
get_lengths(m) = m.l1, m.l2 # DoublePendulumRC and MC
get_radii(m) = fill(.1, 2) # DoublePendulumRC and MC

function visualize!(m,Z,Δt)
    l1,l2 = get_lengths(m)
    r1,r2 = get_radii(m)
    N = length(Z)

    vis = Visualizer()
    open(vis)

    link1 = GeometryBasics.Rect{3,Float64}(Vec(-r1/2, -r1/2, -l1/2), Vec(r1, r1, l1))
    link2 = GeometryBasics.Rect{3,Float64}(Vec(-r2/2, -r2/2, -l2/2), Vec(r2, r2, l2))

    setobject!(vis["link1"], link1, MeshPhongMaterial(color=RGB(0, 1, 0)))
    setobject!(vis["link2"], link2, MeshPhongMaterial(color=RGB(0, 1, 0)))

    # Generate animation
    anim = MeshCat.Animation(Int(1/Δt))
    for k = 1:N
        atframe(anim, k-1) do
            pos1 = Z[k][1:3]
            quat1 = UnitQuaternion(Z[k][4:7])
            pos2 = Z[k][7 .+ (1:3)]
            quat2 = UnitQuaternion(Z[k][7 .+ (4:7)])

            settransform!(vis["link1"], AffineMap(quat1,pos1))
            settransform!(vis["link2"], AffineMap(quat2,pos2))
        end
    end

    setanimation!(vis, anim)
end