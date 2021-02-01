using MeshCat, GeometryTypes, CoordinateTransformations, Colors
using GeometryTypes: HyperRectangle, HyperSphere, Vec, Point
using LinearMaps

# visualize!(model,X,.001)
function visualize!(m::Acrobot3D,Z,Δt)
    l1,l2 = m.lengths
    r1,r2 = m.radii
    N = length(Z)

    vis = Visualizer()
    open(vis)

    link1 = GeometryTypes.Rect{3,Float64}(Vec(-r1/2, -r1/2, -l1/2), Vec(r1, r1, l1))
    link2 = GeometryTypes.Rect{3,Float64}(Vec(-r2/2, -r2/2, -l2/2), Vec(r2, r2, l2))

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