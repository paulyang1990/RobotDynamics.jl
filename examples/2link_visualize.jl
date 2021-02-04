using Rotations
using MeshCat, GeometryTypes, GeometryBasics, CoordinateTransformations, Colors
using GeometryTypes: HyperRectangle, Vec

function get_lengths(m)
    if m isa LieGroupModelMC
        return m.lengths # acrobot3D
    else
        return m.l1, m.l2 # DoublePendulumRC and MC
    end
end

function get_radii(m)
    if m isa LieGroupModelMC
        return m.radii # acrobot3D
    else
        return fill(.1, 2) # DoublePendulumRC and MC
    end
end

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
            if m isa LieGroupModelMC # acrobot3D
                pos1 = Z[k][1:3]
                rot1 = UnitQuaternion(Z[k][4:7])
                pos2 = Z[k][7 .+ (1:3)]
                rot2 = UnitQuaternion(Z[k][7 .+ (4:7)])
            else # double_pendulum
                pos1 = [0; Z[k][1:2]]
                rot1 = RotX(Z[k][3]-pi/2)
                pos2 = [0; Z[k][4:5]]
                rot2 = RotX(Z[k][6]-pi/2)
            end

            settransform!(vis["link1"], AffineMap(rot1,pos1))
            settransform!(vis["link2"], AffineMap(rot2,pos2))
        end
    end

    setanimation!(vis, anim)
end