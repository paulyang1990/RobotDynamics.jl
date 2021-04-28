using MeshCat
using StaticArrays
using FileIO, MeshIO
using Rotations
using MeshCat, GeometryTypes, GeometryBasics, CoordinateTransformations, Colors
using GeometryTypes: HyperRectangle, Vec
using TrajOptPlots

function ModifiedMeshFileObject(obj_path::String, material_path::String; scale::T=0.1) where {T}
    obj = MeshFileObject(obj_path)
    rescaled_contents = rescale_contents(obj_path, scale=scale)
    # material = select_material(material_path)
    material = MeshPhongMaterial(color=RGBA(1,1,1, 0.5))

    mod_obj = MeshFileObject(
        rescaled_contents,
        obj.format,
        material,
        obj.resources,
        )
    return mod_obj
end

function rescale_contents(obj_path::String; scale::T=0.1) where T
    lines = readlines(obj_path)
    rescaled_lines = copy(lines)
    for (k,line) in enumerate(lines)
        if length(line) >= 2
            if line[1] == 'v'
                stringvec = split(line, " ")
                vals = map(x->parse(Float64,x),stringvec[end-2:end])
                rescaled_vals = vals .* scale
                rescaled_lines[k] = join([stringvec[1]; string.(rescaled_vals)], " ")
            end
        end
    end
    rescaled_contents = join(rescaled_lines, "\r\n")
    return rescaled_contents
end

function visualize!(m, Z, Δt)
    l, r, nb = m.lengths, m.radii, m.nb
    P = Lie_P(m)
    N = length(Z)

    vis = Visualizer()
    open(vis)
    
    quad_scaling = 0.3
    obj_path =  joinpath(@__DIR__, "quadrotor.obj")
    rescaled_contents = rescale_contents(obj_path, scale=quad_scaling)
    # obj = MeshFileObject(obj_path)
    # mod_obj = MeshFileObject(
    #         rescaled_contents,
    #         obj.format,
    #         nothing,
    #         obj.resources,
    #         )

    scaled_obj = MeshFileGeometry(rescaled_contents, "obj")
    setobject!(vis["link1"], scaled_obj, MeshPhongMaterial(color=RGB(.2,.2,.2)))

    for i=2:nb
        link = GeometryBasics.Rect{3,Float64}(Vec(-r[i]/2, -r[i]/2, -l[i]/2), Vec(r[i], r[i], l[i]))
        setobject!(vis["link$i"], link, MeshPhongMaterial(color=RGB(.3,.3,.3)))
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

    return vis
end

function TrajOptPlots._set_mesh!(vis, m; color=RGBA(.3,.3,.3,1))
    l, r, nb = m.lengths, m.radii, m.nb

    quad_scaling = 0.3
    obj_path =  joinpath(@__DIR__, "quadrotor.obj")
    rescaled_contents = rescale_contents(obj_path, scale=quad_scaling)

    scaled_obj = MeshFileGeometry(rescaled_contents, "obj")
    setobject!(vis["link1"], scaled_obj, MeshPhongMaterial(color=color))

    for i=2:nb
        link = GeometryBasics.Rect{3,Float64}(Vec(-r[i]/2, -r[i]/2, -l[i]/2), Vec(r[i], r[i], l[i]))
        setobject!(vis["link$i"], link, MeshPhongMaterial(color=color))
    end
end

function TrajOptPlots.visualize!(vis, model::LieGroupModelMC, x::SVector)
    pos = RD.vec_states(model, x) 
    rot = RD.rot_states(model, x) 
    for i=1:model.nb
        settransform!(vis["robot","link$i"], AffineMap(rot[i], pos[i]))
    end 
end
