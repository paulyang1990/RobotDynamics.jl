
config_inds(i) = (7*(i-1) .+ (1:3), 
7*(i-1) .+ (4:7))
vel_inds(i) = (6*(i-1) .+ (1:3), 
6*(i-1) .+ (4:6))

function jan_to_og(x)
    nb = Int(length(x)/13)
    config = zeros(nb*7)
    vel = zeros(nb*6)
    for i=1:nb
        xinds, vinds, qinds, ωinds = fullargsinds(i) 
        _xinds, _qinds = config_inds(i)
        _vinds, _ωinds = vel_inds(i)
        config[_xinds] = x[xinds]
        config[_qinds] = x[qinds]
        vel[_vinds] = x[vinds]
        vel[_ωinds] = x[ωinds]
    end
    return [config; vel]
end

function og_to_jan(_x)
    nb = Int(length(_x)/13)
    config = _x[1:nb*7]
    vel = _x[nb*7+1:end]
    x = copy(_x)
    for i=1:nb
        xinds, vinds, qinds, ωinds = fullargsinds(i) 
        _xinds, _qinds = config_inds(i)
        _vinds, _ωinds = vel_inds(i)
        x[xinds] = config[_xinds]
        x[qinds] = config[_qinds]
        x[vinds] = vel[_vinds]
        x[ωinds] = vel[_ωinds] 
    end
    return x
end
