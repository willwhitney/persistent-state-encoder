require 'nn'
require 'nngraph'

require 'Print'
require 'ChangeLimiter'
require 'Noise'
require 'ScheduledWeightSharpener'

BasicEncoder = require('BasicEncoder')

local AtariEncoder = function(dim_hidden, color_channels, feature_maps, noise, sharpening_rate, scheduler_iteration, num_heads)

    local previous_state = nn.Identity()():annotate{name="previous_state"}
    local input_frame = nn.Identity()():annotate{name="frame"}
    local inputs = {
            previous_state,
            input_frame,
        }

    -- make two copies of an encoder

    local encoder = BasicEncoder(dim_hidden, color_channels, feature_maps)

    encoder = encoder(input_frame):annotate{name="encoder"}

    -- make the heads to analyze the encodings
    local heads = {}
    heads[1] = nn.Sequential()
    heads[1]:add(nn.JoinTable(2))
    heads[1]:add(nn.Linear(dim_hidden * 2, dim_hidden))
    heads[1]:add(nn.Sigmoid())
    heads[1]:add(nn.Noise(noise))
    heads[1]:add(nn.ScheduledWeightSharpener(sharpening_rate, scheduler_iteration))
    heads[1]:add(nn.AddConstant(1e-20))
    heads[1]:add(nn.Normalize(1, 1e-100))

    for i = 2, num_heads do
        heads[i] = heads[1]:clone()
    end

    for i = 1, num_heads do
        heads[i] = heads[i]{previous_state, encoder}:annotate{name="gating_head_"..i}
    end

    local dist
    if num_heads > 1 then
        -- combine the distributions from all heads
        local dist_adder = nn.CAddTable()(heads)
        local dist_clamp = nn.Clamp(0, 1)(dist_adder)
        dist = dist_clamp
    else
        dist = heads[1]
    end

    -- and use it to filter the encodings
    local change_limiter = nn.ChangeLimiter()({dist, previous_state, encoder}):annotate{name="change_limiter"}

    local output = {change_limiter}
    local gmod = nn.gModule(inputs, output)
    -- print(gmod:getParameters():size())
    return gmod
end

return AtariEncoder
