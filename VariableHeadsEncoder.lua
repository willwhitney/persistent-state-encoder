require 'nn'
require 'nngraph'

require 'Print'
require 'ChangeLimiter'
require 'Noise'
require 'ScheduledWeightSharpener'
require 'Accumulator'

BasicEncoder = require('BasicEncoder')

local VariableHeadsEncoder = function(dim_hidden, color_channels, feature_maps, noise, sharpening_rate, encoder_noise, head_cost, max_heads)
    -- print("dim_hidden, color_channels, feature_maps, noise, sharpening_rate, scheduler_iteration, head_cost, max_heads")
    -- print(dim_hidden, color_channels, feature_maps, noise, sharpening_rate, scheduler_iteration, head_cost, max_heads)

    local previous_state = nn.Identity()():annotate{name="previous_state"}
    local input_frame = nn.Identity()():annotate{name="frame"}
    local inputs = {
            previous_state,
            input_frame,
        }

    -- make two copies of an encoder

    local encoder = BasicEncoder(dim_hidden, color_channels, feature_maps, encoder_noise)

    encoder = encoder(input_frame):annotate{name="encoder"}

    local heads_predictor = nn.Sequential()
    heads_predictor:add(nn.JoinTable(2))
    heads_predictor:add(nn.Linear(dim_hidden * 2, max_heads))
    heads_predictor:add(nn.SoftMax()) -- distribution over how many heads to have
    heads_predictor:add(nn.Accumulator(head_cost)) -- prob of having at least n heads
    heads_predictor = heads_predictor{previous_state, encoder}:annotate{name="heads_predictor"}


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

    -- create the rest of the heads
    for i = 2, max_heads do
        heads[i] = heads[1]:clone()
        heads[i]:reset()
    end

    -- name all the heads
    for i = 1, max_heads do
        -- print("head: ", heads[i])
        heads[i] = heads[i]{previous_state, encoder}:annotate{name="gating_head_"..i}
    end

    local dist
    if max_heads > 1 then
        -- combine the distributions from all heads
        -- local heads_tensor = nn.JoinTable(2)(heads)
        -- local heads_mixture = nn.MixtureTable(3){heads_predictor, heads_tensor}
        local heads_table = nn.Identity()(heads)
        local heads_mixture = nn.MixtureTable(){heads_predictor, heads_table}
        local dist_clamp = nn.Clamp(0, 1)(heads_mixture)
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

return VariableHeadsEncoder
