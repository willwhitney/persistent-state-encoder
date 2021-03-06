require 'nngraph'

Encoder = require 'VariableHeadsEncoder'
BasicEncoder = require 'BasicEncoder'

local VariableHeadsModel = function(dim_hidden, color_channels, feature_maps, noise, sharpening_rate, encoder_noise, head_cost, max_heads, timesteps)

    local inputs = {}
    for timestep = 1, timesteps do
        table.insert(inputs, nn.Identity()():annotate{name="input_step_"..timestep})
    end

    local state_initialization_encoder = BasicEncoder(dim_hidden, color_channels, feature_maps, encoder_noise)
    state_initialization_encoder = state_initialization_encoder(inputs[1]):annotate{name="state_initializer"}

    local encoder_prototype = Encoder(dim_hidden, color_channels, feature_maps, noise, sharpening_rate, encoder_noise, head_cost, max_heads)
    -- local decoder_prototype = Decoder(dim_hidden, color_channels, feature_maps)

    if opt.gpu then
        encoder_prototype:cuda()
        -- decoder_prototype:cuda()
    end
    -- print("encoder prototype params")
    -- print(encoder_prototype:getParameters():size())

    -- local model_prototype = nn.Sequential()
    -- model_prototype:add(encoder_prototype)
    -- model_prototype:add(decoder_prototype)

    -- each of these encoders represents a transition function for the state
    -- the state for step 1 is initialized by the state_initialization_encoder
    -- the state for step 2 is a special case since we're using the prototype there
    -- thus we start with 3
    local encoder_clones = {encoder_prototype}
    for _ = 3, timesteps do
        local clone = Encoder(dim_hidden, color_channels, feature_maps, noise, sharpening_rate, encoder_noise, head_cost, max_heads)
        if opt.gpu then
            clone:cuda()
        end
        clone:share(encoder_prototype, 'weight', 'bias', 'gradWeight', 'gradBias')
        table.insert(encoder_clones, clone)
    end

    encoder_clones[1] = encoder_clones[1]{state_initialization_encoder, inputs[2]}:annotate{name="encoder"}
    for i = 2, #encoder_clones do
        encoder_clones[i] = encoder_clones[i]{encoder_clones[i-1], inputs[i+1]}:annotate{name="encoder"}
    end

    output = {state_initialization_encoder, table.unpack(encoder_clones)}

    -- local decoder_clones = {decoder_prototype}
    -- for _ = 2, timesteps do
    --     local clone = Decoder(dim_hidden, color_channels, feature_maps)
    --     if opt.gpu then
    --         clone:cuda()
    --     end
    --     clone:share(decoder_prototype, 'weight', 'bias', 'gradWeight', 'gradBias')
    --     table.insert(decoder_clones, clone)
    -- end

    -- decoder_clones[1] = decoder_clones[1](state_initialization_encoder):annotate{name="decoder"}
    -- for i = 1, #encoder_clones do
    --     decoder_clones[i+1] = decoder_clones[i+1](encoder_clones[i]):annotate{name="decoder"}
    -- end

    -- local output = decoder_clones

    collectgarbage()
    return nn.gModule(inputs, output)
end

return VariableHeadsModel
