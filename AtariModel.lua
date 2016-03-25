require 'nngraph'

Encoder = require 'AtariEncoder'
Decoder = require 'AtariDecoder'
BasicEncoder = require 'BasicEncoder'

local AtariModel = function(dim_hidden, color_channels, feature_maps, noise, sharpening_rate, scheduler_iteration, num_heads, timesteps)

    local inputs = {}
    for timestep = 1, timesteps do
        table.insert(inputs, nn.Identity()():annotate{name="input_step_"..timestep})
    end

    local state_initialization_encoder = BasicEncoder(dim_hidden, color_channels, feature_maps)
    state_initialization_encoder = state_initialization_encoder(inputs[1]):annotate{name="state_initializer"}

    local encoder_prototype = Encoder(dim_hidden, color_channels, feature_maps, noise, sharpening_rate, scheduler_iteration, num_heads)
    local decoder_prototype = Decoder(dim_hidden, color_channels, feature_maps)

    -- local model_prototype = nn.Sequential()
    -- model_prototype:add(encoder_prototype)
    -- model_prototype:add(decoder_prototype)

    -- each of these encoders represents a transition function for the state
    -- the state for step 1 is initialized by the state_initialization_encoder
    -- the state for step 2 is a special case since we're using the prototype there
    -- thus we start with 3
    local encoder_clones = {encoder_prototype}
    for _ = 3, timesteps do
        table.insert(encoder_clones, encoder_prototype:clone('weight', 'bias', 'gradWeight', 'gradBias'))
    end

    encoder_clones[1] = encoder_clones[1]{state_initialization_encoder, inputs[2]}:annotate{name="encoder_1"}
    for i = 2, #encoder_clones do
        encoder_clones[i] = encoder_clones[i]{encoder_clones[i-1], inputs[i+1]}:annotate{name="encoder_"..i}
    end

    local decoder_clones = {decoder_prototype}
    for _ = 2, timesteps do
        table.insert(decoder_clones, decoder_prototype:clone('weight', 'bias', 'gradWeight', 'gradBias'))
    end

    decoder_clones[1] = decoder_clones[1](state_initialization_encoder):annotate{name="decoder_1"}
    for i = 1, #encoder_clones do
        decoder_clones[i+1] = decoder_clones[i+1](encoder_clones[i]):annotate{name="decoder_"..i}
    end

    local outputs = decoder_clones

    return nn.gModule(inputs, outputs)
end

return AtariModel
