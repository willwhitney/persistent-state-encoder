require 'nn'
require 'ProperJacobian'

require 'cutorch'
require 'cunn'

local Model = require 'AtariModel'
local Decoder = require 'AtariDecoder'

torch.manualSeed(1)

-- parameters
local precision = 1e-5
local jac = nn.ProperJacobian

-- define inputs and module
local input = torch.rand(2, 1, 210, 160)

local scheduler_iteration = torch.Tensor{1}
local batch_timesteps = 2
opt = {
        dim_hidden = 10,
        color_channels = 1,
        feature_maps = 4,
        noise = 0,
        sharpening_rate = 1,
        heads = 1,
    }

model = nn.Sequential()
model:add(nn.SplitTable(1))
encoder = Model(opt.dim_hidden, opt.color_channels, opt.feature_maps, opt.noise, opt.sharpening_rate, scheduler_iteration, opt.heads, batch_timesteps)
model:add(encoder)

join = nn.JoinTable(1)
model:add(join)

decoder = Decoder(opt.dim_hidden, opt.color_channels, opt.feature_maps)
model:add(decoder)

print(model)

model:cuda()
input = input:cuda()

-- test backprop, with Jacobian
local err = jac.testJacobian(model, input)
print('==> error: ' .. err)
if err < precision then
    print('==> module OK')
else
    print('==> error too large, incorrect implementation')
end
