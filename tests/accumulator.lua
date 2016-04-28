require 'nn'

require 'Accumulator'

torch.manualSeed(1)

-- parameters
local precision = 1e-5
local jac = nn.Jacobian

-- define inputs and module
local input = torch.rand(1, 100)

local network = nn.Sequential()
-- network:add(nn.SplitTable(1))
network:add(nn.Accumulator())

-- test backprop, with Jacobian
local err = jac.testJacobian(network, input)
print('==> error: ' .. err)
if err<precision then
    print('==> module OK')
else
    print('==> error too large, incorrect implementation')
end
