require 'nn'

local Accumulator, parent = torch.class('nn.Accumulator', 'nn.Module')

function Accumulator:__init(cost_container)
    parent.__init(self)

    -- self.cost_container is a tensor so that this parameter can be annealed
    self.cost_container = cost_container or torch.Tensor{0}
end

function Accumulator:updateOutput(input)
    self.output = input:clone()
    for i = input:size(2)-1, 1, -1 do
        self.output[{{}, i}] = self.output[{{}, i}] + self.output[{{}, i+1}]
    end
    return self.output
end

function Accumulator:updateGradInput(input, gradOutput)
    self.gradInput = self.gradInput:resizeAs(input):typeAs(input)
    self.gradInput:zero()
    self.gradInput[{{}, 1}] = gradOutput[{{}, 1}]
    for i = 2, input:size(2) do
        self.gradInput[{{}, i}] = self.gradInput[{{}, i-1}] + gradOutput[{{}, i}]
    end

    -- penalty is proportional to the number of heads proposed
    local gradPenalty = torch.zeros(input:size()):typeAs(input)
    for i = 1, input:size(2) do
        gradPenalty[{{}, i}] = i
    end
    gradPenalty = gradPenalty * self.cost_container[1]
    self.gradInput = self.gradInput + gradPenalty
    return self.gradInput
end

function Accumulator:type(t)
    parent.type(self, t)
    self.cost_container:type(t)
end
