require 'nn'

Accumulator = torch.class('nn.Accumulator', 'nn.Module')

function Accumulator:init(cost_container)
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
    self.gradInput = torch.zeros(input:size())
    self.gradInput[{{}, i}] = gradOutput[{{}, i}]
    for i = 2, input:size(2) do
        self.gradInput[{{}, i}] = self.gradInput[{{}, i-1}] + gradOutput[{{}, i}]
    end

    -- penalty is proportional to the number of heads proposed
    local gradPenalty = torch.zeros(input:size())
    for i = 1, input:size(2) do
        gradPenalty[{{}, i}] = i
    end
    gradPenalty = gradPenalty * self.cost_container[1]
    self.gradInput = self.gradInput + gradPenalty
    return self.gradInput
end
