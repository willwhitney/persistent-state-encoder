require 'nn'
require 'optim'

require 'MotionBCECriterion'

local Model = require 'VariableHeadsModel'
local Decoder = require 'AtariDecoder'
local data_loaders = require 'data_loaders'
local utils = require 'utils'
local vis = require 'vis'

local cmd = torch.CmdLine()

cmd:option('--name', 'default', 'filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('--checkpoint_dir', 'networks', 'output directory where checkpoints get written')
cmd:option('--import', '', 'initialize network parameters from checkpoint at this path')

-- data
cmd:option('--datasetdir', '/om/user/wwhitney/deep-game-engine', 'dataset source directory')
cmd:option('--dataset_name', 'breakout', 'dataset source directory')
cmd:option('--frame_interval', 3, 'the number of timesteps between input[1] and input[2]')

-- optimization
cmd:option('--learning_rate', 1e-4, 'learning rate')
cmd:option('--learning_rate_decay', 0.97, 'learning rate decay')
cmd:option('--learning_rate_decay_after', 18000, 'in number of examples, when to start decaying the learning rate')
cmd:option('--learning_rate_decay_interval', 4000, 'in number of examples, how often to decay the learning rate')
cmd:option('--decay_rate', 0.95, 'decay rate for rmsprop')
cmd:option('--grad_clip', 3, 'clip gradients at this value')

cmd:option('--cost_decay_rate', 0.95, 'decay rate for the cost per head')
cmd:option('--cost_per_head', 0.01, 'cost per additional head used')
cmd:option('--max_heads', 20, 'how many filtering heads to use')
cmd:option('--motion_scale', 3, 'how much to accentuate loss on changing pixels')

cmd:option('--L2', 0, 'amount of L2 regularization')
cmd:option('--criterion', 'BCE', 'criterion to use')

cmd:option('--dim_hidden', 200, 'dimension of the representation layer')
cmd:option('--feature_maps', 64, 'number of feature maps')
cmd:option('--color_channels', 3, '1 for grayscale, 3 for color')
cmd:option('--sharpening_rate', 10, 'how fast to sharpen the heads')
cmd:option('--noise', 0.1, 'variance of added Gaussian noise')


cmd:option('--max_epochs', 50, 'number of full passes through the training data')

-- bookkeeping
cmd:option('--seed', 123, 'torch manual random number generator seed')
cmd:option('--print_every', 1, 'how many steps/minibatches between printing out the loss')
cmd:option('--eval_val_every', 8000, 'every how many iterations should we evaluate on validation data?')

-- data
cmd:option('--num_train_batches', 8000, 'number of batches to train with per epoch')
cmd:option('--num_test_batches', 900, 'number of batches to test with')

-- GPU/CPU
cmd:option('--gpu', false, 'whether to use GPU')
cmd:text()


-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

if opt.gpu then
    require 'cutorch'
    require 'cunn'
end

if opt.name == 'net' then
    local name = 'unsup_'
    for _, v in ipairs(arg) do
        name = name .. tostring(v) .. '_'
    end
    opt.name = name .. os.date("%b_%d_%H_%M_%S")
end

local savedir = string.format('%s/%s', opt.checkpoint_dir, opt.name)
print("Saving output to "..savedir)
os.execute('mkdir -p '..savedir)
os.execute(string.format('rm %s/*', savedir))

-- log out the options used for creating this network to a file in the save directory.
-- super useful when you're moving folders around so you don't lose track of things.
local f = io.open(savedir .. '/opt.txt', 'w')
for key, val in pairs(opt) do
  f:write(tostring(key) .. ": " .. tostring(val) .. "\n")
end
f:flush()
f:close()

local logfile = io.open(savedir .. '/output.log', 'w')
true_print = print
print = function(...)
    for _, v in ipairs{...} do
        true_print(v)
        logfile:write(tostring(v))
    end
    logfile:write("\n")
    logfile:flush()
end


local scheduler_iteration = torch.zeros(1)

-- this is dumb, but it's the easiest and most portable way
-- to make this a global variable
opt.current_head_cost = 0

local sample_batch = data_loaders.load_random_atari_batch('train')
local batch_timesteps = #sample_batch
-- local batch_timesteps = 10

model = nn.Sequential()
encoder = Model(opt.dim_hidden, opt.color_channels, opt.feature_maps, opt.noise, opt.sharpening_rate, scheduler_iteration, head_cost, opt.max_heads, batch_timesteps)
model:add(encoder)

join = nn.JoinTable(1)
model:add(join)

decoder = Decoder(opt.dim_hidden, opt.color_channels, opt.feature_maps)
model:add(decoder)

print(model)

-- input = torch.ones(10, 3, 210, 160)
-- embeddings = model.modules[1]:forward(input:split(1))
-- print(embeddings[1])
-- graph.dot(model.modules[1].fg, 'atari_varheads', 'reports/atari_varheads')
-- print(model.modules[1])
-- graph.dot(model.modules[1].modules[4].fg, 'atari_varheads_encoder', 'reports/atari_varheads_encoder')

-- [[

if opt.criterion == 'MSE' then
    criterion = nn.CriterionTable(nn.MSECriterion())
elseif opt.criterion == 'BCE' then
    criterion = nn.MotionBCECriterion(opt.motion_scale)
else
    error("Invalid criterion specified!")
end

local stupid_join = nn.JoinTable(1)

if opt.gpu then
    model:cuda()
    criterion:cuda()
    stupid_join:cuda()
end

-- local encoders = utils.findModulesByAnnotation(model, 'encoder')
-- print("Encoders: " .. #encoders)
-- for i = 2, #encoders do
--     encoders[i]:share(encoders[1], 'weight', 'bias', 'gradWeight', 'gradBias')
-- end
-- collectgarbage()

-- local decoders = utils.findModulesByAnnotation(model, 'decoder')
-- print("Decoders: " .. #decoders)
-- for i = 2, #decoders do
--     decoders[i]:share(decoders[1], 'weight', 'bias', 'gradWeight', 'gradBias')
-- end
-- collectgarbage()

params, grad_params = model:getParameters()


local heads_predictor = model.modules[1]:findModules('nn.Accumulator')[1]
function validate()
    local loss = 0
    model:evaluate()

    for i = 1, opt.num_test_batches do -- iterate over batches in the split
        -- fetch a batch
        local input = data_loaders.load_atari_batch(i, 'test')

        local output = model:forward(input)
        step_loss = criterion:forward(output, stupid_join:forward(input))

        if i % 10 == 0 then
            print("Accumulator distribution: ", vis.simplestr(heads_predictor.output[1]))
            -- print("Accumulator head cost: ", heads_predictor.cost_container[1])
        end

        loss = loss + step_loss
    end

    loss = loss / opt.num_test_batches
    return loss
end

-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= params then
        error("Params not equal to given feval argument.")
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local input = data_loaders.load_random_atari_batch('train')

    ------------------- forward pass -------------------
    model:training() -- make sure we are in correct mode

    local loss
    local output = model:forward(input)

    -- print(model.modules[2].output:size())

    loss = criterion:forward(output, stupid_join:forward(input))
    local grad_output = criterion:backward(output, stupid_join:forward(input))

    model:backward(input, grad_output)


    ------------------ regularize -------------------
    if opt.L2 > 0 then
        -- Loss:
        loss = loss + opt.L2 * params:norm(2)^2/2
        -- Gradients:
        grad_params:add( params:clone():mul(opt.L2) )
    end

    grad_params:clamp(-opt.grad_clip, opt.grad_clip)

    collectgarbage()
    return loss, grad_params
end

train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * opt.num_train_batches
-- local iterations_per_epoch = opt.num_train_batches
local loss0 = nil

-- print(cutorch.getMemoryUsage(cutorch.getDevice()))

for step = 1, iterations do
    scheduler_iteration[1] = step
    epoch = step / opt.num_train_batches

    opt.current_head_cost = opt.cost_per_head * (1 - opt.cost_decay_rate ^ epoch)

    local timer = torch.Timer()

    local _, loss = optim.rmsprop(feval, params, optim_state)

    local time = timer:time().real

    -- print(cutorch.getMemoryUsage(cutorch.getDevice()))
    -- print(params:size())
    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[step] = train_loss

    -- exponential learning rate decay
    if step % opt.learning_rate_decay_interval == 0 and opt.learning_rate_decay < 1 then
        if step >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed function learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    if step % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", step, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end

    -- every now and then or on last iteration
    if step % opt.eval_val_every == 0 or step == iterations then
        print(string.format("Head cost at epoch %.3f: %.6f", epoch, opt.current_head_cost))
        -- evaluate loss on validation data
        local val_loss = validate() -- 2 = validation
        val_losses[step] = val_loss
        print(string.format('[epoch %.3f] Validation loss: %6.8f', epoch, val_loss))

        local model_file = string.format('%s/epoch%.2f_%.4f.t7', savedir, epoch, val_loss)
        print('saving checkpoint to ' .. model_file)
        local checkpoint = {}
        checkpoint.model = model
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.step = step
        checkpoint.epoch = epoch
        torch.save(model_file, checkpoint)

        local val_loss_log = io.open(savedir ..'/val_loss.txt', 'a')
        val_loss_log:write(val_loss .. "\n")
        val_loss_log:flush()
        val_loss_log:close()
    end

    if step % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then
        loss0 = loss[1]
    end
    -- if loss[1] > loss0 * 8 then
    --     print('loss is exploding, aborting.')
    --     print("loss0:", loss0, "loss[1]:", loss[1])
    --     break -- halt
    -- end
end
--]]
