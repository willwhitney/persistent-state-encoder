require 'nn'
require 'cutorch'
require 'cunn'
require 'paths'
require 'lfs'

vis = require 'vis'
require 'UnsupervisedEncoder'
require 'Decoder'
local data_loaders = require 'data_loaders'

name = arg[1]
networks = {}
while true do
    local line = io.read()
    if line == nil then break end

    -- strip whitespace
    line = string.gsub(line, "%s+", "")

    table.insert(networks, line)
end

opt = {
        datasetdir = '/om/user/wwhitney/facegen/CNN_DATASET',
        gpu = true,
    }

base_directory = "/om/user/wwhitney/unsupervised-dcign/networks"

local jobname = name ..'_'.. os.date("%b_%d_%H_%M")
local output_path = 'reports/renderings/'..jobname
os.execute('mkdir -p '..output_path)

local dataset_types = {"AZ_VARIED", "EL_VARIED", "LIGHT_AZ_VARIED"}


function getLastSnapshot(network_name)
    local res_file = io.popen("ls -t "..paths.concat(base_directory, network_name).." | grep -i epoch | head -n 1")
    local result = res_file:read():match( "^%s*(.-)%s*$" )
    res_file:close()
    return result
end

for _, network in ipairs(networks) do
    collectgarbage()
    print('')
    print(network)
    local checkpoint = torch.load(paths.concat(base_directory, network, getLastSnapshot(network)))
    local model = checkpoint.model
    local scheduler_iteration = torch.Tensor{checkpoint.step}
    model:evaluate()

    local encoder = model.modules[1]
    local sharpener = encoder:findModules('nn.ScheduledWeightSharpener')[1]
    sharpener.iteration_container = scheduler_iteration
    print("Current sharpening: ", sharpener:getP())

    local weight_predictor = encoder:findModules('nn.Normalize')[1]
    local previous_embedding = encoder:findModules('nn.Linear')[1]
    local current_embedding = encoder:findModules('nn.Linear')[2]
    local decoder = model.modules[2]

    for _, variation in ipairs(dataset_types) do
        local images = {}
        for i = 1, 1 do -- for now only render one batch
            -- fetch a batch
            local input = data_loaders.load_mv_batch(i, variation, 'FT_test')
            local output = model:forward(input):clone()
            local embedding_from_previous = previous_embedding.output:clone()
            local embedding_from_current = current_embedding.output:clone()

            local reconstruction_from_previous = decoder:forward(embedding_from_previous):clone()
            local reconstruction_from_current = decoder:forward(embedding_from_current):clone()

            local weight_norms = torch.zeros(output:size(1))
            for input_index = 1, output:size(1) do
                local weights = weight_predictor.output[input_index]
                local max_weight, varying_index = weights:max(1)
                print("Varying index: " .. vis.simplestr(varying_index), "Weight: " .. vis.simplestr(max_weight))

                -- local embedding_change = embedding_from_current[input_index] - embedding_from_previous[input_index]
                -- local normalized_embedding_change = embedding_change / embedding_change:norm(1)
                -- print("Independence of embedding change: ", normalized_embedding_change:norm())
                -- print("Distance between timesteps: ", embedding_change:norm())

                weight_norms[input_index] = weights:norm()

                local image_row = {}
                table.insert(image_row, input[1][input_index]:float())
                table.insert(image_row, input[2][input_index]:float())
                table.insert(image_row, reconstruction_from_previous[input_index]:float())
                table.insert(image_row, reconstruction_from_current[input_index]:float())
                table.insert(image_row, output[input_index]:float())
                table.insert(images, image_row)
            end
            print("Mean independence of weights: ", weight_norms:mean())

        end
        vis.save_image_grid(paths.concat(output_path, network .."-"..variation..'.png'), images)
    end
    collectgarbage()
end


print("done")
