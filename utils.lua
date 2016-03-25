local utils = {}

utils.findModulesByAnnotation = function(graph, annotation)
    local results = {}
    for _, m in ipairs(graph.forwardnodes) do
        if m.data.annotations.name == annotation then
            table.insert(results, m.data.module)
        end
    end
    return results
end

return utils
