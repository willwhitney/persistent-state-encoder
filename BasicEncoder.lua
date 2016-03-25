local BasicEncoder = function(dim_hidden, color_channels, feature_maps)
    local filter_size = 5

    local encoder = nn.Sequential()
    encoder:add(nn.SpatialConvolution(color_channels, feature_maps, filter_size, filter_size))
    encoder:add(nn.SpatialMaxPooling(2,2,2,2))
    encoder:add(nn.Threshold(0,1e-6))

    encoder:add(nn.SpatialConvolution(feature_maps, feature_maps/2, filter_size, filter_size))
    encoder:add(nn.SpatialMaxPooling(2,2,2,2))
    encoder:add(nn.Threshold(0,1e-6))

    encoder:add(nn.SpatialConvolution(feature_maps/2, feature_maps/4, filter_size, filter_size))
    encoder:add(nn.SpatialMaxPooling(2,2,2,2))
    encoder:add(nn.Threshold(0,1e-6))

    encoder:add(nn.Reshape((feature_maps/4) * 22*16))
    encoder:add(nn.Linear((feature_maps/4) * 22*16, dim_hidden))

    return encoder
end

return BasicEncoder
