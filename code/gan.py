import paddle.v2 as paddle

##
# Models
##

def conv_block(inputs,
               num_filter,
               groups,
               num_channels=None,
               filter_size=3,
               pool=False,
               activation=paddle.activation.Relu(),
               dropout=0.5):
    return paddle.networks.img_conv_group(
        input=inputs,
        num_channels=num_channels,
        pool_size=(2 if pool else None),
        pool_stride=(2 if pool else None),
        pool_type=(paddle.pooling.Max() if pool else None),
        conv_num_filter=[num_filter] * groups,
        conv_filter_size=filter_size,
        conv_act=activation,
        conv_batchnorm_drop_rate=dropout)

def D(inputs):
    # VGG19 Kinda
    conv1 = conv_block(inputs, 64, 2, 1, pool=True)
    conv2 = conv_block(conv1, 128, 2, pool=True)
    conv3 = conv_block(conv2, 256, 4, pool=True)
    conv4 = conv_block(conv3, 512, 4, pool=True)
    conv5 = conv_block(conv4, 512, 4, pool=True)

    fc_dim = 4096
    fc1 = paddle.layer.fc(
        input=conv5,
        size=fc_dim,
        act=paddle.activation.Relu(),
        layer_attr=paddle.attr.Extra(drop_rate=0.5))
    fc2 = paddle.layer.fc(
        input=fc1,
        size=fc_dim,
        act=paddle.activation.Relu(),
        layer_attr=paddle.attr.Extra(drop_rate=0.5))
    out = paddle.layer.fc(input=fc2, size=1, act=paddle.activation.Sigmoid())

    return out

def G(inputs):
    conv1 = conv_block(inputs, 64, 2, 1),
    conv2 = conv_block(conv1, 128, 2),
    conv3 = conv_block(conv2, 256, 2, filter_size=5),
    conv4 = conv_block(conv3, 512, 2, filter_size=5),
    conv5 = conv_block(conv4, 256, 2, filter_size=5),
    conv6 = conv_block(conv5, 128, 2),
    out = conv_block(conv6, 1, 1, dropout=0),

    return out



