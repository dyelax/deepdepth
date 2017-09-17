import paddle.v2 as paddle

# TODO: Dropout?
# TODO: image augmentation?
def eigen_coarse(input):
    conv1 = paddle.layer.img_conv(
        input=input,
        num_channels=3,
        num_filters=96,
        filter_size=(11, 11),
        padding=(5, 5),
        stride=(4, 4),
        act=paddle.activation.Relu(),
        pool_size=2,
        pool_stride=2,
        pool_type=paddle.pooling.Max()
    )
    conv2 = paddle.layer.img_conv(
        input=conv1,
        num_filters=256,
        filter_size=(5, 5),
        padding=(2, 2),
        act=paddle.activation.Relu(),
        pool_size=2,
        pool_stride=2,
        pool_type=paddle.pooling.Max()
    )
    conv3 = paddle.layer.img_conv(
        input=conv2,
        num_filters=384,
        filter_size=(3, 3),
        padding=(1, 1),
        act=paddle.activation.Relu()
    )
    conv4 = paddle.layer.img_conv(
        input=conv3,
        num_filters=384,
        filter_size=(3, 3),
        padding=(1, 1),
        act=paddle.activation.Relu()
    )
    conv5 = paddle.layer.img_conv(
        input=conv4,
        num_filters=384,
        filter_size=(3, 3),
        padding=(1, 1),
        act=paddle.activation.Relu(),
        pool_size=2,
        pool_stride=2,
        pool_type=paddle.pooling.Max()
    )

    fc1 = paddle.layer.fc(
        input=conv5,
        size=4096,
        act=paddle.activation.Relu(),
        layer_attr=paddle.attr.Extra(drop_rate=0.5))
    fc2 = paddle.layer.fc(
        input=fc1,
        size=(74 * 55),
        act=paddle.activation.Linear(),
        layer_attr=paddle.attr.Extra(drop_rate=0.5))

    return fc2

def eigen(input, coarse):
    conv1 = paddle.layer.img_conv(
        input=input,
        num_channels=3,
        num_filters=63,
        filter_size=(9, 9),
        padding=(1, 1),
        stride=(3, 3),
        act=paddle.activation.Relu(),
        pool_size=2,
        pool_stride=2,
        pool_type=paddle.pooling.Max()
    )
    conv2 = paddle.layer.img_conv(
        input=input,
        num_filters=63,
        filter_size=(5, 5),
        padding=(2, 2),
        act=paddle.activation.Relu()
    )

    concat = paddle.layer.concat(
        input=[conv2, coarse(input)],
        layer_attr=,
    )

    conv3 = paddle.layer.img_conv(
        input=concat,
        num_filters=64,
        filter_size=(5, 5),
        padding=(2, 2),
        act=paddle.activation.Relu()
    )
    conv4 = paddle.layer.img_conv(
        input=conv3,
        num_filters=64,
        filter_size=(5, 5),
        padding=(2, 2),
        act=paddle.activation.Linear()
    )
