import paddle.v2 as paddle
import sys
import numpy as np

##
# Models
##

#TODO: CUDNN
# def conv_block(inputs,
#                num_filter,
#                groups,
#                input_channels=None,
#                filter_size=3,
#                activation=paddle.activation.Relu(),
#                dropout=0.5):
#     return paddle.networks.img_conv_group(
#         input=inputs,
#         num_channels=input_channels,
#         conv_num_filter=[num_filter] * groups,
#         conv_filter_size=filter_size,
#         conv_act=activation,
#         conv_batchnorm_drop_rate=dropout)
#
# def D(inputs):
#     # VGG19 Kinda
#     # TODO: Pooling
#     conv1 = conv_block(inputs, 64, 2, input_channels=1)
#     conv2 = conv_block(conv1, 128, 2)
#     conv3 = conv_block(conv2, 256, 4)
#     conv4 = conv_block(conv3, 512, 4)
#     conv5 = conv_block(conv4, 512, 4)
#
#     fc_dim = 4096
#     fc1 = paddle.layer.fc(
#         input=conv5,
#         size=fc_dim,
#         act=paddle.activation.Relu(),
#         layer_attr=paddle.attr.Extra(drop_rate=0.5))
#     fc2 = paddle.layer.fc(
#         input=fc1,
#         size=fc_dim,
#         act=paddle.activation.Relu(),
#         layer_attr=paddle.attr.Extra(drop_rate=0.5))
#     out = paddle.layer.fc(input=fc2, size=1, act=paddle.activation.Sigmoid())
#
#     return out

def conv(inputs, fms, input_fms=None, filter_size=3, activation=paddle.activation.Relu()):
    return paddle.layer.img_conv(
        input=inputs,
        filter_size=filter_size,
        num_filters=fms,
        num_channels=input_fms,
        # act=activation,
        # bias_attr=None,
        # padding=(filter_size // 2)
    )

def G(inputs):
    # conv1 = conv(inputs, 64, input_fms=3)
    # conv2 = conv(conv1, 128)
    # conv3 = conv(conv2, 256, filter_size=5)
    # conv4 = conv(conv3, 512, filter_size=5)
    # conv5 = conv(conv4, 256, filter_size=5)
    # conv6 = conv(conv5, 128)
    # out = conv(conv6, 1, 1, activation=paddle.activation.Linear())

    # out = conv(inputs, 1, input_fms=3)

    out = paddle.layer.img_conv(
        input=inputs,
        num_channels=3,
        filter_size=1,
        num_filters=1)
    return out


##
# Run
##

paddle.init(use_gpu=True, trainer_count=1)

# img_height = 240
# img_width = 320
# img_depth = 3
img_height = 8
img_width = 8
img_depth = 4

inputs = paddle.layer.data(name='inputs', type=paddle.data_type.dense_vector(
    img_height * img_width * img_depth))
labels = paddle.layer.data(name='labels', type=paddle.data_type.dense_vector(
    img_height * img_width * 1))

cost = paddle.layer.square_error_cost(input=G(inputs), label=labels)

parameters = paddle.parameters.create(cost)


def img_reader():
    # TODO: read in data and yield
    yield (
        np.random.random([img_height, img_width, img_depth]) * 2 - 1,
        np.random.random([img_height, img_width, 1]) * 2 - 1
    )


# Create optimizer
optimizer = paddle.optimizer.Adam(learning_rate=0.0001)

# Create trainer
trainer = paddle.trainer.SGD(cost=cost,
                             parameters=parameters,
                             update_equation=optimizer)

batch_size = 1
reader = paddle.minibatch.batch(
    paddle.reader.shuffle(img_reader, batch_size), batch_size)

feeding={'inputs': 0,
         'labels': 1}

# event handler to track training and testing process
def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 100 == 0:
            print "\nEpoch %d, Batch %d, Cost %f, %s" % (
                event.pass_id, event.batch_id, event.cost, event.metrics)
        else:
            sys.stdout.write('.')
            sys.stdout.flush()
    if isinstance(event, paddle.event.EndPass):
        # save parameters
        with open('params_epoch_%d.tar' % event.pass_id, 'w') as f:
            parameters.to_tar(f)

        result = trainer.test(
            reader=paddle.batch(
                paddle.dataset.cifar.test10(), batch_size=128),
            feeding=feeding)
        print "\nTest with Epoch %d, %s" % (event.pass_id, result.metrics)


trainer.train(
    reader=reader,
    num_passes=200,
    event_handler=event_handler,
    feeding=feeding)
