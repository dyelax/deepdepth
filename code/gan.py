import paddle.v2 as paddle
import sys

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


##
# Run
##

paddle.init(use_gpu=True, trainer_count=1)

img_height = 28
img_width = 28
img_depth = 3

inputs = paddle.layer.data(name='inputs', type=paddle.data_type.dense_vector(
    img_height * img_width * img_depth))
labels = paddle.layer.data(name='labels', type=paddle.data_type.dense_vector(
    img_height * img_width * img_depth))

cost = paddle.layer.mse_cost(input=G(inputs), label=labels)

parameters = paddle.parameters.create(cost)


def img_reader():
    # TODO: read in data and yield
    pass


# Create optimizer
optimizer = paddle.optimizer.Adam(learning_rate=0.0001)

# Create trainer
trainer = paddle.trainer.SGD(cost=cost,
                             parameters=parameters,
                             update_equation=optimizer)

batch_size = 32
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
