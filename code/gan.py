import paddle.v2 as paddle
import sys
import numpy as np
import os
from PIL import Image

img_height = 128
img_width = 128
img_depth = 3

##
# Models
##
DIR = '/mnt/processed/classrooms/classroom_0001a'

#TODO: CUDNN

def conv(inputs, fms, input_fms=None, filter_size=3, activation=paddle.activation.Relu()):
    return paddle.layer.img_conv(
        input=inputs,
        filter_size=filter_size,
        num_filters=fms,
        num_channels=input_fms,
        act=activation,
        bias_attr=None,
        padding=(filter_size // 2)
    )

# def D(inputs):
#     # TODO: Pooling
#     conv1 = conv(inputs, 32, input_fms=1)
#     conv2 = conv(conv1, 64)
#     conv3 = conv(conv2, 128)
#     conv4 = conv(conv3, 256)
#
#     fc_dim = 4096
#     fc1 = paddle.layer.fc(
#         input=conv4,
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

def G(inputs):
    conv1 = conv(inputs, 64, input_fms=3)
    conv2 = conv(conv1, 128)
    conv3 = conv(conv2, 256, filter_size=5)
    conv4 = conv(conv3, 512, filter_size=5)
    conv5 = conv(conv4, 256, filter_size=5)
    conv6 = conv(conv5, 128)
    out = conv(conv6, 1, activation=paddle.activation.Linear())

    return out


##
# Run
##

paddle.init(use_gpu=True, trainer_count=1)

inputs = paddle.layer.data(name='inputs', type=paddle.data_type.dense_vector(
    img_height * img_width * img_depth))
labels = paddle.layer.data(name='labels', type=paddle.data_type.dense_vector(
    img_height * img_width * 1))

preds = G(inputs)
cost = paddle.layer.square_error_cost(input=preds, label=labels)

parameters = paddle.parameters.create(cost)


def img_reader():
    # TODO: read in data and yield
    num_files = len(os.listdir(DIR))

    # while True:
    # for i in xrange(num_files / 2): # There is an rgb and d_image image per frame
    for i in xrange(20): # There is an rgb and d_image image per frame
        d_image = Image.open(os.path.join(DIR, 'd-%d.pgm' % i)).resize((171, 128))
        rgb_image = Image.open(os.path.join(DIR, 'r-%d.ppm' % i)).resize((171, 128))
        final_width = 128
        final_height = 128

        width, height = d_image.size   # Get dimensions

        left = (width - final_width)/2
        top = (height - final_height)/2
        right = (width + final_width)/2
        bottom = (height + final_height)/2

        d_image = d_image.crop((left, top, right, bottom))
        rgb_image = rgb_image.crop((left, top, right, bottom))

        depth_arr = np.array(d_image).flatten()
        rgb_arr = np.array(rgb_image).flatten()

        # Normalize between between -1 and 1
        rgb_norm = rgb_arr / np.max(np.abs(rgb_arr), axis=0) # (2 * (rgb_arr - np.max(rgb_arr))) / (-np.ptp(rgb_arr) - 1)
        depth_norm = depth_arr / np.max(np.abs(depth_arr), axis=0) # (2 * (depth_arr - np.max(depth_arr))) / (-np.ptp(depth_arr) - 1)

        yield (
            rgb_norm,
            depth_norm
        )


# Create optimizer
optimizer = paddle.optimizer.Adam(learning_rate=0.0000005)

# Create trainer
trainer = paddle.trainer.SGD(cost=cost,
                             parameters=parameters,
                             update_equation=optimizer)

batch_size = 4
reader = paddle.minibatch.batch(paddle.reader.shuffle(img_reader, batch_size), batch_size)

feeding={'inputs': 0,
         'labels': 1}

# event handler to track training and testing process
def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 1 == 0:
            print "\nEpoch %d, Batch %d, Cost %f, %s" % (
                event.pass_id, event.batch_id, event.cost, event.metrics)
        # else:
        #     sys.stdout.write('.')
        #     sys.stdout.flush()



    if isinstance(event, paddle.event.EndPass):
        if event.pass_id % 10 == 0:
            result = paddle.infer(output_layer=preds, parameters=parameters,
                                  input=[[img_reader().next()[0]]],
                                  feeding=feeding)
            img = result.reshape([img_height, img_width])
            denormed_img = (img + 1) * (255. / 2.)
            pil_img = Image.fromarray(denormed_img.astype('uint8'))
            pil_img.save('/mnt/results/%d.jpg' % event.pass_id)
            print 'Image saved'

            # save parameters
            with open('params_epoch_%d.tar' % event.pass_id, 'w') as f:
                parameters.to_tar(f)

        # result = trainer.test(
        #     reader=paddle.batch(
        #         paddle.dataset.cifar.test10(), batch_size=128),
        #     feeding=feeding)
        # print "\nTest with Epoch %d, %s" % (event.pass_id, result.metrics)


trainer.train(
    reader=reader,
    num_passes=1001,
    event_handler=event_handler,
    feeding=feeding)
