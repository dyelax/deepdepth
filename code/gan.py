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
    conv5 = conv(conv4, 1024, filter_size=5)
    conv6 = conv(conv5, 512, filter_size=5)
    conv7 = conv(conv6, 256, filter_size=5)
    conv8 = conv(conv7, 128)
    out = conv(conv8, 1, activation=paddle.activation.Linear())

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


inference_topology = paddle.topology.Topology(layers=preds)
with open('model-v1.pkl', 'wb') as f:
    inference_topology.serialize_for_inference(f)



def img_reader():
    num_files = len(os.listdir(DIR))
    num_pairs = num_files / 2


    indices = np.arange(num_pairs)
    indices_shuffled = np.random.permutation(indices)

    for i in indices_shuffled: # There is an rgb and d_image image per frame
        d_image = Image.open(os.path.join(DIR, 'd-%d.pgm' % i)).resize((171, 128), Image.ANTIALIAS)

        rgb_image = Image.open(os.path.join(DIR, 'r-%d.ppm' % i)).resize((171, 128), Image.ANTIALIAS)
        final_width = 128
        final_height = 128

        width, height = d_image.size   # Get dimensions

        left = (width - final_width)/2
        top = (height - final_height)/2
        right = (width + final_width)/2
        bottom = (height + final_height)/2

        d_image = d_image.crop((left, top, right, bottom))
        rgb_image = rgb_image.crop((left, top, right, bottom))

        depth_tensor = np.array(d_image, dtype=float)
        rgb_tensor = np.array(rgb_image, dtype=float).transpose((2, 0, 1))

        depth_arr = depth_tensor.flatten()
        rgb_arr = rgb_tensor.flatten()

        # Normalize between between -1 and 1
        rgb_norm = rgb_arr / np.max(np.abs(rgb_arr))
        depth_norm = depth_arr / np.max(np.abs(depth_arr))

        # d_image = Image.fromarray((depth_norm * 255).astype('uint8'))
        # d_image.convert('L').save('/mnt/results/depth.jpg')
        # rgb_image = Image.fromarray((rgb_norm * 255).astype('uint8'))
        # rgb_image.convert('RGB').save('/mnt/results/rgb.jpg')

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
reader = paddle.minibatch.batch(img_reader, batch_size)

feeding={'inputs': 0,
         'labels': 1}

# event handler to track training and testing process
def event_handler(event):
    if isinstance(event, paddle.event.EndIteration):
        if event.batch_id % 1 == 0:
            print "\nEpoch %d, Batch %d, Cost %f, %s" % (
                event.pass_id, event.batch_id, event.cost, event.metrics)

        if event.batch_id % 500 == 0:
            save_img(event.pass_id, event.batch_id)

    if isinstance(event, paddle.event.EndPass):
        # save parameters
        with open('/mnt/results/default/params_epoch_%d.tar' % event.pass_id, 'w') as f:
            parameters.to_tar(f)

        # result = trainer.test(
        #     reader=paddle.batch(
        #         paddle.dataset.cifar.test10(), batch_size=128),
        #     feeding=feeding)
        # print "\nTest with Epoch %d, %s" % (event.pass_id, result.metrics)


def save_img(epoch, step):
    rgb_image = Image.open(os.path.join(DIR, 'r-0.ppm')).resize((171, 128), Image.ANTIALIAS)
    final_width = 128
    final_height = 128

    width, height = rgb_image.size  # Get dimensions

    left = (width - final_width) / 2
    top = (height - final_height) / 2
    right = (width + final_width) / 2
    bottom = (height + final_height) / 2

    rgb_image = rgb_image.crop((left, top, right, bottom))

    rgb_tensor = np.array(rgb_image, dtype=float).transpose((2, 0, 1))
    rgb_arr = rgb_tensor.flatten()

    # Normalize between between -1 (0?) and 1
    rgb_norm = rgb_arr / np.max(np.abs(rgb_arr))

    result = paddle.infer(output_layer=preds, parameters=parameters,
                          input=[[rgb_norm]],
                          feeding=feeding)
    img = result.reshape([img_height, img_width])
    denormed_img = (img + 1) * (255. / 2.)
    pil_img = Image.fromarray(denormed_img.astype('uint8'))
    pil_img.save('/mnt/results/default/epoch-%d_step-%d.png' % (epoch, step))
    print 'Image saved'

trainer.train(
    reader=reader,
    num_passes=1001,
    event_handler=event_handler,
    feeding=feeding)
