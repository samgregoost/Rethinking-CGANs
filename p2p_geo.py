from __future__ import print_function
import tensorflow as tf
import numpy as np
import random
import TensorflowUtils as utils
import read_data as scene_parsing
import datetime
import BatchDatsetReaderAtt as dataset
from six.moves import xrange
from skimage import io, color

import math
from scipy import signal
from scipy.interpolate import interp1d


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "25", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "/scratch1/ram095/nips21/logs_nlrgeon/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "/scratch1/ram095/nips20/datasets/pix2pix/datasets/att", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MAX_ITERATION = int(1e7 + 1)
NUM_OF_CLASSESS = 3
IMAGE_SIZE = 128

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    try:
      matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    except ValueError as err:
        msg = "NOTE: Usually, this is due to an issue with the image dimensions.  Did you correctly set '--crop' or '--input_height' or '--output_height'?"
        err.args = err.args + (msg,)
        raise
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv

def new_conv_layer( bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, name=None ):
        with tf.variable_scope( name ):
            w = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-1],
                    initializer=tf.constant_initializer(0.))

            conv = tf.nn.conv2d( bottom, w, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(conv, b))

        return bias #relu



def new_deconv_layer(bottom, filter_shape, output_shape, activation=tf.identity, padding='SAME', stride=1, name=None):
        with tf.variable_scope(name):
            W = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-2],
                    initializer=tf.constant_initializer(0.))
            deconv = tf.nn.conv2d_transpose( bottom, W, output_shape, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(deconv, b))

        return bias

def batchnorm(bottom, is_train, epsilon=1e-8, name=None):
        bottom = tf.clip_by_value( bottom, -100., 100.)
        depth = bottom.get_shape().as_list()[-1]

        with tf.variable_scope(name):
            gamma = tf.get_variable("gamma", [depth], initializer=tf.constant_initializer(1.))
            beta  = tf.get_variable("beta" , [depth], initializer=tf.constant_initializer(0.))

            batch_mean, batch_var = tf.nn.moments(bottom, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def update():
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            ema_apply_op = ema.apply([batch_mean, batch_var])
            ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
            mean, var = tf.cond(
                    is_train,
                    update,
                    lambda: (ema_mean, ema_var) )

            normed = tf.nn.batch_norm_with_global_normalization(bottom, mean, var, beta, gamma, epsilon, False)
        return normed

def adv_cost(images, keep_prob, condition, is_train):
    encoderLayerNum = int(math.log(IMAGE_SIZE) / math.log(2))
    encoderLayerNum = encoderLayerNum - 1 # minus 1 because the second last layer directly go from 4x4 to 1x1
    print("encoderLayerNum=", encoderLayerNum)
    encoderLayerNum = encoderLayerNum

    decoderLayerNum = int(math.log(IMAGE_SIZE) / math.log(2))
    decoderLayerNum = decoderLayerNum - 1
    print("decoderLayerNum=", decoderLayerNum)
    decoderLayerNum = decoderLayerNum
    print("setting up vgg initialized conv layers ...")

    with tf.variable_scope("adv", reuse = tf.AUTO_REUSE):
  #      up = new_deconv_layer( tf.tile(h,[1,16,16,1]), [4,4,3,1024], [FLAGS.batch_size,128,128,3], stride=2, name="recons")
 #       upsample = tf.nn.leaky_relu(batchnorm(up, is_train, name=("upbn2")))
#    upsample = tf.reshape(tf.nn.leaky_relu(tf.layers.dense(tf.contrib.layers.flatten(h),128*128*3)),[FLAGS.batch_size,128,128,3])
        images = tf.concat([images, condition], axis = 3)
        previousFeatureMap = images
        previousDepth = 3
        depth = 64

        conv1 = tf.nn.dropout(new_conv_layer(previousFeatureMap, [4,4,previousDepth,depth], stride=2, name=("conv1")),keep_prob)
        bn1 = tf.nn.leaky_relu(batchnorm(conv1, is_train, name=("bn1")))
        previousDepth = depth
        depth = depth * 2

        conv2 = tf.nn.dropout(new_conv_layer(bn1, [4,4,previousDepth,depth], stride=2, name=("conv2")),keep_prob)
        bn2 = tf.nn.leaky_relu(batchnorm(conv2, is_train, name=("bn2")))
        previousDepth = depth
        depth = depth * 2

        conv3 = tf.nn.dropout(new_conv_layer(bn2, [4,4,previousDepth,depth], stride=2, name=("conv3")),keep_prob)
        bn3 = tf.nn.leaky_relu(batchnorm(conv3, is_train, name=("bn3")))
        previousDepth = depth
        depth = depth * 2

        conv4 = tf.nn.dropout(new_conv_layer(bn3, [4,4,previousDepth,depth], stride=2, name=("conv4")),keep_prob)
        bn4 = tf.nn.leaky_relu(batchnorm(conv4, is_train, name=("bn4")))
        previousDepth = depth
        depth = depth * 2

        conv5 = tf.nn.dropout(new_conv_layer(bn4, [4,4,previousDepth,depth], stride=2, name=("conv5")),keep_prob)
        bn5 = tf.nn.leaky_relu(batchnorm(conv5, is_train, name=("bn5")))
        previousDepth = depth
        depth = depth * 2

        conv6 = new_conv_layer(bn5, [4,4,previousDepth,4000], stride=2, padding='VALID', name=('conv6'))
        bn6 = tf.nn.leaky_relu(batchnorm(conv6, is_train, name=("bn6")))

        flat = tf.contrib.layers.flatten(bn6)

        fc1 = tf.layers.dense(flat,16)
        fc2 = tf.layers.dense(fc1,1)
        return fc2


def inference(images, keep_prob,z,e,is_train):
    encoderLayerNum = int(math.log(IMAGE_SIZE) / math.log(2))
    encoderLayerNum = encoderLayerNum - 1 # minus 1 because the second last layer directly go from 4x4 to 1x1 
    print("encoderLayerNum=", encoderLayerNum)
    decoderLayerNum = int(math.log(IMAGE_SIZE) / math.log(2))
    decoderLayerNum = decoderLayerNum - 1
    print("decoderLayerNum=", decoderLayerNum)
    
    with tf.variable_scope("encoder", reuse = tf.AUTO_REUSE):

        previousFeatureMap = images
        previousDepth = 3
        depth = 64
        conv1 = tf.nn.dropout(new_conv_layer(previousFeatureMap, [4,4,previousDepth,depth], stride=2, name=("conv1")),keep_prob)
        bn1 = tf.nn.leaky_relu(batchnorm(conv1, is_train, name=("bn1")))
        previousDepth = depth
        depth = depth * 2
        conv2 = tf.nn.dropout(new_conv_layer(bn1, [4,4,previousDepth,depth], stride=2, name=("conv2")),keep_prob)
        bn2 = tf.nn.leaky_relu(batchnorm(conv2, is_train, name=("bn2")))
        previousDepth = depth
        depth = depth * 2
        conv3 = tf.nn.dropout(new_conv_layer(bn2, [4,4,previousDepth,depth], stride=2, name=("conv3")),keep_prob)
        bn3 = tf.nn.leaky_relu(batchnorm(conv3, is_train, name=("bn3")))
        previousDepth = depth
        depth = depth * 2
        conv4 = tf.nn.dropout(new_conv_layer(bn3, [4,4,previousDepth,depth], stride=2, name=("conv4")),keep_prob)
        bn4 = tf.nn.leaky_relu(batchnorm(conv4, is_train, name=("bn4")))
        previousDepth = depth
        depth = depth * 2
        conv5 = tf.nn.dropout(new_conv_layer(bn4, [4,4,previousDepth,depth], stride=2, name=("conv5")),keep_prob)
        bn5 = tf.nn.leaky_relu(batchnorm(conv5, is_train, name=("bn5")))
        previousDepth = depth
        depth = depth * 2
        conv6 = new_conv_layer(bn5, [4,4,previousDepth,4000], stride=2, padding='VALID', name=('conv6'))
        bn6 = tf.nn.leaky_relu(batchnorm(conv6, is_train, name=("bn6")))
        previousDepth = 4000
        depth = 64 * pow(2,decoderLayerNum-2)
        featureMapSize = 4
        deconv6 =  tf.nn.dropout(new_deconv_layer( bn6, [4,4,depth,previousDepth], [FLAGS.batch_size,featureMapSize,featureMapSize,depth], padding='VALID', stride=2, name=("deconv6")),keep_prob)
        #debn_ = tf.nn.relu(batchnorm(deconv, is_train, name=("debn" + str(decoderLayerNum))))
        z_ = z#/tf.norm(z)
        debn_ = tf.nn.relu(batchnorm(deconv6, is_train, name=("debn6")))
        debn6 = tf.concat([debn_,tf.tile(z_,[1,4,4,1])],axis = 3)+ e

    with tf.variable_scope("decoder", reuse = tf.AUTO_REUSE):
        previousDepth = 1088
        depth = int(depth / 2)
        featureMapSize = featureMapSize *2
        print("build_reconstruction decoder layer4")
        deconv5 = new_deconv_layer( debn6, [4,4,depth,previousDepth], [FLAGS.batch_size,featureMapSize,featureMapSize,depth], stride=2, name=("deconv5"))
        debn5 = tf.nn.relu(batchnorm(deconv5, is_train, name=('debn5'))) + bn4
        previousDepth = depth
        depth = int(depth / 2)
        featureMapSize = featureMapSize *2

        deconv4 = new_deconv_layer( debn5, [4,4,depth,previousDepth], [FLAGS.batch_size,featureMapSize,featureMapSize,depth], stride=2, name=("deconv4"))
        debn4 = tf.nn.relu(batchnorm(deconv4, is_train, name=('debn4'))) + bn3
        previousDepth = depth
        depth = int(depth / 2)
        featureMapSize = featureMapSize *2

        deconv3 = new_deconv_layer( debn4, [4,4,depth,previousDepth], [FLAGS.batch_size,featureMapSize,featureMapSize,depth], stride=2, name=("deconv3"))
        debn3 = tf.nn.relu(batchnorm(deconv3, is_train, name=('debn3'))) + bn2
        previousDepth = depth
        depth = int(depth / 2)
        featureMapSize = featureMapSize *2
        deconv2 = new_deconv_layer( debn3, [4,4,depth,previousDepth], [FLAGS.batch_size,featureMapSize,featureMapSize,depth], stride=2, name=("deconv2"))
        debn2 = tf.nn.relu(batchnorm(deconv2, is_train, name=('debn2'))) + bn1
        previousDepth = depth
        depth = int(depth / 2)
        featureMapSize = featureMapSize *2
        recon = tf.nn.tanh(new_deconv_layer( debn2, [4,4,3,previousDepth], [FLAGS.batch_size,128,128,3], stride=2, name="recon"))
    return recon, debn_

def train(loss_val, var_list):
    optimizer = tf.train.RMSPropOptimizer(0.0001)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)

def train_advcost(loss_val, var_list):
    optimizer = tf.train.RMSPropOptimizer(0.0001)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)

def var_net(images, is_train):
    with tf.variable_scope("gaussian", reuse = tf.AUTO_REUSE):
        conv1 = new_conv_layer(images, [4,4,3,2], stride=2, name=("conv1"))
        bn1 = tf.nn.leaky_relu(batchnorm(conv1, is_train, name=("bn1")))

        conv2 = new_conv_layer(bn1, [4,4,2,2], stride=2, name=("conv2"))
        bn2 = tf.nn.leaky_relu(batchnorm(conv2, is_train, name=("bn2")))

        conv3 = new_conv_layer(bn2, [4,4,2,2], stride=2, name=("conv3"))
        bn3 = tf.nn.leaky_relu(batchnorm(conv3, is_train, name=("bn3")))

        conv4 =  new_conv_layer(bn3, [4,4,2,1], stride=2, name=("conv4"))
    #    bn4 = tf.nn.leaky_relu(batchnorm(conv4, is_train, name=("bn4")))

        return conv4

def calc_det(A, n, alpha):
    c1 = n*tf.math.log(alpha)
    C_ = tf.eye(n, batch_shape = [FLAGS.batch_size]) - A/alpha
    S_ = tf.zeros([FLAGS.batch_size])   

    Sf, Cf, If = tf.while_loop(lambda S, C, i: (i < 20) ,
    lambda S, C, i:( S + tf.trace(C)/i, tf.linalg.matmul(C,C), i+1),(S_, C_, 1.0))

    logdet = c1 - Sf

    return logdet

        
def var_loss(z_var_stack, itr, annotation, image, keep_probability, istrain):            
    tensor_list = []
    for i in range(itr):
        z_curr = tf.reshape(tf.slice(z_var_stack, [i,0,0,0,0], [1,FLAGS.batch_size,1,1,64 ]), [FLAGS.batch_size,1,1,64])
        logits, _ = inference(image,keep_probability, z_curr ,0.0,istrain)
        var_img = var_net(logits,istrain)
        tensor_list.append(var_img)


    tensor_stack = tf.stack(tensor_list, axis = 0)

    tensor_reshaped = tf.transpose(tf.reshape(tensor_stack, [itr,FLAGS.batch_size,64]), perm = [1,2,0])  #(20,N,10)

    tensor_mean = tf.reduce_mean(tensor_reshaped, axis = 2, keepdims = True)

    covar = tf.linalg.matmul(tensor_reshaped-tensor_mean,tensor_reshaped-tensor_mean,transpose_b=True)/64.0

    v  = 0.1 * tf.transpose(covar,[0,2,1] )

    inv_mat, p_1 = tf.while_loop(lambda x, i: tf.logical_and(i < 100 ,tf.less(tf.reduce_mean(x),tf.constant([10.0]))[0]),
    lambda x, i:( tf.matmul( tf.eye(64,  batch_shape = [FLAGS.batch_size]) + 1.0/4.0 * tf.matmul(tf.eye(64,  batch_shape = [FLAGS.batch_size])-tf.matmul(x, covar),tf.matmul(3.0*tf.eye(64,  batch_shape = [FLAGS.batch_size])-tf.matmul(x,covar),3.0*tf.eye(64,  batch_shape = [FLAGS.batch_size])-tf.matmul(x,covar))),x),i+1),(v, 0))

    inv_mat =  tf.where(tf.is_nan(inv_mat), tf.zeros_like(inv_mat), inv_mat)

    det = calc_det(covar, 64, 10.0)

    annot_list = []
    for i in range(10):
        annot_r = tf.image.random_brightness(annotation, 0.8, seed=None)
        annot_r = tf.image.random_contrast(annot_r, 0.2, 0.8)

        annot_var = var_net(annot_r, istrain)
        annot_list.append(annot_var)
         
    annot_stack = tf.stack(annot_list, axis = 0) #(10,20,8,8)
    annot_reshaped = tf.transpose(tf.reshape(annot_stack, [itr,FLAGS.batch_size,64]), perm = [1,2,0])  #(20,N,10)
    annot_mean = tf.reduce_mean(annot_reshaped, axis = 2, keepdims = True)
    annot_covar = tf.linalg.matmul(annot_reshaped-annot_mean,annot_reshaped-annot_mean,transpose_b=True)/64.0 
    
    res_tensor = tf.reshape(tensor_mean - annot_mean, [FLAGS.batch_size,64,1])

    annot_det = calc_det(annot_covar, 64, 10.0)
    
    term_1 = (det - annot_det)
    term_2 = tf.trace(tf.matmul(inv_mat, annot_covar)) 
    term_3_1 = tf.linalg.matmul(res_tensor,inv_mat,transpose_a = True)
    term_3 = tf.reshape(tf.linalg.matmul(term_3_1,res_tensor),[20])
    return tf.reduce_mean(term_1+ term_2 + term_3), covar, annot_det 
        

def train_z(loss,Z):
    return tf.gradients(ys = loss, xs = Z)

def random_mask():
    x1 =  random.randint(5, 20)
    w1 = random.randint(20, 34)
    y1 = random.randint(5, 20)
    h1 =  random.randint(20, 34)

    mask = np.zeros((1,128,128,1))
    mask[:,x1:x1+w1,y1:y1+h1,:] = 1.0

    mask2 = np.zeros((1,128,128,1))
    mask2[:,x1-5:x1+w1+5,y1-5:h1+y1+5,:] = 1.0
    mask2 = mask2 - mask

    return mask, mask2

def main(argv=None):
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="input_image")
    annotation = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="annotation")
    z = tf.placeholder(tf.float32, shape=[None, 1, 1, 64], name="z")
    mask = tf.placeholder(tf.float32, shape=[None, 128, 128, 1], name="mask")
    mask2 = tf.placeholder(tf.float32, shape=[None, 128, 128, 1], name="mask2")
    z_new = tf.placeholder(tf.float32, shape=[None, 1, 1, 1], name="z_new")
    istrain = tf.placeholder(tf.bool)


    z_var_stack = tf.placeholder(tf.float32, shape=[None, None, 1, 1, 64], name="z_stack")
    variance_loss, _, _ = var_loss(z_var_stack, 10, annotation, image, keep_probability, istrain)
    e = tf.placeholder(tf.float32, shape=[None, 4, 4, 1088], name="e")
    e_p = tf.placeholder(tf.float32, shape=[None, 1, 1, 16448], name="e_p")
    
    save_itr = 0

    zero = tf.zeros([FLAGS.batch_size,1,1,16448])   
    logits, h  = inference(image,keep_probability,z,0.0,istrain)
    logits_e, h_e = inference(image, keep_probability,z,e,istrain)


    z_pred = predictor(h,z,zero,istrain)
    z_pred_e = predictor(h,z,e_p,istrain)

    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)


    advloss_g = tf.reduce_mean(adv_cost(logits, keep_probability, image, istrain))
    advloss_d = -1.0 * tf.reduce_mean(adv_cost(annotation, keep_probability, image, istrain))

    z_loss_l = tf.reduce_sum(tf.contrib.layers.flatten(tf.abs((annotation - logits))),1)
    loss_ = tf.reduce_mean(tf.reduce_sum(tf.contrib.layers.flatten(tf.abs((annotation - logits))),1)) - 10*advloss_g  # + loss_ssim

    loss_interp =  tf.reduce_mean(tf.reduce_sum(tf.contrib.layers.flatten(tf.abs((annotation - logits))),1))

    loss_summary = tf.summary.scalar("entropy", loss_)
    lip_loss_dec = tf.reduce_mean(tf.reduce_sum(tf.contrib.layers.flatten(tf.abs((logits - logits_e))),1))
    loss = loss_ + 0.1*lip_loss_dec  + 0.0001* variance_loss

    grads = train_z(z_loss_l,z)    
    val_grads = train_z(z_pred,z)
    val_grads2 = train_z( -advloss_g,z)

    trainable_var = tf.trainable_variables()
    trainable_z_pred_var = tf.trainable_variables(scope="predictor")

    

    not_c_var = [s for s in trainable_var if not "adv" in s.name]
    c_var = [s for s in trainable_var if "adv" in s.name]

    print(trainable_z_pred_var)
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)

    train_op = train( loss, not_c_var)

    train_interp = train_predictor(loss_interp,not_c_var)
    
    train_advd = train_advcost((advloss_d + advloss_g), c_var) 
    clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in c_var]

    print("Setting up image reader...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)

    sess = tf.Session()

    print("Setting up Saver...")
    saver2 =  tf.train.Saver()
    saver = tf.train.Saver()

    train_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.logs_dir + '/validation')

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver2.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.mode == "train":
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            z_ = np.random.uniform(low=-1.0, high = 1.0, size=(FLAGS.batch_size,1,1,64))
            r_m, r_m2 = random_mask(64)
            v = 0
            error_dec =  np.random.normal(0.0,0.001,(FLAGS.batch_size,4,4,1088))
            feed_dict_init = {image: train_images, keep_probability: 0.8, z: z_,e:error_dec, mask:r_m,mask2:r_m2, istrain:True}
            train_annot_init = sess.run(logits, feed_dict=feed_dict_init)

            z_list = []
            pred_z_list = []
            for p in range(11):
                train_annot_interp = (10-p)*0.1*train_annot_init + p*0.1*train_annotations
                z_ol = np.copy(z_)
                pred_z_list.append(z_ol)

                feed_dict = {image: train_images, annotation: train_annot_interp, keep_probability: 0.8, z: z_,e:error_dec, mask:r_m,mask2:r_m2, istrain:True }         
               
                _ = sess.run(train_interp, feed_dict=feed_dict)

                z_loss,advloss_g_,advloss_d_, l_lip, summ = sess.run([z_loss_l, advloss_g, advloss_d, lip_loss_dec, loss_summary], feed_dict=feed_dict)
                print("Step: %d, z_step: %d, Train_loss:%g, adv_g:%g, adv_d:%g, lip_loss:%g, " % (itr,p,np.mean(z_loss),advloss_g_,advloss_d_, l_lip))

                g = sess.run([grads],feed_dict=feed_dict)
                v_prev = np.copy(v)
                v = 0.001*v - 0.1*g[0][0]
                z_ += 0.001 * v_prev + (1+0.001)*v
                print(z_)

            for t in range(10):
                z_rand = z_ + np.random.normal(0.0,0.001,(FLAGS.batch_size,1,1,64))
                z_list.append(z_rand)
            z_stack = np.stack(z_list, axis = 0)

            feed_dict = {image: train_images, annotation: train_annotations,z_var_stack:z_stack, keep_probability:0.8,mask:r_m,e:error_dec, z: z_,mask2:r_m2, istrain:True }
            
            _ = sess.run([train_op], feed_dict=feed_dict)
            sess.run(train_advd, feed_dict=feed_dict) 
            sess.run(clip_D)
            sess.run(train_advd, feed_dict=feed_dict)
            sess.run(clip_D)
            sess.run(train_advd, feed_dict=feed_dict)
            sess.run(clip_D)
            sess.run(train_advd, feed_dict=feed_dict)        
            sess.run(clip_D)
            sess.run(train_advd, feed_dict=feed_dict)
            sess.run(clip_D)    
            

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, loss_summary], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                
                train_writer.add_summary(summary_str, itr)
              

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)

                valid_loss, summary_sva = sess.run([loss, loss_summary], feed_dict={image: valid_images,mask:r_m, annotation: valid_annotations,
                                                       keep_probability: 1.0, z: z_,e:error_dec, istrain:False,mask2:r_m2 })
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                # add validation loss to TensorBoard
                validation_writer.add_summary(summary_sva, itr)
                if itr % 3000 == 0:
                    save_itr = save_itr + 3000
               
                saver.save(sess, FLAGS.logs_dir + "modell1.ckpt", save_itr)

    elif FLAGS.mode == "visualize":
      valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
      for k in xrange(20):

        r_m, r_m2 = random_mask(64)
        v= 0
        z_ol = np.random.uniform(low=-15.0, high=15.0, size=(FLAGS.batch_size,1,1,64))
        for p in range(0):
                feed_dict = {image: valid_images, keep_probability: 0.8, z: z_ol, istrain:False }
                
                g = sess.run([val_grads],feed_dict=feed_dict)
                g2 = sess.run([val_grads2],feed_dict=feed_dict)
                v = 0.001*v - 0.1*g[0][0]-0.1*g2[0][0]
                z_ol = np.random.normal(loc=0.0, scale=1.0, size=(FLAGS.batch_size,1,1,64))

        pred = sess.run(logits, feed_dict={image: valid_images,z:z_ol, istrain:False,mask:r_m,mask2:r_m2,
                                                    keep_probability: 1.0})
        

                
        valid_images_masked = (valid_images + 1.)/2.0*255
        pred_ = (pred +1.)/2.0*255

        pred = pred_
        valid_annotations_ = (valid_annotations +1.)/2.0*255

        print(np.max(pred))
        print(valid_images.shape)
        print(valid_annotations.shape)
        print(pred.shape)
        
        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images_masked[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr) )
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="predz_" + str(5+itr) + "_" + str(k)  )     
            utils.save_image(valid_annotations_[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr) )        

if __name__ == "__main__":
    tf.app.run()
