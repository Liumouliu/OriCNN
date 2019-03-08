from input_data_rgb_ori_m1_1_augument import InputData

from encoder_net import sem_encoder

import tensorflow as tf
import numpy as np
import os

batch_size = 12
loss_weight = 10.0
number_of_epoch = 100

learning_rate_val = 1e-5

# -------------------------------------------------------- #


def validate(grd_descriptor, sat_descriptor):
    accuracy = 0.0
    data_amount = 0.0
    dist_array = 2 - 2 * np.matmul(sat_descriptor, np.transpose(grd_descriptor))
    top1_percent = int(dist_array.shape[0] * 0.01) + 1
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, i]
        prediction = np.sum(dist_array[:, i] < gt_dist)
        if prediction < top1_percent:
            accuracy += 1.0
        data_amount += 1.0
    accuracy /= data_amount

    return accuracy


def compute_loss(sat_global, grd_global, batch_hard_count=0):
    '''
    Compute the weighted soft-margin triplet loss
    :param sat_global: the satellite image global descriptor
    :param grd_global: the ground image global descriptor
    :param batch_hard_count: the number of top hard pairs within a batch. If 0, no in-batch hard negative mining
    :return: the loss
    '''
    with tf.name_scope('weighted_soft_margin_triplet_loss'):
        dist_array = 2 - 2 * tf.matmul(sat_global, grd_global, transpose_b=True)
        pos_dist = tf.diag_part(dist_array)
        if batch_hard_count == 0:
            pair_n = batch_size * (batch_size - 1.0)

            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            loss_g2s = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))) / pair_n

            # satellite to ground
            triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
            loss_s2g = tf.reduce_sum(tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))) / pair_n

            loss = (loss_g2s + loss_s2g) / 2.0
        else:
            # ground to satellite
            triplet_dist_g2s = pos_dist - dist_array
            triplet_dist_g2s = tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))
            top_k_g2s, _ = tf.nn.top_k(tf.transpose(triplet_dist_g2s), batch_hard_count)
            loss_g2s = tf.reduce_mean(top_k_g2s)

            # satellite to ground
            triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
            triplet_dist_s2g = tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))
            top_k_s2g, _ = tf.nn.top_k(triplet_dist_s2g, batch_hard_count)
            loss_s2g = tf.reduce_mean(top_k_s2g)

            loss = (loss_g2s + loss_s2g) / 2.0

    return loss

def train(start_epoch=1):
    '''
    Train the network and do the test
    :param start_epoch: the epoch id start to train. The first epoch is 1.
    '''

    # import data
    input_data = InputData()

    # define placeholders
    sat_x = tf.placeholder(tf.float32, [None, 512, 512, 3], name='sat_x')
    grd_x = tf.placeholder(tf.float32, [None, 224, 1232, 3], name='grd_x')

    sat_x_ori = tf.placeholder(tf.float32, [None, 512, 512, 2], name='sat_x_ori')
    grd_x_ori = tf.placeholder(tf.float32, [None, 224, 1232, 2], name='grd_x_ori')

    learning_rate = tf.placeholder(tf.float32)


    # build a encode cnn to extract feature embeddings

    semnet = sem_encoder()

    sat_global = semnet.create_generator_gem_pooling_shallow_concatLast3conv(tf.concat([sat_x,sat_x_ori], axis=3 ), 'sat_rgbori')
    grd_global = semnet.create_generator_gem_pooling_shallow_concatLast3conv(tf.concat([grd_x,grd_x_ori], axis=3 ), 'grd_rgbori')


    # define loss
    localization_loss = compute_loss(sat_global, grd_global, 0)

    # set training

    global_step = tf.Variable(0, trainable=False)
    with tf.device('/gpu:0'):
        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(learning_rate, 0.9, 0.999).minimize(localization_loss, global_step=global_step)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

    # run model
    print('run model...')
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # just uncomment these lines if you want to load pretrained models

        # print('load model...')
        # load_model_path = '/media/liu/6TB/CVPR2019/crossview_localisation/src/ModelonlyRGBOrientation/CVM-NET-I/1/model.ckpt'
        # saver.restore(sess, load_model_path)
        # print("   Model loaded from: %s" % load_model_path)
        # print('load model...FINISHED')

        # Train
        for epoch in range(start_epoch, start_epoch + number_of_epoch):
            iter = 0
            while True:
                batch_sat, batch_grd, batch_sat_ori, batch_grd_ori = input_data.next_pair_batch(batch_size)
                if batch_sat is None:
                    break

                global_step_val = tf.train.global_step(sess, global_step)

                feed_dict = {sat_x: batch_sat, grd_x: batch_grd,
                             sat_x_ori: batch_sat_ori, grd_x_ori: batch_grd_ori,
                             learning_rate: learning_rate_val}
                if iter % 20 == 0:
                    _, loss_val = sess.run([train_step, localization_loss], feed_dict=feed_dict)
                    print('global %d, epoch %d, iter %d: loss : %.4f' %
                          (global_step_val, epoch, iter, loss_val))
                else:
                    sess.run(train_step, feed_dict=feed_dict)


                iter += 1

            # ---------------------- validation ----------------------
            print('validate...')
            print('   compute global descriptors')
            input_data.reset_scan()
            sat_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 512*3])
            grd_global_descriptor = np.zeros([input_data.get_test_dataset_size(), 512*3])
            val_i = 0
            while True:
                print('      progress %d' % val_i)
                batch_sat, batch_grd, batch_sat_ori, batch_grd_ori = input_data.next_batch_scan(batch_size)
                if batch_sat is None:
                    break
                feed_dict = {sat_x: batch_sat, grd_x: batch_grd,
                             sat_x_ori: batch_sat_ori, grd_x_ori: batch_grd_ori}
                sat_global_val, grd_global_val = \
                    sess.run([sat_global, grd_global], feed_dict=feed_dict)

                sat_global_descriptor[val_i: val_i + sat_global_val.shape[0], :] = sat_global_val
                grd_global_descriptor[val_i: val_i + grd_global_val.shape[0], :] = grd_global_val
                val_i += sat_global_val.shape[0]

            print('   compute accuracy')
            val_accuracy = validate(grd_global_descriptor, sat_global_descriptor)

            if not os.path.exists('./Result/'):
                os.makedirs('./Result/')

            with open('./Result/' + '_accuracy_deep6_scratch_m1_1_concat3conv_rgb_ori_gem.txt', 'a') as file:
                file.write(str(epoch) + ' ' + str(iter) + ' : ' + str(val_accuracy) + '\n')
            print('   %d: accuracy = %.1f%%' % (epoch, val_accuracy * 100.0))

            if not os.path.exists('./Modeldeep6_scratch_m1_1_concat3conv_rgb_ori_gem/'):
                os.makedirs('./Modeldeep6_scratch_m1_1_concat3conv_rgb_ori_gem/')

            model_dir = './Modeldeep6_scratch_m1_1_concat3conv_rgb_ori_gem/' + str(epoch) + '/'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            save_path = saver.save(sess, model_dir + 'model.ckpt')
            print("Model saved in file: %s" % save_path)
            # ---------------------------------------------------------


if __name__ == '__main__':
    train()
