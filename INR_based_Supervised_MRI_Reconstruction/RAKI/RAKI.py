import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_v2_behavior()

def weight_variable(shape, vari_name):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial, name=vari_name)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv2d_dilate(x, W, dilate_rate):
    return tf.nn.convolution(x, W, padding='VALID', dilation_rate=[1, dilate_rate])


def learning(ACS_input, target_input, sess, params):
    input_ACS = tf.placeholder(tf.float32, ACS_input.shape)
    input_Target = tf.placeholder(tf.float32, target_input.shape)

    W_conv1 = weight_variable([params['kernel_x_1'], params['kernel_y_1'], ACS_input.shape[-1], params['layer1_channels']], 'W1')
    h_conv1 = tf.nn.relu(conv2d_dilate(input_ACS, W_conv1, params['down_scale']))

    W_conv2 = weight_variable([params['kernel_x_2'], params['kernel_y_2'], params['layer1_channels'], params['layer2_channels']], 'W2')
    h_conv2 = tf.nn.relu(conv2d_dilate(h_conv1, W_conv2, params['down_scale']))

    W_conv3 = weight_variable([params['kernel_last_x'], params['kernel_last_y'], params['layer2_channels'], target_input.shape[-1]], 'W3')
    h_conv3 = conv2d_dilate(h_conv2, W_conv3, params['down_scale'])

    error_norm = tf.norm(input_Target - h_conv3)
    train_step = tf.train.AdamOptimizer(params['LearningRate']).minimize(error_norm)

    if int(tf.__version__.split('.')[1]) < 12 and int(tf.__version__.split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(params['MaxIteration']):

        sess.run(train_step, feed_dict={input_ACS: ACS_input, input_Target: target_input})
        if (i + 1) % 100 == 0:
            error_now = sess.run(error_norm, feed_dict={input_ACS: ACS_input, input_Target: target_input})
            print('The', i + 1, 'th iteration gives an error', error_now)

    error = sess.run(error_norm, feed_dict={input_ACS: ACS_input, input_Target: target_input})
    return [sess.run(W_conv1), sess.run(W_conv2), sess.run(W_conv3), error]


def cnn_3layer(input_kspace, w1, w2, w3, down_scale, sess):
    h_conv1 = tf.nn.relu(conv2d_dilate(input_kspace, w1,down_scale))
    h_conv2 = tf.nn.relu(conv2d_dilate(h_conv1, w2, down_scale))
    h_conv3 = conv2d_dilate(h_conv2, w3, down_scale)
    return sess.run(h_conv3)



def RAKI(params, undersampled_kspace, ACS):

    _, height, width, coil_num = undersampled_kspace.shape
    coil_num //= 2

    if params['load_weight']:
        w = np.load(params['weight_path'])
        w1_all = w['w1_all']
        w2_all = w['w2_all']
        w3_all = w['w3_all']
    else:
        w1_all = np.zeros([params['kernel_x_1'], params['kernel_y_1'], coil_num * 2, params['layer1_channels'], coil_num * 2], dtype=np.float32)
        w2_all = np.zeros([params['kernel_x_2'], params['kernel_y_2'], params['layer1_channels'], params['layer2_channels'], coil_num * 2], dtype=np.float32)
        w3_all = np.zeros([params['kernel_last_x'], params['kernel_last_y'], params['layer2_channels'], params['down_scale'] - 1, coil_num * 2], dtype=np.float32)

        target_x_start = np.int32(np.ceil(params['kernel_x_1'] / 2) + np.floor(params['kernel_x_2'] / 2) + np.floor(params['kernel_last_x'] / 2) - 1)
        target_x_end = ACS.shape[1] - target_x_start
        target_y_start = np.int32((np.ceil(params['kernel_y_1'] / 2) - 1) + (np.ceil(params['kernel_y_2'] / 2) - 1) + (np.ceil(params['kernel_last_y'] / 2) - 1)) * params['down_scale']
        target_y_end = ACS.shape[2] - np.int32((np.floor(params['kernel_y_1'] / 2) + np.floor(params['kernel_y_2'] / 2) + np.floor(params['kernel_last_y'] / 2))) * params['down_scale']
        # ground truth shape
        target_dim_X = target_x_end - target_x_start
        target_dim_Y = target_y_end - target_y_start
        target_dim_Z = params['down_scale'] - 1

        errorSum = 0
        for i in range(coil_num * 2):
            sess = tf.Session()
            # target build
            target = np.zeros([1, target_dim_X, target_dim_Y, target_dim_Z])
            for j in range(params['down_scale'] - 1):
                target_y_start = np.int32((np.ceil(params['kernel_y_1'] / 2) - 1) + (np.ceil(params['kernel_y_2'] / 2) - 1) + (np.ceil(params['kernel_last_y'] / 2) - 1)) * params['down_scale'] + j + 1
                target_y_end = ACS.shape[-2] - np.int32((np.floor(params['kernel_y_1'] / 2) + (np.floor(params['kernel_y_2'] / 2)) + np.floor(params['kernel_last_y'] / 2))) * params['down_scale'] + j + 1
                target[0, :, :, j] = ACS[0, target_x_start:target_x_end, target_y_start:target_y_end, i]
            # learning
            w1, w2, w3, error = learning(ACS, target, sess, params)
            w1_all[:, :, :, :, i] = w1
            w2_all[:, :, :, :, i] = w2
            w3_all[:, :, :, :, i] = w3
            errorSum = errorSum + error

            sess.close()
            tf.reset_default_graph()

        if params['save_weight']:
            np.savez(params['weight_path'], w1_all=w1_all, w2_all=w2_all, w3_all=w3_all)

    # reconstruction
    recon_kspace = undersampled_kspace.copy()
    for i in range(coil_num * 2):

        sess = tf.Session()
        if int(tf.__version__.split('.')[1]) < 12 and int(tf.__version__.split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)

        # grab w and b
        w1 = np.float32(w1_all[:, :, :, :, i])
        w2 = np.float32(w2_all[:, :, :, :, i])
        w3 = np.float32(w3_all[:, :, :, :, i])

        res = cnn_3layer(undersampled_kspace, w1, w2, w3, params['down_scale'], sess)

        target_x_start = np.int32(np.ceil(params['kernel_x_1'] / 2) + np.floor(params['kernel_x_2'] / 2) + np.floor(params['kernel_last_x'] / 2) - 1)
        target_x_end_kspace = undersampled_kspace.shape[1] - target_x_start

        for j in range(params['down_scale'] - 1):
            target_y_start = np.int32((np.ceil(params['kernel_y_1'] / 2) - 1) + np.int32((np.ceil(params['kernel_y_2'] / 2) - 1)) + np.int32(np.ceil(params['kernel_last_y'] / 2) - 1)) * params['down_scale'] + j + 1
            target_y_end_kspace = undersampled_kspace.shape[1] - np.int32((np.floor(params['kernel_y_1'] / 2)) + (np.floor(params['kernel_y_2'] / 2)) + np.floor(params['kernel_last_y'] / 2)) * params['down_scale'] + j + 1
            recon_kspace[0, target_x_start:target_x_end_kspace, target_y_start:target_y_end_kspace:params['down_scale'], i] = res[0, :, ::params['down_scale'], j]

        sess.close()
        tf.reset_default_graph()

    recon_kspace = np.squeeze(recon_kspace)
    idx_lower = int((1 - params['center_ratio']) * undersampled_kspace.shape[-2] / 2)
    idx_upper = int((1 + params['center_ratio']) * undersampled_kspace.shape[-2] / 2)
    recon_kspace[:, idx_lower:idx_upper:, ] = ACS
    recon_kspace = recon_kspace[..., :coil_num] + 1j * recon_kspace[..., coil_num:]

    return recon_kspace