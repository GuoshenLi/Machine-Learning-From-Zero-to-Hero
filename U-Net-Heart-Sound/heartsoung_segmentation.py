import numpy as np
import tensorflow as tf


def train_data_label_shuffle(Xtrain_normalize, Ytrain_onehot):
    train_num = np.shape(Xtrain_normalize)[0]
    new_order = np.random.permutation(train_num)

    Xtrain_normalize = Xtrain_normalize[new_order]
    Ytrain_onehot = Ytrain_onehot[new_order]

    return Xtrain_normalize, Ytrain_onehot

def get_batch(Xtrain_normalize, Ytrain_onehot, number, batch_size):
    return Xtrain_normalize[number * batch_size:(number + 1) * batch_size], \
           Ytrain_onehot[number * batch_size:(number + 1) * batch_size]

def interpolation(input_tensor, ref_tensor):  # resizes input_tensor wrt. ref_tensor

    H = ref_tensor.get_shape()[1]
    input_tensor = tf.reshape(input_tensor, (-1, input_tensor.get_shape()[1], 1, input_tensor.get_shape()[2]))
    input_tensor = tf.image.resize_nearest_neighbor(input_tensor, [H.value, 1])[:,:,0,:]

    return input_tensor

def FCN_Heart_Segmentation(x, y):
    x = tf.layers.conv1d(x, filters=8, strides = 1, kernel_size = 32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(x, filters=8, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    conv_1 = tf.nn.relu(x)

    pool1 = tf.layers.max_pooling1d(conv_1, pool_size=2, strides=2, padding='same')

    x = tf.layers.conv1d(pool1, filters=16, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(x, filters=16, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    conv_2 = tf.nn.relu(x)

    pool2 = tf.layers.max_pooling1d(conv_2, pool_size=2, strides=2, padding='same')

    x = tf.layers.conv1d(pool2, filters=32, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(x, filters=32, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    conv_3 = tf.nn.relu(x)

    pool3 = tf.layers.max_pooling1d(conv_3, pool_size=2, strides=2, padding='same')

    x = tf.layers.conv1d(pool3, filters=64, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(x, filters=64, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    conv_4 = tf.nn.relu(x)

    pool4 = tf.layers.max_pooling1d(conv_4, pool_size=2, strides=2, padding='same')

    x = tf.layers.conv1d(pool4, filters=128, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(x, filters=128, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    ############## up sampling #####

    up_sample_1 = interpolation(x, ref_tensor=conv_4)
    x = tf.layers.conv1d(up_sample_1, filters=64, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    x = tf.concat((conv_4, x), axis=-1)

    x = tf.layers.conv1d(x, filters=64, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(x, filters=64, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    up_sample_2 = interpolation(x, ref_tensor=conv_3)
    x = tf.layers.conv1d(up_sample_2, filters=32, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    x = tf.concat((conv_3, x), axis=-1)

    x = tf.layers.conv1d(x, filters=32, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(x, filters=32, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    up_sample_3 = interpolation(x, ref_tensor=conv_2)
    x = tf.layers.conv1d(up_sample_3, filters=16, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    x = tf.concat((conv_2, x), axis=-1)

    x = tf.layers.conv1d(x, filters=16, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(x, filters=16, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    up_sample_4 = interpolation(x, ref_tensor=conv_1)
    x = tf.layers.conv1d(up_sample_4, filters=8, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    x = tf.concat((conv_1, x), axis=-1)

    x = tf.layers.conv1d(x, filters=8, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(x, filters=8, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')
    x = tf.nn.relu(x)

    x = tf.layers.conv1d(x, filters=4, strides=1, kernel_size=32, padding='same', kernel_initializer = 'he_normal')

    pred = tf.nn.softmax(x, axis=-1)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=x))


    return pred, loss


if __name__ == '__main__':

    tf.reset_default_graph()
    data = np.load('heartsound_segmentation_256.npz')
    audio_feature = data['audio_feature']
    audio_label = data['audio_label']
    split_count = data['split_count']

    print(audio_label.shape)
    print(audio_label.shape)
    print(split_count.shape)

    split_count = split_count[np.nonzero(split_count)[0]]

    #the 0~379 ahead is training set, the 380 ~ 426 is test set
    all_training_num = np.sum(split_count[:380])
    all_test_num = np.sum(split_count[380:])

    test_count = split_count[380:]
    print(np.sum(test_count))

    x_train = audio_feature[:all_training_num]
    y_train = audio_label[:all_training_num]

    x_test = audio_feature[all_training_num:]
    y_test = audio_label[all_training_num:]

    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    input_x = tf.placeholder(tf.float32, [None, x_train.shape[1], x_train.shape[2]])
    input_y = tf.placeholder(tf.float32, [None, y_train.shape[1], y_train.shape[2]])

    pred, loss = FCN_Heart_Segmentation(input_x, input_y)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    optimizer = tf.train.AdamOptimizer(0.001)
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]


    with tf.control_dependencies(update_ops):
        train = optimizer.apply_gradients(capped_gvs)


    init = tf.global_variables_initializer()

    sess = tf.Session()
    saver = tf.train.Saver()
    checkpoint = tf.train.get_checkpoint_state("saved_networks/FCN")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")



    batch_size = 8
    max_epoch = 50
    total_batch_train = int(x_train.shape[0] / batch_size)

    sess.run(init)
    test_acc_draw_list = []
    train_loss_draw_list = []
    for epoch in range(max_epoch):
        train_acc_batch_list = []
        test_acc_batch_list = []
        loss_batch_list = []

        for batch_num in range(total_batch_train):
            x, y = get_batch(x_train, y_train, batch_num, batch_size)
            (_, train_loss) = sess.run((train, loss), feed_dict={
                input_x: x,
                input_y: y
            })

            loss_batch_list.append(train_loss)
        train_loss_draw_list.append(np.mean(np.array(loss_batch_list)))

        print('train_loss:', train_loss_draw_list)

        correct_all = 0
        length_all = 0
        test_pred = []
        test_pred_all = []
        for i in range(x_test.shape[0]):

            x, y = get_batch(x_test, y_test, i, batch_size=1)
            pred_ = sess.run(pred, feed_dict={
                input_x: x,
                input_y: y
            })

            test_pred_all.append(pred_)
        test_pred_all = np.array(test_pred_all)

        offset = 0
        for i in range(test_count.shape[0]):

            single_pred = test_pred_all[offset:offset + test_count[i]]
            single_pred = single_pred.reshape(-1, 4)

            single_y_test = y_test[offset:offset + test_count[i]]
            single_y_test = single_y_test.reshape(-1, 4)

            offset += test_count[i]

            correct_all += np.sum(np.equal(np.argmax(single_pred, axis = -1), np.argmax(single_y_test, axis = -1)).astype(np.float32))
            length_all += np.shape(np.argmax(single_y_test, axis = -1))[0]


        print('test_acc', correct_all / length_all)

        print('finish ', epoch + 1, ' epoch!')
        saver.save(sess, "saved_networks/FCN", global_step = epoch + 1)
        x_train, y_train = train_data_label_shuffle(x_train, y_train)


    # pred_ = sess.run(pred, feed_dict={
    #     input_x: x_test[0,:,:].reshape(-1,128,4),
    # })
    #

    #
    # GT_ = np.argmax(y_test[0,:,:].reshape(-1,128,4), axis = -1).ravel()
    #
    # plt.plot(pred_[0,:])
    # plt.show()







