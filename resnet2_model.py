import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import glob
import sys
import tensorflow as tf
from datetime import datetime

def build_resnet2_model():
    with tf.device('/gpu:0'):

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        with tf.name_scope('input') as scope:
            X = tf.placeholder(tf.float32, shape = (None, 3072))
            y = tf.placeholder(tf.int32, shape = (None))
    #         dropout_rate = tf.placeholder(tf.float32, shape=())
            training = tf.placeholder(tf.bool)
            input_layer = tf.reshape(X, [-1, 32, 32, 3])

        with tf.name_scope('conv1') as scope:
            conv1 = tf.layers.conv2d(inputs=input_layer, filters=64, kernel_size=3, strides=1, 
                                     padding = 'SAME', activation = tf.nn.relu, name='conv1',
                                     kernel_regularizer = regularizer)
            batch_norm1 = tf.layers.batch_normalization(conv1, axis = 1, name='batch_norm1')
    #         pool1 = tf.layers.max_pooling2d(inputs=batch_norm1, pool_size=2, strides=2, padding='SAME', name='pool1')
            dropout1 = tf.layers.dropout(batch_norm1, rate=0.3, training=training, name='dropout1')


        with tf.name_scope('conv2') as scope:
            conv2 = tf.layers.conv2d(inputs=dropout1, filters=64, kernel_size=3, strides=1, 
                                    padding = 'SAME', activation = tf.nn.relu, name='conv2',
                                    kernel_regularizer = regularizer)
            batch_norm2 = tf.layers.batch_normalization(conv2, axis = 1, name='batch_norm2')
            pool2 = tf.layers.max_pooling2d(inputs=batch_norm2, pool_size=2, strides=2, padding='SAME', name='pool2')

        with tf.name_scope('conv3') as scope:
            conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=3, strides=1, 
                                    padding = 'SAME', activation = tf.nn.relu, name='conv3',
                                    kernel_regularizer = regularizer)
            batch_norm3 = tf.layers.batch_normalization(conv3, axis = 1, name='batch_norm3')
            dropout3 = tf.layers.dropout(batch_norm3, rate=0.4, training=training, name='dropout3')

    #         pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=2, strides=2, padding='SAME', name='pool3')

        with tf.name_scope('conv4') as scope:
            conv4 = tf.layers.conv2d(inputs=dropout3, filters=128, kernel_size=3, strides=1, 
                                    padding = 'SAME', activation = tf.nn.relu, name='conv4',
                                    kernel_regularizer = regularizer)
            batch_norm4 = tf.layers.batch_normalization(conv4, axis = 1, name='batch_norm4')
            pool4 = tf.layers.max_pooling2d(inputs=batch_norm4, pool_size=2, strides=2, padding='SAME', name='pool4')

        with tf.name_scope('conv5') as scope:
            conv5 = tf.layers.conv2d(inputs=pool4, filters=256, kernel_size=3, strides=1, 
                                    padding = 'SAME', activation = tf.nn.relu, name='conv5',
                                    kernel_regularizer = regularizer)
            batch_norm5 = tf.layers.batch_normalization(conv5, axis = 1, name='batch_norm5')
            dropout5 = tf.layers.dropout(batch_norm5, rate=0.4, training=training, name='dropout5')

        with tf.name_scope('conv6') as scope:
            conv6 = tf.layers.conv2d(inputs=dropout5, filters=256, kernel_size=3, strides=1, 
                                    padding = 'SAME', activation = tf.nn.relu, name='conv6',
                                    kernel_regularizer = regularizer)
            batch_norm6 = tf.layers.batch_normalization(conv6, axis = 1, name='batch_norm6')
            dropout6 = tf.layers.dropout(batch_norm6, rate=0.4, training=training, name='dropout6')

        with tf.name_scope('conv7') as scope:
            conv7 = tf.layers.conv2d(inputs=dropout6, filters=256, kernel_size=3, strides=1, 
                                    padding = 'SAME', activation = tf.nn.relu, name='conv7',
                                    kernel_regularizer = regularizer)
            batch_norm7 = tf.layers.batch_normalization(conv7, axis = 1, name='batch_norm7')
            pool7 = tf.layers.max_pooling2d(inputs=batch_norm7, pool_size=2, strides=2, padding='SAME', name='pool7')

        with tf.name_scope('conv8') as scope:
            conv8 = tf.layers.conv2d(inputs=pool7, filters=512, kernel_size=3, strides=1, 
                                    padding = 'SAME', activation = tf.nn.relu, name='conv8',
                                    kernel_regularizer = regularizer)
            batch_norm8 = tf.layers.batch_normalization(conv8, axis = 1, name='batch_norm8')
            dropout8 = tf.layers.dropout(batch_norm8, rate=0.4, training=training, name='dropout8')

        with tf.name_scope('conv9') as scope:
            conv9 = tf.layers.conv2d(inputs=dropout8, filters=512, kernel_size=3, strides=1, 
                                    padding = 'SAME', activation = tf.nn.relu, name='conv9',
                                    kernel_regularizer = regularizer)
            batch_norm9 = tf.layers.batch_normalization(conv9, axis = 1, name='batch_norm9')
            dropout9 = tf.layers.dropout(batch_norm9, rate=0.4, training=training, name='dropout9')

        with tf.name_scope('conv10') as scope:
            conv10 = tf.layers.conv2d(inputs=dropout9, filters=512, kernel_size=3, strides=1, 
                                    padding = 'SAME', activation = tf.nn.relu, name='conv10',
                                    kernel_regularizer = regularizer)
            batch_norm10 = tf.layers.batch_normalization(conv10, axis = 1, name='batch_norm10')
            pool10 = tf.layers.max_pooling2d(inputs=batch_norm10, pool_size=2, strides=2, padding='SAME', name='pool10')

        with tf.name_scope('conv11') as scope:
            conv11 = tf.layers.conv2d(inputs=pool10, filters=512, kernel_size=3, strides=1, 
                                    padding = 'SAME', activation = tf.nn.relu, name='conv11',
                                    kernel_regularizer = regularizer)
            batch_norm11 = tf.layers.batch_normalization(conv11, axis = 1, name='batch_norm11')
            dropout11 = tf.layers.dropout(batch_norm11, rate=0.4, training=training, name='dropout11')

        with tf.name_scope('conv12') as scope:
            conv12 = tf.layers.conv2d(inputs=dropout11, filters=512, kernel_size=3, strides=1, 
                                    padding = 'SAME', activation = tf.nn.relu, name='conv12',
                                    kernel_regularizer = regularizer)
            batch_norm12 = tf.layers.batch_normalization(conv12, axis = 1, name='batch_norm12')
            dropout12 = tf.layers.dropout(batch_norm12, rate=0.4, training=training, name='dropout12')

        with tf.name_scope('conv13') as scope:
            conv13 = tf.layers.conv2d(inputs=dropout12, filters=512, kernel_size=3, strides=1, 
                                    padding = 'SAME', activation = tf.nn.relu, name='conv13',
                                    kernel_regularizer = regularizer)
            batch_norm13 = tf.layers.batch_normalization(conv13, axis = 1, name='batch_norm13')
            pool13 = tf.layers.max_pooling2d(inputs=batch_norm13, pool_size=2, strides=2, padding='SAME', name='pool13')

            dropout13 = tf.layers.dropout(pool13, rate=0.5, training=training, name='dropout13')



        with tf.name_scope('fc14') as scope:
            pool13_flat = tf.reshape(dropout13, [-1, 512], name='pool13_flat')
    #         dropout1 = tf.layers.dropout(pool3_flat, rate=dropout_rate, training=training, name='dropout1')
    #         dense1 = tf.layers.dense(pool3_flat, units = 8*128, activation = tf.nn.relu, name='dense1')
    #         batch_norm4 = tf.layers.batch_normalization(dense1, axis = 1, name='batch_norm4')
    #         dropout2 = tf.layers.dropout(dense1, rate=dropout_rate, training=training, name='dropout2')
            dense14 = tf.layers.dense(pool13_flat, units = 512, activation = tf.nn.relu, name='dense14',
                                     kernel_regularizer = regularizer)
    #         dropout3 = tf.layers.dropout(dense2, rate=dropout_rate, training=training, name='dropout3')
            batch_norm14 = tf.layers.batch_normalization(dense14, axis = 1, name='batch_norm14')
            dropout14 = tf.layers.dropout(batch_norm14, rate=0.5, training=training, name='dropout14')

        with tf.name_scope('logits') as scope:
            logits = tf.layers.dense(dropout14, units = 10, name='logits')

        with tf.name_scope('loss') as scope:
            softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name = 'softmax')
            loss = tf.reduce_mean(softmax) 

        with tf.name_scope('train') as scope:
            optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        with tf.name_scope('eval') as scope:
            correct = tf.nn.in_top_k(logits, y, 1)
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        with tf.name_scope('summary') as scope:
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(logdir + '/train', tf.get_default_graph())
            test_writer = tf.summary.FileWriter(logdir + '/test', tf.get_default_graph())
            
    return optimizer, merged, loss, accuracy, test_writer, logits
    
def save_variable(var, name):
    with tf.name_scope(name) as scope:
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(var - mean))
        tf.summary.scalar('stddev', stddev)
        tf.summary.histogram('histgram', var)
    
    
def train_resnet2(X_train, y_train, X_validation, y_validation, iteration, X_test, model_list):

    optimizer, merged, loss, accuracy, test_writer, logits = model_list
    
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    init = tf.global_variables_initializer()
    
    with tf.Session(config=config) as sess:
        sess.run(init)
        for step in range(iteration):
            X_batch, y_batch = get_batch(X_train, y_train, batch_size)
            sess.run(optimizer, feed_dict={X: X_batch, y:y_batch, training:True})
                
            if step % 100 == 0:
                summary, loss_, acc = sess.run([merged, loss, accuracy],
                                               feed_dict={X: X_validation, y:y_validation, training:False})
                test_writer.add_summary(summary, step)
                print('###################################')
                print('validation! after '+ str(step) + ' iterations' + 
                              ' the loss is ' + str(loss_) + ', the accuracy is ' + str(acc))
                        
                summary, loss_, acc = sess.run([merged, loss, accuracy], 
                                               feed_dict={X: X_batch, y:y_batch, training:False})
                train_writer.add_summary(summary, step)
                        
                print('training! after '+ str(step) + ' iterations' + 
                          ' the loss is ' + str(loss_) + ', the accuracy is ' + str(acc))
            
        y_labels = logits.eval(feed_dict={X: X_test, training:False})
        return y_labels.T

