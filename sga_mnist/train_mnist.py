import os
import time
import numpy as np
import tensorflow as tf
from mnist_gan import generator, discriminator
import sys
import tqdm
from tensorflow.python.client import timeline
from tensorflow.contrib.kfac.python.ops.utils import fwd_gradients
from utils import grid_x
from scipy.misc import imsave

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 100, "batch size [100]")
flags.DEFINE_string('data_dir', './data/cifar-10-python', 'data directory')
flags.DEFINE_string('logdir', './log_mnist/000', 'log directory')
flags.DEFINE_integer('labeled', 10, 'labeled image per class[100]')
flags.DEFINE_float('learning_rate_d', 0.003, 'learning_rate dis[0.003]')
flags.DEFINE_float('learning_rate_g', 0.003, 'learning_rate gen[0.003]')
flags.DEFINE_float('ma_decay', 0.9999 , 'moving average [0.9999]')

flags.DEFINE_float('step_print', 1200 , 'scale perturbation')
flags.DEFINE_float('freq_print', 600, 'scale perturbation')
flags.DEFINE_integer('save_im', 600, 'scale perturbation')

flags.DEFINE_integer('seed', 111, 'seed')
flags.DEFINE_integer('seed_data', 111, 'seed data')
flags.DEFINE_integer('seed_tf', 111, 'tf random seed')

flags.DEFINE_float('scale', 0.15 , 'scale perturbation')
flags.DEFINE_boolean('nabla', False , 'enable nabla reg')
flags.DEFINE_float('nabla_w', 0.1 , 'weight nabla reg')
flags.DEFINE_boolean('soft', True , 'enable nabla reg softmaxed')

optimistic = True

def jac_vec(ys, xs, vs):
    return fwd_gradients(ys, xs, grad_xs=vs, stop_gradients=xs)

def jac_tran_vec(ys, xs, vs):
    dydxs = tf.gradients(ys, xs, grad_ys=vs, stop_gradients=xs)
    return [tf.zeros_like(x) if dydx is None else dydx for (x, dydx) in zip(xs, dydxs)]

def get_sym_adj(Ls, xs):
    xi = [tf.gradients(l, x)[0] for (l, x) in zip(Ls, xs)]
    H_xi = jac_vec(xi, xs, xi)
    Ht_xi = jac_tran_vec(xi, xs, xi)
    At_xi = [(ht - h) / 2 for (h, ht) in zip(H_xi, Ht_xi)]
    return At_xi


def display_progression_epoch(j, id_max):
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % epoch' + chr(13))
    _ = sys.stdout.flush


def get_getter(ema):  # to update neural net with moving avg variables, suitable for ss learning cf Saliman
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter


def main(_):
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.lower(), value))
    print("")

    if not os.path.exists(FLAGS.logdir):
        os.mkdir(FLAGS.logdir)

    # Random seed
    rng = np.random.RandomState(FLAGS.seed)  # seed labels
    rng_data = np.random.RandomState(FLAGS.seed_data)  # seed shuffling
    print('loading data')
    # load MNIST data
    data = np.load('./data/mnist.npz')
    trainx = np.concatenate([data['x_train'], data['x_valid']], axis=0).astype(np.float32)
    trainx_unl = trainx.copy()
    trainx_unl2 = trainx.copy()
    trainy = np.concatenate([data['y_train'], data['y_valid']]).astype(np.int32)
    nr_batches_train = int(trainx.shape[0] / FLAGS.batch_size)
    testx = data['x_test'].astype(np.float32)
    testy = data['y_test'].astype(np.int32)
    nr_batches_test = int(testx.shape[0] / FLAGS.batch_size)

    # select labeled data
    inds = rng_data.permutation(trainx.shape[0])
    trainx = trainx[inds]
    trainy = trainy[inds]
    txs = []
    tys = []
    for j in range(10):
        txs.append(trainx[trainy == j][:FLAGS.labeled])
        tys.append(trainy[trainy == j][:FLAGS.labeled])
    txs = np.concatenate(txs, axis=0)
    tys = np.concatenate(tys, axis=0)

    print("Data:") # sanity check input data
    print('train shape %d | batch training %d \ntest shape %d  |  batch  testing %d' \
          % (trainx.shape[0], nr_batches_train, testx.shape[0], nr_batches_test))
    print('histogram train', np.histogram(trainy, bins=10)[0])
    print('histogram test ', np.histogram(testy, bins=10)[0])

    '''construct graph'''
    print('constructing graph')
    inp = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28], name='labeled_data_input_pl')
    unl = tf.placeholder(tf.float32, [FLAGS.batch_size, 28 * 28], name='unlabeled_data_input_pl')
    lbl = tf.placeholder(tf.int32, [FLAGS.batch_size], name='lbl_input_pl')
    noise_pl = tf.placeholder(tf.float32,[5,100],name='noise_pl')
    is_training_pl = tf.placeholder(tf.bool, [], name='is_training_pl')
    acc_train_pl = tf.placeholder(tf.float32, [], 'acc_train_pl')
    acc_test_pl = tf.placeholder(tf.float32, [], 'acc_test_pl')
    acc_test_pl_ema = tf.placeholder(tf.float32, [], 'acc_test_pl')
    kl_weight = tf.placeholder(tf.float32, [], 'kl_weight')

    random_z = tf.random_uniform([FLAGS.batch_size, 100], name='random_z')
    d = tf.random_normal([FLAGS.batch_size, 100], mean=0, stddev=1)

    generator(random_z,is_training_pl,init=True)
    gen_inp = generator(random_z, is_training=is_training_pl,reuse=True)

    discriminator(inp, is_training_pl, init=True)
    logits_real, layer_real = discriminator(unl, is_training_pl,reuse=True)
    logits_fake, layer_fake = discriminator(gen_inp, is_training_pl,reuse=True)

    with tf.name_scope('loss_functions'):
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_real, labels=tf.ones_like(logits_real)) +
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_fake, labels=tf.zeros_like(logits_fake)))

        # accuracy_dis = tf.reduce_mean(tf.cast(tf.less(l_unl, 0), tf.float32))
        # correct_pred = tf.equal(tf.cast(tf.argmax(logits_lab, 1), tf.int32), tf.cast(lbl, tf.int32))
        # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        loss_dis = loss
        loss_gen = -loss

        # fool_rate = tf.reduce_mean(tf.cast(tf.less(l_gen, 0), tf.float32))


    with tf.name_scope('optimizers'):

        # control op dependencies for batch norm and trainable variables
        tvars = tf.trainable_variables()
        dvars = [var for var in tvars if 'discriminator_model' in var.name]
        gvars = [var for var in tvars if 'generator_model' in var.name]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops_gen = [x for x in update_ops if ('generator_model' in x.name)]
        update_ops_dis = [x for x in update_ops if ('discriminator_model' in x.name)]
        
        ################## normal #################
        # optimizer_dis = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_d, beta1=0.5, name='dis_optimizer')
        # optimizer_gen = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_g, beta1=0.5, name='gen_optimizer')
        #
        # with tf.control_dependencies(update_ops_gen):
        #     train_gen_op = optimizer_gen.minimize(loss_gen, var_list=gvars)
        #
        # train_dis_op = optimizer_dis.minimize(loss_dis, var_list=dvars)
        
        ################## sga #################
        # d_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_d, beta1=0.5, name='dis_optimizer')
        # g_opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_g, beta1=0.5, name='gen_optimizer')
        d_opt = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate_d, name='dis_optimizer')
        g_opt = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate_g, name='gen_optimizer')
        with tf.control_dependencies(update_ops_gen):
            gvs = g_opt.compute_gradients(-loss, var_list=gvars)
        dvs = d_opt.compute_gradients(loss, var_list=dvars)
        adj = get_sym_adj([loss] * len(dvars) + [-loss] * len(gvars), dvars + gvars)
        d_adj = adj[:len(dvars)]
        g_adj = adj[-len(gvars)::]
        dvs_sga = [(grad + adj, var) for (grad, var), adj in zip(dvs, d_adj)]
        gvs_sga = [(grad + adj, var) for (grad, var), adj in zip(gvs, g_adj)]
        train_dis_op = d_opt.apply_gradients(dvs_sga)
        train_gen_op = g_opt.apply_gradients(gvs_sga)

        ################## optimistic #################
        d_opt = tf.train.RMSPropOptimizer(learning_rate=1e-4)
        g_opt = tf.train.RMSPropOptimizer(learning_rate=1e-4)

        optimizer = tf.train.RMSPropOptimizer(1e-4, use_locking=True)

        d_grads = tf.gradients(loss, dvars)
        g_grads = tf.gradients(-loss, gvars)

        variables = dvars + gvars
        grads = d_grads + g_grads

        reg = 0.5 * sum(tf.reduce_sum(tf.square(g)) for g in grads)
        # Jacobian times gradient
        Jgrads = tf.gradients(reg, variables)

        apply_vec = [(g + 10. * Jg, v) for (g, Jg, v) in zip(grads, Jgrads, variables) if Jg is not None]
        with tf.control_dependencies([g for (g, v) in apply_vec]):
            train_op = optimizer.apply_gradients(apply_vec)

    with tf.name_scope('summary'):
        with tf.name_scope('discriminator'):
            tf.summary.scalar('loss_discriminator', loss_dis, ['dis'])
            # tf.summary.scalar('cross_entrop', tf.reduce_mean(d_xentropy),['gen'])


        with tf.name_scope('generator'):
            tf.summary.scalar('loss_generator', loss_gen, ['gen'])

        with tf.name_scope('images'):
            tf.summary.image('gen_images', tf.reshape(gen_inp,[-1,28,28,1]),5, ['image'])

        with tf.name_scope('epoch'):
            tf.summary.scalar('accuracy_train', acc_train_pl, ['epoch'])
            tf.summary.scalar('accuracy_test_moving_average', acc_test_pl_ema, ['epoch'])
            tf.summary.scalar('accuracy_test_raw', acc_test_pl, ['epoch'])


        sum_op_dis = tf.summary.merge_all('dis')
        sum_op_gen = tf.summary.merge_all('gen')
        sum_op_im = tf.summary.merge_all('image')
        sum_op_epoch = tf.summary.merge_all('epoch')


    init_gen = [var.initializer for var in gvars][:-3]
    # [print(var.name) for var in gvars]
    saver = tf.train.Saver()
    '''//////perform training //////'''
    print('start training')
    with tf.Session() as sess:
        tf.set_random_seed(rng.randint(2**10))
        sess.run(init_gen)
        init = tf.global_variables_initializer()
        #Data-Dependent Initialization of Parameters as discussed in DP Kingma and Salimans Paper
        sess.run(init, feed_dict={inp: trainx_unl[0:FLAGS.batch_size], is_training_pl: True, kl_weight:0})
        print('initialization done')

        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        train_batch = 0
        for epoch in range(200):
            begin = time.time()
            trainx_unl = trainx_unl[rng.permutation(trainx_unl.shape[0])]  # shuffling unl dataset
            trainx_unl2 = trainx_unl2[rng.permutation(trainx_unl2.shape[0])]

            train_loss_dis, train_loss_unl, train_loss_gen, train_acc, test_acc, test_acc_ma, train_j_loss = [0, 0, 0, 0, 0, 0,0]
            # training
            for t in range(nr_batches_train):
                display_progression_epoch(t, nr_batches_train)

                ran_from = t * FLAGS.batch_size
                ran_to = (t + 1) * FLAGS.batch_size

                # # train discriminator
                # feed_dict = {unl: trainx_unl[ran_from:ran_to],
                #              is_training_pl: True}
                # _,ld,sm = sess.run([train_dis_op, loss_dis,sum_op_dis], feed_dict=feed_dict)
                # train_loss_dis += ld
                # if (train_batch % FLAGS.step_print) == 0:
                #     writer.add_summary(sm, train_batch)
                #
                # # train generator
                # _, lg, sm = sess.run([train_gen_op, loss_gen, sum_op_gen], feed_dict={unl: trainx_unl2[ran_from:ran_to],
                #                                                                       is_training_pl: True,
                #                                                                       noise_pl:np.random.rand(5,100)})
                # train_loss_gen += lg
                # if ((train_batch % FLAGS.step_print) == 0):
                #     writer.add_summary(sm, train_batch)

############################## optimistic #############################3
                # train discriminator
                feed_dict = {unl: trainx_unl[ran_from:ran_to],
                             is_training_pl: True,
                             noise_pl: np.random.rand(5, 100)}
                _,ld,lg,smg,smd = sess.run([train_op, loss_dis, loss_gen, sum_op_gen ,sum_op_dis], feed_dict=feed_dict)
                train_loss_dis += ld
                train_loss_gen += lg

                if (train_batch % FLAGS.step_print) == 0:
                    writer.add_summary(smd, train_batch)
                if ((train_batch % FLAGS.step_print) == 0):
                    writer.add_summary(smg, train_batch)


                if ((train_batch % FLAGS.freq_print) == 0):
                    sm = sess.run(sum_op_im, feed_dict={is_training_pl: False})
                    writer.add_summary(sm, train_batch)

                if ((train_batch % FLAGS.save_im) == 0):
                    X = sess.run(gen_inp,feed_dict={is_training_pl: True,noise_pl:np.random.rand(5,100)})
                    file = 'im'+str(epoch)+'.png'
                    imsave( os.path.join(FLAGS.logdir,file),grid_x(X))
                    print('im saved')


                train_batch += 1
            train_loss_dis /= nr_batches_train

            print("epoch %d | time = %ds | jloss = %0.4f | loss gen = %.4f | loss lab = %.4f"
                  % (epoch, time.time() - begin, train_j_loss, train_loss_gen, train_loss_dis))

        # save_path = saver.save(sess, os.path.join(FLAGS.logdir, 'model_final.ckpt'))
        # print('model saved in %s'%(save_path))

if __name__ == '__main__':
    tf.app.run()

