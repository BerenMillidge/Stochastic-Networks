import os
import time
import shutil
from datetime import timedelta

import numpy as np
import tensorflow as tf

try:
    from ... import infrastructure


TF_VERSION = float('.'.join(tf.__version__.split('.')[:2]))


class DenseNet:
    def __init__(self, data_provider, growth_rate, depth,
                 total_blocks, keep_prob,
                 weight_decay, nesterov_momentum, model_type, dataset,
                 should_save_logs, should_save_model,
                 renew_logs=False,
                 reduction=1.0,
                 bc_mode=False, email_after = 20,
                 **kwargs):
        """
        Class to implement networks from this paper
        https://arxiv.org/pdf/1611.05552.pdf

        Args:
            data_provider: Class, that have all required data sets
            growth_rate: `int`, variable from paper
            depth: `int`, variable from paper
            total_blocks: `int`, paper value == 3
            keep_prob: `float`, keep probability for dropout. If keep_prob = 1
                dropout will be disables
            weight_decay: `float`, weight decay for L2 loss, paper = 1e-4
            nesterov_momentum: `float`, momentum for Nesterov optimizer
            model_type: `str`, 'DenseNet' or 'DenseNet-BC'. Should model use
                bottle neck connections or not.
            dataset: `str`, dataset name
            should_save_logs: `bool`, should logs be saved or not
            should_save_model: `bool`, should model be saved or not
            renew_logs: `bool`, remove previous logs for current model
            reduction: `float`, reduction Theta at transition layer for
                DenseNets with bottleneck layers. See paragraph 'Compression'
                https://arxiv.org/pdf/1608.06993v3.pdf#4
            bc_mode: `bool`, should we use bottleneck layers and features
                reduction or not.
        """
        self.data_provider = data_provider
        self.batch_size = self.data_provider.batch_size
        self.data_shape = data_provider.data_shape
        self.n_classes = data_provider.n_classes
        self.depth = depth
        self.growth_rate = growth_rate
        # how many features will be received after first convolution
        # value the same as in the original Torch code
        self.first_output_features = growth_rate * 2
        self.total_blocks = total_blocks
        self.layers_per_block = (depth - (total_blocks + 1)) // total_blocks
        self.bc_mode = bc_mode
        self.email_after = email_after
        # compression rate at the transition layers
        self.reduction = reduction
        if not bc_mode:
            print("Build %s model with %d blocks, "
                  "%d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block))
        if bc_mode:
            self.layers_per_block = self.layers_per_block // 2
            print("Build %s model with %d blocks, "
                  "%d bottleneck layers and %d composite layers each." % (
                      model_type, self.total_blocks, self.layers_per_block,
                      self.layers_per_block))
        print("Reduction at transition layers: %.1f" % self.reduction)

        self.keep_prob = keep_prob
        self.weight_decay = weight_decay
        self.nesterov_momentum = nesterov_momentum
        self.model_type = model_type
        self.dataset_name = dataset
        self.should_save_logs = should_save_logs
        self.should_save_model = should_save_model
        self.renew_logs = renew_logs
        self.batches_step = 0
        self.is_training = tf.constant(True, dtype=tf.bool)
        self.alpha = 0.05
        self.beta = 0.1
        self.zeta = 0.01
        self.use_sdr = kwargs['use_sdr']
        self._define_inputs()
#        self._build_graph()
#        self._initialize_session()
#        self._count_trainable_params()

    def _initialize_session(self):
        """Initialize session, variables, saver"""
        config = tf.ConfigProto()
        # restrict model GPU memory utilization to min required
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        tf_ver = int(tf.__version__.split('.')[1])

        with tf.variable_scope("summaries"):
    
            if TF_VERSION <= 0.10:
                self.sess.run(tf.initialize_all_variables())
                logswriter = tf.train.SummaryWriter
            else:
                self.sess.run(tf.global_variables_initializer())
                logswriter = tf.summary.FileWriter
            self.saver = tf.train.Saver()
            self.summary_writer = logswriter(self.logs_path)
            self.summary_writer.add_graph(self.sess.graph)

    def _count_trainable_params(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            total_parameters += variable_parametes
        print("Total training params: %.1fM" % (total_parameters / 1e6))

    @property
    def save_path(self):
        try:
            save_path = self._save_path
        except AttributeError:
            save_path = 'saves/%s' % self.model_identifier
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, 'model.chkpt')
            self._save_path = save_path
        return save_path

    @property
    def logs_path(self):
        try:
            logs_path = self._logs_path
        except AttributeError:
            current_time = time.strftime("%Y_%m_%d_%H%M%S", time.gmtime())
            logs_path = 'logs/%s/%s' % (self.model_identifier, current_time)
            if self.renew_logs:
                shutil.rmtree(logs_path, ignore_errors=True)
            os.makedirs(logs_path, exist_ok=True)
            self._logs_path = logs_path
        return logs_path

    @property
    def model_identifier(self):
        if self.use_sdr:
            return "{}-SDR_growth_rate={}_depth={}_dataset_{}".format(
            self.model_type, self.growth_rate, self.depth, self.dataset_name)
        else:
            return "{}_growth_rate={}_depth={}_dataset_{}".format(
            self.model_type, self.growth_rate, self.depth, self.dataset_name)

    def save_model(self, global_step=None):
        self.saver.save(self.sess, self.save_path, global_step=global_step)

    def load_model(self):
        try:
            self.saver.restore(self.sess, self.save_path)
        except Exception as e:
            raise IOError("Failed to to load model "
                          "from save path: %s" % self.save_path)
        self.saver.restore(self.sess, self.save_path)
        print("Successfully load model from save path: %s" % self.save_path)

    def log_loss_accuracy(self, loss, accuracy, epoch, prefix,
                          should_print=True):
        with tf.variable_scope("summaries"):

            if should_print:
                print("mean cross_entropy: %f, mean accuracy: %f" % (
                    loss, accuracy))
            summary = tf.Summary(value=[
                tf.Summary.Value(
                    tag='loss_%s' % prefix, simple_value=float(loss)),
                tf.Summary.Value(
                    tag='accuracy_%s' % prefix, simple_value=float(accuracy))
            ])
            self.summary_writer.add_summary(summary, epoch)

    def add_histograms(self, tensor):
        with tf.variable_scope("summaries"):
            tf.summary.histogram(tensor.name + "_hist", tensor)

    def _define_inputs(self):
        shape = [self.batch_size]
        shape.extend(self.data_shape)

        self.images = tf.get_variable('input_images',
            shape=shape, initializer=tf.zeros_initializer(dtype=tf.float32))

        #self.images = tf.placeholder(
        #    tf.float32,
        #    shape=shape,
        #    name='input_images')
        #self.labels = tf.placeholder(
        #    tf.float32,
        #    shape=[None, self.n_classes],
        #    name='labels')
        #labels_zeros = tf.zeros([None, self.n_classes], dtype=tf.float32)
        self.labels = tf.get_variable('labels',
            shape=[self.batch_size, self.n_classes],
            initializer=tf.zeros_initializer(dtype=tf.float32))

        self.learning_rate = tf.constant(0.1, dtype=tf.float32)
        #self.learning_rate = tf.placeholder(
        #    tf.float32,
        #    shape=[],
        #    name='learning_rate')
        #self.is_training = tf.placeholder(tf.bool, shape=[])
        #tf.assign(self.is_training, True)

    def composite_function(self, _input, out_features, kernel_size=3, stochastic=False, sigma=2):
        """Function from paper H_l that performs:
        - batch normalization
        - ReLU nonlinearity
        - convolution with required kernel
        - dropout, if required
        """
        with tf.variable_scope("composite_function"):
            # BN
            output = self.batch_norm(_input)
            # ReLU
            output = tf.nn.relu(output)
            # convolution
            output = self.conv2d(
                output, out_features=out_features, kernel_size=kernel_size)
            # dropout(in case of training and in case it is no 1.0)
            output = self.dropout(output)
            if stochastic:
                # add stochasticity here... not totally sure what shapes should be
                # this should hopefully be enough for all reasonable sorts of thigns... but I don't know!
                # could perhaps add stochasticity into the bottleneck just to see what is happening!
                rand = tf.random_normal(output.get_shape(), tf.constant(0, tf.float32), tf.reduce_mean(output) * tf.constant(sigma, tf.float32))
                output += rand
        return output

    def bottleneck(self, _input, out_features, stochastic = False, sigma=2):
        with tf.variable_scope("bottleneck"):
            output = self.batch_norm(_input)
            output = tf.nn.relu(output)
            inter_features = out_features * 4
            output = self.conv2d(
                output, out_features=inter_features, kernel_size=1,
                padding='VALID')
            if stochastic:
                rand = tf.random_normal(output.get_shape(), tf.constant(0, tf.float32), tf.reduce_mean(output) * tf.constant(sigma, tf.float32))
                output += rand
            output = self.dropout(output)
        return output

    def add_internal_layer(self, _input, growth_rate):
        """Perform H_l composite function for the layer and after concatenate
        input with output from composite function.
        """
        # call composite function with 3x3 kernel
        if not self.bc_mode:
            comp_out = self.composite_function(
                _input, out_features=growth_rate, kernel_size=3)
        elif self.bc_mode:
            bottleneck_out = self.bottleneck(_input, out_features=growth_rate)
            comp_out = self.composite_function(
                bottleneck_out, out_features=growth_rate, kernel_size=3)
        # concatenate _input with out from composite function
        if TF_VERSION >= 1.0:
            output = tf.concat(axis=3, values=(_input, comp_out))
        else:
            output = tf.concat(3, (_input, comp_out))
        return output

    def add_block(self, _input, growth_rate, layers_per_block):
        """Add N H_l internal layers"""
        output = _input
        for layer in range(layers_per_block):
            with tf.variable_scope("layer_%d" % layer):
                output = self.add_internal_layer(output, growth_rate)
        return output

    def transition_layer(self, _input, stochastic=False, sigma=2):
        """Call H_l composite function with 1x1 kernel and after average
        pooling
        """
        # call composite function with 1x1 kernel
        out_features = int(int(_input.get_shape()[-1]) * self.reduction)
        output = self.composite_function(
            _input, out_features=out_features, kernel_size=1)
        # run average pooling
        output = self.avg_pool(output, k=2)
        if stochastic:
                rand = tf.random_normal(output.get_shape(), tf.constant(0, tf.float32), tf.reduce_mean(output) * tf.constant(sigma, tf.float32))
                output += rand
        # hopefully this will work... but who knows?
        # should be sufficient to achieve both of the things I want, though not ridiculous!
        # at least should be easy to run and test now... which is fantastic!
        # will be easy to apply them and can run fairly straightforwardly so that is nice!
        return output

    def transition_layer_to_classes(self, _input):
        """This is last transition to get probabilities by classes. It perform:
        - batch normalization
        - ReLU nonlinearity
        - wide average pooling
        - FC layer multiplication
        """
        # BN
        output = self.batch_norm(_input)
        # ReLU
        output = tf.nn.relu(output)
        # average pooling
        last_pool_kernel = int(output.get_shape()[-2])
        output = self.avg_pool(output, k=last_pool_kernel)
        # FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        W = self.weight_variable_xavier(
            [features_total, self.n_classes], name='W')
        bias = self.bias_variable([self.n_classes])
        logits = tf.matmul(output, W) + bias
        return logits

    def conv2d(self, _input, out_features, kernel_size,
               strides=[1, 1, 1, 1], padding='SAME'):
        in_features = int(_input.get_shape()[-1])
        kernel = self.weight_variable_msra(
            [kernel_size, kernel_size, in_features, out_features],
            name='kernel')
        output = tf.nn.conv2d(_input, kernel, strides, padding)
        return output

    def avg_pool(self, _input, k):
        ksize = [1, k, k, 1]
        strides = [1, k, k, 1]
        padding = 'VALID'
        output = tf.nn.avg_pool(_input, ksize, strides, padding)
        return output

    def batch_norm(self, _input):
        output = tf.contrib.layers.batch_norm(
            _input, scale=True, is_training=self.is_training,
            updates_collections=None)
        return output

    def dropout(self, _input):
        if self.keep_prob < 1:
            output = tf.cond(
                self.is_training,
                lambda: tf.nn.dropout(_input, self.keep_prob),
                lambda: _input
            )
        else:
            output = _input
        return output

    def weight_variable_msra(self, shape, name):
        return tf.get_variable(
            name=name,
            shape=shape,
            initializer=tf.contrib.layers.variance_scaling_initializer())

    def weight_variable_xavier(self, shape, name):
        return tf.get_variable(
            name,
            shape=shape,
            initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name='bias'):
        initial = tf.constant(0.0, shape=shape)
        return tf.get_variable(name, initializer=initial)

    def _build_graph(self, batch_size):
        images, labels = self.input_pipeline(batch_size, self.data_provider.train)
        self.images = tf.cast(images, tf.float32)
        #self.sess.run(self.images)
        self.labels = tf.cast(labels, tf.float32)
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block
        # first - initial 3 x 3 conv to first_output_features
        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                self.images,
                out_features=self.first_output_features,
                kernel_size=3)

        # add N required blocks
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self.add_block(output, growth_rate, layers_per_block)
            # last block exist without transition layer
            if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self.transition_layer(output)

        with tf.variable_scope("Transition_to_classes"):
            logits = self.transition_layer_to_classes(output)
        prediction = tf.nn.softmax(logits)



        # Losses
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels))
        self.cross_entropy = cross_entropy
        l2_loss = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()])

        # optimizer and train step
        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        self.optimizer = optimizer
        self.train_step = self.optimizer.minimize(
            cross_entropy + l2_loss * self.weight_decay)

        correct_prediction = tf.equal(
            tf.argmax(prediction, 1),
            tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def _build_graph_sdr(self, batch_size):
        images, labels = self.input_pipeline(batch_size, self.data_provider.train)
        self.images = tf.cast(images, tf.float32)
        #self.sess.run(self.images)
        self.labels = tf.cast(labels, tf.float32)
        growth_rate = self.growth_rate
        layers_per_block = self.layers_per_block
        # first - initial 3 x 3 conv to first_output_features
        with tf.variable_scope("Initial_convolution"):
            output = self.conv2d(
                self.images,
                out_features=self.first_output_features,
                kernel_size=3)
            tf.summary.histogram('pre_gradients_weights_' + output.name, output)
        # add N required blocks
        for block in range(self.total_blocks):
            with tf.variable_scope("Block_%d" % block):
                output = self.add_block(output, growth_rate, layers_per_block)
                tf.summary.histogram('pre_gradients_weights_' +
                    output.name + str(block), output)
            # last block exist without transition layer
            if block != self.total_blocks - 1:
                with tf.variable_scope("Transition_after_block_%d" % block):
                    output = self.transition_layer(output)
                    tf.summary.histogram('pre_gradients_weights_' +
                        output.name + str(block), output)

        with tf.variable_scope("Transition_to_classes"):
            logits = self.transition_layer_to_classes(output)
            tf.summary.histogram('pre_gradients_weights_' + logits.name, logits)
            self.add_histograms(logits)
            
        prediction = tf.nn.softmax(logits)

        tst = 0
        with tf.variable_scope("means_sd") as m_sd:
            sds_ = [tf.get_variable("sds_" + str(k),
                initializer=tf.random_uniform(v.shape, minval=0, maxval=0), trainable=False)
                for k, v in enumerate(tf.trainable_variables())]
            #trains = [v for v in tf.trainable_variables()]
            self.apply_ = []
            for k in range(len(tf.trainable_variables())):
                dist = tf.distributions.Normal(
                    loc=tf.trainable_variables()[k], scale=sds_[k])
                new_trainable = tf.reshape(dist.sample([1]),
                    tf.trainable_variables()[k].shape)
                
                self.apply_.append(tf.assign(tf.trainable_variables()[k],
                    new_trainable))

        # Losses
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=self.labels))
        self.cross_entropy = cross_entropy

        l2_loss = tf.add_n(
            [tf.nn.l2_loss(var) for var in tf.trainable_variables()])


        optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self.nesterov_momentum, use_nesterov=True)
        self.optimizer = optimizer
        
        grads_and_vars = self.optimizer.compute_gradients(
            cross_entropy + l2_loss * self.weight_decay)


        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
          raise ValueError(
            "No gradients provided for any variable, check your graph for ops"
            " that do not support gradients, between variables %s and loss %s." %
            ([str(v) for _, v in grads_and_vars], loss))

        self.sd_asn = []
        with tf.variable_scope(m_sd.original_name_scope):
            for k, (g, v) in enumerate(grads_and_vars):
    
                sd_tmp = tf.multiply(tf.constant(
                    self.zeta, dtype=tf.float32), tf.add(
                    tf.abs(tf.multiply(tf.constant(self.beta,
                    dtype=tf.float32), g)), sds_[k]))
                self.sd_asn.append(tf.assign(sds_[k], sd_tmp))

            for var in tf.trainable_variables():
                tf.summary.histogram('post_sdr_weights_' + var.name, var)
            
        self.train_step = self.optimizer.apply_gradients(
            grads_and_vars)

        self.summaries = tf.summary.merge_all()

        correct_prediction = tf.equal(
            tf.argmax(prediction, 1),
            tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        

    def input_pipeline(self, batch_size, data, test=False):
    
        inputs, labels = data.images, data.labels
        # min_after_dequeue defines how big a buffer we will randomly sample
        #   from -- bigger means better shuffling but slower start up and more
        #   memory used.
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
        min_after_dequeue = 1000
        capacity = min_after_dequeue + 3 * batch_size
        
        if test:
            example_batch, label_batch = tf.train.batch(
                [inputs, labels], batch_size=batch_size, capacity=capacity,
                enqueue_many=True)
        else:
            example_batch, label_batch = tf.train.shuffle_batch(
                [inputs, labels], batch_size=batch_size, num_threads=3, capacity=capacity,
                min_after_dequeue=min_after_dequeue, enqueue_many=True)
        return example_batch, label_batch

    def train_all_epochs(self, train_params):
        n_epochs = train_params['n_epochs']
        learning_rate = train_params['initial_learning_rate']
        batch_size = train_params['batch_size']
        reduce_lr_epoch_1 = train_params['reduce_lr_epoch_1']
        reduce_lr_epoch_2 = train_params['reduce_lr_epoch_2']
        total_start_time = time.time()
        if self.use_sdr:
            self._build_graph_sdr(batch_size)
        else:
            self._build_graph(batch_size)
        self._initialize_session()
        self._count_trainable_params()

        for epoch in range(1, n_epochs + 1):
            print("\n", '-' * 30, "Train epoch: %d" % epoch, '-' * 30, '\n')
            start_time = time.time()
            if epoch == reduce_lr_epoch_1 or epoch == reduce_lr_epoch_2:
                learning_rate = learning_rate / 10
                print("Decrease learning rate, new lr = %f" % learning_rate)

            print("Training...")
            if self.use_sdr:
                loss, acc = self.train_one_epoch_sdr(
                    self.data_provider.train, batch_size, learning_rate)
            else:
                loss, acc = self.train_one_epoch(
                    self.data_provider.train, batch_size, learning_rate)

            if self.should_save_logs:
                self.log_loss_accuracy(loss, acc, epoch, prefix='train')

            if train_params.get('validation_set', False):
                print("Validation...")
                loss, acc = self.test(
                    self.data_provider.validation, batch_size)
                if self.should_save_logs:
                    self.log_loss_accuracy(loss, acc, epoch, prefix='valid')

            time_per_epoch = time.time() - start_time
            seconds_left = int((n_epochs - epoch) * time_per_epoch)
            print("Time per epoch: %s, Est. complete in: %s" % (
                str(timedelta(seconds=time_per_epoch)),
                str(timedelta(seconds=seconds_left))))

            if self.should_save_model:
                self.save_model()

            # send mail with the logs add this here
            if epoch % self.email_after == 0:
                logs = {
                    "loss": loss,
                    "acc": acc,
                    "epoch:", epoch
                }
                infrastructure.send_mail("Stochastic nets training logs:" , infrastructure.format_results_log(logs))
                if np.isnan(loss) or np.isnan(acc):
                    raise ValueError('Nans detected in loss or accuracy.')


        total_training_time = time.time() - total_start_time
        print("\nTotal training time: %s" % str(timedelta(
            seconds=total_training_time)))

    def train_one_epoch(self, data, batch_size, learning_rate, stochastic_learning_rate=False, lr_sigma = 5):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        num_examples = data.num_examples
        self.learning_rate = learning_rate
        if stochastic_learning_rate:
            #assume it's just an arbitrary number
            self.learning_rate = np.random.normal(loc=0, size = learning_rate * lr_sigma)
        total_loss = []
        total_accuracy = []
        self.is_training = tf.constant(True, dtype=tf.bool)
        
        for i in range(num_examples // batch_size):
            result = self.sess.run([self.train_step, self.cross_entropy, self.accuracy])
            _, loss, accuracy = result
            print("Iteration %d: loss=%f, accuracy=%f" % (i, loss, accuracy))
            total_loss.append(loss)
            total_accuracy.append(accuracy)
            if self.should_save_logs:
                self.batches_step += 1
                self.log_loss_accuracy(
                    loss, accuracy, self.batches_step, prefix='per_batch',
                    should_print=False)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy


    def train_one_epoch_sdr(self, data, batch_size, learning_rate, stochastic_learning_rate=False, lr_sigma=5):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        num_examples = data.num_examples
        self.learning_rate = learning_rate
        if stochastic_learning_rate:
            #assume it's just an arbitrary number
            self.learning_rate = np.random.normal(loc=0, size = learning_rate * lr_sigma)
        total_loss = []
        total_accuracy = []
        self.is_training = tf.constant(True, dtype=tf.bool)
        
        for i in range(num_examples // batch_size):
            
            sess_list1 = self.apply_ + self.sd_asn + [self.train_step] #+ self.equal
            sess_list2 = [self.cross_entropy, self.accuracy]
            result1 = self.sess.run(sess_list1)
            result2 = self.sess.run(sess_list2)
            #record histograms, etc. every epoch
            if i % (num_examples // batch_size) - 1 == 0:
                summ_ = self.sess.run(self.summaries)
                self.summary_writer.add_summary(summ_)
            loss, accuracy = result2
            print("Iteration %d: loss=%f, accuracy=%f" % (i, loss, accuracy))
            total_loss.append(loss)
            total_accuracy.append(accuracy)
            if self.should_save_logs:
                self.batches_step += 1
                self.log_loss_accuracy(
                    loss, accuracy, self.batches_step, prefix='per_batch',
                    should_print=False)

        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        return mean_loss, mean_accuracy


    def test(self, data, batch_size):
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        num_examples = data.num_examples
        print("Length of set: " + str(num_examples))
        total_loss = []
        total_accuracy = []

        images, labels = self.input_pipeline(batch_size,
            data, test=True)
        self.is_training = tf.constant(False, dtype=tf.bool)
        train_images = self.images
        train_labels = self.labels
        self.images = tf.cast(images, tf.float32)
        self.labels = tf.cast(labels, tf.float32)
        for i in range(num_examples // batch_size):
            loss, accuracy = self.sess.run([self.cross_entropy, self.accuracy])
            total_loss.append(loss)
            total_accuracy.append(accuracy)
        mean_loss = np.mean(total_loss)
        mean_accuracy = np.mean(total_accuracy)
        self.images = train_images
        self.labels = train_labels
        self.is_training = tf.constant(True, dtype=tf.bool)
        return mean_loss, mean_accuracy
