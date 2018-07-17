import logging

import tensorflow as tf
import tensorflow.contrib.layers as c_layers
import numpy as np
from unitytrainers.models import LearningModel
from unityagents import UnityException, AllBrainInfo,BrainInfo

logger = logging.getLogger("unityagents")


class MARLModel(LearningModel):
    def __init__(self, brain, lr=1e-4, h_size=128, epsilon=0.2, beta=1e-3, max_step=5e6,
                 normalize=False, use_recurrent=False, num_layers=2, m_size=None,agent_cnt=1):
        """
        Takes a Unity environment and model-specific hyper-parameters and returns the
        appropriate PPO agent model for the environment.
        :param brain: BrainInfo used to generate specific network graph.
        :param lr: Learning rate.
        :param h_size: Size of hidden layers
        :param epsilon: Value for policy-divergence threshold.
        :param beta: Strength of entropy regularization.
        :return: a sub-class of PPOAgent tailored to the environment.
        :param max_step: Total number of training steps.
        :param normalize: Whether to normalize vector observation input.
        :param use_recurrent: Whether to use an LSTM layer in the network.
        :param num_layers Number of hidden layers between encoded input and policy & value layers
        :param m_size: Size of brain memory.
        """
        LearningModel.__init__(self, m_size, normalize, use_recurrent, brain)
        if num_layers < 1:
            num_layers = 1
       
        self.num_layers = num_layers
        self.h_size = h_size
        self.lr = lr
        self.beta = beta
        self.max_step = max_step
        self.vepsilon = epsilon
        self.created_model = False

        self.create_model(agent_cnt,None,"")


    @staticmethod
    def create_reward_encoder():
        """Creates TF ops to track and increment recent average cumulative reward."""
        last_reward = tf.Variable(0, name="last_reward", trainable=False, dtype=tf.float32)
        new_reward = tf.placeholder(shape=[], dtype=tf.float32, name='new_reward')
        update_reward = tf.assign(last_reward, new_reward)
        return last_reward, new_reward, update_reward

    def create_ppo_optimizer(self, probs, old_probs, value, entropy, beta, epsilon, lr, max_step):
        """
        Creates training-specific Tensorflow ops for PPO models.
        :param probs: Current policy probabilities
        :param old_probs: Past policy probabilities
        :param value: Current value estimate
        :param beta: Entropy regularization strength
        :param entropy: Current policy entropy
        :param epsilon: Value for policy-divergence threshold
        :param lr: Learning rate
        :param max_step: Total number of training steps.
        """

        self.returns_holder = tf.placeholder(shape=[None], dtype=tf.float32, name='discounted_rewards')
        self.advantage = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='advantages')
        self.learning_rate = tf.train.polynomial_decay(lr, self.global_step, max_step, 1e-10, power=1.0)

        self.old_value = tf.placeholder(shape=[None], dtype=tf.float32, name='old_value_estimates')
        self.mask_input = tf.placeholder(shape=[None], dtype=tf.float32, name='masks')

        decay_epsilon = tf.train.polynomial_decay(epsilon, self.global_step, max_step, 0.1, power=1.0)
        decay_beta = tf.train.polynomial_decay(beta, self.global_step, max_step, 1e-5, power=1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        self.mask = tf.equal(self.mask_input, 1.0)

        clipped_value_estimate = self.old_value + tf.clip_by_value(tf.reduce_sum(value, axis=1) - self.old_value,
                                                                   - decay_epsilon, decay_epsilon)

        v_opt_a = tf.squared_difference(self.returns_holder, tf.reduce_sum(value, axis=1))
        v_opt_b = tf.squared_difference(self.returns_holder, clipped_value_estimate)
        self.value_loss = tf.reduce_mean(tf.boolean_mask(tf.maximum(v_opt_a, v_opt_b), self.mask))

        # Here we calculate PPO policy loss. In continuous control this is done independently for each action gaussian
        # and then averaged together. This provides significantly better performance than treating the probability
        # as an average of probabilities, or as a joint probability.
        self.r_theta = probs / (old_probs + 1e-10)
        self.p_opt_a = self.r_theta * self.advantage
        self.p_opt_b = tf.clip_by_value(self.r_theta, 1.0 - decay_epsilon, 1.0 + decay_epsilon) * self.advantage
        self.policy_loss = -tf.reduce_mean(tf.boolean_mask(tf.minimum(self.p_opt_a, self.p_opt_b), self.mask))

        self.loss = self.policy_loss + 0.5 * self.value_loss - decay_beta * tf.reduce_mean(
            tf.boolean_mask(entropy, self.mask))
        self.update_batch = optimizer.minimize(self.loss)


 
    def create_model(self,agent_cnt,sess,model_path):
        if self.created_model == True:
            return

        if self.brain.vector_action_space_type == "continuous":
            self.create_marl_cc_actor_critic(agent_cnt,self.h_size, self.num_layers)
            self.entropy = tf.ones_like(tf.reshape(self.value, [-1])) * self.entropy
        else:
            self.create_marl_dc_actor_critic(agent_cnt,self.h_size, self.num_layers)
        self.create_ppo_optimizer(self.probs, self.old_probs, self.value,
                                  self.entropy, self.beta, self.vepsilon, self.lr, self.max_step)

        if model_path =="":
            if sess != None:
                init = tf.global_variables_initializer()
                sess.run(init)
        else:
            saver = tf.train.Saver()
            # Instantiate model parameters     
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt is None:
                self.logger.info('The model {0} could not be found. Make sure you specified the right '
                                    '--run-id'.format(self.model_path))
            saver.restore(sess, ckpt.model_checkpoint_path)

        self.last_reward, self.new_reward, self.update_reward = self.create_reward_encoder()

        self.created_model = True


    def create_marl_dc_actor_critic(self,agent_cnt ,h_size, num_layers):
        num_streams = 1
        hidden_streams = self.create_new_obs(num_streams, h_size, num_layers)
        hidden = hidden_streams[0]

        if self.use_recurrent:
            tf.Variable(self.m_size, name="memory_size", trainable=False, dtype=tf.int32)
            self.prev_action = tf.placeholder(shape=[None], dtype=tf.int32, name='prev_action')
            self.prev_action_oh = c_layers.one_hot_encoding(self.prev_action, self.a_size)
            hidden = tf.concat([hidden, self.prev_action_oh], axis=1)

            self.memory_in = tf.placeholder(shape=[None, self.m_size], dtype=tf.float32, name='recurrent_in')
            hidden, self.memory_out = self.create_recurrent_encoder(hidden, self.memory_in)
            self.memory_out = tf.identity(self.memory_out, name='recurrent_out')

        self.policy = tf.layers.dense(hidden, self.a_size, activation=None, use_bias=False,
                                      kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01))

        self.all_probs = tf.nn.softmax(self.policy, name="action_probs")
        self.output = tf.multinomial(self.policy, 1)
        self.output = tf.identity(self.output, name="action")

        brain = self.brain
        s_size = brain.vector_observation_space_size * brain.num_stacked_vector_observations

        self.team_obs = tf.placeholder(shape=[None,s_size*agent_cnt], dtype=tf.float32, name='team_obs')

        actions = tf.identity(c_layers.flatten(self.all_probs),name="team_actions");
  
        hidden = tf.concat([hidden,self.vector_in,actions, self.team_obs], axis=1)

        self.value = tf.layers.dense(hidden, 1, activation=None)
        self.value = tf.identity(self.value, name="value_estimate")
        self.entropy = -tf.reduce_sum(self.all_probs * tf.log(self.all_probs + 1e-10), axis=1)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)
        self.selected_actions = c_layers.one_hot_encoding(self.action_holder, self.a_size)

        self.all_old_probs = tf.placeholder(shape=[None, self.a_size], dtype=tf.float32, name='old_probabilities')

        # We reshape these tensors to [batch x 1] in order to be of the same rank as continuous control probabilities.
        self.probs = tf.expand_dims(tf.reduce_sum(self.all_probs * self.selected_actions, axis=1), 1)
        self.old_probs = tf.expand_dims(tf.reduce_sum(self.all_old_probs * self.selected_actions, axis=1), 1)

    def create_marl_cc_actor_critic(self,agent_cnt, h_size, num_layers):
        num_streams = 2
        hidden_streams = self.create_new_obs(num_streams, h_size, num_layers)

        if self.use_recurrent:
            tf.Variable(self.m_size, name="memory_size", trainable=False, dtype=tf.int32)
            self.memory_in = tf.placeholder(shape=[None, self.m_size], dtype=tf.float32, name='recurrent_in')
            _half_point = int(self.m_size / 2)
            hidden_policy, memory_policy_out = self.create_recurrent_encoder(
                hidden_streams[0], self.memory_in[:, :_half_point], name='lstm_policy')

            hidden_value, memory_value_out = self.create_recurrent_encoder(
                hidden_streams[1], self.memory_in[:, _half_point:], name='lstm_value')
            self.memory_out = tf.concat([memory_policy_out, memory_value_out], axis=1, name='recurrent_out')
        else:
            hidden_policy = hidden_streams[0]
            hidden_value = hidden_streams[1]

        self.mu = tf.layers.dense(hidden_policy, self.a_size, activation=None, use_bias=False,
                                  kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01))

        self.log_sigma_sq = tf.get_variable("log_sigma_squared", [self.a_size], dtype=tf.float32,
                                            initializer=tf.zeros_initializer())

        self.sigma_sq = tf.exp(self.log_sigma_sq)
        self.epsilon = tf.random_normal(tf.shape(self.mu), dtype=tf.float32)
        self.output = self.mu + tf.sqrt(self.sigma_sq) * self.epsilon
        self.output = tf.identity(self.output, name='action')
        a = tf.exp(-1 * tf.pow(tf.stop_gradient(self.output) - self.mu, 2) / (2 * self.sigma_sq))
        b = 1 / tf.sqrt(2 * self.sigma_sq * np.pi)
        self.all_probs = tf.multiply(a, b, name="action_probs")
        self.entropy = tf.reduce_mean(0.5 * tf.log(2 * np.pi * np.e * self.sigma_sq))

        
       
        brain = self.brain
        s_size = brain.vector_observation_space_size * brain.num_stacked_vector_observations
        self.team_obs = tf.placeholder(shape=[None,s_size*agent_cnt], dtype=tf.float32, name='team_obs')
        hidden_value = tf.concat([hidden_value,self.vector_in, c_layers.flatten(self.all_probs),self.team_obs ], axis=1)


        self.value = tf.layers.dense(hidden_value, 1, activation=None)
        self.value = tf.identity(self.value, name="value_estimate")
        self.all_old_probs = tf.placeholder(shape=[None, self.a_size], dtype=tf.float32,
                                            name='old_probabilities')
        # We keep these tensors the same name, but use new nodes to keep code parallelism with discrete control.
        self.probs = tf.identity(self.all_probs)
        self.old_probs = tf.identity(self.all_old_probs)


    def create_marl_new_obs(self,brain_info:BrainInfo,num_streams, h_size, num_layers):
        brain = self.brain
        s_size = brain.vector_observation_space_size * brain.num_stacked_vector_observations*len(brain_info.agents)+1
        if brain.vector_action_space_type == "continuous":
            activation_fn = tf.nn.tanh
        else:
            activation_fn = self.swish

        self.visual_in = []
        for i in range(brain.number_visual_observations):
            height_size, width_size = brain.camera_resolutions[i]['height'], brain.camera_resolutions[i]['width']
            bw = brain.camera_resolutions[i]['blackAndWhite']
            visual_input = self.create_visual_input(height_size, width_size, bw, name="visual_observation_" + str(i))
            self.visual_in.append(visual_input)
        self.create_vector_input(s_size)

        final_hiddens = []
        for i in range(num_streams):
            visual_encoders = []
            hidden_state, hidden_visual = None, None
            if brain.number_visual_observations > 0:
                for j in range(brain.number_visual_observations):
                    encoded_visual = self.create_visual_encoder(h_size, activation_fn, num_layers)
                    visual_encoders.append(encoded_visual)
                hidden_visual = tf.concat(visual_encoders, axis=1)
            if brain.vector_observation_space_size > 0:
                s_size = brain.vector_observation_space_size * brain.num_stacked_vector_observations
                if brain.vector_observation_space_type == "continuous":
                    hidden_state = self.create_continuous_state_encoder(h_size, activation_fn, num_layers)
                else:
                    hidden_state = self.create_discrete_state_encoder(s_size, h_size,
                                                                      activation_fn, num_layers)
            if hidden_state is not None and hidden_visual is not None:
                final_hidden = tf.concat([hidden_visual, hidden_state], axis=1)
            elif hidden_state is None and hidden_visual is not None:
                final_hidden = hidden_visual
            elif hidden_state is not None and hidden_visual is None:
                final_hidden = hidden_state
            else:
                raise Exception("No valid network configuration possible. "
                                "There are no states or observations in this brain")
            final_hiddens.append(final_hidden)
        return final_hiddens


  
