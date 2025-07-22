import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
import tensorflow_probability as tfp
from maddpg.trainer.prioritized_rb.replay_buffer import ReplayBuffer
from maddpg import AgentTrainer
 

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma * r
        r = r * (1. - done)
        discounted.append(r)
    return discounted[::-1]

class TargetUpdateManager:
    def __init__(self, polyak=0.01):
        self.polyak = polyak
        
    def update_target_networks(self, main_variables, target_variables):
        """Update target network parameters polyak averaging"""
        for main_var, target_var in zip(main_variables, target_variables):
            target_var.assign(self.polyak * main_var + (1.0 - self.polyak) * target_var)



class MLPModel(keras.Model):
    def __init__(self, num_outputs, num_units=64, name="mlp_model"):
        super(MLPModel, self).__init__(name=name)

        #batch norm
        self.batch_norm = layers.BatchNormalization()

        #hidden layer with regularization
        self.dense1 = layers.Dense(num_units * 2,
                                    activation='relu',
                                      kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                      bias_regularizer=tf.keras.regularizers.l2(1e-3),
                                      name='dense1')
        
        self.dropout1 = layers.Dropout(0.1)

        self.dense2 = layers.Dense(num_units,
                                    activation='relu',
                                      kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                      bias_regularizer=tf.keras.regularizers.l2(1e-3),
                                      name='dense2')
        self.dropout2 = layers.Dropout(0.1)

        self.dense3 = layers.Dense(num_units // 2,
                                    activation='relu',
                                      kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                      bias_regularizer=tf.keras.regularizers.l2(1e-3),
                                      name='dense3')
        
        # Output layer
        self.output_layer = layers.Dense(num_outputs, 
                                         activation=None,
                                         kernel_regularizer=tf.keras.regularizers.l2(1e-3),
                                         bias_regularizer=tf.keras.regularizers.l2(1e-3),
                                           name='output')
        
        def call(self, inputs, training=None):
            x = self.batch_norm(inputs, training=training)
            x = self.dense1(x)
            x = self.dropout1(x, training=training)
            x = self.dense2(x)
            x = self.dropout2(x, training=training)
            x = self.dense3(x)
            out = self.output_layer(x)
            return out
                                   

class ActorNetwork(keras.Model):
    #continous action space
    def __init__(self, action_dim, num_units=64, name="actor"):
        super(ActorNetwork, self).__init__(name=name)
        self.action_dim = action_dim

        #mlp model for feature extraction
        self.mlp = MLPModel(num_outputs=action_dim * 2, num_units=num_units) #x2 for mean and log_Std

    def call(self, state, training=None):
        x = self.mlp(state, training=training)
        mean, log_std = tf.split(x, num_or_size_splits=2, axis=-1)  # split the output into mean and log_std
        log_std = tf.clip_by_value(log_std, -20, 2)  # clip log_std to avoid numerical issues
        
        return mean, log_std
    
    def sample_action(self, state, training=None):
        mean, log_std = self.call(state, training=training)
        std = tf.exp(log_std)
        normal = tfp.distributions.Normal(loc=mean, scale=std)
        action = normal.sample()  # sample from the distribution
        action = tf.tanh(action)  # apply tanh to bound the action

        return action, normal  # return action and the distribution for log_prob calculation

class CriticNetwork(keras.Model):
    #value estimation
    def __init__(self, num_units=64, name="critic"):
        super(CriticNetwork, self).__init__(name=name)

        #mlp model for q-value estimation
        self.mlp = MLPModel(num_outputs=1, num_units=num_units)

    def call(self, state_action, training=None):
        q_value = self.mlp(state_action, training=training)  # state_action is concatenated state and action
        return tf.squeeze(q_value, axis=-1)  # squeeze to remove the last dimension

class MADDPGAgent(AgentTrainer):
    def __init__(self, name, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.n = len(obs_shape_n)
        self.name = name
        self.args = args
        self.local_q_func = local_q_func

        #action dim
        self.action_dim = act_space_n[agent_index].shape[0] if hasattr(act_space_n[agent_index], 'shape') else act_space_n[agent_index]
        self.obs_dim = obs_shape_n[agent_index][0] 

        #networks
        self.actor = ActorNetwork(self.action_dim, num_units=args.num_units, name=f"{name}_actor")
        self.critic = CriticNetwork(num_units=args.num_units, name=f"{name}_critic")
        self.target_actor = ActorNetwork(self.action_dim, num_units=args.num_units, name=f"{name}_target_actor")
        self.target_critic = CriticNetwork(num_units=args.num_units, name=f"{name}_target_critic")

        #lr with decay
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=args.lr,
            decay_steps=args.decay_steps,
            decay_rate=args.decay_rate,
            staircse=True
        )

        #optimizers
        self.actor_optimizer = optimizers.Adam(learning_rate=self.lr_schedule)
        self.critic_optimizer = optimizers.Adam(learning_rate=self.lr_schedule)

        #target update manager
        self.target_updater = TargetUpdateManager(polyak=args.polyak)

        #replay buffer
        self.buffer_size = args.buffer_size
        self.beta = args.beta
        self.replay_buffer = ReplayBuffer(int(self.buffer_size), 
                                          int(args.batch_size),
                                          args.alpha,
                                          args.epsilon)
        self.replay_sample_index = None  # for prioritized replay buffer

        #init network
        self._initialize_networks()

        #metrics track
        self.actor_loss_metric = tf.keras.metrics.Mean(name='actor_loss')
        self.critic_loss_metric = tf.keras.metrics.Mean(name='critic_loss')

    def _initialize_networks(self):
        dummy_obs = tf.zeros((1, self.obs_dim))  # dummy observation for initialization
        dummy_action = tf.zeros((1, self.action_dim))

        if self.local_q_func:
            # local Q-function
            dummy_critic_input = tf.concat([dummy_obs, dummy_action], axis=1)
        else: 
            # global Q-function
            dummy_critic_input = tf.concat([dummy_obs] * self.n + [dummy_action] * self.n, axis=1)        

        _ = self.actor(dummy_obs)  # initialize actor
        _ = self.critic(dummy_critic_input)  # initialize critic
        _ = self.target_actor(dummy_obs)  # initialize target actor
        _ = self.target_critic(dummy_critic_input)  # initialize target critic

        #copy weights to target networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())


    @property
    def filled_size(self):
        return len(self.replay_buffer)

    def action(self, obs):
        """
        Get action for UAV given observation
        Args:
            obs: observation vector of shape (obs_dim,)
        Returns:
            action: continuous action vector [delta_x, delta_y, power]
        """
        if obs.ndim == 1:
            obs = tf.expand_dims(obs, 0)
        
        obs_tensor = tf.convert_to_tensor(obs, dtype=tf.float32)
        action, _ = self.actor.sample_action(obs_tensor, training=False)
        
        return action.numpy()[0]
    
    def experience(self, obs, act, rew, new_obs, done, terminal, num_actor_workers):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done), self.args.N, self.args.gamma, num_actor_workers)

    def preupdate(self):
        self.replay_sample_index = None

    @tf.function
    def _train_critic(self, obs_n, act_n, target_q):
        with tf.GradientTape() as tape:
            if self.local_q_func:
                critic_input = tf.concat([obs_n[self.agent_index], act_n[self.agent_index]], axis=1)  # local Q-function
            else:
                # global Q-function
                critic_input = tf.concat(obs_n + act_n, axis=1)

            q_value = self.critic(critic_input, training=True)  # forward pass
            critic_loss = tf.reduce_mean(tf.square(target_q - q_value))  # MSE loss

            #regularization loss
            critic_loss += sum(self.critic.losses)  # add L2 regularization losses
        
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)  # compute gradients

        #gradient clipping
        critic_gradients = [tf.clip_by_value(g, self.args.grad_clip) for g in critic_gradients if g is not None]
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))  # apply gradients
        return critic_loss, q_value
    
    @tf.function
    def _train_actor(self, obs_n, act_n):
        with tf.GradientTape() as tape:
            new_actions, _ = self.actor.sample_action(obs_n[self.agent_index], training=True)  # sample new actions

            act_n_new = act_n.copy() if isinstance(act_n, list) else [act_n[i] for i in range(len(self.n))]
            act_n_new[self.agent_index] = new_actions  # replace the action of the current

            if self.local_q_func:
                critic_input = tf.concat([obs_n[self.agent_index], new_actions], axis=1)  
            else:
                critic_input = tf.concat(obs_n + act_n_new, axis=1)
            
            q_value = self.critic(critic_input, training=False)  # forward pass
            actor_loss = -tf.reduce_mean(q_value)  # maximize Q-value (minimize negative Q-value)
            #regularization loss
            actor_loss += sum(self.actor.losses)  * 1e-3

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)  # compute gradients
        #gradient clipping
        actor_gradients = [tf.clip_by_value(g, self.args.grad_clip) for g in actor_gradients if g is not None]
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        return actor_loss

    def update(self, env, agents, t):
        # replay buffer is not large enough 没填满的时候不训练
        # if len(self.replay_buffer) < 10000:
        if len(self.replay_buffer) < 100 * self.args.batch_size:
            return [0]
        if not t % 10 == 0:  # only update every 10 steps
            return [0]

        # 随着训练的进行，让β从某个小于1的值渐进地靠近1
        if self.beta < 1.:
            self.beta *= 1. + 1e-4

        # sample from one agent(batch:1024)  之后根据β算出来的weights没有用到呢！！！
        (obs, act, rew, obs_next, done), weights, priorities, self.replay_sample_index = self.replay_buffer.sample(
            self.args.batch_size, self.beta, self.args.num_actor_workers, 0)  
        
        #convert to tensors
        obs = tf.convert_to_tensor(obs, dtype=tf.float32)
        act = tf.convert_to_tensor(act, dtype=tf.float32)
        rew = tf.convert_to_tensor(rew, dtype=tf.float32)
        obs_next = tf.convert_to_tensor(obs_next, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        weights = tf.convert_to_tensor(weights, dtype=tf.float32)

        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []

        index = self.replay_sample_index  # index数组
        for i in range(self.n):
            obs_, _, rew_, obs_next_, _ = agents[i].replay_buffer.sample_index(index, self.args.num_actor_workers,
                                                                               0)
            _, act_, _, _, done_ = agents[i].replay_buffer.sample_index(index, 0, 0)

            obs_n.append(tf.convert_to_tensor(obs_, dtype=tf.float32))
            obs_next_n.append(tf.convert_to_tensor(obs_next_, dtype=tf.float32))
            act_n.append(tf.convert_to_tensor(act_, dtype=tf.float32))

        # calculate target Q-value
        target_act_next_n  = []
        for i in range(self.n):
            target_action, _ = agents[i].target_actor.sample_action(obs_next_n[i], training=False)
            target_act_next_n.append(target_action)

        if self.local_q_func:
            target_critic_input = tf.concat([obs_next_n[self.agent_index], target_act_next_n[self.agent_index]], axis=1)
        else:
            target_critic_input = tf.concat(obs_next_n + target_act_next_n, axis=1)
        
        target_q_next = self.target_critic(target_critic_input, training=False)  # get target Q-value
        target_q = rew + self.args.gamma ** self.args.N * target_q_next * (1. - done)

        critic_loss, q_value = self._train_critic(obs_n, act_n, target_q)  # train critic
        actor_loss = self._train_actor(obs_n, act_n)  # train actor

        self.target_updater.update_target_networks(
            self.actor.trainable_variables, 
            self.target_actor.trainable_variables)
        
        self.target_updater.update_target_networks(
            self.critic.trainable_variables, 
            self.target_critic.trainable_variables)
        
        # update replay buffer priorities
        td_errors = tf.abs(target_q - q_value)  # TD errors
        self.replay_buffer.priority_update(self.replay_sample_index, td_errors.numpy())

        self.actor_loss_metric.update_state(actor_loss)
        self.critic_loss_metric.update_state(critic_loss)

        #febug
        if hasattr(env, 'log_dir'):
            debug_dir = env.log_dir + getattr(self.args, 'debug_dir', 'debug/')
            if not os.path.exists(debug_dir):
                os.makedirs(debug_dir)
            
            with open(debug_dir + f"current_step_info_{self.name}.txt", 'w+') as f:
                for i, r, p, q, w in zip(index, rew.numpy(), priorities.numpy(), q_value.numpy(), weights.numpy()):
                    print(f"{self.name} step: {t} index: {i} reward: {r:.4f} "
                          f"priority: {p:.4f} Q: {q:.4f} weight: {w:.4f}", file=f)
                    
        return [
            critic_loss.numpy(),
            actor_loss.numpy(), 
            tf.reduce_mean(target_q).numpy(),
            tf.reduce_mean(rew).numpy(),
            tf.reduce_mean(target_q_next).numpy(),
            tf.math.reduce_std(target_q).numpy()
        ]
    
    def save_weights(self, path):
        self.actor.save_weights(f"{path}_actor")
        self.critic.save_weights(f"{path}_critic")

    def load_weights(self, path):
        self.actor.load_weights(f"{path}_actor")
        self.critic.load_weights(f"{path}_critic")
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        