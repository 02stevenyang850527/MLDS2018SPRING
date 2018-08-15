from agent_dir.agent import Agent
import scipy
import numpy as np
import tensorflow as tf
import os


# Ref: https://github.com/mrahtz/tensorflow-rl-pong


# Action values to send to gym environment to move paddle up/down
UP_ACTION = 2
DOWN_ACTION = 3
# Mapping from action values to outputs from the policy network
action_dict = {DOWN_ACTION: 0, UP_ACTION: 1}


def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)

def dense_layer(input_tensor, input_dim, output_dim, name, activation=tf.nn.relu):
    with tf.variable_scope(name):
        W = tf.get_variable('weights', [input_dim, output_dim], dtype=tf.float32)
        b = tf.get_variable('bias', [output_dim], dtype=tf.float32)
        output = tf.nn.bias_add(tf.matmul(input_tensor, W), b)
        return activation(output)

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)


        ##################
        # YOUR CODE HERE #
        ##################
        print('Build Model Architecture...')
        self.lr = 1e-4
        self.hidden_size = 256
        self.env = env
        self.ep2save = 10
        self.batch_episodes = 1
        if args.render:
            self.render=True
        else:
            self.render=False

        self.sess = tf.InteractiveSession()
        self.states = tf.placeholder(tf.float32, [None, 80*80], name='input_state')
        self.sampled_actions = tf.placeholder(tf.float32, [None, 1], name='sampled_actions')
        self.advantage = tf.placeholder(tf.float32, [None, 1], name='advantage')

        fc1 = dense_layer(self.states, 80*80, self.hidden_size, 'dense1')
        self.up_prob = dense_layer(fc1, self.hidden_size, 1, 'up_prob', tf.nn.sigmoid)
        self.loss = tf.losses.log_loss(labels=self.sampled_actions, predictions=self.up_prob,
                                       weights=self.advantage)
        opt = tf.train.AdamOptimizer(self.lr)
        self.train_op = opt.minimize(self.loss)
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess.run(init)

        self.model_dir = 'save/'

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if args.train_pg:
            self.log_dir = 'log/'
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

        if args.test_pg:
            #you can load your model here
            model_path = 'save/pg_net.ckpt-9990' # Greatest rewards: 9990
            self.saver.restore(self.sess, model_path)
            print('loading trained model')


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        self.observation_memory = []


    def forward_pass(self, states):
        prob = self.sess.run(self.up_prob, feed_dict={self.states: states.reshape([1, -1])})
        return prob


    def train(self):
        """
        Implement your training algorithm here
        """
        # Action list: https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py
        ##################
        # YOUR CODE HERE #
        ##################
        episode_n = 1
        discount_factor = 0.99
        smoothed_reward = None

        states_history = []
        actions_history = []
        rewards_history = []

        while True:
            print("starting episode",episode_n)
            n_steps = 1
            done = False
            episode_reward_sum = 0
            round_n = 1
            last_observation = self.env.reset()
            last_observation = prepro(last_observation)
            action = self.env.action_space.sample()
            observation, _, _, _ = self.env.step(action)
            observation = prepro(observation)
            while not done:
                if self.render:
                    self.env.env.render()
                residual = observation - last_observation
                last_observation = observation
                up_prob = self.forward_pass(residual)[0]

                if np.random.uniform() < up_prob:
                    action = UP_ACTION
                else:
                    action = DOWN_ACTION

                observation, reward, done, _ = self.env.step(action)
                observation = prepro(observation)
                episode_reward_sum += reward
                n_steps += 1

                states_history.append(residual)
                actions_history.append(action_dict[action])
                rewards_history.append(reward)
                '''
                if reward == -1:
                    print("Round {}: {} time steps; lost...".format(round_n, n_steps))
                elif reward == +1:
                    print("Round {}: {} time steps; won!".format(round_n, n_steps))
                '''
                if reward != 0:
                    round_n += 1
                    n_steps = 0

            summary = tf.Summary(value=[tf.Summary.Value(tag="Episode_Rewards", simple_value=episode_reward_sum)])
            self.writer.add_summary(summary, episode_n)
            print("Episode {} finished after {} rounds".format(episode_n, round_n))

            if smoothed_reward is None:
                smoothed_reward = episode_reward_sum
            else:
                smoothed_reward = smoothed_reward * 0.99 + episode_reward_sum * 0.01
            print("Reward total was {}; discounted moving average of reward is {}".format(episode_reward_sum, smoothed_reward))

            if (episode_n % self.ep2save == 0 and episode_n > 1000):
                self.saver.save(self.sess, os.path.join(self.model_dir, 'pg_net.ckpt'), global_step=episode_n)

            if episode_n % self.batch_episodes == 0:
                rewards = self.discount_rewards(rewards_history, discount_factor)
                rewards -= np.mean(rewards)
                rewards /= np.std(rewards)
                self.update(states_history, actions_history, rewards)
                states_history = []
                actions_history = []
                rewards_history = []

            episode_n += 1


    def discount_rewards(self, rewards, discount_factor):
        discounted_rewards = np.zeros_like(rewards)
        for t in range(len(rewards)):
            discounted_reward_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                discounted_reward_sum += rewards[k] * discount
                discount *= discount_factor
                if rewards[k] != 0:
                # Don't count rewards from subsequent rounds
                    break
            discounted_rewards[t] = discounted_reward_sum
        return discounted_rewards


    def update(self, states, actions, rewards):
        print('Update with {} (states, actions, rewards)'.format(len(actions)))
        
        states = np.array(states).reshape(-1,6400)
        actions = np.array(actions).reshape(-1,1)
        rewards = np.array(rewards).reshape(-1,1)
        feed_dict = {
            self.states: states,
            self.sampled_actions: actions,
            self.advantage: rewards
        }
        self.sess.run(self.train_op, feed_dict)


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        if self.observation_memory == []:
            init_observation = prepro(observation)
            action = self.env.get_random_action()
            second_observation, _, _, _ = self.env.step(action)
            second_observation = prepro(second_observation)
            residual = second_observation - init_observation
            self.observation_memory = second_observation
        else:
            observation = prepro(observation)
            residual = observation - self.observation_memory
            self.observation_memory = observation

        up_prob = self.forward_pass(residual)[0]
        if up_prob > 0.5:
            action = UP_ACTION
        else:
            action = DOWN_ACTION

        return action

