########################
# importing libraries
########################
# system libraries
import os
import numpy as np
import pandas as pd

# custom libraries
from lib.MetaQNN.cnn import parse as cnn_parse
import lib.MetaQNN.state_enumerator as se
from lib.MetaQNN.state_string_utils import StateStringUtils
import lib.Models.state_space_parameters as state_space_parameters


class QValues:
    """
    Q-value table for discrete state action space

    Attributes:
        q_values (dict): a dictionary with start state as keys, and a list of actions and a list of corresponding
                         Q-values as the elements
    """

    def __init__(self):
        self.q_values = {}

    def save_to_csv(self, q_values_csv_path):
        """
        puts the Q-value table in a data frame and saves to a csv
        
        Parameters:
            q_values_csv_path (string): path to csv for saving the data frame
        """

        start_layer_type = []
        start_layer_depth = []
        start_filter_depth = []
        start_filter_size = []
        start_stride = []
        start_image_size = []
        start_fc_size = []
        start_terminate = []

        end_layer_type = []
        end_layer_depth = []
        end_filter_depth = []
        end_filter_size = []
        end_stride = []
        end_image_size = []
        end_fc_size = []
        end_terminate = []

        utility = []

        for start_state_list in self.q_values.keys():
            start_state = se.State(state_list=start_state_list)
            for to_state_ix in range(len(self.q_values[start_state_list]['actions'])):
                to_state = se.State(state_list=self.q_values[start_state_list]['actions'][to_state_ix])
                utility.append(self.q_values[start_state_list]['utilities'][to_state_ix])
                start_layer_type.append(start_state.layer_type)
                start_layer_depth.append(start_state.layer_depth)
                start_filter_depth.append(start_state.filter_depth)
                start_filter_size.append(start_state.filter_size)
                start_stride.append(start_state.stride)
                start_image_size.append(start_state.image_size)
                start_fc_size.append(start_state.fc_size)
                start_terminate.append(start_state.terminate)
                end_layer_type.append(to_state.layer_type)
                end_layer_depth.append(to_state.layer_depth)
                end_filter_depth.append(to_state.filter_depth)
                end_filter_size.append(to_state.filter_size)
                end_stride.append(to_state.stride)
                end_image_size.append(to_state.image_size)
                end_fc_size.append(to_state.fc_size)
                end_terminate.append(to_state.terminate)

        q_values_csv = pd.DataFrame({'start_layer_type': start_layer_type,
                                     'start_layer_depth': start_layer_depth,
                                     'start_filter_depth': start_filter_depth,
                                     'start_filter_size': start_filter_size,
                                     'start_stride': start_stride,
                                     'start_image_size': start_image_size,
                                     'start_fc_size': start_fc_size,
                                     'start_terminate': start_terminate,
                                     'end_layer_type': end_layer_type,
                                     'end_layer_depth': end_layer_depth,
                                     'end_filter_depth': end_filter_depth,
                                     'end_filter_size': end_filter_size,
                                     'end_stride': end_stride,
                                     'end_image_size': end_image_size,
                                     'end_fc_size': end_fc_size,
                                     'end_terminate': end_terminate,
                                     'utility': utility})

        q_values_csv.to_csv(q_values_csv_path, index=False)

    def load_q_values(self, q_values_csv_path):
        """
        reading a stored Q-value csv and making a Q-value dictionary
        
        Parameters:
            q_values_csv_path (string): path to csv for loading the Q-values from
        """

        self.q_values = {}
        q_values_csv = pd.read_csv(q_values_csv_path)

        for row in zip(*[q_values_csv[col].values.tolist() for col in ['start_layer_type',
                                                                       'start_layer_depth',
                                                                       'start_filter_depth',
                                                                       'start_filter_size',
                                                                       'start_stride',
                                                                       'start_image_size',
                                                                       'start_fc_size',
                                                                       'start_terminate',
                                                                       'end_layer_type',
                                                                       'end_layer_depth',
                                                                       'end_filter_depth',
                                                                       'end_filter_size',
                                                                       'end_stride',
                                                                       'end_image_size',
                                                                       'end_fc_size',
                                                                       'end_terminate',
                                                                       'utility']]):
            start_state = se.State(layer_type=row[0],
                                   layer_depth=row[1],
                                   filter_depth=row[2],
                                   filter_size=row[3],
                                   stride=row[4],
                                   image_size=row[5],
                                   fc_size=row[6],
                                   terminate=row[7]).as_tuple()

            end_state = se.State(layer_type=row[8],
                                 layer_depth=row[9],
                                 filter_depth=row[10],
                                 filter_size=row[11],
                                 stride=row[12],
                                 image_size=row[13],
                                 fc_size=row[14],
                                 terminate=row[15]).as_tuple()
            utility = row[16]

            if start_state not in self.q_values:
                self.q_values[start_state] = {'actions': [end_state], 'utilities': [utility]}
            else:
                self.q_values[start_state]['actions'].append(end_state)
                self.q_values[start_state]['utilities'].append(utility)


class QLearner:
    """
    generate net states, checks if net states present in replay buffer, adds net states to replay buffer and updates
    Q-values using the iterative Bellman update

    Parameters:
        args (argparse.Namespace): parsed command line arguments
        num_classes (int): number of data classes
        save_path (string): directory path to log the results

    Attributes:
        state_space_parameters (lib.Models.state_space_parameters): defines search rules like discrete search space,
                                                                    epsilon schedule, replay number etc.
        enum (lib.MetaQNN.state_enumerator.StateEnumerator): populates the discrete state-action space
        state_string_utils (lib.MetaQNN.state_string_utils.StateStringUtils): parses state list to state string and vice
                                                                              versa
        q_values_obj (QValues): defines a dictionary to hold the Q-values and methods to store it to and load it from
                                csv
        replay_buffer (pandas.DataFrame): replay buffer for Q-learning
        state (lib.MetaQNN.state_enumerator.State): starting state in net states
        state_list (list): list for storing sequence of states
        state_string (string): string for a sequence of states
        epsilon (float): epsilon value for sampling states to produce a state list
        arc_count (int): starting architecture index to continue search
        fixed_net_buffer (pandas.DataFrame): data frame for storing results from full training of a fixed net, to csv
    """
    def __init__(self, args, num_classes, save_path):
        self.args = args
        self.num_classes = num_classes
        self.save_path = save_path

        self.state_space_parameters = state_space_parameters
        self.enum = se.StateEnumerator(args)
        self.state_string_utils = StateStringUtils()
        self.q_values_obj = QValues()
        # epsilon -> epsilon value at which this architecture has been sampled
        # net -> network architecture
        # train_flag -> flag that says if this network has been trained till convergence (full training)
        # spp_size -> number of SPP scales
        # reward -> reward from evaluating the network
        # hard_best_val -> best validation hard precision
        # hard_val_all_epochs -> list of hard validation precision from all epochs
        # soft_best_val -> best validation soft precision
        # soft_val_all_epochs -> list of soft validation precision from all  epochs
        # hard_best_background -> best hard precision for 'background' class
        # hard_best_crack -> best hard precision for 'crack' class
        # hard_best_spallation -> best hard precision for 'spallation' class
        # hard_best_exposed_bars -> best hard precision for 'exposed bars' class
        # hard_best_efflorescence -> best hard precision for 'efflorescence' class
        # hard_best_corrosion_stain -> best hard precision for 'corrosion stain' class
        self.replay_buffer = pd.DataFrame(columns=['epsilon',
                                                   'net',
                                                   'train_flag',
                                                   'spp_size',
                                                   'reward',
                                                   'hard_best_val',
                                                   'hard_val_all_epochs',
                                                   'soft_best_val',
                                                   'soft_val_all_epochs',
                                                   'hard_best_background',
                                                   'hard_best_crack',
                                                   'hard_best_spallation',
                                                   'hard_best_exposed_bars',
                                                   'hard_best_efflorescence',
                                                   'hard_best_corrosion_stain'])

        # initial instatiation of state_list, state_string and state
        self.state_list = []
        self.state_string = ''
        self.state = se.State(layer_type='start',
                              layer_depth=0,
                              filter_depth=0,
                              filter_size=0,
                              stride=0,
                              image_size=args.patch_size,
                              fc_size=0,
                              terminate=0,
                              state_list=self.state_list)

        # initial values of epsilon and count of no. of networks searched over
        self.epsilon = 1.0
        self.arc_count = 0

        # continue arc search
        if int(args.task) == 1 and (args.continue_search is True):
            self.q_values_obj.load_q_values(args.q_values_csv_path)
            self.replay_buffer = pd.read_csv(args.replay_buffer_csv_path, index_col=0)
            for episode in state_space_parameters.epsilon_schedule:
                if episode[0] == args.continue_epsilon:
                    self.arc_count += args.continue_ite - 1
                    break
                else:
                    self.arc_count += episode[1]
            self.epsilon = args.continue_epsilon
        # train fixed net
        elif int(args.task) == 2:
            self.fixed_net_buffer = pd.DataFrame(columns=['net',
                                                          'spp_size',
                                                          'hard_best_val',
                                                          'hard_val_all_epochs',
                                                          'soft_best_val',
                                                          'soft_val_all_epochs',
                                                          'hard_best_background',
                                                          'hard_best_crack',
                                                          'hard_best_spallation',
                                                          'hard_best_exposed_bars',
                                                          'hard_best_efflorescence',
                                                          'hard_best_corrosion_stain'
                                                          ])
            self.replay_buffer = pd.read_csv(args.replay_buffer_csv_path, index_col=0)
            if args.fixed_net_index_no > self.replay_buffer.shape[0] - 1:
                raise ValueError('fixed net index not  within limits of replay buffer')

    def generate_search_net_states(self, epsilon):
        """
        generates states for a search net
        
        Parameters:
            epsilon (float): epsilon value for sampling states in a state sequence for a search net
        """
        self.epsilon = epsilon
        self._reset_for_new_walk()
        self._run_agent()
        self.state_string = self.state_string_utils.state_list_to_string(self.state_list, num_classes=self.num_classes)

    def check_search_net_in_replay_buffer(self):
        """
        checks if search net is present in replay buffer
        
        Returns:
            boolean, True if search net present and False otherwise
        """
        if self.state_string in self.replay_buffer['net'].values:
            return True
        else:
            return False

    def add_search_net_to_replay_buffer(self, search_net_in_replay_buffer, spp_size=None, reward=None,
                                        hard_best_val=None, hard_val_all_epochs=None, soft_best_val=None,
                                        soft_val_all_epochs=None, train_flag=None, hard_best_background=None,
                                        hard_best_crack=None, hard_best_spallation=None, hard_best_exposed_bars=None,
                                        hard_best_efflorescence=None, hard_best_corrosion_stain=None, verbose=True):
        """
        appends search net at the end of replay buffer
        
        Parameters:
            search_net_in_replay_buffer (bool): True if search net is present in replay buffer and False otherwise
            spp_size (int): number of spp scales in the net
            reward (float): reward from the evaluating the search net
            hard_best_val (float): best validation hard precision
            hard_val_all_epochs (list): list of hard validation precision from all epochs
            soft_best_val (float): best validation soft precision
            soft_val_all_epochs (list): list of soft validation precision from all  epochs
            train_flag (bool): True if search net completely trained, False if early-stopped
            hard_best_background (float): best hard precision for 'background' class
            hard_best_crack (float): best hard precision for 'crack' class
            hard_best_spallation (float): best hard precision for 'spallation' class
            hard_best_exposed_bars (float): best hard precision for 'exposed bars' class
            hard_best_efflorescence (float): best hard precision for 'efflorescence' class
            hard_best_corrosion_stain (float): best hard precision for 'corrosion stain' class
            verbose (bool): True for printing search net performance measures to stdout
        """
        if search_net_in_replay_buffer:
            spp_size = self.replay_buffer[self.replay_buffer['net'] == self.state_string]['spp_size'].values[0]
            reward = self.replay_buffer[self.replay_buffer['net'] == self.state_string]['reward'].values[0]
            hard_best_val = self.replay_buffer[self.replay_buffer['net']
                                               == self.state_string]['hard_best_val'].values[0]
            hard_val_all_epochs = self.replay_buffer[self.replay_buffer['net']
                                                     == self.state_string]['hard_val_all_epochs'].values[0]
            soft_best_val = self.replay_buffer[self.replay_buffer['net']
                                               == self.state_string]['soft_best_val'].values[0]
            soft_val_all_epochs = self.replay_buffer[self.replay_buffer['net']
                                                     == self.state_string]['soft_val_all_epochs'].values[0]
            train_flag = self.replay_buffer[self.replay_buffer['net'] == self.state_string]['train_flag'].values[0]
            hard_best_background = self.replay_buffer[self.replay_buffer['net']
                                                      == self.state_string]['hard_best_background'].values[0]
            hard_best_crack = self.replay_buffer[self.replay_buffer['net']
                                                 == self.state_string]['hard_best_crack'].values[0]
            hard_best_spallation = self.replay_buffer[self.replay_buffer['net']
                                                      == self.state_string]['hard_best_spallation'].values[0]
            hard_best_exposed_bars = self.replay_buffer[self.replay_buffer['net']
                                                        == self.state_string]['hard_best_exposed_bars'].values[0]
            hard_best_efflorescence = self.replay_buffer[self.replay_buffer['net']
                                                         == self.state_string]['hard_best_efflorescence'].values[0]
            hard_best_corrosion_stain = self.replay_buffer[self.replay_buffer['net']
                                                           == self.state_string]['hard_best_corrosion_stain'].values[0]

        self.replay_buffer = self.replay_buffer.append(pd.DataFrame([[self.epsilon,
                                                                      self.state_string,
                                                                      train_flag,
                                                                      spp_size,
                                                                      reward,
                                                                      hard_best_val,
                                                                      hard_val_all_epochs,
                                                                      soft_best_val,
                                                                      soft_val_all_epochs,
                                                                      hard_best_background,
                                                                      hard_best_crack,
                                                                      hard_best_spallation,
                                                                      hard_best_exposed_bars,
                                                                      hard_best_efflorescence,
                                                                      hard_best_corrosion_stain]],
                                                                    columns=['epsilon',
                                                                             'net',
                                                                             'train_flag',
                                                                             'spp_size',
                                                                             'reward',
                                                                             'hard_best_val',
                                                                             'hard_val_all_epochs',
                                                                             'soft_best_val',
                                                                             'soft_val_all_epochs',
                                                                             'hard_best_background',
                                                                             'hard_best_crack',
                                                                             'hard_best_spallation',
                                                                             'hard_best_exposed_bars',
                                                                             'hard_best_efflorescence',
                                                                             'hard_best_corrosion_stain']),
                                                       ignore_index=True, sort=False)

        if verbose:
            print('reward {reward:.3f}, best hard validation accuracy {hard_best_val:.3f},'
                  'last hard validation accuracy {hard_last_val:.3f}'.format(reward=reward, hard_best_val=hard_best_val,
                                                                             hard_last_val=hard_val_all_epochs[-1]))

    def update_q_values_and_save_partial(self):
        """
        updates Q-values using the Bellman iterative update, and saves partial results during search
        """
        self.arc_count += 1
        self.replay_buffer.to_csv(os.path.join(self.save_path, 'replay_buffer_' + str(self.arc_count) + '.csv'))
        self._sample_replay_for_update()
        self.q_values_obj.save_to_csv(os.path.join(self.save_path, 'q_values_' + str(self.arc_count) + '.csv'))

    def save_final(self):
        """
        save final Q-values and replay buffer at the end of search
        """
        if int(self.args.task) == 1:
            self.replay_buffer.to_csv(os.path.join(self.save_path, 'replay_buffer_final.csv'))
            self.q_values_obj.save_to_csv(os.path.join(self.save_path, 'q_values_final.csv'))
        elif int(self.args.task) == 2:
            self.fixed_net_buffer.to_csv(os.path.join(self.save_path, 'fixed_net.csv'))

    def generate_fixed_net_states(self):
        """
        generates states for fixed net
        """
        self.state_string = self.replay_buffer.iloc[int(self.args.fixed_net_index_no)]['net']
        self.state_list = self.state_string_utils.parsed_list_to_state_list(cnn_parse('net', self.state_string),
                                                                            self.args.patch_size)

    def add_fixed_net_to_fixed_net_buffer(self, spp_size=None, hard_best_val=None, hard_val_all_epochs=None,
                                          soft_best_val=None, soft_val_all_epochs=None, hard_best_background=None,
                                          hard_best_crack=None, hard_best_spallation=None, hard_best_exposed_bars=None,
                                          hard_best_efflorescence=None, hard_best_corrosion_stain=None, verbose=True):
        """
        appends fixed net and its performance measures to the fixed net buffer
        
        Parameters:
            spp_size (int): number of spp scales in the net
            hard_best_val (float): best validation hard precision
            hard_val_all_epochs (list): list of hard validation precision from all epochs
            soft_best_val (float): best validation soft precision
            soft_val_all_epochs (list): list of soft validation precision from all  epochs
            hard_best_background (float): best hard precision for 'background' class
            hard_best_crack (float): best hard precision for 'crack' class
            hard_best_spallation (float): best hard precision for 'spallation' class
            hard_best_exposed_bars (float): best hard precision for 'exposed bars' class
            hard_best_efflorescence (float): best hard precision for 'efflorescence' class
            hard_best_corrosion_stain (float): best hard precision for 'corrosion stain' class
            verbose (bool): True for printing search net performance measures to stdout
        """
        self.fixed_net_buffer = self.fixed_net_buffer.append(pd.DataFrame([[self.state_string,
                                                                            spp_size,
                                                                            hard_best_val,
                                                                            hard_val_all_epochs,
                                                                            soft_best_val,
                                                                            soft_val_all_epochs,
                                                                            hard_best_background,
                                                                            hard_best_crack,
                                                                            hard_best_spallation,
                                                                            hard_best_exposed_bars,
                                                                            hard_best_efflorescence,
                                                                            hard_best_corrosion_stain]],
                                                                          columns=['net',
                                                                                   'spp_size',
                                                                                   'hard_best_val',
                                                                                   'hard_val_all_epochs',
                                                                                   'soft_best_val',
                                                                                   'soft_val_all_epochs',
                                                                                   'hard_best_background',
                                                                                   'hard_best_crack',
                                                                                   'hard_best_spallation',
                                                                                   'hard_best_exposed_bars',
                                                                                   'hard_best_efflorescence',
                                                                                   'hard_best_corrosion_stain']),
                                                             ignore_index=True, sort=False)
        if verbose:
            print('best hard validation accuracy {hard_best_val:.3f}, last hard validation accuracy {hard_last_val:.3f}'
                  .format(hard_best_val=hard_best_val, hard_last_val=hard_val_all_epochs[-1]))

    def _reset_for_new_walk(self):
        """
        resets starting state and state list for generating a new state sequence
        """
        self.state_list = []
        self.state = se.State(layer_type='start',
                              layer_depth=0,
                              filter_depth=0,
                              filter_size=0,
                              stride=0,
                              image_size=self.args.patch_size,
                              fc_size=0,
                              terminate=0,
                              state_list=self.state_list)

    def _run_agent(self):
        """generate states in sequence until the terminal state is encountered"""
        while self.state.terminate == 0:
            self._transition_q_learning()

    def _transition_q_learning(self):
        """ samples the action given a starting state"""
        # populates actions for a state if that state not previously encountered during search
        if self.state.as_tuple() not in self.q_values_obj.q_values:
            # if state not in Q-val dict, enumerate the
            # state using enumerator
            self.enum.enumerate_state(self.state, self.q_values_obj.q_values)

        action_values = self.q_values_obj.q_values[self.state.as_tuple()]

        # samples next state as per the epsilon greedy schedule
        if np.random.random() < self.epsilon:
            action = se.State(state_list=action_values['actions'][np.random.randint(len(action_values['actions']))])
        else:
            max_q_value = max(action_values['utilities'])
            max_q_indices = [i for i in range(len(action_values['actions'])) if
                             action_values['utilities'][i] == max_q_value]
            max_actions = [action_values['actions'][i] for i in max_q_indices]
            action = se.State(state_list=max_actions[np.random.randint(len(max_actions))])

        self.state = self._action_to_state(action)
        self._post_transition_updates()

    def _post_transition_updates(self):
        """
        produces the next state given a starting state
        """
        state_copy = self.state.copy()
        self.state_list.append(state_copy)

    def _sample_replay_for_update(self):
        """
        samples replay_number nets from the replay buffer and updates Q-values
        """
        # always samples last net for update
        state_string = self.replay_buffer.iloc[-1]['net']
        reward = self.replay_buffer.iloc[-1]['reward']
        state_list = self.state_string_utils.parsed_list_to_state_list(cnn_parse('net', state_string),
                                                                       self.args.patch_size)
        self._update_q_value_sequence(state_list, reward)

        # samples replay_number - 1 random nets
        for i in range(self.state_space_parameters.replay_number - 1):
            state_string = np.random.choice(self.replay_buffer['net'])
            reward = self.replay_buffer[self.replay_buffer['net'] == state_string]['reward'].values[0]
            state_list = self.state_string_utils.parsed_list_to_state_list(cnn_parse('net', state_string),
                                                                           self.args.patch_size)
            self._update_q_value_sequence(state_list, reward)

    def accuracies_to_reward(self, val_acc_list):
        """
        metric to compute reward from a net's validation accuracy list
        
        Parameters:
            val_acc_list (list): validation accuracy list of a net
        
        Returns:
            reward (float): reward to be used in the Bellman equation update
        """
        # average of last 5 validation accuracies
        reward = np.mean(val_acc_list[-5:]) / 100.
        return reward

    def _transition_to_action(self, to_state):
        """
        simple use case to convert transition to action
        
        Parameters:
            to_state (lib.MetaQNN.state_enumerator.State): state for the current state to transition into
        
        Returns:
            action (lib.MetaQNN.state_enumerator.State): action corresponding to the transition
        """
        action = to_state.copy()
        return action

    def _action_to_state(self, action):
        """
        simple use case to convert action to transition
        
        Parameters:
            action (lib.MetaQNN.state_enumerator.State): action from the current state
        
        Returns:
            to_state (ib.MetaQNN.state_enumerator.State): transition (next state) corresponding to the current state
        """
        to_state = action.copy()
        return to_state

    def _update_q_value_sequence(self, state_list, reward):
        """
        updates Q-values for a sequence of state action pairs
        
        Parameters:
            state_list (list): list of states in a state sequence
            reward (float): reward to be used in the Bellman equation update
        """
        # updates the Q-value for the last state pair (update taking place in the reverse order)
        self._update_q_value(state_list[-2], state_list[-1], reward)

        # updates the Q-values for the remaining state action pairs in the reverse order
        for i in reversed(range(len(state_list) - 2)):
            # Q-learning Bellman update
            self._update_q_value(state_list[i], state_list[i + 1], 0)

    def _update_q_value(self, start_state, to_state, reward):
        """
        updates Q-value for single state action pair
        
        Parameters:
            start_state (lib.MetaQNN.state_enumerator.State): start state in the state action pair
            to_state (lib.MetaQNN.state_enumerator.State): end state in the state action pair
            reward (float): reward to be used in the Q-learning Bellman iterative update
        """
        # populate transitions if either start state/ end state not encountered previously (useful to continue search)
        if start_state.as_tuple() not in self.q_values_obj.q_values:
            self.enum.enumerate_state(start_state, self.q_values_obj.q_values)
        if to_state.as_tuple() not in self.q_values_obj.q_values:
            self.enum.enumerate_state(to_state, self.q_values_obj.q_values)

        actions = self.q_values_obj.q_values[start_state.as_tuple()]['actions']
        values = self.q_values_obj.q_values[start_state.as_tuple()]['utilities']

        max_over_next_states = max(self.q_values_obj.q_values[to_state.as_tuple()]['utilities']) if to_state.terminate \
            != 1 else 0

        action_between_states = self._transition_to_action(to_state).as_tuple()

        # Q-learning iterative Bellman update
        values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
            self.args.q_learning_rate * (reward + self.args.q_discount_factor * max_over_next_states -
                                         values[actions.index(action_between_states)])

        self.q_values_obj.q_values[start_state.as_tuple()] = {'actions': actions, 'utilities': values}
