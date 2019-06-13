########################
# importing libraries
########################
# system libraries
import os
import numpy as np
from time import gmtime, strftime
import torch

# custom libraries
from lib.cmdparser import parser
import lib.Datasets.datasets as datasets
from lib.MetaQNN.q_learner import QLearner as QLearner
import lib.Models.state_space_parameters as state_space_parameters
from lib.Models.initialization import WeightInit
from lib.Training.train_model import train_val_net


def main():
    # set device for torch computations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_path = './runs/' + strftime("%Y-%m-%d_%H-%M-%S", gmtime())
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # parse command line arguments
    args = parser.parse_args()
    print("Command line options:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # create log file
    log_file = os.path.join(save_path, "stdout")

    # write parsed args to log file
    log = open(log_file, "a")
    for arg in vars(args):
        print(arg, getattr(args, arg))
        log.write(arg + ':' + str(getattr(args, arg)) + '\n')
    log.close()

    # instantiate the weight initializer
    print("Initializing network with: " + args.weight_init)
    weight_initializer = WeightInit(args.weight_init)

    # instantiate dataset object
    data_init_method = getattr(datasets, args.dataset)
    dataset = data_init_method(torch.cuda.is_available(), args)

    # instantiate a tabular Q-learner
    q_learner = QLearner(args, dataset.num_classes, save_path)

    # start new architecture search
    if int(args.task) == 1:
        if args.continue_search is True:
            # raise exceptions if requirements to start new search not met
            if args.continue_epsilon not in np.array(state_space_parameters.epsilon_schedule)[:, 0]:
                raise ValueError('continue-epsilon {} not in epsilon schedule!'.format(args.continue_epsilon))
            if (args.replay_buffer_csv_path is None) or (not os.path.exists(args.replay_buffer_csv_path)):
                raise ValueError('specify correct path to replay buffer to continue ')
            if (args.q_values_csv_path is None) or (not os.path.exists(args.q_values_csv_path)):
                raise ValueError('wrong path is specified for Q-values')

        # iterate as per the epsilon-greedy schedule
        for episode in state_space_parameters.epsilon_schedule:
            epsilon = episode[0]
            m = episode[1]

            # raise exception if net number to continue from greater than number of nets for the continue_epsilon
            if epsilon == args.continue_epsilon and args.continue_ite > m:
                raise ValueError('continue-ite {} not within range of continue-epsilon {} in epsilon schedule!'
                                 .format(args.continue_ite, epsilon))

            # iterate through number of nets for an epsilon
            for ite in range(1, m + 1):
                # check conditions to generate and train arc
                if (epsilon == args.continue_epsilon and ite >= args.continue_ite) or (epsilon < args.continue_epsilon):
                    print('ite:{}, epsilon:{}'.format(ite, epsilon))

                    # generate net states for search
                    q_learner.generate_search_net_states(epsilon)

                    # check if net already trained before
                    search_net_in_replay_dict = q_learner.check_search_net_in_replay_buffer()

                    # add to the end of the replay buffer if net already trained before
                    if search_net_in_replay_dict:
                        q_learner.add_search_net_to_replay_buffer(search_net_in_replay_dict, verbose=True)
                    # train net if net not trained before
                    else:
                        # train/val search net
                        mem_fit, spp_size, hard_best_val, hard_val_all_epochs, soft_best_val, soft_val_all_epochs,\
                        train_flag, hard_best_background, hard_best_crack, hard_best_spallation,\
                        hard_best_exposed_bars, hard_best_efflorescence, hard_best_corrosion_stain =\
                            train_val_net(q_learner.state_list, dataset, weight_initializer, device, args, save_path)

                        # check if net fits memory
                        while mem_fit is False:
                            print("net failed mem check even with batch splitting, sampling again!")

                            q_learner.generate_search_net_states(epsilon)
                            net_in_replay_dict = q_learner.check_search_net_in_replay_buffer()

                            if search_net_in_replay_dict:
                                q_learner.add_search_net_to_replay_buffer(net_in_replay_dict)
                                break
                            else:
                                mem_fit, spp_size, hard_best_val, hard_val_all_epochs, soft_best_val, \
                                soft_val_all_epochs, train_flag, hard_best_background, hard_best_crack,\
                                hard_best_spallation, hard_best_exposed_bars, hard_best_efflorescence,\
                                hard_best_corrosion_stain =\
                                    train_val_net(q_learner.state_list, dataset, weight_initializer, device, args,
                                                  save_path)

                        # add new net and performance measures to replay buffer if it fits in memory after splitting
                        # batch
                        if mem_fit:
                            reward = q_learner.accuracies_to_reward(hard_val_all_epochs)
                            q_learner.add_search_net_to_replay_buffer(search_net_in_replay_dict, spp_size=spp_size,
                                                                      reward=reward, hard_best_val=hard_best_val,
                                                                      hard_val_all_epochs=hard_val_all_epochs,
                                                                      soft_best_val=soft_best_val,
                                                                      soft_val_all_epochs=soft_val_all_epochs,
                                                                      train_flag=train_flag,
                                                                      hard_best_background=hard_best_background,
                                                                      hard_best_crack=hard_best_crack,
                                                                      hard_best_spallation=hard_best_spallation,
                                                                      hard_best_exposed_bars=hard_best_exposed_bars,
                                                                      hard_best_efflorescence=hard_best_efflorescence,
                                                                      hard_best_corrosion_stain
                                                                      =hard_best_corrosion_stain, verbose=True)
                    # sample nets from replay buffer, update Q-values and save partially filled replay buffer and
                    # Q-values
                    q_learner.update_q_values_and_save_partial()

        # save fully filled replay buffer and final Q-values
        q_learner.save_final()

    # load single architecture config from replay buffer and train till convergence
    elif int(args.task) == 2:
        # raise exceptions if requirements to continue incomplete search not met
        if (args.replay_buffer_csv_path is None) or (not os.path.exists(args.replay_buffer_csv_path)):
            raise ValueError('wrong path specified for replay buffer')
        if int(args.fixed_net_index_no) < 0:
            raise ValueError('specify a non negative integer for fixed net index')

        # generate states for fixed net from a complete search
        q_learner.generate_fixed_net_states()

        # train/val fixed net exhaustively
        mem_fit, spp_size, hard_best_val, hard_val_all_epochs, soft_best_val, soft_val_all_epochs, train_flag,\
        hard_best_background, hard_best_crack, hard_best_spallation, hard_best_exposed_bars, hard_best_efflorescence, \
        hard_best_corrosion_stain = train_val_net(q_learner.state_list, dataset, weight_initializer, device, args,
                                                  save_path)

        # add fixed net and performance measures to a data frame and save it
        q_learner.add_fixed_net_to_fixed_net_buffer(spp_size=spp_size, hard_best_val=hard_best_val,
                                                    hard_val_all_epochs=hard_val_all_epochs,
                                                    soft_best_val=soft_best_val,
                                                    soft_val_all_epochs=soft_val_all_epochs,
                                                    hard_best_background=hard_best_background,
                                                    hard_best_crack=hard_best_crack,
                                                    hard_best_spallation=hard_best_spallation,
                                                    hard_best_exposed_bars=hard_best_exposed_bars,
                                                    hard_best_efflorescence=hard_best_efflorescence,
                                                    hard_best_corrosion_stain=hard_best_corrosion_stain)

        # save fixed net buffer
        q_learner.save_final()

    # raise exception if no matching task
    else:
        raise NotImplementedError('Given task no. not implemented.')


if __name__ == '__main__':
    main()
