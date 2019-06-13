"""
config file for defining search space rules and Q-learning hyperparameters
"""

########################
# search space rules
########################

# possible square conv kernel size
# kernel size from [3, 5] => stride = 1; conv size from [7, 9, 11] => stride = 2
conv_sizes = [3, 5, 7, 9, 11]
conv_features = [32, 64, 128, 256]  # possible number of conv features
conv_padding = 'VALID'  # conv_padding = 'VALID' => no padding applied; 'SAME' => padding for same output dimensionality

spp_sizes = [3, 4, 5]   # possible number of spp scales

fc_sizes = [32, 64, 128]    # possible FC layer sizes

################################
# q-learning hyperparameters
################################

# epsilon schedule for Q-learning
# Format : [[epsilon, # unique models]]
epsilon_schedule = [[1.0, 100],
                    [0.9, 10],
                    [0.8, 10],
                    [0.7, 10],
                    [0.6, 10],
                    [0.5, 10],
                    [0.4, 10],
                    [0.3, 10],
                    [0.2, 15],
                    [0.1, 15]]

replay_number = 40  # number of samples drawn from the replay buffer for Q-values update
