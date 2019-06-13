########################
# importing libraries
########################
# system libraries
import math

# custom libraries
import lib.Models.state_space_parameters as state_space_parameters


class State:
    """
    creates a state for a layer in the net

    Parameters:
        layer_type (string): type of layer
        layer_depth (int): position of layer in the network
        filter_depth (int): number of channels/features in the output of the layer
        filter_size (int): height/width of the square filter
        stride (int): stride value for strided convolutional layer
        image_size (int): height/width of the feature space output by the layer
        fc_size (int): size of layer if fully connected
        terminate (int): 1 if the state is the last state, else 0
        state_list (list): sequence of states can be built from a state list which takes precedence if specified
    """
    def __init__(self,
                 layer_type=None,
                 layer_depth=None,
                 filter_depth=None,
                 filter_size=None,
                 stride=None,
                 image_size=None,
                 fc_size=None,
                 terminate=None,
                 state_list=None):
        if not state_list:
            self.layer_type = layer_type
            self.layer_depth = layer_depth
            self.filter_depth = filter_depth
            self.filter_size = filter_size
            self.stride = stride
            self.image_size = image_size
            self.fc_size = fc_size
            self.terminate = terminate
        else:
            self.layer_type = state_list[0]
            self.layer_depth = state_list[1]
            self.filter_depth = state_list[2]
            self.filter_size = state_list[3]
            self.stride = state_list[4]
            self.image_size = state_list[5]
            self.fc_size = state_list[6]
            self.terminate = state_list[7]

    def as_tuple(self):
        """
        Returns:
            tuple containing the state attributes
        """
        return (self.layer_type,
                self.layer_depth,
                self.filter_depth,
                self.filter_size,
                self.stride,
                self.image_size,
                self.fc_size,
                self.terminate)

    def as_list(self):
        """
        Returns:
            list containing the state attributes
        """
        return list(self.as_tuple())

    def copy(self):
        """
        Returns:
            copy of the state
        """
        return State(self.layer_type,
                     self.layer_depth,
                     self.filter_depth,
                     self.filter_size,
                     self.stride,
                     self.image_size,
                     self.fc_size,
                     self.terminate)


class StateEnumerator:
    """
    populates the state-action space, possible transitions given a state

    Parameters:
        args (argparse..Namespace): parsed command line arguments

    Attributes:
        init_utility (float): initial Q-values, set to the early stopping threshold heuristically
    """
    def __init__(self, args):
        self.args = args
        # this assignment is redundant but only present for verbosity
        self.init_utility = args.early_stopping_thresh

    def enumerate_state(self, state, q_values):
        """
        defines all state transitions, populates q_values where actions are valid
        
        Parameters:
            state (lib.MetaQNN.state_enumerator.State): current state
            q_values (): partially populated Q-value dictionary
        """

        actions = []

        if state.terminate == 0:

            # fc / spp -> terminal
            if state.layer_type == 'fc' or state.layer_type == 'spp':
                actions += [State(layer_type=state.layer_type,
                                  layer_depth=state.layer_depth + 1,
                                  filter_depth=state.filter_depth,
                                  filter_size=state.filter_size,
                                  stride=state.stride,
                                  image_size=state.image_size,
                                  fc_size=state.fc_size,
                                  terminate=1)]

            # spp -> first fc
            if state.layer_type == 'spp':
                for fc_size in self._fc_sizes(state):
                    actions += [State(layer_type='fc',
                                      layer_depth=state.layer_depth + 1,
                                      filter_depth=0,
                                      filter_size=0,
                                      stride=0,
                                      image_size=0,
                                      fc_size=fc_size,
                                      terminate=0)]

            # fc -> fc
            if state.layer_type == 'fc' and state.filter_depth < self.args.max_fc - 1:
                for fc_size in self._fc_sizes(state):
                    actions += [State(layer_type='fc',
                                      layer_depth=state.layer_depth + 1,
                                      filter_depth=state.filter_depth + 1,
                                      filter_size=0,
                                      stride=0,
                                      image_size=0,
                                      fc_size=fc_size,
                                      terminate=0)]

            if (state.layer_type in ['start', 'conv', 'wrn'] and state.layer_depth < self.args.conv_layer_min_limit) or\
               (state.layer_type in ['conv', 'wrn'] and self.args.conv_layer_min_limit <= state.layer_depth
                < self.args.conv_layer_max_limit):
                # (start /) conv / wrn -> conv if current image_size (= new image_size if conv_padding = 'SAME')
                # < minimum possible number of spp scales
                for conv_feature in state_space_parameters.conv_features:
                    # iterate over conv sizes which are less than the current image size
                    for conv_size in self._conv_sizes(state.image_size):
                        if state_space_parameters.conv_padding == 'SAME' and state.image_size >=\
                                min(state_space_parameters.spp_sizes):
                            actions += [State(layer_type='conv',
                                              layer_depth=state.layer_depth + 1,
                                              filter_depth=conv_feature,
                                              filter_size=conv_size,
                                              stride=1 if conv_size <= 5 else 2,
                                              image_size=state.image_size,
                                              fc_size=0,
                                              terminate=0)]
                        elif state_space_parameters.conv_padding == 'VALID'\
                                and self._calc_new_image_size(state.image_size, conv_size)\
                                >= min(state_space_parameters.spp_sizes):
                            actions += [State(layer_type='conv',
                                              layer_depth=state.layer_depth + 1,
                                              filter_depth=conv_feature,
                                              filter_size=conv_size,
                                              stride=1 if conv_size <= 5 else 2,
                                              image_size=self._calc_new_image_size(state.image_size, conv_size),
                                              fc_size=0,
                                              terminate=0)]

                # (start /) conv / wrn -> wrn if current image_size (= new image_size for wrn)
                # < minimum possible number of spp scales
                for conv_feature in state_space_parameters.conv_features:
                    if state.image_size > 3:
                        actions += [State(layer_type='wrn',
                                          layer_depth=state.layer_depth + 2,
                                          filter_depth=conv_feature,
                                          filter_size=3,
                                          stride=1,
                                          image_size=state.image_size,
                                          fc_size=0,
                                          terminate=0)]

            # conv / wrn -> spp
            if state.layer_type in ['conv', 'wrn'] and self.args.conv_layer_min_limit <= state.layer_depth:
                for spp_size in state_space_parameters.spp_sizes:
                    if state.image_size >= spp_size:
                        actions += [State(layer_type='spp',
                                          layer_depth=state.layer_depth + 1,
                                          filter_depth=state.filter_depth,
                                          filter_size=spp_size,
                                          stride=0,
                                          image_size=int(spp_size * (spp_size + 1) * (2 * spp_size + 1) / 6.),
                                          fc_size=0,
                                          terminate=0)]

        # Add states to transition and Q-values dictionary
        q_values[state.as_tuple()] = {'actions': [to_state.as_tuple() for to_state in actions],
                                      'utilities': [self.init_utility for _ in range(len(actions))]}

    def _fc_sizes(self, state=None):
        """
        get possible fc sizes given a state
        
        Parameters:
            state (lib.MetaQNN.state_enumerator.State): current state
        
        Returns:
            fc_sizes (list): all possible fc sizes for the next state
        """
        # for fc layer, next fc layers have smaller fc_size
        if not isinstance(state, type(None)) and state.layer_type == 'fc':
            fc_sizes = [i for i in state_space_parameters.fc_sizes if i <= state.fc_size]
        else:
            fc_sizes = state_space_parameters.fc_sizes

        return fc_sizes

    def _conv_sizes(self, image_size):
        """
        get possible conv sizes given a state
        
        Parameters:
            image_size (int): current image size
        
        Returns:
            conv_sizes (list): all possible conv square kernel sizes for the given image size
        """
        conv_sizes = [conv_size for conv_size in state_space_parameters.conv_sizes if conv_size < image_size]

        return conv_sizes

    def _calc_new_image_size(self, image_size, filter_size):
        """
        get new image size

        Parameters:
            image_size (int): current image size
            filter_size (int): conv square kernel size

        Returns:
            new_size (int): image size after applying this conv filter
        """
        if filter_size <= 5:
            new_size = int(math.floor((image_size - filter_size) / 1 + 1))
        else:
            new_size = int(math.floor((image_size - filter_size) / 2 + 1))

        return new_size
