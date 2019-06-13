########################
# importing libraries
########################
# system libraries
import math

# custom libraries
import lib.MetaQNN.state_enumerator as se
import lib.Models.state_space_parameters as state_space_parameters


class StateStringUtils:
    """
    parses state lists to strings and vice versa
    """
    def __init__(self):
        pass

    def state_list_to_string(self, state_list, num_classes):
        """
        parses state list to string
        
        Parameters:
            state_list (list): state list for parsing
            num_classes (int): number of classes in the dataset
        
        Returns:
            string describing states from the state list
        """

        strings = []
        i = 0
        while i < len(state_list):
            state = state_list[i]
            if self._state_to_string(state, num_classes):
                strings.append(self._state_to_string(state, num_classes))
            i += 1
        return str('[' + ', '.join(strings) + ']')

    def parsed_list_to_state_list(self, parsed_list, patch_size):
        """
        converts a list output by cnn.py after parsing a string, to a state list
        
        Parameters:
            parsed_list (list): list output by cnn.py upon parsing a string
            patch_size (int): patch size of image input to the network
        
        Returns:
            states (list): list of states from parsed list
        """

        states = [se.State('start', 0, 0, 0, 0, patch_size, 0, 0)]

        for layer in parsed_list:
            # conv layer
            if layer[0] == 'conv' and layer[4] == 1:
                states.append(se.State(layer_type='conv',
                                       layer_depth=states[-1].layer_depth + 1,
                                       filter_depth=layer[1],
                                       filter_size=layer[2],
                                       stride=layer[3],
                                       image_size=states[-1].image_size if state_space_parameters.conv_padding == 'SAME'
                                       else self._calc_new_image_size(states[-1].image_size, layer[2]),
                                       fc_size=0,
                                       terminate=0))
            # wrn block
            elif layer[0] == 'conv' and layer[4] == 0:
                states.append(se.State(layer_type='wrn',
                                       layer_depth=states[-1].layer_depth + 2,
                                       filter_depth=layer[1],
                                       filter_size=3,
                                       stride=1,
                                       image_size=states[-1].image_size,
                                       fc_size=0,
                                       terminate=0))
            # spp layer
            elif layer[0] == 'conv' and layer[4] == 2:
                states.append(se.State(layer_type='spp',
                                       layer_depth=states[-1].layer_depth + 1,
                                       filter_depth=layer[1],
                                       filter_size=layer[2],
                                       stride=0,
                                       image_size=int(layer[2] * (layer[2] + 1) * (2 * layer[2] + 1) / 6.),
                                       fc_size=0,
                                       terminate=0))
            elif layer[0] == 'fc':
                states.append(se.State(layer_type='fc',
                                       layer_depth=states[-1].layer_depth+1,
                                       filter_depth=len([state for state in states if state.layer_type == 'fc']),
                                       filter_size=0,
                                       stride=0,
                                       image_size=0,
                                       fc_size=layer[1],
                                       terminate=0))
            elif layer[0] == 'softmax':
                termination_state = states[-1].copy()
                termination_state.terminate = 1
                termination_state.layer_depth += 1
                states.append(termination_state)

        return states

    def _state_to_string(self, state, num_classes):
        """
        parses an individual state to string
        
        Parameters:
            state (lib.MetaQNN.state_enumerator.State): state for parsing
            num_classes (int): number of classes in the dataset
        
        Returns:
            string for an individual state
        """
        if state.terminate == 1:
            return 'SM(%i)' % num_classes
        # 0 at the end => wrn block
        elif state.layer_type == 'wrn':
            return 'C(%i,%i,%i,%i)' % (state.filter_depth, state.filter_size, state.stride, 0)
        # 1 at the end => conv layer
        elif state.layer_type == 'conv':
            return 'C(%i,%i,%i,%i)' % (state.filter_depth, state.filter_size, state.stride, 1)
        # 2 at the end => spp layer
        elif state.layer_type == 'spp':
            return 'C(%i,%i,%i,%i)' % (state.filter_depth, state.filter_size, state.stride, 2)
        elif state.layer_type == 'fc':
            return 'FC(%i)' % state.fc_size

    def _calc_new_image_size(self, image_size, filter_size):
        """
        return new image size
        
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
