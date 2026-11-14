"""GPEM12 (trainer name nnUNetTrainerGPEM15)

Same image, more preprocessing as input (see geom2.py): more radial-profile
scales, gradient and local orientation.

Train: nnUNetv2_train <ID> 2d <FOLD> -tr nnUNetTrainerGPEM15
"""
try:
    from .accept_trainer import nnUNetTrainerAccept
    from .maps12 import maps as maps2
    from .gpem7 import install
except ImportError:
    from accept_trainer import nnUNetTrainerAccept
    from maps12 import maps as maps2
    from gpem7 import install


class nnUNetTrainerGPEM15(nnUNetTrainerAccept):
    #The delivered method: nnUNetTrainerGPEM15, GPEM with a 12-channel prior.

    @staticmethod
    def build_network_architecture(plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        net = nnUNetTrainerAccept.build_network_architecture(plans_manager, configuration_manager,
                                                             num_input_channels, num_output_channels,
                                                             enable_deep_supervision)
        install(net, tuple(max(p // 2, 64) for p in configuration_manager.patch_size), maps2)
        return net