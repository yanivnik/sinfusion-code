import os
from config_base import Config
from main import load_resnet50
########################################
# Some pre-made useful example configs #
########################################

# Niv NNF check
X2_NNF_SIMPLE = Config()
X2_NNF_SIMPLE.input_path = '/home/nivh/data/projects_data/torch_zssr/image_SRF_2_gt_part'
X2_NNF_SIMPLE.result_path = '/home/nivh/data/projects_data/torch_zssr/'
X2_NNF_SIMPLE.scale_factors = [[2.0, 2.0]]
X2_NNF_SIMPLE.back_projection_iters = [0]
X2_NNF_SIMPLE.name = 'simple_zssr'


# Simple
X2_TORCH_SIMPLE = Config()
X2_TORCH_SIMPLE.input_path = '/home/nivh/data/remote_projects/PytorchNeuralStyleTransfer/Images/SR/images/lr'
X2_TORCH_SIMPLE.result_path = '/home/nivh/data/remote_projects/PytorchNeuralStyleTransfer/Images/SR/images/'
X2_TORCH_SIMPLE.scale_factors = [[2.0, 2.0]]
X2_TORCH_SIMPLE.back_projection_iters = [0]
X2_TORCH_SIMPLE.name = 'simple_zssr'


# Debug
X2_DEBUG_CONF = Config()
X2_DEBUG_CONF.input_path = os.path.dirname(__file__) + '/example_with_gt'
X2_DEBUG_CONF.result_path = '/home/labs/waic/nivh/data/torch_zssr/results'
X2_DEBUG_CONF.scale_factors = [[2.0, 2.0]]
X2_DEBUG_CONF.back_projection_iters = [10]
X2_DEBUG_CONF.name = 'debug'

# Check torch vs. original ZSSR
X2_TORCH_VS_ORIG_CONF = Config()
X2_TORCH_VS_ORIG_CONF.input_path = '/home/nivh/data/datasets/SelfExSR/BSD100/image_SRF_2_gt' #/home/nivh/data/datasets/BSDS300_png/images/test'
X2_TORCH_VS_ORIG_CONF.result_path = '/home/nivh/data/torch_zssr/perceptual'
X2_TORCH_VS_ORIG_CONF.scale_factors = [[2.0, 2.0]]
X2_TORCH_VS_ORIG_CONF.back_projection_iters = [10]
X2_TORCH_VS_ORIG_CONF.name = 'perceptual_vgg16'
X2_TORCH_VS_ORIG_CONF.original_loss = False  # only L1 loss is used as in original ZSSR
X2_TORCH_VS_ORIG_CONF.content_weight = 1e5
X2_TORCH_VS_ORIG_CONF.style_weight = 0
X2_TORCH_VS_ORIG_CONF.perceptual_model = load_resnet50()

# Basic default config (same as not specifying), non-gradual SRx2 with default bicubic kernel (Ideal case)
# example is set to run on set14
X2_ONE_JUMP_IDEAL_CONF = Config()
X2_ONE_JUMP_IDEAL_CONF.input_path = os.path.dirname(__file__) + '/set14'

# Same as above but with visualization (Recommended for one image, interactive mode, for debugging)
X2_IDEAL_WITH_PLOT_CONF = Config()
X2_IDEAL_WITH_PLOT_CONF.plot_losses = True
X2_IDEAL_WITH_PLOT_CONF.run_test_every = 20
X2_IDEAL_WITH_PLOT_CONF.input_path = os.path.dirname(__file__) + '/example_with_gt'

# Gradual SRx2, to achieve superior results in the ideal case
X2_GRADUAL_IDEAL_CONF = Config()
X2_GRADUAL_IDEAL_CONF.scale_factors = [[1.0, 1.5], [1.5, 1.0], [1.5, 1.5], [1.5, 2.0], [2.0, 1.5], [2.0, 2.0]]
X2_GRADUAL_IDEAL_CONF.back_projection_iters = [6, 6, 8, 10, 10, 12]
X2_GRADUAL_IDEAL_CONF.input_path = os.path.dirname(__file__) + '/set14'

# Applying a given kernel. Rotations are canceled sense kernel may be non-symmetric
X2_GIVEN_KERNEL_CONF = Config()
X2_GIVEN_KERNEL_CONF.output_flip = False
X2_GIVEN_KERNEL_CONF.augment_allow_rotation = False
X2_GIVEN_KERNEL_CONF.back_projection_iters = [2]
X2_GIVEN_KERNEL_CONF.input_path = os.path.dirname(__file__) + '/kernel_example'

# An example for a typical setup for real images. (Kernel needed + mild unknown noise)
# back-projection is not recommended because of the noise.
X2_REAL_CONF = Config()
X2_REAL_CONF.output_flip = False
X2_REAL_CONF.back_projection_iters = [0]
X2_REAL_CONF.input_path = os.path.dirname(__file__) + '/real_example'
X2_REAL_CONF.noise_std = 0.0125
X2_REAL_CONF.augment_allow_rotation = False
X2_REAL_CONF.augment_scale_diff_sigma = 0
X2_REAL_CONF.augment_shear_sigma = 0
X2_REAL_CONF.augment_min_scale = 0.75
