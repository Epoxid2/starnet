import tensorflow as tf
from PIL import Image as img
import logging
tf.get_logger().setLevel(logging.ERROR)
from starnet_v1_TF2 import StarNet

import tifffile as tiff

starnet = StarNet(mode = 'RGB', window_size = 512, stride = 128)
starnet.load_model('./weights', './history')


Folder = "/Users/stephan/Pictures/stack/"

name = "Flaming_1_16"
#name = "P4c_16"
#name = "cropped_r_mixed_stacked2_16"
#name = "Andromeda2"

in_name = name+".tif"
out_name = name+"_ns.tif"

starnet.transform(Folder+in_name, Folder+out_name)
