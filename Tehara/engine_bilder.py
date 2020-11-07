from tensorrt_uff import Tensorrt

model = Tensorrt(weights_file, num_colors, net_height, net_width, inputs=None, outputs=None)