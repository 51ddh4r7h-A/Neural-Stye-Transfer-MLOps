# Training configurations
NUM_STYLE = 16
STYLE_WEIGHT = 5.0
TV_WEIGHT = 1e-5
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
ITERATIONS = 40_000

# Data configurations
CONTENT_PATH = 'data/content'
STYLE_PATH = 'data/style'
IMSIZE = 256
CROPSIZE = 240
CENCROP = False

# Model configurations
CONTENT_NODES = ['relu_3_3']
STYLE_NODES = ['relu_1_2', 'relu_2_2', 'relu_3_3', 'relu_4_2']
RETURN_NODES = {3: 'relu_1_2',
                8: 'relu_2_2',
                15: 'relu_3_3',
                22: 'relu_4_2'}
MODEL_PATH = 'modelv2.ckpt'