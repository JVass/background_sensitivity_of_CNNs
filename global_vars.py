PATH_TO_DATASET = ""

SR = 48000

# STFT parameters
N_FFT = 512
NOVERLAP = 256
WINDOW = "hamming"

# Chunk sizes
CHUNK_SIZE = NOVERLAP*254

# Learning parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 100

# Early stopping parameters
PATIENCE = 3
MIN_DELTA = 0.01

# Loss function weights
TIME_WEIGHT = 10
FREQ_WEIGHT = 0.05