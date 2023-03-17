import torch.nn as nn
import torch.functional as F
import torch
from global_vars import *

class Autoencoder(nn.Module):
    def __init__(self, n_channels = 1, number_of_filters = 128, device_name = "cpu"):
        super(Autoencoder, self).__init__()
        self.n_channels = n_channels

        nof = number_of_filters

        if WINDOW == "hamming":
            self.window_tensor = torch.hamming_window(N_FFT)
        elif WINDOW == "bartlett":
            self.window_tensor = torch.bartlett_window(N_FFT)
        elif WINDOW == "blackman":
            self.window_tensor = torch.blackman_window(N_FFT)
        else:
            print("Non valid window type. Proceeding with Boxcar.")
            self.window_tensor = torch.ones_like(torch.blackman_window(N_FFT))

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.tanh = nn.Tanh()

        self.featurizer = nn.Conv2d(1, nof, kernel_size = 3, padding = "same")
        self.down1 = nn.Conv2d(nof, 16, kernel_size = 3, padding = "same")
        self.down2 = nn.Conv2d(16, 8, kernel_size = 3, padding = "same")
        # self.down3 = nn.Conv2d(8, 4, kernel_size = 3, padding = "same")

        self.upsample = nn.Upsample(scale_factor = 2, mode = "bilinear", align_corners=True)

        # self.up1 = nn.ConvTranspose2d(4, 8, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.up2 = nn.ConvTranspose2d(8, 16, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.up3 = nn.ConvTranspose2d(16, 32, kernel_size = 3, stride = 2, output_padding = 1, padding = 1)
        self.output_layer = nn.Conv2d(32, 1, kernel_size = 1, padding = "same")

    def stft(self, audio):
        return torch.stft(audio, n_fft = N_FFT, hop_length = NOVERLAP, window = self.window_tensor, return_complex = True)

    def forward(self, x):
        # spectrogram generation
        spectrogram = self.stft(x)

        magnitude = torch.abs(spectrogram)
        phase = torch.angle(spectrogram)

        if len(magnitude.shape) < 3:
            magnitude = torch.reshape(magnitude, shape = (1,1, magnitude.shape[0], magnitude.shape[1]))
            phase = torch.reshape(phase, shape = (1,1, phase.shape[0], phase.shape[1]))

        else:
            magnitude = torch.reshape(magnitude, shape = (magnitude.shape[0], 1, magnitude.shape[1], magnitude.shape[2]))
            phase = torch.reshape(phase, shape = (phase.shape[0], 1, phase.shape[1], phase.shape[2]))

        magnitude = magnitude[:,:,0:-1,:]

        # Encoder
        y = self.featurizer(magnitude)
        y = self.max_pool(y)

        y = self.down1(y)
        y = self.max_pool(y)

        # Bottleneck
        y_bottleneck = self.down2(y)

        y = self.up2(y_bottleneck)

        y = self.up3(y)

        # output spectrogram
        output_magnitude = self.output_layer(y)

        # audio reconstruction in time domain
        predicted_spectrogram = torch.cat((output_magnitude, 
                                           torch.zeros(size = (output_magnitude.shape[0], output_magnitude.shape[1], 1, output_magnitude.shape[-1])).to(self.device)),
                                    dim = 2)
        
        predicted_spectrogram = predicted_spectrogram * torch.exp(1j * phase)

        predicted_spectrogram = torch.squeeze(predicted_spectrogram)

        output = torch.istft(predicted_spectrogram, n_fft = N_FFT, hop_length = NOVERLAP, window = self.window_tensor)

        return output, y_bottleneck
    
    def set_device(self, device):
        self.window_tensor = self.window_tensor.to(device)
        self.device = device