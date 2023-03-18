from models import *
from global_vars import *
from data_parser import *

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import auraloss
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensorboard_writer = SummaryWriter(f"runs/denoising")

# Metrics
multires_spec_similarity = auraloss.freq.MelSTFTLoss(n_mels = 64,
                                                     device = "cuda",
                                                     sample_rate = SR)
snr_similarity = auraloss.time.SNRLoss()
sdsdr_similarity = auraloss.time.SDSDRLoss()
mse_similarity = nn.MSELoss()

train_parser = UrbanSound8K_parser(chunk_size=CHUNK_SIZE)
test_parser = UrbanSound8K_parser(chunk_size=CHUNK_SIZE)

model = Autoencoder(n_channels=1, number_of_filters=64, device_name=device)
model.load_state_dict(torch.load("models/denoise_ckpt.pth"))
model.set_device(device)
model.to(device)
model.eval()

train_parser.prepare_folds(test_fold_no=1)
train_parser.set_device(device)

test_parser.prepare_folds(test_fold_no=1)
test_parser.set_as_annotations(test_parser.test_annotations)
test_parser.set_device(device)

train_loader = torch.utils.data.DataLoader(train_parser, batch_size = 1, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_parser, batch_size = 1, shuffle = False)

noise_levels = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


with open("denoising_results.csv", "w") as file:
    writer = csv.writer(file)

    writer.writerow(["Noise level (%RMS)", "Train - SNR (Noisy/Gt)", "Train - SNR (Cleaned/Gt)", 
                                            "Train - SDSDR (Noisy/Gt)", "Train - SDSDR (Cleaned/Gt)", 
                                            "Train - MultiResMel (Noisy/Gt)", "Train - MultiResMel (Cleaned/Gt)", 
                                            "Train - MSE (Noisy/Gt)", "Train - MSE (Cleaned/Gt)",
                                            
                                            "Test - SNR (Noisy/Gt)", "Test - SNR (Cleaned/Gt)", 
                                            "Test - SDSDR (Noisy/Gt)", "Test - SDSDR (Cleaned/Gt)", 
                                            "Test - MultiResMel (Noisy/Gt)", "Test - MultiResMel (Cleaned/Gt)", 
                                            "Test - MSE (Noisy/Gt)", "Test - MSE (Cleaned/Gt)"])
    with torch.no_grad():
        for noise_level in noise_levels:
            # Noisy to gt, Reconstructed to gt
            train_snr = [0,0]
            train_sdsdr = [0,0]
            train_mres_spec = [0,0]
            train_mse = [0,0]

            test_snr = [0,0]
            test_sdsdr = [0,0]
            test_mres_spec = [0,0]
            test_mse = [0,0]

            # Train set
            for iteration, (clean_audio, _) in enumerate(tqdm(train_loader)):
                audio_rms = torch.sqrt(torch.mean(torch.sum(torch.pow(clean_audio, 2))))

                noisy_audio = clean_audio + torch.rand_like(clean_audio)*noise_level*audio_rms

                # Clipping.
                noisy_audio[noisy_audio > 1] = 1
                noisy_audio[noisy_audio < -1] = -1

                outputs, _ = model(noisy_audio)

                train_snr[0] += snr_similarity(noisy_audio, clean_audio).item()
                train_snr[1] += snr_similarity(outputs, clean_audio).item()

                train_sdsdr[0] += sdsdr_similarity(noisy_audio, clean_audio).item()
                train_sdsdr[1] += sdsdr_similarity(outputs, clean_audio).item()

                train_mres_spec[0] += multires_spec_similarity(noisy_audio, clean_audio).item()
                train_mres_spec[1] += multires_spec_similarity(outputs, clean_audio).item()

                train_mse[0] += mse_similarity(torch.unsqueeze(noisy_audio, dim = 0), clean_audio).item()
                train_mse[1] += mse_similarity(torch.unsqueeze(outputs, dim =0), clean_audio).item()

            for iteration, (clean_audio, _) in enumerate(tqdm(test_loader)):
                audio_rms = torch.sqrt(torch.mean(torch.sum(torch.pow(clean_audio, 2))))

                noisy_audio = clean_audio + torch.rand_like(clean_audio)*noise_level*audio_rms
                outputs, _ = model(noisy_audio)

                test_snr[0] += snr_similarity(noisy_audio, clean_audio).item()
                test_snr[1] += snr_similarity(outputs, clean_audio).item()

                test_sdsdr[0] += sdsdr_similarity(noisy_audio, clean_audio).item()
                test_sdsdr[1] += sdsdr_similarity(outputs, clean_audio).item()

                test_mres_spec[0] += multires_spec_similarity(noisy_audio, clean_audio).item()
                test_mres_spec[1] += multires_spec_similarity(outputs, clean_audio).item()

                test_mse[0] += mse_similarity(torch.unsqueeze(noisy_audio, dim = 0), clean_audio).item()
                test_mse[1] += mse_similarity(torch.unsqueeze(outputs, dim =0), clean_audio).item()

            train_number_of_examples = len(train_loader)
            test_number_of_examples = len(test_loader)
            writer.writerow([noise_level, train_snr[0]/train_number_of_examples, train_snr[1]/train_number_of_examples,
                                            train_sdsdr[0]/train_number_of_examples, train_sdsdr[1]/train_number_of_examples,
                                            train_mres_spec[0]/train_number_of_examples, train_mres_spec[1]/train_number_of_examples,
                                            train_mse[0]/train_number_of_examples, train_mse[1]/train_number_of_examples,
                                            
                                            test_snr[0]/test_number_of_examples, test_snr[1]/test_number_of_examples,
                                            test_sdsdr[0]/test_number_of_examples, test_sdsdr[1]/test_number_of_examples,
                                            test_mres_spec[0]/test_number_of_examples, test_mres_spec[1]/test_number_of_examples,
                                            test_mse[0]/test_number_of_examples, test_mse[1]/test_number_of_examples,
                                            ])