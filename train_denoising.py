from models import *
from global_vars import *
from data_parser import *

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import auraloss
import matplotlib.pyplot as plt
import torchvision

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensorboard_writer = SummaryWriter(f"runs/denoising")


time_loss = torch.nn.L1Loss()
freq_loss = auraloss.freq.STFTLoss(w_log_mag = 1.0,
                                       w_lin_mag = 1.0,
                                       w_sc = 0.0,
                                       device = "cuda")

# Metrics
multires_spec_similarity = auraloss.freq.MelSTFTLoss(n_mels = 64,
                                                     device = "cuda",
                                                     sample_rate = SR)
snr_similarity = auraloss.time.SNRLoss()
sdsdr_similarity = auraloss.time.SDSDRLoss()

parser = UrbanSound8K_parser(chunk_size=CHUNK_SIZE)

model = Autoencoder(n_channels=1, number_of_filters=64, device_name=device)
model.set_device(device)
model.window_tensor = model.window_tensor.to(device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

# Setting the first fold as the test case.
parser.prepare_folds(test_fold_no=1)
parser.set_device(device)

train_loader = torch.utils.data.DataLoader(parser, batch_size = BATCH_SIZE, shuffle = True)

for epoch in range(EPOCHS):
    print(f"Epoch: {epoch + 1}")
    train_loss = 0
    prev_train_loss = 10e5
    best_epoch = 0
    best_loss = 10e5

    for iteration, (audio, _) in enumerate(tqdm(train_loader)):    
        optimizer.zero_grad()

        outputs, _ = model(audio)

        loss_time = TIME_WEIGHT*time_loss(audio, outputs) 
        loss_freq = FREQ_WEIGHT*freq_loss(audio, outputs)
        loss = loss_time + loss_freq

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        snr_sim = snr_similarity(outputs, audio)
        sdsdr_sim = sdsdr_similarity(outputs, audio)
        multires_spec_sim = multires_spec_similarity(outputs, audio)

        iter_no = iteration + epoch*len(train_loader)
        tensorboard_writer.add_scalars("Loss scalar", {"Train loss": loss,
                                                        "Train loss (time)": loss_time,
                                                        "Train loss (freq)": loss_freq,
                                                       "Train snr" : snr_sim.item(),
                                                       "Train SDSDR": sdsdr_sim.item(),
                                                       "Train MresSTFT": multires_spec_sim.item()},
                                                        iter_no)
        
        # Visualization
        if (iter_no % 5 == 0):
            # Train-set
            audio, _ = parser[5]

            outputs, _ = model(audio)
            
            gt_spectrogram = torch.abs(model.stft(audio[:]))
            pred_spectrogram = torch.abs(model.stft(outputs[:]))

            max_db_value = gt_spectrogram.max()

            plt.pcolormesh(torch.squeeze(torch.log10(gt_spectrogram)).cpu().numpy(), vmin = -10, vmax = max_db_value)
            plt.savefig("images/gt_spec.png")
            plt.close()

            plt.pcolormesh(torch.squeeze(torch.log10(pred_spectrogram)).detach().cpu().numpy(), vmin = -10, vmax = max_db_value)
            plt.savefig("images/pred_spec.png")
            plt.close()

            plt.plot(audio.detach().cpu().numpy())
            plt.savefig("images/gt_audio.png")
            plt.close()

            plt.plot(outputs.detach().cpu().numpy())
            plt.savefig("images/pred_audio.png")
            plt.close()

            spec_photo = torchvision.io.read_image("images/gt_spec.png")
            tensorboard_writer.add_image("Groundtruth_spec", spec_photo[0:3, ::, ::], 0)

            spec_photo = torchvision.io.read_image("images/pred_spec.png")
            tensorboard_writer.add_image("Predicted_spec", spec_photo[0:3, ::, ::], iter_no)

            spec_photo = torchvision.io.read_image("images/gt_audio.png")
            tensorboard_writer.add_image("Groundtruth_audio", spec_photo[0:3, ::, ::], 0)

            spec_photo = torchvision.io.read_image("images/pred_audio.png")
            tensorboard_writer.add_image("Predicted_audio", spec_photo[0:3, ::, ::], iter_no)


    train_loss_per_epoch = train_loss / len(train_loader)

    # Early stopping
    if best_loss > train_loss_per_epoch:
        building_rage = 0
        best_loss = train_loss_per_epoch
        torch.save(model.state_dict(), "models/denoise_ckpt.pth")
    elif torch.abs(prev_train_loss - train_loss_per_epoch) < MIN_DELTA:
        building_rage += 1
    elif building_rage == PATIENCE:
        print(f"Early stopping at: {epoch}")
        break

    prev_train_loss = train_loss_per_epoch
    print(f"Mean loss: {train_loss / len(train_loader)}")
    print("---------------------")

torch.save(model.state_dict(), "models/denoiser.pth")