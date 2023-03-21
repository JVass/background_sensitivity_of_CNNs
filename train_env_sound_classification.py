from models import *
from global_vars import *
from data_parser import *

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score, multiclass_confusion_matrix
from torcheval.metrics.functional import multiclass_precision, multiclass_recall 

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensorboard_writer = SummaryWriter(f"runs/env_sound_classification")

classification_loss = torch.nn.CrossEntropyLoss()

# Metrics (input, target)
f1_score = multiclass_f1_score()
accuracy = multiclass_accuracy
precision = multiclass_precision
recall = multiclass_recall
confusion_matrix = multiclass_confusion_matrix

train_parser = UrbanSound8K_parser(chunk_size=CHUNK_SIZE)
test_parser = UrbanSound8K_parser(chunk_size=CHUNK_SIZE)

model = DenseNet()
model.freeze_layers()
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

train_parser.prepare_folds(test_fold_no=1)
train_parser.set_device(device)

test_parser.prepare_folds(test_fold_no=1)
test_parser.set_as_annotations(test_parser.test_annotations)
test_parser.set_device(device)

train_loader = torch.utils.data.DataLoader(train_parser, batch_size = BATCH_SIZE, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_parser, batch_size = BATCH_SIZE, shuffle = False)

prev_train_loss = 10e5
prev_test_loss = 10e5
best_loss = 10e5
built_rage = 0

for epoch in range(EPOCHS):
    print(f"Epoch: {epoch + 1}")
    train_loss = 0

    for iteration, (audio, true_labels) in enumerate(tqdm(train_loader)):    
        optimizer.zero_grad()

        output_labels = model(audio)

        loss_value = classification_loss(output_labels, true_labels)

        loss_value.backward()
        optimizer.step()

#         train_loss += loss.item()
#         snr_sim = snr_similarity(outputs, audio)
#         sdsdr_sim = sdsdr_similarity(outputs, audio)
#         multires_spec_sim = multires_spec_similarity(outputs, audio)

#         iter_no = iteration + epoch*len(train_loader)
#         tensorboard_writer.add_scalars("Train", {"Train loss": loss,
#                                                         "Train loss (time)": loss_time,
#                                                         "Train loss (freq)": loss_freq,
#                                                        "Train snr" : snr_sim.item(),
#                                                        "Train SDSDR": sdsdr_sim.item(),
#                                                        "Train MresSTFT": multires_spec_sim.item()},
#                                                         iter_no)
        
#         # Visualization
#         if (iter_no % 50 == 0):
#             # Train-set
#             audio, _ = train_parser[5]

#             outputs, _ = model(audio)
            
#             gt_spectrogram = torch.abs(model.stft(audio[:]))
#             pred_spectrogram = torch.abs(model.stft(outputs[:]))

#             max_db_value = gt_spectrogram.max()

#             plt.pcolormesh(torch.squeeze(torch.log10(gt_spectrogram)).cpu().numpy(), vmin = -10, vmax = max_db_value)
#             plt.savefig("images/gt_spec.png")
#             plt.close()

#             plt.pcolormesh(torch.squeeze(torch.log10(pred_spectrogram)).detach().cpu().numpy(), vmin = -10, vmax = max_db_value)
#             plt.savefig("images/pred_spec.png")
#             plt.close()

#             plt.plot(audio.detach().cpu().numpy())
#             plt.savefig("images/gt_audio.png")
#             plt.close()

#             plt.plot(outputs.detach().cpu().numpy())
#             plt.savefig("images/pred_audio.png")
#             plt.close()

#             spec_photo = torchvision.io.read_image("images/gt_spec.png")
#             tensorboard_writer.add_image("Train groundtruth_spec", spec_photo[0:3, ::, ::], 0)

#             spec_photo = torchvision.io.read_image("images/pred_spec.png")
#             tensorboard_writer.add_image("Train predicted_spec", spec_photo[0:3, ::, ::], iter_no)

#             spec_photo = torchvision.io.read_image("images/gt_audio.png")
#             tensorboard_writer.add_image("Train groundtruth_audio", spec_photo[0:3, ::, ::], 0)

#             spec_photo = torchvision.io.read_image("images/pred_audio.png")
#             tensorboard_writer.add_image("Train predicted_audio", spec_photo[0:3, ::, ::], iter_no)

#             # Test-set
#             audio, _ = test_parser[8]

#             outputs, _ = model(audio)
            
#             gt_spectrogram = torch.abs(model.stft(audio[:]))
#             pred_spectrogram = torch.abs(model.stft(outputs[:]))

#             max_db_value = gt_spectrogram.max()

#             plt.pcolormesh(torch.squeeze(torch.log10(gt_spectrogram)).cpu().numpy(), vmin = -10, vmax = max_db_value)
#             plt.savefig("images/test_gt_spec.png")
#             plt.close()

#             plt.pcolormesh(torch.squeeze(torch.log10(pred_spectrogram)).detach().cpu().numpy(), vmin = -10, vmax = max_db_value)
#             plt.savefig("images/test_pred_spec.png")
#             plt.close()

#             plt.plot(audio.detach().cpu().numpy())
#             plt.savefig("images/test_gt_audio.png")
#             plt.close()

#             plt.plot(outputs.detach().cpu().numpy())
#             plt.savefig("images/test_pred_audio.png")
#             plt.close()

#             spec_photo = torchvision.io.read_image("images/test_gt_spec.png")
#             tensorboard_writer.add_image("Test groundtruth_spec", spec_photo[0:3, ::, ::], 0)

#             spec_photo = torchvision.io.read_image("images/test_pred_spec.png")
#             tensorboard_writer.add_image("Test predicted_spec", spec_photo[0:3, ::, ::], iter_no)

#             spec_photo = torchvision.io.read_image("images/test_gt_audio.png")
#             tensorboard_writer.add_image("Test groundtruth_audio", spec_photo[0:3, ::, ::], 0)

#             spec_photo = torchvision.io.read_image("images/test_pred_audio.png")
#             tensorboard_writer.add_image("Test predicted_audio", spec_photo[0:3, ::, ::], iter_no)

#     train_loss_per_epoch = train_loss / len(train_loader)
#     test_loss = 0
#     snr_sim = 0
#     sdsdr_sim = 0
#     multires_spec_sim = 0

#     # Test loss
#     with torch.no_grad():
#         for iteration, (audio, _) in enumerate(tqdm(test_loader)):
#             outputs, _ = model(audio)

#             loss_time = TIME_WEIGHT*time_loss(audio, outputs) 
#             loss_freq = FREQ_WEIGHT*freq_loss(audio, outputs)
#             loss = loss_time + loss_freq

#             test_loss += loss.item()
#             snr_sim = snr_similarity(outputs, audio)
#             sdsdr_sim = sdsdr_similarity(outputs, audio)
#             multires_spec_sim = multires_spec_similarity(outputs, audio)

#         test_loss_per_epoch = test_loss / len(test_loader)
#         snr_sim /= len(test_loader)
#         sdsdr_sim /= len(test_loader)
#         multires_spec_sim /= len(test_loader) 

#         tensorboard_writer.add_scalars("Test", {"Test loss": test_loss,
#                                                         "Test loss (time)": loss_time,
#                                                         "Test loss (freq)": loss_freq,
#                                                     "Test snr" : snr_sim.item(),
#                                                     "Test SDSDR": sdsdr_sim.item(),
#                                                     "Test MresSTFT": multires_spec_sim.item()},
#                                                         epoch)

    
#     # Early stopping
#     if best_loss > test_loss_per_epoch:
#         built_rage = 0
#         best_loss = test_loss_per_epoch
#         torch.save(model.state_dict(), "models/env_sound_classification_ckpt.pth")
#     elif np.abs(prev_test_loss - test_loss_per_epoch) < MIN_DELTA:
#         built_rage += 1
    
    
#     if built_rage >= PATIENCE:
#         print(f"Early stopping at: {epoch}")
#         print(f"Last checkpoint at epoch: {epoch - PATIENCE}")
#         break

#     prev_test_loss = test_loss_per_epoch
#     print(f"Rage: {built_rage}")
#     print(f"Mean loss (train): {train_loss_per_epoch}")
#     print(f"Mean loss (test): {test_loss_per_epoch}")
#     print("---------------------")

# torch.save(model.state_dict(), "models/env_sound_classification.pth")