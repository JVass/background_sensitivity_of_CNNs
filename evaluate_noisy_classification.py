from models import *
from global_vars import *
from data_parser import *

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import auraloss
import pandas as pd
import seaborn as sns
import numpy as np
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score, multiclass_confusion_matrix
from torcheval.metrics.functional import multiclass_precision 

import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Metrics
f1_score = multiclass_f1_score
accuracy = multiclass_accuracy
precision = multiclass_precision
confusion_matrix = multiclass_confusion_matrix

train_parser = UrbanSound8K_parser(chunk_size=CHUNK_SIZE)
test_parser = UrbanSound8K_parser(chunk_size=CHUNK_SIZE)

densenet_model = DenseNet()
densenet_model.load_state_dict(torch.load("models/env_sound_classification_ckpt.pth"))
densenet_model.set_device(device)
densenet_model = densenet_model.to(device)
densenet_model.eval()

autoencoder_model = Autoencoder(n_channels=1, number_of_filters=64, device_name=device)
autoencoder_model.load_state_dict(torch.load("models/denoise_ckpt.pth"))
autoencoder_model.set_device(device)
autoencoder_model.to(device)
autoencoder_model.eval()

train_parser.prepare_folds(test_fold_no=1)
train_parser.set_device(device)

test_parser.prepare_folds(test_fold_no=1)
test_parser.set_as_annotations(test_parser.test_annotations)
test_parser.set_device(device)

train_loader = torch.utils.data.DataLoader(train_parser, batch_size = 64, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_parser, batch_size = 64, shuffle = False)

with open("results/noisy_autoenc_to_densenet_classification_results.csv", "w") as file:
    writer = csv.writer(file)

    writer.writerow(["Train - F1 Score", "Train - Accuracy", "Train - Precision",
                     
                     "Test - F1 Score", "Test - Accuracy", "Test - Precision"])
    
    with torch.no_grad():
        # Noisy to gt, Reconstructed to gt
        train_f1_score = 0
        train_accuracy_score = 0
        train_precision = 0

        test_f1_score = 0
        test_accuracy_score = 0
        test_precision = 0
        test_confusion_matrix = torch.zeros(size = (10,10)).type(torch.int64).to(device)

        # Train set
        for iteration, (audio, true_labels) in enumerate(tqdm(train_loader)):
            # adding 0.01% of audio rms noise to audio
            rms_per_song = torch.sqrt(torch.mean(torch.pow(audio, 2), dim = 1)) * 0.1

            for song_index in range(rms_per_song.shape[0]):
                audio[song_index, :] = audio[song_index, :] + torch.rand_like(audio[song_index, :]) * rms_per_song[song_index]

            # clip
            audio[audio > 1] = 1
            audio[audio < -1] = -1

            # use autoencoder first
            audio, _ = autoencoder_model(audio)

            # test classification with autoencoded ('denoised/simplified')
            output_labels = densenet_model(audio)

            true_labels_one_hot = torch.nn.functional.one_hot(torch.squeeze(true_labels).type(torch.int64), 10).type(torch.float32)

            f1_value = f1_score(output_labels, torch.squeeze(true_labels).type(torch.int64), average="macro", num_classes=10)
            accuracy_value = accuracy(output_labels, torch.squeeze(true_labels).type(torch.int64), average="macro", num_classes=10)
            precision_value = precision(output_labels, torch.squeeze(true_labels).type(torch.int64), average="macro", num_classes=10)

            train_f1_score += f1_value.item()
            train_accuracy_score += accuracy_value.item()
            train_precision += precision_value.item()

        for iteration, (audio, true_labels) in enumerate(tqdm(test_loader)):
            rms_per_song = torch.sqrt(torch.mean(torch.pow(audio, 2), dim = 1)) * 0.1

            for song_index in range(rms_per_song.shape[0]):
                audio[song_index, :] = audio[song_index, :] + torch.rand_like(audio[song_index, :]) * rms_per_song[song_index]

            # clip
            audio[audio > 1] = 1
            audio[audio < -1] = -1

            audio, _ = autoencoder_model(audio)
            output_labels = densenet_model(audio)

            true_labels_one_hot = torch.nn.functional.one_hot(torch.squeeze(true_labels).type(torch.int64), 10).type(torch.float32)

            f1_value = f1_score(output_labels, torch.squeeze(true_labels).type(torch.int64), average="macro", num_classes=10)
            accuracy_value = accuracy(output_labels, torch.squeeze(true_labels).type(torch.int64), average="macro", num_classes=10)
            precision_value = precision(output_labels, torch.squeeze(true_labels).type(torch.int64), average="macro", num_classes=10)

            test_f1_score += f1_value.item()
            test_accuracy_score += accuracy_value.item()
            test_precision += precision_value.item()
            test_confusion_matrix += confusion_matrix(output_labels, torch.squeeze(true_labels).type(torch.int64), num_classes=10)

        train_number_of_examples = len(train_loader)
        test_number_of_examples = len(test_loader)

        writer.writerow([train_f1_score/train_number_of_examples, train_accuracy_score/train_number_of_examples, train_precision/train_number_of_examples,
                            
                            test_f1_score/test_number_of_examples, test_accuracy_score/test_number_of_examples, test_precision/test_number_of_examples])
    
    # Saving the confusion matrix
    confusion_matrix_df = pd.DataFrame(test_confusion_matrix.cpu() / torch.sum(test_confusion_matrix, dim = 1).cpu())
    sns.heatmap(confusion_matrix_df, annot=True)
    plt.savefig("results/noisy_autoenc_to_densenet_confusion_matrix.png")
    plt.close()
                            