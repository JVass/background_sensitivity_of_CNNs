from models import *
from global_vars import *
from data_parser import *

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score, multiclass_confusion_matrix
from torcheval.metrics.functional import multiclass_precision 

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensorboard_writer = SummaryWriter(f"runs/env_sound_classification")

classification_loss = torch.nn.CrossEntropyLoss()

# Metrics (input, target)
f1_score = multiclass_f1_score
accuracy = multiclass_accuracy
precision = multiclass_precision
confusion_matrix = multiclass_confusion_matrix

train_parser = UrbanSound8K_parser(chunk_size=CHUNK_SIZE)
test_parser = UrbanSound8K_parser(chunk_size=CHUNK_SIZE)

model = DenseNet()
model.freeze_layers()
model.set_device(device)
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

        true_labels_one_hot = torch.nn.functional.one_hot(torch.squeeze(true_labels).type(torch.int64), 10).type(torch.float32)

        loss_value = classification_loss(output_labels, true_labels_one_hot)

        loss_value.backward()
        optimizer.step()

        train_loss += loss_value.item()

        f1_value = f1_score(output_labels, torch.squeeze(true_labels).type(torch.int64), average="macro", num_classes=10)
        accuracy_value = accuracy(output_labels, torch.squeeze(true_labels).type(torch.int64), average="macro", num_classes=10)
        precision_value = precision(output_labels, torch.squeeze(true_labels).type(torch.int64), average="macro", num_classes=10)

        iter_no = iteration + epoch*len(train_loader)
        tensorboard_writer.add_scalars("Train", {"Train loss": loss_value.item(),
                                                       "Train F1" : f1_value.item(),
                                                       "Train Accuracy": accuracy_value.item(),
                                                       "Train Precision": precision_value.item()},
                                                        iter_no)
        
    train_loss_per_epoch = train_loss / len(train_loader)
    test_loss = 0
    f1_value = 0
    accuracy_value = 0
    precision_value = 0

    # Test loss
    with torch.no_grad():
        confusion_matrix_test = torch.zeros(size = (10,10)).type(torch.int64).to(device)
        for iteration, (audio, true_labels) in enumerate(tqdm(test_loader)):
            output_labels = model(audio)

            true_labels_one_hot = torch.nn.functional.one_hot(torch.squeeze(true_labels).type(torch.int64), 10).type(torch.float32)

            loss_value = classification_loss(output_labels, true_labels_one_hot)

            test_loss += loss_value.item()

            f1_value += f1_score(output_labels, torch.squeeze(true_labels).type(torch.int64), average="macro", num_classes=10)
            accuracy_value += accuracy(output_labels, torch.squeeze(true_labels).type(torch.int64), average="macro", num_classes=10)
            precision_value += precision(output_labels, torch.squeeze(true_labels).type(torch.int64), average="macro", num_classes=10)

            confusion_matrix_test += confusion_matrix(output_labels, torch.squeeze(true_labels).type(torch.int64), num_classes=10)

        test_loss_per_epoch = test_loss / len(test_loader)
        f1_value /= len(test_loader)
        accuracy_value /= len(test_loader)
        precision_value /= len(test_loader) 

        tensorboard_writer.add_scalars("Test", {"Test loss": test_loss_per_epoch,
                                                       "Train F1" : f1_value.item(),
                                                       "Train Accuracy": accuracy_value.item(),
                                                       "Train Precision": precision_value.item()},
                                                        epoch)
        
        confusion_matrix_df = pd.DataFrame(confusion_matrix_test.cpu() / torch.sum(confusion_matrix_test, dim = 1).cpu())

        tensorboard_writer.add_figure("Test confusion matrix",
                                      sns.heatmap(confusion_matrix_df, annot=True).get_figure(),
                                      epoch)

    
    # Early stopping
    if best_loss > test_loss_per_epoch:
        built_rage = 0
        best_loss = test_loss_per_epoch
        torch.save(model.state_dict(), "models/env_sound_classification_ckpt.pth")
    elif np.abs(prev_test_loss - test_loss_per_epoch) < MIN_DELTA:
        built_rage += 1
    
    if built_rage >= PATIENCE:
        print(f"Early stopping at: {epoch}")
        print(f"Last checkpoint at epoch: {epoch - PATIENCE}")
        break

    prev_test_loss = test_loss_per_epoch
    print(f"Rage: {built_rage}")
    print(f"Mean loss (train): {train_loss_per_epoch}")
    print(f"Mean loss (test): {test_loss_per_epoch}")
    print("---------------------")

torch.save(model.state_dict(), "models/env_sound_classification.pth")