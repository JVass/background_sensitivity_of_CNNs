# Deep Learning for Audio and Music assignment
## Motivation
In this module we were asked to implement two different models for two different tasks (or inference types) and then combine those.

Based on the work of Palanisamy et al (https://arxiv.org/abs/2007.11154), off-the-shelf CNNs trained on ImageNet outperformed musically informed CNNs and provided SOTA results in 2/3 datasets.

Also, based on the survey of Moayeri et al (https://arxiv.org/abs/2201.10766) on the sensitivity of CNNs to non-informative sections of an image, and especially the background, I will implement an Autoencoder to 'denoise' but more specifically, alter the spectrogram in a non-obvious way.

Therefore,

1. A simple Autoencoder will be trained on reproducing the spectrogram of an image. Later on, it will be tested on its denoising capabilities (additive white-noise).
2. A DenseNet will be used as a frozen pre-trained module and an MLP (1920x10) will be fine-tuned.
3. The combination of Autoencoder and DenseNet will be used to ('partially') empirically prove that the CNN trained on ImageNet is sensitive to background information, as the Autoencoder succesfuly reconstructs the spectrogram's most informative sections but not the background.

---

## How to run
1. On the file *global_vars.py*, insert the path to UrbanSound8k the variable ``` PATH_TO_DATASET = "" ```.
2. Generate the csv through the metadata from the file by running *generate_useable_csv.py*
3. Train the denoising Autoencoder with running * train_denoising.py*
4. Train the DenseNet with *train_env_sound_classification.py*
5. Run the evaluation scripts.