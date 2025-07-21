# Synergistic PET/CT Reconstruction Using Deep Learning

This repository contains the official implementation of the code developed for the thesis **"Synergistic PET/CT Reconstruction Using Deep Learning"**.

The objective of this work is to explore joint PET and CT reconstruction via deep generative priors, leveraging the complementary nature of both modalities. Specifically, the method integrates learned representations from PET and CT to improve image quality in low-dose or undersampled acquisition scenarios.

This code supports the experiments presented in the following publications:

- N. J. Pinton, A. Bousse, C. Cheze-Le-Rest, and D. Visvikis, ***“Multi-Branch Generative Models for Multichannel Imaging With an Application to PET/CT Synergistic Reconstruction,”*** IEEE Trans. Radiat. Plasma Med. Sci., 2025. DOI: [10.1109/TRPMS.2025.3532176](https://doi.org/10.1109/TRPMS.2025.3532176)

- N. J. Pinton, A. Bousse, Z. Wang, C. Cheze-Le-Rest, V. Maxim, C. Comtat, F. Sureau, and D. Visvikis, ***“Synergistic PET/CT Reconstruction Using a Joint Generative Model,”*** in *Proc. Int. Conf. Fully Three-Dimensional Image Reconstruction in Radiology and Nuclear Medicine (Fully3D)*, 2023. Available: [https://arxiv.org/abs/2411.07339](https://arxiv.org/abs/2411.07339)

- N. J. Pinton, A. Bousse, C. Cheze-Le-Rest, and D. Visvikis, ***“Joint PET/CT Reconstruction Using a Double Variational Autoencoder,”*** in *IEEE Nucl. Sci. Symp. Med. Imag. Conf. (NSS/MIC)*, 2023. DOI: [10.1109/NSSMICRTSD49126.2023.10337812](https://doi.org/10.1109/NSSMICRTSD49126.2023.10337812)

- N. J. Pinton, ***“Synergistic PET/CT Reconstruction Using Deep Learning,”*** Ph.D. dissertation, Univ. Bretagne Occidentale, Brest, France, Dec. 2024. [Online]. Available: [https://theses.hal.science/tel-04996652](https://theses.hal.science/tel-04996652)

The implemented models are based on **Variational Autoencoders (VAEs)** and **Generative Adversarial Networks (GANs)**. These models are designed to learn multi-modal latent priors and act as deep regularizers within a synergistic PET/CT reconstruction framework.
