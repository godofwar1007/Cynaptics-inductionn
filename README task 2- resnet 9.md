

🎵 Audio GAN with ResNet-9 Discriminator
This is a PyTorch project to train a Generative Adversarial Network (GAN) to generate audio spectrograms. The key part of this project is the use of a powerful ResNet-9 model as the Discriminator.

🚀 Core Concept: The "Adversarial Game"
A GAN works as a "two-player game" between two neural networks:

The Generator (G): A "forger" that takes random noise and tries to create a realistic audio spectrogram "image" that looks like it came from our real dataset.

The Discriminator (D): A "detective" that takes a spectrogram "image" and tries to guess if it's real (from our dataset) or fake (from the Generator).

Through this process, the Generator gets better at making fakes, and the Discriminator gets better at catching them.

🧠 Models & Architecture
This project was built by cleverly combining new and existing code:

Generator (G): This was a new model built from scratch. It uses nn.ConvTranspose2d layers (also called "deconvolutions") to up-sample a small random vector (the latent_size) into a full-sized 3-channel spectrogram.

Discriminator (D): For this, we reused our ResNet-9 model from a previous classification task. A powerful classifier is the perfect tool to be a "detective." We just modified its final layer to output a single number (a probability between 0 for "fake" and 1 for "real") instead of 10 class predictions.

AudioDataset: We also reused our dataset class from the previous task. This was used to load our real audio files, convert them to mel spectrograms, and feed them to the Discriminator for training.

📈 The Big Challenge: Stabilizing the "GAN Fight"
When we first trained the models, we immediately hit a classic GAN problem: model collapse.

Problem: The "Unnerfed" Fair Fight
Initially, we set the learning rates to be equal for both models (d_lr = lr and g_lr = lr).

The ResNet-9 Discriminator was simply too smart, too fast. It learned to identify fakes almost instantly, and the Generator couldn't keep up.

The Evidence (from our logs):

D_loss (Discriminator loss) crashed to ~0.0024. This means it was "winning" the game easily.

G_loss (Generator loss) exploded to ~6.6772. This means it was "losing" so badly it couldn't learn anything.

The GAN was failing because the "detective" was a super-computer and the "forger" was a 5-year-old. The Generator had no chance to learn.

Solution: "Nerfing" the Discriminator
To fix this, we had to balance the fight. We "nerfed" the "detective" by making it learn 10 times slower than the Generator.

The Code Fix:

g_lr = 0.0002 (Generator learns at normal speed)

d_lr = 0.00002 (Discriminator learns 10x slower)

This single change worked perfectly.

The Result (from our new logs):

D_loss stabilized to ~0.4578

G_loss stabilized to ~1.3039

These new losses show that both models were now in a balanced fight. The Discriminator was still a good opponent, but the Generator was no longer being instantly defeated and finally had a chance to learn and improve. This is what made the GAN training successful.

🛠️ How to Run This
1. Dependencies
You'll need the standard data science and audio libraries:

pip install torch
pip install librosa
pip install numpy
pip install pandas
2. Data
Place your audio files (e.g., .wav, .mp3) into a data folder. The code will scan this folder to build the dataset.

3. Training
Just run the cells in the Task_2.ipynb notebook from top to bottom. The code will:

Define the Generator and Discriminator (ResNet-9) models.

Set up the optimizers with the "nerfed" learning rate for the Discriminator.

Load the data using the AudioDataset.

Run the main training loop, printing the D_loss and G_loss at each step.

💡 What I Learned from This
How to build a Generator model from scratch using ConvTranspose2d layers.

How to repurpose a powerful classifier (ResNet-9) as a GAN Discriminator.

That GAN training is a delicate balance, and the Discriminator can often be "too strong."

The most important trick for debugging GANs: How to stabilize training by adjusting the relative learning rates (e.g., "nerfing" the Discriminator) to keep the "fight" fair.
