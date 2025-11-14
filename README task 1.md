# Cynaptics-inductionn

🎵 Audio Classification with ResNet-9
Hey! This is a project from my AI/ML work to classify audio files using a custom ResNet-9 model.

🚀 What This Project Does
The goal was to take audio files, process them, and train a model to figure out what "class" they belong to (e.g., "cat", "dog", "car", "music", etc.).

Here's the basic pipeline:

Load Audio: Reads an audio file (like a .wav file) using librosa.

Pre-process: Converts the audio waveform into a mel spectrogram. This turns the audio into a 2D "image" that an image model can understand.

Classify: Feeds this spectrogram "image" into our ResNet-9 model to get a prediction.

🧠 Model Used
Model: ResNet-9

Why? It's a lightweight but very powerful version of the famous ResNet. It's fast to train but still uses the crucial residual "skip" connections that make ResNets so effective at learning complex patterns.

Framework: PyTorch

📈 Performance & Results
The model trained very quickly and effectively. We achieved strong accuracy results:

At 10 epochs: Reached ~95% validation accuracy.

At 20 epochs: Reached a final validation accuracy of ~98%.

These results were achieved using the following key hyperparameters:

Optimizer: Adam

Learning Rate (lr): 0.001 (or 1e-3)

Batch Size: 32

Loss Function: nn.CrossEntropyLoss()

🛠️ How to Run This
1. Dependencies
You'll need a few key libraries. You can install them with pip:

pip install torch
pip install librosa  # For loading/processing audio
pip install numpy
pip install pandas   # For handling file lists
2. Data
The code expects a folder of audio files and a CSV or DataFrame that maps each file to its correct label.

3. Training
Just run the cells in the Jupyter Notebook in order. It will:

Load the data and create the AudioDataset (which converts audio to spectrograms).

Create the DataLoader.

Define the ResNet-9 model.

Run the training loop and print the loss and accuracy.

💡 What I Learned from This
How to load and handle audio data in Python using librosa.

The concept of mel spectrograms and why they're the "bridge" between audio and image models.

How to implement a ResNet-9 in PyTorch.

How to use residual connections to build a deep, effective model that trains fast.

The standard PyTorch training loop: Dataset, DataLoader, model, optimizer, and loss function.
