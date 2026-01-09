🎵 Audio GAN (Simple CNN Discriminator Attempt)
This project is our second attempt at training a Generative Adversarial Network (GAN) to create audio spectrograms.

This version was created as a direct response to the problems we faced in our first attempt (using a ResNet-9 Discriminator).

🚀 Core Concept
The goal is the same: create a "forger" (Generator) to make fake audio and a "detective" (Discriminator) to catch the fakes.

Generator (G): The same nn.ConvTranspose2d model from our first attempt. We know this part works.

Discriminator (D): This is the new part. Instead of a complex ResNet-9, this is a simple, sequential CNN.

🧠 Why a New Model? The ResNet-9 Failure
Our first experiment with the ResNet-9 Discriminator was a failure.

We discovered that the ResNet-9 model was too powerful for the job. It learned way too fast.

The Problem: Even when we "nerfed" its learning rate by 10x (setting d_lr = lr / 10), the Discriminator's loss still crashed to almost zero. It was simply too "smart."

The Result: The Generator's loss would explode. It couldn't learn anything because it was like a new artist trying to fool a world-class art expert. It had no chance. The "fight" was completely unbalanced.

Our New Hypothesis
The ResNet-9 was "too smart," so... what if we used a "dumber" Discriminator?

The plan was to build a much simpler CNN from scratch. The idea was that a less complex "detective" would be easier for the "forger" to fool, creating a more balanced and fair fight from the beginning.

This new, simple CNN Discriminator was just a few Conv2d -> LeakyReLU -> BatchNorm layers, with no residual connections.

🛑 Current Status: Not Run
We were not able to run or train this new model.

This was not a code problem. We were stopped by two real-world project limits:

Time Crunch: We were running out of time to complete the task.

Google Colab GPU Limit: After so many experiments (restarting the runtime for the AudioDataset bug, running the classifier, training the ResNet-9 GAN), Google Colab stopped giving us access to free GPUs.

We hit our usage limit, and without a GPU, it was impossible to train a new GAN in the time we had left.

💡 What I Learned from This
Even though we couldn't run this final experiment, the process taught us some critical lessons about GANs:

The "Balancing Act" is Everything: GANs are extremely sensitive. The "fight" between the Generator and Discriminator is the most important part, and it's very easy to break.

"Smarter" is Not Always "Better": We proved that just dropping in a powerful, state-of-the-art model (like ResNet-9) can be worse than a simple, custom-built model if it unbalances the system.

Practical Limits: We hit the very real-world wall of compute limits. This is a major part of ML engineering—managing your time and your available hardware (like Colab's GPU quota).
