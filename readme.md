# Variational Autoencoder (VAE) Implementation: From Theory to Latent Space

This project implements a Variational Autoencoder (VAE) based on the foundational paper *Auto-Encoding Variational Bayes* (Kingma & Welling, 2013). 

## 1. The Core Concept
Unlike a standard Autoencoder that maps an image to a single point, a VAE maps an image to a **probability distribution** in the latent space. This allows us to sample from the space and generate new, unseen data.

* **Likelihood $p_{\theta}(x|z)$**: The Decoder's job. It treats the image $x$ as the effect caused by the latent factor $z$.
* **Posterior $p(z|x)$**: The "True" mapping from image to latent code. This is mathematically "intractable" (impossible to calculate directly).
* **Approximate Posterior $q_{\phi}(z|x)$**: The Encoder's job. It tries to guess the true posterior by predicting a Mean ($\mu$) and a Variance ($\sigma^2$).

## 2. The Reparameterization Trick
To allow backpropagation through a random sampling layer, we move the "noise" to an external variable $\epsilon$.
Instead of sampling $z \sim \mathcal{N}(\mu, \sigma^2)$ directly, we calculate:
$$z = \mu + \sigma \odot \epsilon$$
Where $\epsilon \sim \mathcal{N}(0, 1)$. This makes the operation differentiable.

## 3. The "Log-Variance" Stability Trick
To prevent the model from outputting negative values for Standard Deviation or crashing with `log(0)`, we train the network to predict **Log-Variance** ($\log \sigma^2$).
* $\sigma = \exp(0.5 \cdot \text{log\_var})$
* This ensures numerical stability and allows the network to output values in the range $(-\infty, \infty)$.

## 4. The Loss Function (ELBO)
The VAE is trained to maximize the Evidence Lower Bound (ELBO), which we implement as a **Minimization Loss**:

$$\text{Total Loss} = \text{Reconstruction Loss} (BCE) + \text{KL Divergence}$$

### The Components:
1. **Reconstruction Loss (Likelihood)**: 
   - Measured using Binary Cross Entropy (BCE).
   - Forces the decoder to reconstruct the input as accurately as possible.
2. **KL Divergence ($D_{KL}(q_{\phi}(z|x) || p(z))$):** - Measures how much the encoder's distribution differs from the **Prior** $p(z) \sim \mathcal{N}(0, 1)$.
   - Acts as a regularizer, forcing the latent space to be centered, symmetric, and smooth.
   - **Formula:** $0.5 \cdot \sum(\mu^2 + \exp(\text{log\_var}) - 1 - \text{log\_var})$

## 5. Key Lessons
* **The Tug-of-War:** If the KL loss is too strong, the output becomes a "blurry mess" (Posterior Collapse). If the Reconstruction loss is too strong, the latent space becomes fragmented "islands."
* **Scaling Matters:** Summing the losses (rather than taking the mean) respects the information scale between 784 pixels and 2 latent dimensions.