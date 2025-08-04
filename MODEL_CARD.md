# Latent Diffusion MNIST Experiment Model Card

## Model Details
- **Model type:** Latent diffusion model using a VAE encoder-decoder and conditional U-Net.
- **Developed by:** Contributors to this repository.
- **License:** MIT
- **Repository:** https://github.com/codinglabsong/ldm-hf

## Intended Use
- Research and education on generative modeling techniques.
- Demonstrates diffusion models on the MNIST dataset.
- Not intended for production, biometric, or classification purposes.

## Training Data
- Trained solely on the MNIST dataset of handwritten digits.
- Images are resized to 32×32 pixels and normalized to the `[-1, 1]` range.
- No personally identifiable information is included.

## Evaluation
- Qualitative evaluation via reconstruction and sample visualizations.
- No comprehensive quantitative metrics (e.g., FID) are reported.

## Ethical Considerations
- See [ETHICS.md](ETHICS.md) for details on data usage, potential risks, and responsible deployment.

## Limitations
- The model only learns from digit images and cannot generalize to complex scenes.
- Small training corpus may lead to limited diversity or memorization.
- Does not include safety filtering; users must implement their own safeguards.