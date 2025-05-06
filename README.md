# Joint PET/CT Reconstruction

This project performs joint PET/CT image reconstruction from simulated sinograms using a multi-branch β-VAE as a learned deep prior. The method combines classical analytical updates (e.g., MLEM for PET and PWLS for CT) with deep latent-space regularization.

The reconstruction pipeline is based on the framework described in the following preprint:

**Joint PET/CT Reconstruction Using Multi-Branch VAEs**  
[https://arxiv.org/abs/2404.08748](https://arxiv.org/abs/2404.08748)

---

## Optimization Formulation

We formulate the joint reconstruction as a regularized optimization problem:


```
(x₁*, x₂*) = argmin_{x₁, x₂} ∑ₖ ηₖ Lₖ(Mₖ(xₖ), yₖ) + β R(x₁, x₂)
```


Where:
- `yₖ` is the observed sinogram data for modality `k ∈ {PET, CT}`,
- `Mₖ` is the forward projector for each modality,
- `Lₖ` is the likelihood loss (e.g., Poisson NLL for PET),
- `R(x₁, x₂)` is a regularizer coupling the two modalities.

---

## Deep Prior via Multi-Branch β-VAE

We introduce a shared latent code `z ∈ ℝ^d` and a decoder `G_θ` trained to reconstruct PET/CT image pairs. The regularization term becomes:

```
R(x₁, x₂) = min_z ∑ₖ ηₖ / 2 * || xₖ - Gₖ(z) ||²
```

This encourages the reconstructed images to lie on the learned manifold of PET/CT pairs modeled by the decoder branches of a pretrained β-VAE.

---

## Implementation Summary

- PET images are reconstructed via MLEM, using attenuation correction derived from the CT image.
- CT images are reconstructed via PWLS (Penalized Weighted Least Squares).
- The latent code `z` is optimized using L-BFGS to minimize the mismatch between the current reconstruction and the generative model output.
- A pretrained multi-branch β-VAE provides the learned PET/CT prior through its decoder.
