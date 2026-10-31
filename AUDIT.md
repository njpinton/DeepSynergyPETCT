# Code Audit: DeepSynergyPETCT

> **Status (v2, 2026-10-24):** this audit covers the thesis-era code (git tag `v0-thesis-snapshot`).
> v2 is a rewrite based on `recon_joint_patches.py` and addresses every item below:
> C1 `physics/attenuation.attenuation_factors` (unscaled line integrals, with a regression test) · C2 `experiment.simulate` (attenuated PET data) ·
> C3 `physics/ct.CTProblem.prior_step` · H1–H5, P1–P4 (single package, tested CLIs, no hard-coded paths) ·
> M1 NIST μ/ρ at 80 keV · M2/P12 scales stored in `meta.json`, phantom demo · M3 optional MSE loss · M4 capacity schedule in `train_step` only ·
> M5–M7, M9 fixed · P5–P7 batched latent fit, one decoder pass per outer iteration · P8 `alpha` · P9 seeded RNG · P10 border-aligned patches.
> M8 (approximate adjoint of the rotation projector) is kept and documented. The original text follows.

**Date:** 2026-10-23
**Scope:** this repo (`src/`) and the local research scripts in `~/projects/petct-synergy/code/pet-ct-vae/` (not on GitHub).
**Verdict:** the method is implemented, but the public repo can't reproduce the thesis yet. Three of the issues below break the PET/CT reconstruction itself, and the most up-to-date code (`recon_joint_patches.py`) isn't on GitHub.

Line numbers refer to the files as they are after the fixes in section 0.

---

## 0. Already fixed (uncommitted)

| # | Issue | Fix |
|---|---|---|
| F1 | `requirements.txt` could not install: `numpy~=1.20.4` doesn't exist, TF 2.4 + Keras 3.9 conflict, and `astra-toolbox`, `imutils`, `opencv`, `pydicom` were missing | Pinned to TF 2.15 (Keras 2, which loads the SavedModel `decoder/` dirs), numpy <2, Python 3.9–3.11. ASTRA has pip wheels for Linux/Windows only; on macOS use conda |
| F2 | `utils/recon_ct.py:9` used `from lbfgs_vae import`, which raised ImportError whenever `recon_joint.py` ran | `from utils.lbfgs_vae import` |
| F3 | `config.json` was **gitignored** (so it wasn't on GitHub) and was missing 7 keys that `recon_joint.py` reads (`beam_type`, `det_width`, `det_count`, `source_origin`, `origin_det`, `proj_type`, `beta_recon`). `param_recon['I']` / `['bckg']` raised TypeError on a plain object | Un-ignored, added the keys (values from `ParamProj` defaults / `recon_ct.main`), switched to attribute access, `mode: petct` |
| F4 | CT units: `recon_joint.py` projected the **raw HU** image; `i_all: 0` meant 0 photons (all-zero CT data); `mu_air` was `2.407e-2` (missing the `× 1.205e-3` air density), which inflated HU about 13× | Convert HU → attenuation before projecting; `i_all: 1000`; `mu_air = 2.407e-2 × 1.205e-3`; CT PSNR now uses the image max |

Verified: all modules import and the config parses in a clean Python 3.11 environment with the fixed requirements (ASTRA replaced by a stand-in). An end-to-end run still needs a trained model, data, and ASTRA.

---

## 1. Critical: wrong reconstructions

**C1. PET attenuation map is projected with the PET count scale, so attenuation correction collapses** (`src/recon_joint.py:77`)
`forward_projection(mu_511, params)` uses `params.scale = 30000`. Line integrals become about 10⁴, so `lac = exp(-…) ≈ 0`, `norm_ac ≈ 0`, and the MLEM update divides by about zero, which blows up or NaNs the PET image. The local `recon_joint_v2.py` / `recon_joint_patches.py` correctly use a separate `params_ac` with `scale=1` (and clip `pmu_511 ≥ 0`). That fix never reached GitHub.

**C2. PET data is simulated without attenuation but reconstructed with it** (`src/recon_joint.py:191`, also `recon_joint_v2.py:275`)
`sinogram_pet = Poisson(A·x)` has no `lac` factor, while the update divides by `Aᵀ·lac`. The resulting activity is biased (overestimated where attenuation is high). `recon_joint_patches.py:532` does it correctly: `Poisson(mu_map · A·x)`.

**C3. `recon_joint_v2.py:153`: the CT data step is thrown away** (local script only)
```python
x_rec = np.where(D1 == 0, 0, x_ct - grad / D1)
x_rec = np.where(x_ct > 0, x_ct, 0)   # overwrites x_rec with the OLD x_ct
```
During the joint iterations, the CT image never sees the sinogram again. It only gets pulled toward the generator output. **Any thesis numbers produced with `recon_joint_v2.py` should be re-checked.** `recon_joint_patches.py:382` and the GitHub version are correct.

---

## 2. High: crashes in the shipped code paths

- **H1.** `src/utils/recon_ct.py:181` calls `get_z(..., bounds=None)`, but the GitHub `get_z` (`lbfgs_vae.py`) has no `bounds` parameter, so it raises TypeError. The local `lbfgs_vae.py:244` has it.
- **H2.** `src/fit_model.py:133` passes 10 positional args to `build_betavae`, which takes 6, so `--mode pet` / `--mode ct` crash. `build_betavae` (`models/betavae.py:239`) also calls `BetaVAE(encoder, decoder)` without the required `g, c_max, c_stop_iter, loss_type`.
- **H3.** `src/utils/lbfgs_vae.py`: `mode_='pet'` has no branch inside the GradientTape, so `_y` is undefined (NameError).
- **H4.** `lbfgs_vae.py:65,99`: the fallback `old_dydx` is a global that is undefined on the first call (NameError if the first gradient is 0). The NaN guard `dy_dx == np.nan` is always False; it should be `np.isnan(dy_dx).any()`.
- **H5.** `recon_joint_v2.py`, `recon_joint_petct.py` and `recon_joint_patches.py` import `recon_ct` and `recon_mlem` from the root. `recon_mlem.py` exists nowhere on this laptop, so these scripts can't run as they are.

---

## 3. Medium: correctness and scientific validity

- **M1. Attenuation coefficients.** Most scripts use `mu_water = 0.02597`, `mu_air = 0.02407·ρ`. These are NIST **energy-absorption** coefficients (μen/ρ) at 80 keV, not attenuation (μ/ρ = 0.1837 / 0.1662). `recon_joint_patches.py:506-507` switched to the correct values and rescales `scale_ct` to match. The GitHub code (and the F4 fix) still uses the old pair, which stays consistent with the trained model's `scale_ct = 96.94` but makes the simulated CT about 7× too transparent (unrealistic noise level).
- **M2. Magic constant** `scale_ct = 96.93775373893472` (`recon_joint.py:93`) is the training-set `1/ct.max()`. It is hard-coded, so it is silently wrong for any other model or dataset. Save it with the model (e.g. a JSON next to `decoder/`).
- **M3. VAE loss.** `binary_crossentropy` (`models/betavae.py:77,118`) on targets outside [0, 1]: PET isn't normalized in `fit_model.prepare_data`, and CT in HU × `1/max` is negative below 0 HU. The decoder uses ReLU outputs, not sigmoid. Either normalize both to [0, 1] with sigmoid outputs, or switch to MSE.
- **M4. Capacity schedule (loss B).** `self.C += 1/140` per step is hard-coded and ignores `c_max` / `c_stop_iter`. It also runs inside `call()`, so evaluation and prediction keep increasing C.
- **M5. `merge_type='average'`** is accepted by the CLI but silently falls back to concatenate (`betavae.py:169`).
- **M6. `normalise_zero_one`** (`data_utils.py:62`): `if not min_` treats `0` as missing and recomputes the min from data. Use `is None`.
- **M7. `hu_to_511`** (`attenuationCT_to_511.py:15`): `image.shape == 3` is always False (should be `len(image.shape) == 3`), so 3D input crashes.
- **M8. Projector pair.** `radon2` / `bradon2` (rotation-based, `projectors.py:106`) are not an exact adjoint pair, so MLEM isn't strictly ML-EM. That's acceptable for simulation, but worth a sentence in the methods.
- **M9. PSNR.** The default `max_pixel = 1.0` is meaningless for attenuation (about 0.02/voxel) or raw PET counts. Pass the data range explicitly and report it.

---

## 4. Reproducibility

- **R1. The latest method isn't public.** `recon_joint_patches.py` (Nov 2024, patch-based, corrected physics) and `fit_betavae_patches2d.py` exist only locally. The GitHub `recon_joint.py` is an older, simplified variant.
- **R2.** No trained weights, no data loader for a public dataset, and no instructions for producing the `.npy` input. Hard-coded `/home/noel/...` paths appear in `data_utils.py:27`, `lbfgs_vae.py:120`, `recon_ct.py:215` and `attenuationCT_to_511.py:76`. One of them includes a **patient name** in a DICOM path: remove it before the repo gets attention.
- **R3.** No random seeds in the GitHub version (the local v2 seeds `np.random.seed(42)`); Poisson noise and `z` init differ every run.
- **R4.** The README has no usage section (install, train, reconstruct, expected output).

---

## 5. Code quality (low)

- `tf.config.run_functions_eagerly(True)` at import time in every model file slows training globally.
- `models/vae.py`, `vae512.py`, `betavae.py` and `betavae3d.py` duplicate `Sampling`, the encoder and the decoder. `vqvae.py` is unused.
- `get_objf` runs the decoder twice per call (once for the value, once inside the tape). Compute both in one pass to halve L-BFGS cost.
- Bare `except:` (`betavae.py`, `recon_joint_v2.py`), `print('0')` on missing GPU, interactive `plt.ion()` inside the reconstruction loops.
- No tests. Even a tiny MLEM/PWLS sanity test on a phantom would have caught C1 and F4.

---

## 6. Local reference code: `pet-ct-vae/recon_joint_patches.py` + `fit_betavae_patches2d.py`

This is the patch-based method from thesis Ch. 5 / TRPMS 2025, and the physics is right: the PET data includes attenuation (`:532`), the attenuation map is projected with `params_ac` (scale 1, `:257`), μ/ρ values are corrected with `scale_ct` rescaled (`:295-300`, `:503-511`), and the CT separable-surrogate update is correct (`:381-393`). **This should become the public implementation.** Its problems:

**Doesn't run as is**
- **P1.** Imports `recon_ct` and `recon_mlem` from the root. `recon_mlem.py` exists nowhere on this laptop (only `display_images` is used, so it's replaceable).
- **P2.** `main()` only runs the **baseline** (`nogen=True`, `:567`); the joint reconstruction call is commented out (`:559`).
- **P3.** `:613-615` always `np.load`s a z-array from a hard-coded previous run, even when `z_fit=True`, which crashes on a fresh machine. `os.mkdir('images/…')` (`:553`) fails if `images/` is missing.
- **P4.** `get_objf` (`lbfgs_vae.py`) prints `diff_pet` on every call, which is a NameError in `pet` / `ct` modes and floods stdout in `petct` mode.

**Very slow (but correct)**
- **P5.** 512² image, 64² patches, stride 16 gives **841 patches**, each with its own L-BFGS (up to 1000 iterations, `factr=10`) and one decoder call per patch per evaluation. Batching all patches into one decoder call (one `(841, 64)` latent tensor, one L-BFGS over the stacked z, since the patch problems are separable) would give about a 100× speed-up.
- **P6.** `get_Fz` (one decoder call per patch) is re-run inside every PET (30) and CT (100) sub-iteration, although `z` is fixed there. Compute it once per outer iteration.
- **P7.** The inner `iter_recon = 250` loops (`:365`, `:386`) are no-ops after the first pass: `x_gen = x − (D2·x − R(Fz))/D2 = R(Fz)/D2` doesn't depend on `x`, so 249 of 250 passes repeat the same computation.

**Method / fidelity**
- **P8.** The thesis z-update (eq. z-update) includes `α·H(z)` (latent prior); the code uses only data fit. Add `α‖z‖²` as an option, or document that α = 0.
- **P9.** Sinogram noise is drawn **before** `np.random.seed(42)` (which runs inside `recon_joint`), so noisy data differs every run.
- **P10.** `D2` (patch overlap count) is zero wherever patches don't cover the image (image size not ≡ patch size mod stride), which divides by zero at the borders.
- **P11.** PSF `fwhm = 0.1` mm (essentially none), while older scripts use 5 mm. Decide which one is the reported setting and make it a parameter.
- **P12.** Training data (CHU Poitiers) is private, so the repo needs a synthetic/phantom demo path to be runnable by others.

---

## Suggested order

1. C1 + C2 (port `params_ac` and attenuated simulation from `recon_joint_patches.py`)
2. Check whether any thesis or paper figures came from `recon_joint_v2.py` (C3)
3. Remove the patient-identifying path (R2)
4. Publish `recon_joint_patches.py` + `fit_betavae_patches2d.py` as the reference implementation, with a usage section
5. H1–H5, then the rest
