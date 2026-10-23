import numpy as np
import pytest
import tensorflow as tf

from deepsynergy.config import Config, ReconConfig
from deepsynergy.data import make_training_patches, thorax_phantom
from deepsynergy.model_io import load_prior, save_model
from deepsynergy.models import build_dual_betavae
from deepsynergy.prior import LatentFit, PatchOperator, PatchPrior, patch_starts


def test_patch_starts_cover_image():
    assert patch_starts(10, 4, 3) == [0, 3, 6]
    assert patch_starts(11, 4, 3) == [0, 3, 6, 7]


@pytest.mark.parametrize("n,patch,stride", [(32, 8, 4), (30, 8, 5)])
def test_patch_operator_adjoint_and_average(n, patch, stride):
    rng = np.random.default_rng(0)
    op = PatchOperator((n, n), patch, stride)
    x, p = rng.random((n, n)), rng.random((op.n_patches, patch, patch))
    np.testing.assert_allclose(np.vdot(op.extract(x), p), np.vdot(x, op.adjoint(p)))
    assert op.overlap.min() >= 1  # every pixel covered (P10)
    np.testing.assert_allclose(op.average(op.extract(x)), x)


@pytest.fixture(scope="module")
def tiny_prior():
    tf.keras.utils.set_random_seed(0)
    vae = build_dual_betavae(patch=16, latent_dim=4, inter_dim=16, filters=8, n_layers=2)
    return PatchPrior(vae.decoder, pet_scale=1.0, ct_scale=1.0)


def test_batched_objective_is_sum_of_patch_objectives(tiny_prior):
    """The z-update solves all patch problems with one L-BFGS: valid because the batched
    objective/gradient are the per-patch objectives/gradients stacked."""
    rng = np.random.default_rng(1)
    z = tf.constant(rng.normal(size=(3, 4)), tf.float32)
    u1 = tf.constant(rng.random((3, 16, 16)), tf.float32)
    u2 = tf.constant(rng.random((3, 16, 16)), tf.float32)
    eta, alpha = tf.constant(0.3), tf.constant(0.1)
    f, g = tiny_prior._value_and_grad(z, u1, u2, eta, alpha)
    parts = [tiny_prior._value_and_grad(z[i:i + 1], u1[i:i + 1], u2[i:i + 1], eta, alpha) for i in range(3)]
    np.testing.assert_allclose(float(f), sum(float(fi) for fi, _ in parts), rtol=1e-5)
    np.testing.assert_allclose(g.numpy(), np.concatenate([gi.numpy() for _, gi in parts]), rtol=1e-4, atol=1e-6)


def test_latent_fit_reduces_objective(tiny_prior):
    rng = np.random.default_rng(2)
    pet, ct = tiny_prior.generate(rng.normal(size=(5, 4)))
    z0 = rng.normal(size=(5, 4))
    obj = lambda z: float(tiny_prior._value_and_grad(  # noqa: E731
        tf.constant(z, tf.float32), tf.constant(pet, tf.float32), tf.constant(ct, tf.float32),
        tf.constant(0.5), tf.constant(0.0))[0])
    z = tiny_prior.fit_latents(pet, ct, z0, 0.5, LatentFit(max_iter=200, factr=10.0))
    assert obj(z) < 0.1 * obj(z0)


def test_training_step_and_save_load(tmp_path):
    rng = np.random.default_rng(0)
    pairs = [thorax_phantom(32, rng) for _ in range(4)]
    pet, hu = np.stack([p for p, _ in pairs]), np.stack([c for _, c in pairs])
    p_pet, p_ct, meta = make_training_patches(pet, hu, vox_mm=15.6, patch=16, stride=8)
    assert 0 <= p_pet.min() and p_pet.max() <= 1 and 0 <= p_ct.min() and p_ct.max() <= 1

    vae = build_dual_betavae(patch=16, latent_dim=4, inter_dim=16, filters=8, n_layers=2,
                             loss_type="B", c_max=10, c_stop_iter=4)
    vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    ds = tf.data.Dataset.from_tensor_slices({"input_1": p_pet[..., None], "input_2": p_ct[..., None]}).batch(16)
    hist = vae.fit(ds, epochs=2, verbose=0)
    assert np.isfinite(hist.history["total_loss"]).all()
    assert int(vae.train_iter.numpy()) == 2 * len(ds)

    save_model(vae, tmp_path / "m", meta)
    prior = load_prior(tmp_path / "m")
    assert prior.patch == 16 and prior.latent_dim == 4
    assert prior.ct_scale == pytest.approx(meta["ct_scale"])


def test_config_roundtrip(tmp_path):
    cfg = Config()
    cfg.recon.beta_pet = 7.0
    cfg.save(tmp_path / "c.json")
    assert Config.load(tmp_path / "c.json") == cfg
    (tmp_path / "bad.json").write_text('{"recon": {"betta_pet": 1}}')  # typo must be rejected
    with pytest.raises(KeyError):
        Config.load(tmp_path / "bad.json")


def test_joint_reconstruction_runs(tiny_prior):
    from deepsynergy.experiment import simulate
    from deepsynergy.recon import joint_reconstruct

    cfg = Config()
    cfg.image.size, cfg.image.vox_mm = 32, 15.6
    cfg.pet.n_angles = cfg.ct.n_angles = 30
    cfg.recon = ReconConfig(init_iters=5, outer_iters=2, z_fit_iters=1, pet_subiters=3, ct_subiters=3,
                            stride=8, lbfgs_max_iter=20, lbfgs_factr=1e7)
    rng = np.random.default_rng(0)
    pet, hu = thorax_phantom(32, rng)
    sim = simulate(cfg, pet, hu, rng)
    seen = []
    res = joint_reconstruct(sim.pet, sim.ct, tiny_prior, cfg.recon, rng=rng, callback=lambda i, r: seen.append(i))
    assert seen == [0, 1]
    for img in (res.pet, res.ct, res.pet_gen, res.ct_gen):
        assert img.shape == (32, 32) and np.isfinite(img).all() and img.min() >= 0
