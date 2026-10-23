import numpy as np
import pytest

from deepsynergy.data import thorax_phantom
from deepsynergy.metrics import psnr
from deepsynergy.physics import (CTProblem, PETProblem, RotationProjector, attenuation_factors, hu_to_mu,
                                 hu_to_mu511, mu_to_hu, simulate_ct)

N, VOX = 64, 7.8


@pytest.fixture(scope="module")
def phantom():
    return thorax_phantom(N, np.random.default_rng(0))


def test_projector_is_approximately_adjoint():
    rng = np.random.default_rng(1)
    proj = RotationProjector(N, 60, fwhm_mm=10.0, vox_mm=VOX, scale=3.0)
    x, y = rng.random((N, N)), rng.random(proj.sino_shape)
    lhs, rhs = np.vdot(proj.forward(x), y), np.vdot(x, proj.back(y))
    assert abs(lhs - rhs) / abs(lhs) < 0.02


def test_hu_mu_roundtrip():
    hu = np.array([-1000.0, -500.0, 0.0, 1000.0])
    np.testing.assert_allclose(mu_to_hu(hu_to_mu(hu, VOX), VOX), hu, atol=1e-6)


def test_mu511_values():
    mu = hu_to_mu511(np.array([-1000.0, 0.0, 1000.0]), vox_mm=10.0)  # 1 cm pixels -> cm^-1
    np.testing.assert_allclose(mu, [0.0, 0.093, 0.093 + 1000 * 4.71e-5], atol=1e-9)


def test_attenuation_factors_not_scaled(phantom):
    """C1 regression: the PET count scale must not enter the attenuation line integrals."""
    _, hu = phantom
    a1 = attenuation_factors(hu_to_mu511(hu, VOX), RotationProjector(N, 30, vox_mm=VOX, scale=1.0))
    a2 = attenuation_factors(hu_to_mu511(hu, VOX), RotationProjector(N, 30, vox_mm=VOX, scale=3e4))
    np.testing.assert_allclose(a1, a2)
    assert a1.min() > 0.01


def test_mlem_noiseless_recovers_activity(phantom):
    pet, hu = phantom
    proj = RotationProjector(N, 60, vox_mm=VOX, scale=100.0)
    att = attenuation_factors(hu_to_mu511(hu, VOX), proj)
    prob = PETProblem(att * proj.forward(pet), proj, att)
    x10 = prob.mlem(np.ones((N, N)), 10)
    x100 = prob.mlem(x10, 90)
    assert psnr(pet, x100) > psnr(pet, x10)
    assert np.isfinite(x100).all()
    assert abs(x100.sum() / pet.sum() - 1) < 0.01


def test_pwls_converges(phantom):
    _, hu = phantom
    mu = hu_to_mu(hu, VOX)
    proj = RotationProjector(N, 60, vox_mm=VOX)
    prob = CTProblem(simulate_ct(mu, proj, 1e6, np.random.default_rng(2)), proj, 1e6)
    x = prob.pwls(np.ones((N, N)), 20)
    p20 = psnr(mu, x)
    x = prob.pwls(x, 180)
    assert psnr(mu, x) > p20
    assert psnr(mu, x) > 25


def test_prior_steps_reduce_to_data_steps_without_prior(phantom):
    pet, hu = phantom
    proj = RotationProjector(N, 30, vox_mm=VOX, scale=50.0)
    att = attenuation_factors(hu_to_mu511(hu, VOX), proj)
    prob = PETProblem(att * proj.forward(pet), proj, att)
    x = np.ones((N, N))
    np.testing.assert_allclose(prob.prior_step(x, pet, np.zeros((N, N))), prob.em_step(x))


def test_pet_prior_step_pulls_towards_generated(phantom):
    pet, hu = phantom
    proj = RotationProjector(N, 30, vox_mm=VOX, scale=50.0)
    att = attenuation_factors(hu_to_mu511(hu, VOX), proj)
    prob = PETProblem(att * proj.forward(pet), proj, att)
    x = np.ones((N, N))
    strong = prob.prior_step(x, pet, np.full((N, N), 1e12))
    np.testing.assert_allclose(strong, pet, atol=1e-3)


def test_pet_prior_step_stable_for_tiny_weight(phantom):
    pet, hu = phantom
    proj = RotationProjector(N, 30, vox_mm=VOX, scale=50.0)
    att = attenuation_factors(hu_to_mu511(hu, VOX), proj)
    prob = PETProblem(att * proj.forward(pet), proj, att)
    x = np.ones((N, N))
    np.testing.assert_allclose(prob.prior_step(x, pet, np.full((N, N), 1e-12)), prob.em_step(x), rtol=1e-6)
