"""Substrate tests. Run: python test_dna.py   (pytest-compatible too)."""
import math
import torch
import torch.nn as nn

from dna import DNAConfig, DNALinear, SeesawActivation, InfeasibleConfig, capacity_report


def _linear(i=50, o=20, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(i, o)


def test_identity_linear():
    """THE gate: an ideal substrate must be nn.Linear, bit-for-bit up to float tolerance."""
    lin = _linear()
    cfg = DNAConfig(sigma=0.0, leak_rate=0.0, n_levels=None, threshold=0.0, gain=1.0)
    dna = DNALinear(lin, cfg, seed=123)
    x = torch.randn(64, 50)
    torch.testing.assert_close(dna(x), lin(x), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(DNALinear(lin, DNAConfig().ideal())(x), lin(x), rtol=1e-5, atol=1e-6)


def test_identity_seesaw_is_relu():
    act = SeesawActivation(DNAConfig().ideal())
    x = torch.randn(1000)
    torch.testing.assert_close(act(x), torch.relu(x))


def test_rails_non_negative_and_disjoint():
    dna = DNALinear(_linear(), DNAConfig(sigma=0.2))
    assert (dna.W_plus >= 0).all() and (dna.W_minus >= 0).all()
    assert ((dna.W_plus > 0) & (dna.W_minus > 0)).sum() == 0


def test_noise_is_multiplicative_lognormal():
    lin = _linear(200, 200)
    ideal = DNALinear(lin, DNAConfig(sigma=0.0, leak_rate=0.0, n_levels=None))
    noisy = DNALinear(lin, DNAConfig(sigma=0.1, leak_rate=0.0, n_levels=None), seed=1)
    nz = ideal.W_plus > 0
    log_ratio = torch.log(noisy.W_plus[nz] / ideal.W_plus[nz])
    assert abs(log_ratio.mean()) < 0.01 and abs(log_ratio.std() - 0.1) < 0.01
    # zeros stay zero and signs never flip: impossible under additive Gaussian
    assert (noisy.W_plus[~nz] == 0).all()


def test_seed_determinism():
    lin, cfg = _linear(), DNAConfig(sigma=0.1)
    assert torch.equal(DNALinear(lin, cfg, seed=5).W_plus, DNALinear(lin, cfg, seed=5).W_plus)
    assert not torch.equal(DNALinear(lin, cfg, seed=5).W_plus, DNALinear(lin, cfg, seed=6).W_plus)


def test_quantisation_levels():
    dna = DNALinear(_linear(), DNAConfig(sigma=0.0, leak_rate=0.0, n_levels=4))
    vals = torch.unique(torch.cat([dna.W_plus.flatten(), dna.W_minus.flatten()]))
    assert len(vals) <= 4


def test_leak_is_signal_independent_floor():
    lin = _linear()
    cfg = DNAConfig(sigma=0.0, leak_rate=1e-2, n_levels=None)
    dna = DNALinear(lin, cfg)
    x = torch.zeros(3, 50)
    expect = lin(x) + 1e-2 * lin.weight.abs().sum(1) + 1e-2 * lin.bias.abs()
    torch.testing.assert_close(dna(x), expect)


def test_capacity_ceiling():
    model = nn.Sequential()
    model.l1 = DNALinear(_linear(200, 100), DNAConfig(max_orthogonal_domains=1e3))
    rep = capacity_report(model, 200, DNAConfig(max_orthogonal_domains=1e3))
    assert not rep["feasible"]
    try:
        capacity_report(model, 200, DNAConfig(max_orthogonal_domains=1e3, strict_capacity=True))
        raise AssertionError("should have raised")
    except InfeasibleConfig:
        pass


def test_counters():
    dna = DNALinear(_linear(), DNAConfig())
    dna(torch.randn(10, 50))
    assert dna.tokens == 10 and dna.ops == 10 * dna.n_weight_gates * 2.0


def test_fuel_depletion():
    x = torch.full((1000, 4), 0.5)
    assert torch.equal(SeesawActivation(DNAConfig())(x), x)  # infinite fuel: no effect
    act = SeesawActivation(DNAConfig(fuel_budget=100.0, fuel_chunk=10))
    y = act(x)
    assert torch.equal(y[:10], x[:10])                      # fresh fuel: unchanged
    assert (y[1:, 0] <= y[:-1, 0]).all() and y[-1, 0] < 0.05 # monotone decay toward empty
    assert act.consumed[0] <= 100.0 + 1e-4                   # never burns more than it has
    act.refuel()
    assert torch.equal(act(x)[:10], x[:10])


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} passed")
