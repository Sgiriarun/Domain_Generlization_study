import pytest

torch = pytest.importorskip("torch")

from rq1_hr.models.timeppg.model import LogCoshLoss, TimePPGBigPPGOnly


def test_timeppg_accepts_frozen_window_shape():
    model = TimePPGBigPPGOnly(input_samples=512)
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(3, 1, 512))
    assert output.shape == (3,)
    assert torch.isfinite(output).all()


def test_timeppg_rejects_wrong_input_channels():
    model = TimePPGBigPPGOnly(input_samples=512)
    with pytest.raises(ValueError):
        model(torch.randn(2, 4, 512))


def test_log_cosh_is_zero_for_exact_prediction():
    loss = LogCoshLoss()(torch.tensor([70.0, 80.0]), torch.tensor([70.0, 80.0]))
    assert loss.item() == pytest.approx(0.0, abs=1e-7)
