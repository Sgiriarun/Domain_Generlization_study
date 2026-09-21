import pytest

from rq1_hr.evaluation import hr_metrics


def test_hr_metrics_have_expected_direction_and_units():
    result = hr_metrics([60, 70, 80], [62, 68, 80])
    assert result["windows"] == 3
    assert result["mae_bpm"] == pytest.approx(4 / 3)
    assert result["rmse_bpm"] == pytest.approx((8 / 3) ** 0.5)
    assert result["within_5_bpm_percent"] == 100.0
    assert result["bland_altman_bias_bpm"] == 0.0
