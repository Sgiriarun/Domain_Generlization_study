import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset

from rq1_hr.models.timeppg.model import LogCoshLoss
from rq1_hr.models.timeppg.trainer import run_epoch


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, values):
        return self.linear(values).squeeze(-1)


def test_run_epoch_updates_a_trainable_model():
    values = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    targets = torch.tensor([0.0, 2.0, 4.0, 6.0])
    indices = torch.arange(4)
    loader = DataLoader(TensorDataset(values, targets, indices), batch_size=4)
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = run_epoch(model, loader, LogCoshLoss(), optimizer=optimizer, device=torch.device("cpu"))
    assert metrics["mae_bpm"] >= 0
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
