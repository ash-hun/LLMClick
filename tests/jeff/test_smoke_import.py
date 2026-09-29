import pytest
import torch

STUDENT = "Qwen/Qwen3.5-0.8B"
STUDENT_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def test_cuda_available() -> None:
    assert torch.cuda.is_available()
    assert "RTX PRO 6000" in torch.cuda.get_device_name(0)


@pytest.mark.slow
def test_student_loads_and_scores() -> None:
    from jeff.model import DecisionModel

    model = DecisionModel(base_model=STUDENT, revision=STUDENT_REVISION)
    row = {"state": "The sky is green.", "question": {"type": "noul", "instructions": "Is the statement true?"}}
    [probabilities] = model.predict([row])
    assert len(probabilities) == 2
    assert abs(sum(probabilities) - 1) < 1e-4
