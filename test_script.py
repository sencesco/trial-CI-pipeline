import pytest
from train import run_training

def test_training_runs():
    """
    Tests if the training function executes without raising an exception.
    """
    try:
        run_training()
    except Exception as e:
        pytest.fail(f"run_training() raised an exception: {e}")

