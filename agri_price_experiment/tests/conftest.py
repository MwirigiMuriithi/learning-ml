import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pytest
from agriprice.config import load_config, SCENARIOS
from agriprice.generator import generate_clean_world
from agriprice.degrade import degrade


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def world(cfg):
    return generate_clean_world(cfg)


@pytest.fixture(scope="session")
def panels(cfg, world):
    return {s: degrade(world, cfg, s)[0] for s in SCENARIOS}
