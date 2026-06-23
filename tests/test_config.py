import pytest

from armada.config import Config, load_instance


def test_load_known_instance():
    label, price = load_instance("graviton4-c8g")
    assert "Graviton4" in label
    assert price > 0


def test_load_unknown_instance_raises():
    with pytest.raises(ValueError):
        load_instance("nope-not-real")


def test_instance_preset_feeds_cost_model():
    label, price = load_instance("cobalt-d4ps")
    cfg = Config.load(None, [f"cost.usd_per_hour={price}", f"cost.instance_label={label}"])
    assert cfg.cost.usd_per_hour == price
    assert cfg.cost.instance_label == label
