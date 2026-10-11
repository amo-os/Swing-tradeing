import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_project_configuration_is_valid():
    with (ROOT / "config" / "settings.toml").open("rb") as file:
        settings = tomllib.load(file)

    assert settings["project"]["market"] == "India"
    assert settings["project"]["timeframe"] == "1d"
    assert settings["project"]["execution_mode"] == "manual_only"


def test_risk_defaults_are_conservative_placeholders():
    with (ROOT / "config" / "settings.toml").open("rb") as file:
        settings = tomllib.load(file)

    assert settings["risk"]["risk_fraction"] == 0.005
    assert settings["risk"]["capital_inr"] == 0
    assert settings["risk"]["max_open_positions"] >= 1


def test_market_data_provider_is_not_assumed_ready():
    with (ROOT / "config" / "settings.toml").open("rb") as file:
        settings = tomllib.load(file)

    assert settings["data"]["provider"] == "unconfigured"
    assert settings["data"]["require_validated_data"] is True
