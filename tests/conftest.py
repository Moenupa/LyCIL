import os

import pytest

from lycil.constants import is_env_enabled

SUPPORTED_DEVICES: set[str] = {"cpu"}
VISIBLE_DEVICE_ENV: str | None = None
for _env_var, _device_type in [
    ("CUDA_VISIBLE_DEVICES", "cuda"),
    ("ASCEND_RT_VISIBLE_DEVICES", "npu"),
]:
    if os.getenv(_env_var):
        VISIBLE_DEVICE_ENV = _env_var
        SUPPORTED_DEVICES.add(_device_type)


@pytest.fixture(scope="session", autouse=True)
def supported_devices():
    return SUPPORTED_DEVICES


@pytest.fixture(scope="session", autouse=True)
def is_dummy_training():
    return is_env_enabled("DUMMY", default="1")


def pytest_generate_tests(metafunc):
    """Parametrize 'device' based on 'runs_on' marker."""
    if "device" in metafunc.fixturenames:
        marker = metafunc.definition.get_closest_marker("runs_on")
        if marker:
            requested_devices = marker.args[0]
            if not isinstance(requested_devices, (list, tuple)):
                requested_devices = [requested_devices]

            devices = [d for d in requested_devices if d in SUPPORTED_DEVICES]
            if devices:
                metafunc.parametrize("device", devices)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]):
    """Modify test collection based on markers and environment."""
    _handle_slow_tests(items)
    _handle_runs_on(items)
    _handle_device_visibility(items)


def _handle_runs_on(items: list[pytest.Item]):
    """Skip tests if none of the specified device TYPES are available."""
    for item in items:
        marker = item.get_closest_marker("runs_on")
        if not marker:
            continue

        requested_devices = marker.args[0]
        if not isinstance(requested_devices, (list, tuple)):
            requested_devices = [requested_devices]
        requested_devices = set(requested_devices)
        request_not_satisfied_marker = pytest.mark.skip(
            reason=f"test requires one of {requested_devices} (available: {SUPPORTED_DEVICES})"
        )

        if SUPPORTED_DEVICES.isdisjoint(requested_devices):
            item.add_marker(request_not_satisfied_marker)


def _handle_slow_tests(items: list[pytest.Item]):
    """Skip slow tests unless RUN_SLOW is enabled."""
    if is_env_enabled("RUN_SLOW"):
        return

    skip_slow = pytest.mark.skip(reason="slow test (set RUN_SLOW=1 to run)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def _handle_device_visibility(items: list[pytest.Item]):
    """Handle device visibility based on test markers."""
    if VISIBLE_DEVICE_ENV is None:
        return

    # Parse visible devices
    visible_devices_env = os.getenv(VISIBLE_DEVICE_ENV)
    assert (
        visible_devices_env is not None
    )  # Should not be None since we checked earlier

    visible_devices = [v for v in visible_devices_env.split(",") if v != ""]
    available = len(visible_devices)

    for item in items:
        marker = item.get_closest_marker("distributed")
        if not marker:
            continue

        required = marker.args[0] if marker.args else 2
        if available < required:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"test requires {required} devices, but only {available} visible"
                )
            )
