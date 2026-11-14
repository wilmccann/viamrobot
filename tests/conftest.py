"""Shared fixtures. Tests marked `machine` connect to armfarm01 using the VIAM_* values in .env."""

import os

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from viam.robot.client import RobotClient

load_dotenv()

REQUIRED_ENV = ("VIAM_MACHINE_ADDRESS", "VIAM_API_KEY_ID", "VIAM_API_KEY")


def pytest_collection_modifyitems(config, items):
    missing = [name for name in REQUIRED_ENV if not os.getenv(name)]
    skip_machine = pytest.mark.skip(reason=f"not set in .env: {', '.join(missing)}") if missing else None
    # Moving the real arm is opt-in per run, never picked up by a plain `pytest`.
    skip_moves = pytest.mark.skip(reason="moves the arm; set VIAM_ALLOW_MOVES=1 with someone at the E-stop")
    allow_moves = os.getenv("VIAM_ALLOW_MOVES") == "1"
    for item in items:
        if skip_machine and "machine" in item.keywords:
            item.add_marker(skip_machine)
        elif "moves" in item.keywords and not allow_moves:
            item.add_marker(skip_moves)


@pytest_asyncio.fixture(scope="session")
async def robot():
    """One connection to the real machine, shared by every machine test in the run."""
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"],
        api_key_id=os.environ["VIAM_API_KEY_ID"],
    )
    client = await RobotClient.at_address(os.environ["VIAM_MACHINE_ADDRESS"], opts)
    yield client
    await client.close()
