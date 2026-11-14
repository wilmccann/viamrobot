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
    if not missing:
        return
    skip = pytest.mark.skip(reason=f"not set in .env: {', '.join(missing)}")
    for item in items:
        if "machine" in item.keywords:
            item.add_marker(skip)


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
