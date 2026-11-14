# Python project setup (VS Code + tests)

How to set up this project locally so VS Code shows the Viam SDK's APIs, and so you can run unit tests and integration tests against armfarm01.

Your code is a **client**: it talks to the machine through the Viam Python SDK (`viam-sdk`). The modules (anomalib, openvino, etc.) run on the NUC, so they don't need to be installed locally.

## 1. Create the Python environment

From a terminal (needs [uv](https://docs.astral.sh/uv/)):

```bash
cd ~/projects/viamrobot && uv init --app --python 3.12 --name viamrobot
```

```bash
uv add viam-sdk python-dotenv
```

```bash
uv add --dev pytest pytest-asyncio
```

This creates `.venv/` (Python 3.12), `pyproject.toml` and `uv.lock`. Python 3.12 is used instead of the system Python 3.14, which `viam-sdk`'s dependencies may not support yet.

Add this to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "session"
asyncio_default_test_loop_scope = "session"
markers = [
  "machine: talks to the real armfarm01 (needs VIAM_* env vars)",
  "moves: physically moves the arm",
]
```

## 2. VS Code

### Extensions
Open the Extensions view (⇧⌘X) and install **Python** (Microsoft). **Pylance** usually installs with it; if not, install it too. Pylance provides autocomplete, hover docs and go-to-definition.

### Open the project folder
```bash
cd ~/projects/viamrobot && code .
```
Open `viamrobot` itself, not a parent folder. VS Code only finds `.venv` automatically in the folder it has open.

### Select the interpreter
This points VS Code at `.venv`, where `viam-sdk` is installed.

1. ⇧⌘P → **Python: Select Interpreter**
2. Pick `./.venv/bin/python` (3.12.x).

**If `.venv` isn't listed** (for example, VS Code was open before `.venv` was created):
- Click the refresh icon (↻) at the top right of the picker, or run ⇧⌘P → **Developer: Reload Window**, then try again; or
- Choose **Enter interpreter path…** and paste `/Users/wilmccann/projects/viamrobot/.venv/bin/python`.

To check: with a `.py` file open, the status bar (bottom right) shows **3.12.x ('.venv')**. Hovering over `Arm` in `from viam.components.arm import Arm` shows its docs, and ⌘-click (or F12) opens the SDK source.

### `.vscode/settings.json`
Saves the interpreter choice and turns on the **Testing** panel (flask icon in the sidebar):

```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"]
}
```

The Testing panel stays empty until `tests/` exists.

### Optional: module source for reference
`viam-sdk` documents the generic service methods, but not what a specific module returns or which `do_command` calls it accepts. To read the anomalib module's source:

```bash
cd ~/projects && git clone https://github.com/viam-labs/viam-anomalib-service
```

Add it in VS Code with **File → Add Folder to Workspace…**

## 3. Credentials

1. In the Viam app, open armfarm01 → **CONNECT** tab. Copy the machine address, API key ID and API key.
2. Create `.env` in the project root:
   ```
   VIAM_MACHINE_ADDRESS=armfarm1-main.xxxx.viam.cloud
   VIAM_API_KEY_ID=...
   VIAM_API_KEY=...
   ```
3. Add `.env` to `.gitignore` so it never gets committed.

The SDK connects through Viam's cloud; it doesn't need `viam machines part tunnel`. The tunnel is only for the machine's HTTP ports (for example 8888).

## 4. Tests

### `tests/conftest.py`
Opens one machine connection shared by all machine tests. Without credentials, those tests are skipped.

```python
import os, pytest, pytest_asyncio
from dotenv import load_dotenv
from viam.robot.client import RobotClient

load_dotenv()

def pytest_collection_modifyitems(config, items):
    if os.getenv("VIAM_API_KEY"):
        return
    skip = pytest.mark.skip(reason="VIAM_* env vars not set")
    for item in items:
        if "machine" in item.keywords:
            item.add_marker(skip)

@pytest_asyncio.fixture(scope="session")
async def robot():
    opts = RobotClient.Options.with_api_key(
        api_key=os.environ["VIAM_API_KEY"], api_key_id=os.environ["VIAM_API_KEY_ID"])
    robot = await RobotClient.at_address(os.environ["VIAM_MACHINE_ADDRESS"], opts)
    yield robot
    await robot.close()
```

### `tests/integration/test_machine.py`
Read-only checks against the real machine. Nothing here moves the arm.

```python
import pytest
from viam.components.arm import Arm
from viam.services.vision import VisionClient

pytestmark = pytest.mark.machine

async def test_expected_resources(robot):
    names = {r.name for r in robot.resource_names}
    assert {"arm", "cam", "overview-cam", "vision-1"} <= names

async def test_arm_reports_pose(robot):
    pose = await Arm.from_robot(robot, "arm").get_end_position()
    assert pose.z > -200

async def test_anomaly_vision(robot):
    vision = VisionClient.from_robot(robot, "vision-1")
    props = await vision.get_properties()
    if props.classifications_supported:
        out = await vision.get_classifications_from_camera("cam", count=1)
    else:
        out = await vision.get_detections_from_camera("cam")
    print(out)
```

The names `arm` and `cam` come from the config fragments; check them against the first test's output.

### Unit tests
Unit tests don't need the machine. Write decision and sorting logic as plain functions or classes that take the arm and vision clients as arguments. In tests, pass `unittest.mock.AsyncMock(spec=Arm)` and assert what was called (for example, that a low anomaly score sends the item to the box pose).

### Running
Unit tests only:
```bash
uv run pytest -m "not machine"
```

Machine tests (prints output):
```bash
uv run pytest -m machine -s
```

Everything (machine tests are skipped if `.env` isn't filled in):
```bash
uv run pytest -v
```

Or use the Testing panel in VS Code to run or debug single tests.

## Safety

armfarm01 is shared with teammates and Viam staff. Mark any test that moves the arm `@pytest.mark.moves` and run those only on purpose (`uv run pytest -m moves`), with someone at the red E-stop on the xArm controller.
