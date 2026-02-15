import sys
import logging
from pathlib import Path
import importlib.util
from hdfs import InsecureClient
import numpy as np
import torch

from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path
from PIL import Image
import joblib

@dataclass
class TestContext:
    model: Any | None
    models_dir: Path
    metadata: Dict[str, Any]

# -------------------------------------------------------------------
# Logging configuration (stdout → Airflow logs)
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("test_runner")

# -------------------------------------------------------------------
# Prepare execution context
# -------------------------------------------------------------------
models_dir = Path("/tmp/models")
models_dir.mkdir(parents=True, exist_ok=True)

logger.info("Execution context prepared")
logger.info("Artifacts dir: %s", models_dir)


# -------------------------------------------------------------------
# Input validation
# -------------------------------------------------------------------
if len(sys.argv) < 1:
    logger.error("No script name provided. Usage: test_runner.py <script_name.py>")
    sys.exit(1)

logger.info("Starting test runner")

script_name = sys.argv[1]
logger.info("Requested test script: %s", script_name)


# -------------------------------------------------------------------
# HDFS download
# -------------------------------------------------------------------
hdfs_url = "http://namenode:9870"
hdfs_script_path = f"/ml/scripts/{script_name}"
local_path = Path("/tmp/test.py")

logger.info("Connecting to HDFS: %s", hdfs_url)
client = InsecureClient(hdfs_url, user="hdfs")

logger.info("Downloading script from HDFS: %s → %s", hdfs_script_path, local_path)
client.download(hdfs_script_path, str(local_path), overwrite=True)

if not local_path.exists():
    logger.error("Downloaded script not found at %s", local_path)
    sys.exit(1)

logger.info("Script downloaded successfully (%d bytes)", local_path.stat().st_size)

# -------------------------------------------------------------------
# HDFS model download
# -------------------------------------------------------------------
hdfs_modrl_path = f"/ml/models/{script_name}"
local_path = Path("/tmp/test.py")

logger.info("Connecting to HDFS: %s", hdfs_url)
client = InsecureClient(hdfs_url, user="hdfs")

logger.info("Downloading script from HDFS: %s → %s", hdfs_script_path, local_path)
client.download(hdfs_script_path, str(local_path), overwrite=True)

if not local_path.exists():
    logger.error("Downloaded script not found at %s", local_path)
    sys.exit(1)

logger.info("Script downloaded successfully (%d bytes)", local_path.stat().st_size)


# -------------------------------------------------------------------
# Dynamic import
# -------------------------------------------------------------------
logger.info("Loading test module dynamically")

spec = importlib.util.spec_from_file_location("user_test", local_path)
if spec is None or spec.loader is None:
    logger.error("Failed to create import spec for %s", local_path)
    sys.exit(1)

test_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_module)

logger.info("Module loaded: %s", test_module.__name__)

# -------------------------------------------------------------------
# Contract check
# -------------------------------------------------------------------
if not hasattr(test_module, "train"):
    logger.error("Test module does not define train()")
    raise RuntimeError("Test must define train()")

logger.info("train() function found, executing test")

# -------------------------------------------------------------------
# Execute test
# -------------------------------------------------------------------
try:
    result = test_module.train()
    logger.info("Script execution completed successfully")
except Exception as e:
    logger.exception("Script execution failed")
    raise

# -------------------------------------------------------------------
# Result handling
# -------------------------------------------------------------------
if result is None:
    logger.warning("Script returned None")
else:
    logger.info("Script returned result of type: %s", type(result))
    logger.debug("Script result content: %s", result)

logger.info("Model trainer finished")

# -------------------------
# Handling the result
# -------------------------

model = result.get("model")
model_name = result.get("model_name")
hdfs_model_path = f"/ml/models/{script_name[:-3]}/{str(model_name)}"

if model is not None:

    local_model_path = models_dir / "model.pt"
    hdfs_model_path = f"{hdfs_model_path}/{str(model_name)}.pt"

    torch.save(model.state_dict(), local_model_path)

    client.upload(
        hdfs_model_path,
        str(local_model_path),
        overwrite=True,
    )

    logger.info("Model successfully saved to HDFS")
else:
    logger.warning("No model returned, skipping model save")


# -------------------------
# Save artifacts (images)
# -------------------------

artifacts = result.get("artifacts")
hdfs_model_path = f"/ml/models/{script_name[:-3]}/{str(model_name)}"

if artifacts is not None:
    for key, value in artifacts.items():
        local_artifact_path = models_dir / f"{key}.pt"
        hdfs_artifact_path = f"{hdfs_model_path}/artifacts/{str(key)}.pt"

        torch.save(value, local_artifact_path)

        client.upload(
            hdfs_artifact_path,
            str(local_artifact_path),
            overwrite=True,
        )

    logger.info("Artifacts successfully saved to HDFS")
else:
    logger.warning("No artifacts returned, skipping artifacts save")

# -------------------------
# Save metrics
# -------------------------

