import sys
import logging
from pathlib import Path
import importlib.util
from hdfs import InsecureClient
import numpy as np

from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path
from PIL import Image

@dataclass
class TestContext:
    model_name: Any | None
    model_path: str | None
    artifacts_dir: Path
    metadata: Dict[str, Any]
    local_model_path: Path | None


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
artifacts_dir = Path("/tmp/artifacts")
artifacts_dir.mkdir(parents=True, exist_ok=True)

logger.info("Execution context prepared")
logger.info("Artifacts dir: %s", artifacts_dir)

# -------------------------------------------------------------------
# Input validation
# -------------------------------------------------------------------
if len(sys.argv) < 2:
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
local_script_path = Path("/tmp/test.py")

logger.info("Connecting to HDFS: %s", hdfs_url)
client = InsecureClient(hdfs_url, user="hdfs")

logger.info("Downloading script from HDFS: %s → %s", hdfs_script_path, local_script_path)
client.download(hdfs_script_path, str(local_script_path), overwrite=True)

if not local_script_path.exists():
    logger.error("Downloaded script not found at %s", local_script_path)
    sys.exit(1)

logger.info("Script downloaded successfully (%d bytes)", local_script_path.stat().st_size)

# -------------------------------------------------------------------
# Download model from HDFS
# -------------------------------------------------------------------

if len(sys.argv) < 3:
    logger.error("No model name provided. Usage: test_runner.py <script.py> <model_file>")
    sys.exit(1)

model_name = sys.argv[2]
hdfs_model_path = f"/ml/models/{model_name}"
local_model_path = Path("/tmp/model.pt")

logger.info("Downloading model from HDFS: %s → %s", hdfs_model_path, local_model_path)

client.download(hdfs_model_path, str(local_model_path), overwrite=True)

if not local_model_path.exists():
    logger.error("Downloaded model not found at %s", local_model_path)
    sys.exit(1)

logger.info("Model downloaded successfully (%d bytes)", local_model_path.stat().st_size)


# -------------------------------------------------------------------
# Dynamic import
# -------------------------------------------------------------------
logger.info("Loading test module dynamically")

spec = importlib.util.spec_from_file_location("user_test", local_script_path)
if spec is None or spec.loader is None:
    logger.error("Failed to create import spec for %s", local_script_path)
    sys.exit(1)

test_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_module)

logger.info("Module loaded: %s", test_module.__name__)

# -------------------------------------------------------------------
# Contract check
# -------------------------------------------------------------------
if not hasattr(test_module, "run"):
    logger.error("Test module does not define run()")
    raise RuntimeError("Test must define run()")

logger.info("run() function found, executing test")

# -------------------------------------------------------------------
# Make context
# -------------------------------------------------------------------

try:
    ctx = TestContext(model_name = model_name,
                      model_path=hdfs_model_path,
                      artifacts_dir=artifacts_dir,
                      local_model_path = local_model_path,
                      metadata={
                          "script": script_name,
                      },
    )
except Exception as e:
    logger.exception("Test execution failed")

# -------------------------------------------------------------------
# Execute test
# -------------------------------------------------------------------

try:
    result = test_module.run(ctx)
    logger.info("Test execution completed successfully")
except Exception as e:
    logger.exception("Test execution failed")
    raise

# -------------------------------------------------------------------
# Result handling
# -------------------------------------------------------------------
if result is None:
    logger.warning("Test returned None")
else:
    logger.info("Test returned result of type: %s", type(result))
    logger.debug("Test result content: %s", result)

logger.info("Test runner finished")

# -------------------------
# Handling the result
# -------------------------

import json
from datetime import datetime

run_id = datetime.utcnow().isoformat().replace(":", "-")
hdfs_base = f"/ml/results/{script_name.replace('.py', '')}/{run_id}"

logger.info("Saving results to HDFS: %s", hdfs_base)

# Ensure directories
client.makedirs(f"{hdfs_base}/artifacts")

# -------------------------
# Save metrics
# -------------------------
metrics_path = artifacts_dir / "metrics.json"

with metrics_path.open("w") as f:
    json.dump(result.get("metrics", {}), f, indent=2)

client.upload(
    f"{hdfs_base}/metrics.json",
    str(metrics_path),
    overwrite=True,
)

logger.info("Metrics saved to HDFS")

# -------------------------
# Save artifacts (images)
# -------------------------
for name, image in result.get("artifacts", {}).items():

    hdfs_target = f"{hdfs_base}/artifacts/{name}.png"
    img_local_path = Path(f"/tmp/{name}.png")

    if not isinstance(image, np.ndarray):
        logger.warning("Artifact '%s' is not a numpy array, skipping.", name)
        continue

    if np.isnan(image).any():
        logger.warning("NaNs found in original or reconstructed images; replacing with 0")
        image = np.nan_to_num(image, nan=0.0)

    if isinstance(image, np.ndarray):
        img_local_path = Path(f"/tmp/{name}.png")

        # Normalize to 0-255 dynamically
        min_val, max_val = image.min(), image.max()
        if min_val == max_val:
            image_uint8 = np.zeros_like(image, dtype=np.uint8)
        else:
            image_uint8 = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        im = Image.fromarray(image_uint8)
        im.save(img_local_path)


    client.upload(
        hdfs_target,
        str(img_local_path),
        overwrite=True,
    )

    logger.info("Uploaded artifact: %s", hdfs_target)

# -------------------------
# Sharing image
# -------------------------

import json
from pathlib import Path

xcom_path = Path("/airflow/xcom/return.json")
xcom_path.parent.mkdir(parents=True, exist_ok=True)

payload = {
    "metrics_hdfs_path": f"/ml/results/gradient_inversion/{run_id}/metrics.json",
    "images_hdfs_dir": f"/ml/results/gradient_inversion/{run_id}/artifacts",
    "report_id": "gradient_inversion_2026-01-27"
}

with open(xcom_path, "w") as f:
    json.dump(payload, f)

print(f"XCOM_RESULT={payload}")