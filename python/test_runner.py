import sys
import logging
from pathlib import Path
import importlib.util
from hdfs import InsecureClient

from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path

@dataclass
class TestContext:
    model: Any | None
    artifacts_dir: Path
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
artifacts_dir = Path("/tmp/artifacts")
artifacts_dir.mkdir(parents=True, exist_ok=True)

# ctx = TestContext(
#     model=None,  # will plug models later
#     artifacts_dir=artifacts_dir,
#     metadata={
#         "script_name": script_name,
#         "hdfs_source": hdfs_path,
#     }
# )

logger.info("Execution context prepared")
logger.info("Artifacts dir: %s", artifacts_dir)

# -------------------------------------------------------------------
# Input validation
# -------------------------------------------------------------------
if len(sys.argv) < 2:
    logger.error("No script name provided. Usage: test_runner.py <script_name.py>")
    sys.exit(1)

script_name = sys.argv[1]
logger.info("Starting test runner")
logger.info("Requested test script: %s", script_name)

# -------------------------------------------------------------------
# HDFS download
# -------------------------------------------------------------------
hdfs_url = "http://namenode:9870"
hdfs_path = f"/ml/scripts/{script_name}"
local_path = Path("/tmp/test.py")

logger.info("Connecting to HDFS: %s", hdfs_url)
client = InsecureClient(hdfs_url, user="hdfs")

logger.info("Downloading script from HDFS: %s → %s", hdfs_path, local_path)
client.download(hdfs_path, str(local_path), overwrite=True)

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
if not hasattr(test_module, "run"):
    logger.error("Test module does not define run()")
    raise RuntimeError("Test must define run()")

logger.info("run() function found, executing test")

# -------------------------------------------------------------------
# Execute test
# -------------------------------------------------------------------
try:
    result = test_module.run()
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
for name, local_path in result.get("artifacts", {}).items():
    local_path = Path(local_path)

    if not local_path.exists():
        logger.warning("Artifact not found: %s", local_path)
        continue

    hdfs_target = f"{hdfs_base}/artifacts/{name}"

    client.upload(
        hdfs_target,
        str(local_path),
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
    "images_hdfs_dir": f"/ml/results/gradient_inversion/{run_id}/images",
    "report_id": "gradient_inversion_2026-01-27"
}

with open(xcom_path, "w") as f:
    json.dump(payload, f)

print(f"XCOM_RESULT={payload}")