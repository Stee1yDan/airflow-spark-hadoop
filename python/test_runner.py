import sys
import logging
from pathlib import Path
import importlib.util
from hdfs import InsecureClient

# -------------------------------------------------------------------
# Logging configuration (stdout → Airflow logs)
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("test_runner")

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