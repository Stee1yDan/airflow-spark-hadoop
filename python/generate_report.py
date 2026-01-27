import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("report")

raw = os.environ.get("TEST_RESULT_JSON")
if not raw:
    raise RuntimeError("No TEST_RESULT_JSON provided")

logger.info(f"Raw JSON: {raw}")

result = json.loads(raw)

metrics_path = result["metrics_hdfs_path"]
#images_dir = result["images_hdfs_dir"]

logger.info("Metrics path: %s", metrics_path)
#logger.info("Images dir: %s", images_dir)