import os
import json
import ast
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("report")

raw = os.environ.get("TEST_RESULT_JSON")
if not raw:
    raise RuntimeError("No TEST_RESULT_JSON provided")

logger.info(f"Raw JSON: {raw}")
logger.info(f"Raw repr: {repr(raw)}")  # Debug: Show the exact string

# The raw string looks like: "XCOM_RESULT={\u0027metrics_hdfs_path\u0027: ...}"
# It has double quotes around the entire string

# First, strip the outer double quotes if present
raw = raw.strip('"')

# Now remove XCOM_RESULT= prefix
if raw.startswith('XCOM_RESULT='):
    raw = raw[len('XCOM_RESULT='):]

# The string now should be: {\u0027metrics_hdfs_path\u0027: ...}
# Decode escaped Unicode
raw = raw.encode().decode('unicode_escape')

logger.info(f"After processing: {repr(raw)}")

# Now parse with ast.literal_eval
result = ast.literal_eval(raw)

metrics_path = result["metrics_hdfs_path"]
#images_dir = result["images_hdfs_dir"]

logger.info("Metrics path: %s", metrics_path)
#logger.info("Images dir: %s", images_dir)