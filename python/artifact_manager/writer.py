# artifact_store/writer.py
import uuid
import numpy as np
from PIL import Image
import io
import json

def np_to_png_bytes(arr: np.ndarray) -> bytes:
    arr = np.clip(arr, 0, 1)
    img = Image.fromarray((arr * 255).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def png_bytes_to_np(data: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(data)).convert("L")
    return np.array(img) / 255.0

def metrics_to_json(metrics: dict) -> str:
    return json.dumps(metrics)

def metrics_from_json(s: str) -> dict:
    return json.loads(s)


HDFS_BASE_PATH = "/artifacts/gradient_inversion"

def _get_hdfs():
    return fs.HadoopFileSystem(
        host="namenode",
        port=8020,
        user="root"
    )

def save_run_artifacts(metrics: dict, images: dict) -> str:
    hdfs = _get_hdfs()
    run_id = str(uuid.uuid4())
    run_path = f"{HDFS_BASE_PATH}/{run_id}"

    # mkdir
    hdfs.create_dir(run_path)

    # ---- metrics ----
    metrics_path = f"{run_path}/metrics.json"
    with hdfs.open_output_stream(metrics_path) as f:
        f.write(metrics_to_json(metrics).encode("utf-8"))

    # ---- images ----
    for name, np_img in images.items():
        img_path = f"{run_path}/{name}.png"
        png_bytes = np_to_png_bytes(np_img)
        with hdfs.open_output_stream(img_path) as f:
            f.write(png_bytes)

    return run_id
