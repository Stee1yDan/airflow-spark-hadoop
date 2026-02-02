# artifact_store/reader.py
from artifact_manager.serializers import png_bytes_to_np, metrics_from_json

HDFS_BASE_PATH = "/artifacts/gradient_inversion"

def _get_hdfs():
    return fs.HadoopFileSystem(
        host="namenode",
        port=8020,
        user="root"
    )

def load_run_artifacts(run_id: str):
    hdfs = _get_hdfs()
    run_path = f"{HDFS_BASE_PATH}/{run_id}"

    # ---- metrics ----
    metrics_path = f"{run_path}/metrics.json"
    with hdfs.open_input_file(metrics_path) as f:
        metrics = metrics_from_json(
            f.read().decode("utf-8")
        )

    # ---- images ----
    images = {}
    selector = fs.FileSelector(run_path, recursive=False)
    for info in hdfs.get_file_info(selector):
        if info.path.endswith(".png"):
            name = info.path.split("/")[-1].replace(".png", "")
            with hdfs.open_input_file(info.path) as f:
                images[name] = png_bytes_to_np(f.read())

    return metrics, images
