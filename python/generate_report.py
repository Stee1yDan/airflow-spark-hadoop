import os
import json
import ast
import sys
import logging
from pathlib import Path
from hdfs import InsecureClient

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
hdfs_path = metrics_path
local_path = Path("/tmp/test.py")

logger.info("Connecting to HDFS: %s", hdfs_url)
client = InsecureClient(hdfs_url, user="hdfs")

logger.info("Downloading script from HDFS: %s → %s", hdfs_path, local_path)
client.download(hdfs_path, str(local_path), overwrite=True)

if not local_path.exists():
    logger.error("Downloaded script not found at %s", local_path)
    sys.exit(1)

logger.info("Script downloaded successfully (%d bytes)", local_path.stat().st_size)

json_data_path = Path("/tmp/input_data.json")
with open(json_data_path, 'w') as f:
    json.dump(result, f)

# -------------------------------------------------------------------
# Report Generate
# -------------------------------------------------------------------

from reportlab.platypus import Image as RLImage, Spacer, SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

pdfmetrics.registerFont(
    TTFont("DejaVu", "/fonts/DejaVuSans.ttf")
)

title_style = ParagraphStyle(
    "TitleDejaVu",
    fontName="DejaVu",
    fontSize=16,
    leading=20,
)

normal_style = ParagraphStyle(
    "NormalDejaVu",
    fontName="DejaVu",
    fontSize=11,
    leading=14,
)

heading_style = ParagraphStyle(
    "HeadingDejaVu",
    fontName="DejaVu",
    fontSize=13,
    leading=16,
)

table_header_style = ParagraphStyle(
    "TableHeader",
    fontName="DejaVu",
    fontSize=11,
    leading=14,
    alignment=1,  # center
)

table_cell_style = ParagraphStyle(
    "TableCell",
    fontName="DejaVu",
    fontSize=10,
    leading=13,
)

def generate_pdf_report(metrics, output_path):
    doc = SimpleDocTemplate(output_path)
    elements = []

    # ===== Title =====
    elements.append(
        Paragraph(
            "Атака восстановления данных по градиентам (Gradient Inversion). Отчёт",
            title_style
        )
    )
    elements.append(Spacer(1, 14))

    # ===== Metrics table =====
    table_data = [
        [
            Paragraph("Метрика", table_header_style),
            Paragraph("Значение", table_header_style),
        ]
    ]

    for k, v in metrics.items():
        table_data.append(
            [
                Paragraph(str(k), table_cell_style),
                Paragraph(
                    f"{v:.4f}" if isinstance(v, float) else str(v),
                    table_cell_style
                ),
            ]
        )

    metrics_table = Table(table_data, colWidths=[8 * cm, 5 * cm])
    metrics_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    elements.append(metrics_table)
    elements.append(Spacer(1, 18))

    # ===== Reconstruction section =====
    elements.append(
        Paragraph("Результаты реконструкции", heading_style)
    )
    elements.append(Spacer(1, 12))
    elements.append(Spacer(1, 18))

    # ===== Conclusion =====
    elements.append(
        Paragraph(
            "Восстановленное изображение демонстрирует возможность утечки информации "
            "из градиентов",
            normal_style,
        )
    )

    doc.build(elements)

output_pdf = "./gradient_inversion_report.pdf"

logger.info("Resolved JSON")
logger.info(result)


generate_pdf_report(result,output_pdf)