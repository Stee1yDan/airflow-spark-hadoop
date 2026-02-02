import os
import json
import ast
import sys
import logging
from pathlib import Path
from hdfs import InsecureClient
import tempfile

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
images_hdfs_dir = result.get("images_hdfs_dir")
report_id = result.get("report_id", "gradient_inversion_report")

logger.info("Metrics path: %s", metrics_path)
logger.info("Images HDFS dir: %s", images_hdfs_dir)
logger.info("Report ID: %s", report_id)

# -------------------------------------------------------------------
# HDFS Setup
# -------------------------------------------------------------------
hdfs_url = "http://namenode:9870"
logger.info("Connecting to HDFS: %s", hdfs_url)
client = InsecureClient(hdfs_url, user="hdfs")



# -------------------------------------------------------------------
# Download and parse metrics
# -------------------------------------------------------------------
# Create temp directory for working files
temp_dir = Path(tempfile.mkdtemp())
local_metrics_path = temp_dir / "metrics.json"

logger.info("Downloading metrics from HDFS: %s → %s", metrics_path, local_metrics_path)
try:
    client.download(metrics_path, str(local_metrics_path), overwrite=True)
    logger.info("Metrics downloaded successfully (%d bytes)", local_metrics_path.stat().st_size)

    # Read and parse metrics
    with open(local_metrics_path, 'r') as f:
        metrics_content = f.read()
        logger.info(f"Metrics file content (first 500 chars): {metrics_content[:500]}")

        # Try to parse as JSON
        try:
            metrics = json.loads(metrics_content)
            logger.info(f"Parsed metrics type: {type(metrics)}")

            # Debug: Print metrics structure
            if isinstance(metrics, dict):
                logger.info(f"Metrics keys: {list(metrics.keys())}")
            elif isinstance(metrics, list):
                logger.info(f"Metrics list length: {len(metrics)}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse metrics JSON: {e}")
            # If JSON parsing fails, use the result dict instead
            metrics = result
            logger.info("Using input JSON as fallback metrics")

except Exception as e:
    logger.error(f"Failed to download metrics file: {e}")
    # Use the result dict as fallback
    metrics = result
    logger.info("Using input JSON as fallback metrics")

# -------------------------------------------------------------------
# Download images from HDFS directory
# -------------------------------------------------------------------
local_images_dir = temp_dir / "images"
local_images_dir.mkdir(parents=True, exist_ok=True)

downloaded_images = []

if images_hdfs_dir:
    logger.info("Listing images in HDFS dir: %s", images_hdfs_dir)

    try:
        hdfs_files = client.list(images_hdfs_dir, status=True)

        for fname, status in hdfs_files:
            if status["type"] != "FILE":
                continue

            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            hdfs_img_path = f"{images_hdfs_dir}/{fname}"
            local_img_path = local_images_dir / fname

            logger.info(
                "Downloading image from HDFS: %s → %s",
                hdfs_img_path,
                local_img_path,
            )

            client.download(
                hdfs_img_path,
                str(local_img_path),
                overwrite=True,
            )

            downloaded_images.append((local_img_path, fname))

        logger.info(
            "Downloaded %d image(s) from HDFS",
            len(downloaded_images),
        )

    except Exception as e:
        logger.error("Failed to download images from HDFS: %s", e)

else:
    logger.warning("No images_hdfs_dir provided in XCOM result")

# -------------------------------------------------------------------
# Report Generate
# -------------------------------------------------------------------
from reportlab.platypus import Image as RLImage, Spacer, SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Try to register font with fallback
try:
    pdfmetrics.registerFont(TTFont("DejaVu", "/fonts/DejaVuSans.ttf"))
    font_name = "DejaVu"
except:
    try:
        # Try system font path
        pdfmetrics.registerFont(TTFont("DejaVu", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"))
        font_name = "DejaVu"
    except:
        # Use standard Helvetica
        font_name = "Helvetica"
        logger.warning("DejaVu font not found, using Helvetica")

title_style = ParagraphStyle(
    "TitleStyle",
    fontName=font_name,
    fontSize=16,
    leading=20,
)

normal_style = ParagraphStyle(
    "NormalStyle",
    fontName=font_name,
    fontSize=11,
    leading=14,
)

heading_style = ParagraphStyle(
    "HeadingStyle",
    fontName=font_name,
    fontSize=13,
    leading=16,
)

table_header_style = ParagraphStyle(
    "TableHeader",
    fontName=font_name,
    fontSize=11,
    leading=14,
    alignment=1,  # center
)

table_cell_style = ParagraphStyle(
    "TableCell",
    fontName=font_name,
    fontSize=10,
    leading=13,
)


def generate_pdf_report(metrics_data, output_path):
    """Generate PDF report from metrics data"""
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

    # Handle different metrics data structures
    if isinstance(metrics_data, dict):
        metrics_items = metrics_data.items()
    elif isinstance(metrics_data, list):
        # Convert list to dict with indices as keys
        metrics_items = [(str(i), str(item)) for i, item in enumerate(metrics_data)]
    else:
        metrics_items = [("value", str(metrics_data))]

    for k, v in metrics_items:
        # Format the value
        if isinstance(v, float):
            value_str = f"{v:.4f}"
        elif isinstance(v, (dict, list)):
            value_str = json.dumps(v, ensure_ascii=False)[:100] + ("..." if len(json.dumps(v)) > 100 else "")
        else:
            value_str = str(v)

        table_data.append(
            [
                Paragraph(str(k), table_cell_style),
                Paragraph(value_str, table_cell_style),
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

    elements.append(
        Paragraph("Изображения", heading_style)
    )

    for image, image_name in downloaded_images:

        elements.append(Spacer(1, 12))

        img_table = Table(
            [
                [
                    RLImage(image, width=5 * cm, height=5 * cm)
                ],
                [
                    Paragraph(str(image_name), table_cell_style)
                ],
            ],
            colWidths=[7 * cm, 7 * cm],
        )

        img_table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                    ("ALIGN", (0, 1), (-1, 1), "CENTER"),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )

        elements.append(img_table)
        elements.append(Spacer(1, 18))

    # ===== Conclusion =====
    elements.append(
        Paragraph(
            "Восстановленное изображение демонстрирует возможность утечки информации "
            "из градиентов",
            normal_style,
        )
    )

    # ===== Metadata =====
    elements.append(Spacer(1, 24))
    elements.append(Paragraph(f"Отчёт сгенерирован: {report_id}", ParagraphStyle(
        "Metadata", fontName=font_name, fontSize=9, textColor=colors.gray
    )))

    doc.build(elements)


# -------------------------------------------------------------------
# Generate PDF locally and upload to HDFS
# -------------------------------------------------------------------
# Create local PDF file
local_pdf_path = temp_dir / f"{report_id}.pdf"
logger.info(f"Generating PDF locally: {local_pdf_path}")

try:
    generate_pdf_report(metrics, str(local_pdf_path))
    logger.info(f"PDF generated successfully: {local_pdf_path} ({local_pdf_path.stat().st_size} bytes)")
except Exception as e:
    logger.error(f"Failed to generate PDF: {e}")
    import traceback

    logger.error(traceback.format_exc())
    sys.exit(1)

# -------------------------------------------------------------------
# Upload PDF to HDFS
# -------------------------------------------------------------------
# Determine HDFS path for PDF
if images_hdfs_dir:
    # Use parent directory of images directory
    base_dir = "/".join(images_hdfs_dir.split("/")[:-1])
    hdfs_pdf_dir = f"{base_dir}/reports"
else:
    # Use parent directory of metrics file
    base_dir = "/".join(metrics_path.split("/")[:-1])
    hdfs_pdf_dir = f"{base_dir}/reports"

hdfs_pdf_path = f"{hdfs_pdf_dir}/{report_id}.pdf"

logger.info(f"Uploading PDF to HDFS: {local_pdf_path} → {hdfs_pdf_path}")

try:
    # Create directory if it doesn't exist
    try:
        client.status(hdfs_pdf_dir)
        logger.info(f"HDFS directory exists: {hdfs_pdf_dir}")
    except:
        logger.info(f"Creating HDFS directory: {hdfs_pdf_dir}")
        client.makedirs(hdfs_pdf_dir)

    # Upload PDF
    client.upload(hdfs_pdf_path, str(local_pdf_path), overwrite=True)
    logger.info(f"PDF successfully uploaded to HDFS: {hdfs_pdf_path}")

    # Verify upload
    hdfs_status = client.status(hdfs_pdf_path)
    logger.info(f"PDF on HDFS: {hdfs_status['length']} bytes")

    # Add PDF path to result
    result["report_pdf_hdfs_path"] = hdfs_pdf_path
    logger.info(f"Report PDF available at: {hdfs_pdf_path}")

except Exception as e:
    logger.error(f"Failed to upload PDF to HDFS: {e}")
    sys.exit(1)

# -------------------------------------------------------------------
# Clean up
# -------------------------------------------------------------------
try:
    # Clean up temp directory
    import shutil

    shutil.rmtree(temp_dir)
    logger.info(f"Cleaned up temp directory: {temp_dir}")
except Exception as e:
    logger.warning(f"Failed to clean up temp directory: {e}")

logger.info("Report generation completed successfully!")