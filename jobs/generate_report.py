from reportlab.platypus import Image as RLImage, Spacer, SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.units import cm

pdfmetrics.registerFont(
    TTFont("DejaVu", "fonts/DejaVuSans.ttf")
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

def generate_pdf_report(metrics, orig_img_path, recon_img_path, output_path):
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

    img_table = Table(
        [
            [
                RLImage(orig_img_path, width=5 * cm, height=5 * cm),
                RLImage(recon_img_path, width=5 * cm, height=5 * cm),
            ],
            [
                Paragraph("Оригинальное изображение", table_cell_style),
                Paragraph("Восстановленное изображение", table_cell_style),
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

    doc.build(elements)

output_pdf = "./gradient_inversion_report.pdf"

generate_pdf_report(
    metrics=metrics,
    orig_img_path=orig_img_path,
    recon_img_path=recon_img_path,
    output_path=output_pdf
)

print(f"REPORT_PATH={output_pdf}")
