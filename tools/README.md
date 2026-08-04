# `tools`

Repository tooling that is unrelated to the analysis pipeline. These scripts
generate project documentation artifacts (for example a formatted task-brief
reference) and are not imported by `gcamp_analysis`.

| File | Purpose |
|---|---|
| `build_optimal_task_brief.py` | Render a styled `.docx` reference into `artifacts/` using `python-docx`. |
| `render_pdf_pages.py` | Rasterize/convert document pages to PDF images. |
| `export_docx_pdf.ps1` | PowerShell helper to export the generated `.docx` to PDF. |

## Notes

- Outputs are written under `artifacts/`.
- `build_optimal_task_brief.py` requires `python-docx`; the PDF steps require a
  local Word/PDF toolchain.
