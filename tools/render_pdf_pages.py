from pathlib import Path
import sys

import pypdfium2 as pdfium


pdf_path = Path(sys.argv[1]).resolve()
out_dir = Path(sys.argv[2]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)

pdf = pdfium.PdfDocument(str(pdf_path))
for index in range(len(pdf)):
    page = pdf[index]
    bitmap = page.render(scale=2.0)
    image = bitmap.to_pil()
    image.save(out_dir / f"page-{index + 1}.png")
print(len(pdf))
