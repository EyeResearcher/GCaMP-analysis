"""Build an inline HTML fragment from generated wave figure previews."""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image


def _preview_data(path: Path) -> str:
    with Image.open(path) as image:
        preview = image.convert("RGB")
        preview.thumbnail((1100, 650))
        buffer = io.BytesIO()
        preview.save(buffer, format="JPEG", quality=58, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_csv", type=Path)
    parser.add_argument("output_html", type=Path)
    args = parser.parse_args()
    index = pd.read_csv(args.index_csv)
    items = []
    for row in index.itertuples(index=False):
        path = Path(row.path)
        repeat = (
            f" · repeat pattern {int(row.recurrence_cluster) + 1}"
            if int(row.recurrence_cluster) >= 0
            else ""
        )
        items.append(
            {
                "label": f"Day {row.day} · {row.treatment} {row.recording} · {row.center_seconds:.1f} s",
                "detail": (
                    f"{row.model.capitalize()} front · {row.n_participants} participating cells · "
                    f"distance timing R² {row.simple_distance_r2:.2f}{repeat}"
                ),
                "src": _preview_data(path),
            }
        )
    data = json.dumps(items, separators=(",", ":"))
    fragment = f"""<div id="retinal-wave-gallery">
  <div class="viz-controls">
    <label class="form-label" for="wave-gallery-select">Wave
      <select class="form-select" id="wave-gallery-select"></select>
    </label>
  </div>
  <div id="wave-gallery-detail" class="text-small text-muted" aria-live="polite"></div>
  <img id="wave-gallery-image" alt="" style="display:block;max-width:100%;height:auto;margin-top:var(--spacing-3,12px);">
</div>
<script>
(() => {{
  const root = document.getElementById("retinal-wave-gallery");
  const select = root.querySelector("#wave-gallery-select");
  const image = root.querySelector("#wave-gallery-image");
  const detail = root.querySelector("#wave-gallery-detail");
  const items = {data};
  items.forEach((item, index) => {{
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = item.label;
    select.appendChild(option);
  }});
  function update() {{
    const item = items[Number(select.value) || 0];
    image.src = item.src;
    image.alt = item.label + ". Cell masks colored by propagation distance beside distance-sorted fluorescence traces.";
    detail.textContent = item.detail;
  }}
  select.addEventListener("change", update);
  update();
}})();
</script>
"""
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(fragment, encoding="utf-8")
    print(f"{args.output_html} ({args.output_html.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
