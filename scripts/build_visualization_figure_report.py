from pathlib import Path

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter, landscape, portrait
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "notebooks" / "visualization_outputs"
OUTPUT_DIR = ROOT / "output" / "pdf"
OUTPUT_PATH = OUTPUT_DIR / "gcamp_visualization_figure_report.pdf"


FIGURES = [
    {
        "file": "01_roi_detected_vs_active.png",
        "title": "ROI detection and activity by day",
        "data": (
            "Per-video detected ROI counts, active ROI counts, and the active/detected fraction "
            "are stratified by day and treatment. Small points are individual videos; large "
            "markers and error bars show treatment means with 95% bootstrap confidence intervals."
        ),
        "conclusion": (
            "Detected ROI counts decline over time, especially for IOBP, while active ROI counts "
            "and active fractions rise sharply after day 4 in both treatments. The parallel rise "
            "suggests a strong time effect; no BP-versus-IOBP comparison survives FDR correction."
        ),
    },
    {
        "file": "02_spike_rate_recording_summary.png",
        "title": "Recording-level spike-rate summaries",
        "data": (
            "Each point is one video's mean or median spike rate across its active ROIs. Values are "
            "shown by day and treatment, with treatment means and 95% bootstrap confidence intervals."
        ),
        "conclusion": (
            "Video-level spike rates increase with later timepoints in both treatments. BP shows the "
            "largest late increase at days 7 and 10, while IOBP rises more moderately and peaks near day 7."
        ),
    },
    {
        "file": "03_spike_rate_neuron_distributions.png",
        "title": "Active-ROI spike-rate distributions",
        "data": (
            "Every active ROI with a spike-rate measurement is plotted by day and treatment. "
            "Horizontal jitter reveals overlaps; diamond/plus markers and capped bars show the mean +/- SEM "
            "across the displayed ROI observations."
        ),
        "conclusion": (
            "The full distributions shift upward at later days rather than changing only through a few "
            "outliers. BP has a higher late-day center and upper tail than IOBP, although ROI-level points "
            "are descriptive and are not independent biological replicates."
        ),
    },
    {
        "file": "04_connectivity_recording_summary.png",
        "title": "Recording-level connectivity summaries",
        "data": (
            "Per-video connectivity group count, mean group size, and fraction of active ROIs assigned "
            "to a group are shown by day and treatment. Points are videos; large markers and bars are "
            "means with 95% bootstrap confidence intervals."
        ),
        "conclusion": (
            "Later videos contain more connectivity groups, particularly BP at day 7, while typical mean "
            "group size remains close to two ROIs. The main temporal change therefore appears to be the "
            "number of groups rather than a uniform expansion of group size."
        ),
    },
    {
        "file": "04b_connectivity_normalized_to_detected_rois.png",
        "title": "Connectivity normalized to detected ROI exposure",
        "data": (
            "Each video contributes groups per 1,000 detected ROIs and the fraction of detected ROIs "
            "assigned to connectivity groups. These recording-level measures retain zero-group videos "
            "and adjust for unequal opportunities to detect groups."
        ),
        "conclusion": (
            "Normalized connectivity increases at later timepoints in both treatments, with the clearest "
            "BP elevation at day 7 and later IOBP elevations around days 6-10. Considerable between-video "
            "variation remains."
        ),
    },
    {
        "file": "05_connectivity_group_size_distributions.png",
        "title": "Connectivity-group size distributions",
        "data": (
            "Every detected connectivity group is plotted according to its number of member ROIs, day, "
            "and treatment. Jitter displays coincident groups; overlaid markers and capped bars show the "
            "mean +/- SEM across displayed groups."
        ),
        "conclusion": (
            "Most groups contain two ROIs at every timepoint. A small number of larger BP groups appear at "
            "later days, but mean group-size day comparisons do not survive FDR correction."
        ),
    },
    {
        "file": "05b_connectivity_group_number_vs_size_joint.png",
        "title": "Video-level group count versus mean group size",
        "data": (
            "Each point is one video, colored by day and shaped by treatment. The x-axis is the number of "
            "connectivity groups and the y-axis is mean group size; zero-group videos are retained at y=0. "
            "Small display-only jitter reveals overlapping videos, and marginal histograms show overall distributions."
        ),
        "conclusion": (
            "High group-count videos occur mainly at later timepoints, but their mean group size usually "
            "remains near two ROIs. This supports group proliferation, rather than broadly larger groups, "
            "as the dominant source of increased connectivity."
        ),
    },
    {
        "file": "06_longitudinal_recording_trajectories.png",
        "title": "Within-video trajectories across days",
        "data": (
            "Repeated measurements are linked within each recording ID, separately for BP and IOBP. "
            "Panels show active/detected fraction, mean spike rate, group count, and mean group size; faint "
            "lines are individual recordings and thick lines are treatment means."
        ),
        "conclusion": (
            "The same recordings generally show rising active fractions and spike rates over time, arguing "
            "against the trend being caused only by different videos sampled at each day. Group counts rise "
            "but vary strongly between recordings, whereas mean group size is comparatively stable."
        ),
    },
    {
        "file": "07_bp_vs_iobp_significance_heatmap.png",
        "title": "BP versus IOBP comparisons by day",
        "data": (
            "For each outcome and day, BP and IOBP videos are compared with Mann-Whitney U tests. The left "
            "heatmap shows nominal p-values; the right shows Benjamini-Hochberg FDR-adjusted q-values. "
            "Tile color encodes -log10 significance and annotations report p or q values and stars."
        ),
        "conclusion": (
            "Several day 7-10 comparisons are nominally significant, but none remain significant after FDR "
            "correction. The dataset therefore supports temporal changes more strongly than a treatment effect "
            "at any individual day."
        ),
    },
    {
        "file": "08_day_pairwise_detected_rois.png",
        "title": "Within-treatment day changes: detected ROIs",
        "data": (
            "Paired recordings are compared between every day pair separately for BP and IOBP. Tile color "
            "encodes signed mean change from the row day to the column day; each cell reports the change, "
            "FDR-adjusted q-value, and significance stars. Pairwise n is four or five videos."
        ),
        "conclusion": (
            "IOBP detected ROI counts fall significantly from day 1 to every later day, with the largest drop "
            "by days 7 and 10. BP shows downward trends, but no detected-ROI day comparison survives FDR correction."
        ),
    },
    {
        "file": "08_day_pairwise_active_rois.png",
        "title": "Within-treatment day changes: active ROIs",
        "data": (
            "Paired video-level active ROI counts are compared across all day pairs within BP and IOBP. "
            "Colors show signed mean changes and cell labels show FDR-adjusted q-values and stars."
        ),
        "conclusion": (
            "Active ROI counts rise at later timepoints in both treatments. BP shows broad significant gains "
            "from days 1-4 into days 5-10, while IOBP gains are clearest by days 6 and 10."
        ),
    },
    {
        "file": "08_day_pairwise_active_fraction.png",
        "title": "Within-treatment day changes: active fraction",
        "data": (
            "The fraction of detected ROIs classified as active is compared between paired days within each "
            "treatment. Signed mean differences are shown by color; annotations contain FDR-adjusted q-values."
        ),
        "conclusion": (
            "Active fraction increases strongly and consistently over time in both BP and IOBP. The broad set "
            "of FDR-significant early-versus-late comparisons makes this the clearest longitudinal effect in the dataset."
        ),
    },
    {
        "file": "08_day_pairwise_mean_spike_rate.png",
        "title": "Within-treatment day changes: mean spike rate",
        "data": (
            "Each video's mean active-ROI spike rate is compared between paired days within BP and IOBP. "
            "Heatmap color shows the signed change in Hz, with FDR-adjusted q-values and stars in each cell."
        ),
        "conclusion": (
            "Mean spike rate increases over time in both treatments. BP shows its strongest increases at days "
            "7 and 10; IOBP peaks at day 7 and then decreases significantly from day 7 to day 10."
        ),
    },
    {
        "file": "08_day_pairwise_connectivity_groups.png",
        "title": "Within-treatment day changes: connectivity group count",
        "data": (
            "Raw numbers of connectivity groups per paired video are compared across all day pairs within each "
            "treatment. Colors show signed mean changes and annotations show FDR-adjusted q-values."
        ),
        "conclusion": (
            "Raw group counts tend to rise at later days, but large between-video variability prevents any day "
            "comparison from surviving FDR correction. Normalized connectivity measures provide stronger evidence."
        ),
    },
    {
        "file": "08_day_pairwise_mean_group_size.png",
        "title": "Within-treatment day changes: mean group size",
        "data": (
            "Mean connectivity-group size is computed per video and compared between paired days within BP and "
            "IOBP. Videos without groups have undefined mean size and do not contribute to this specific outcome."
        ),
        "conclusion": (
            "Mean group size remains close to two ROIs across most days, and no pairwise change survives FDR "
            "correction. Apparent later increases are driven by a small number of videos with larger groups."
        ),
    },
    {
        "file": "08_day_pairwise_fraction_active_grouped.png",
        "title": "Within-treatment day changes: fraction of active ROIs grouped",
        "data": (
            "For each video, the fraction of active ROIs assigned to connectivity groups is compared across "
            "paired days. Tiles encode signed mean change and report FDR-adjusted q-values."
        ),
        "conclusion": (
            "The fraction of active ROIs grouped fluctuates across days and videos, but no day-pair comparison "
            "survives FDR correction. This denominator is sensitive to simultaneous changes in the active ROI pool."
        ),
    },
    {
        "file": "08_day_pairwise_groups_per_1000_detected_rois.png",
        "title": "Within-treatment day changes: groups per 1,000 detected ROIs",
        "data": (
            "Connectivity group count is normalized to each video's detected ROI exposure and compared between "
            "paired days. The metric remains defined for zero-group videos; cells show signed change and FDR q-values."
        ),
        "conclusion": (
            "Normalized group abundance rises significantly at later timepoints. BP is highest at day 7 and "
            "declines by day 10, while IOBP shows significant increases by days 6 and 10 relative to several early days."
        ),
    },
    {
        "file": "08_day_pairwise_fraction_detected_rois_grouped.png",
        "title": "Within-treatment day changes: fraction of detected ROIs grouped",
        "data": (
            "The number of unique grouped ROIs is divided by total detected ROIs for each video, then compared "
            "across paired days. This exposure-based metric includes zero-group videos and reports signed changes "
            "with FDR-adjusted q-values."
        ),
        "conclusion": (
            "The fraction of detected ROIs participating in groups increases significantly at later days. BP "
            "shows its clearest elevation at day 7, and IOBP shows later increases around days 6 and 10."
        ),
    },
]


def draw_report() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    missing = [item["file"] for item in FIGURES if not (FIGURE_DIR / item["file"]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing figure files: {missing}")

    pdf = canvas.Canvas(str(OUTPUT_PATH), pagesize=letter)
    title_style = ParagraphStyle(
        "FigureTitle",
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=19,
        textColor=colors.HexColor("#1f2937"),
        alignment=TA_LEFT,
        spaceAfter=0,
    )
    caption_style = ParagraphStyle(
        "Caption",
        fontName="Helvetica",
        fontSize=9.2,
        leading=12.0,
        textColor=colors.HexColor("#263238"),
        alignment=TA_LEFT,
    )

    total = len(FIGURES)
    for page_number, item in enumerate(FIGURES, start=1):
        image_path = FIGURE_DIR / item["file"]
        with Image.open(image_path) as im:
            pixel_width, pixel_height = im.size
        aspect = pixel_width / pixel_height
        page_size = landscape(letter) if aspect >= 1.25 else portrait(letter)
        page_width, page_height = page_size
        pdf.setPageSize(page_size)

        margin_x = 34
        title_top = page_height - 30
        footer_y = 18
        caption_bottom = 34
        available_width = page_width - 2 * margin_x

        title = Paragraph(f"Figure {page_number}. {item['title']}", title_style)
        title_width, title_height = title.wrap(available_width, 44)
        title.drawOn(pdf, margin_x, title_top - title_height)

        caption_text = (
            f"<b>Data used:</b> {item['data']}<br/><br/>"
            f"<b>Possible conclusion:</b> {item['conclusion']}"
        )
        caption = Paragraph(caption_text, caption_style)
        caption_width, caption_height = caption.wrap(available_width, 150)
        caption.drawOn(pdf, margin_x, caption_bottom)

        separator_y = caption_bottom + caption_height + 8
        pdf.setStrokeColor(colors.HexColor("#d1d5db"))
        pdf.setLineWidth(0.6)
        pdf.line(margin_x, separator_y, page_width - margin_x, separator_y)

        image_bottom = separator_y + 12
        image_top = title_top - title_height - 12
        max_image_width = available_width
        max_image_height = image_top - image_bottom
        scale = min(max_image_width / pixel_width, max_image_height / pixel_height)
        draw_width = pixel_width * scale
        draw_height = pixel_height * scale
        image_x = (page_width - draw_width) / 2
        image_y = image_bottom + (max_image_height - draw_height) / 2
        pdf.drawImage(
            ImageReader(str(image_path)), image_x, image_y,
            width=draw_width, height=draw_height, preserveAspectRatio=True, mask="auto",
        )

        footer = f"GCaMP visualization report | Page {page_number} of {total}"
        pdf.setFont("Helvetica", 7.5)
        pdf.setFillColor(colors.HexColor("#6b7280"))
        footer_width = stringWidth(footer, "Helvetica", 7.5)
        pdf.drawString(page_width - margin_x - footer_width, footer_y, footer)
        pdf.showPage()

    pdf.save()
    print(OUTPUT_PATH)


if __name__ == "__main__":
    draw_report()
