"""Example: build a GlasanaDocument from the existing pipeline outputs.

Usage:
    python example_build_doc.py <pdf_path>

Example:
    python example_build_doc.py ./pdfs/ac30fbcf...pdf

Inputs (already produced by the pipeline):
    dataset/annotations.json  — detected regions (COCO format, bbox in pixels)
    dataset/connections.json  — reading order per page
"""

import json
import sys
from pathlib import Path

import pdfplumber

from glasana_doc import (
    AnyDocItem,
    Article,
    BBox,
    CaptionItem,
    ContentLayer,
    FigureItem,
    GlasanaDocument,
    PageInfo,
    Provenance,
    RegionCategory,
    TextItem,
    to_html,
    to_markdown,
)

# Map string label → GlasanaDocument item class
from glasana_doc import (
    AbandonItem, AdvertisementItem, AuthorItem, BylineItem, DatelineItem,
    DeckItem, DropcapItem, EditNoteItem, FigBylineItem, FooterItem,
    FootnoteItem, HeaderItem, HeadlineItem, KickerItem, LiteraryItem,
    LiteratureItem, MarginNoteItem, OrderedListItem, PageNumItem,
    ParagraphItem, QuoteItem, SectionItem, SubheadItem, SubsubheadItem,
    TableItem, TOCItem, TranslatorItem, UnorderedListItem, QuestionItem,
)

LABEL_TO_CLASS: dict[str, type] = {
    "Header": HeaderItem,        "Footer": FooterItem,
    "PageNum": PageNumItem,      "Section": SectionItem,
    "Dateline": DatelineItem,    "EditNote": EditNoteItem,
    "MarginNote": MarginNoteItem,"Headline": HeadlineItem,
    "Kicker": KickerItem,        "Deck": DeckItem,
    "Subhead": SubheadItem,      "Subsubhead": SubsubheadItem,
    "Author": AuthorItem,        "Byline": BylineItem,
    "Translator": TranslatorItem,"Paragraph": ParagraphItem,
    "Quote": QuoteItem,          "Dropcap": DropcapItem,
    "Figure": FigureItem,        "Caption": CaptionItem,
    "FigByline": FigBylineItem,  "Table": TableItem,
    "OrderedList": OrderedListItem, "UnorderedList": UnorderedListItem,
    "Footnote": FootnoteItem,    "TOC": TOCItem,
    "Literary": LiteraryItem,    "Literature": LiteratureItem,
    "Advertisement": AdvertisementItem, "Question": QuestionItem,
    "Abandon": AbandonItem,
}


def _extract_text_for_page(
    pdf: pdfplumber.PDF,
    page_no: int,
    regions: list[dict],
    img_width: float,
    img_height: float,
) -> dict[str, str]:
    """Return {region_id: text} for all regions on a page using pdfplumber.

    Converts bbox_norm_1000 → PDF point coordinates by scaling to the PDF
    page dimensions (which may differ from the rendered image dimensions).
    """
    pdf_page = pdf.pages[page_no]
    pw, ph = pdf_page.width, pdf_page.height  # PDF points, origin bottom-left
    results = {}

    for region in regions:
        x0n, y0n, x1n, y1n = region["bbox_norm_1000"]
        # Scale to PDF point space
        x0 = x0n / 1000.0 * pw
        x1 = x1n / 1000.0 * pw
        y0_pdf = y0n / 1000.0 * ph
        y1_pdf = y1n / 1000.0 * ph
        try:
            cropped = pdf_page.within_bbox((x0, y0_pdf, x1, y1_pdf))
            results[region["region_id"]] = (cropped.extract_text() or "").strip()
        except Exception:
            results[region["region_id"]] = ""

    return results


def build_document(pdf_path: str) -> GlasanaDocument:
    pdf_stem = Path(pdf_path).stem  # e.g. "ac30fbcf..."

    # --- Load pipeline outputs ---
    with open("dataset/annotations.json") as f:
        coco = json.load(f)
    with open("dataset/connections.json") as f:
        connections = json.load(f)

    # Index connections entries by filename stem (e.g. "hash_9" → entry)
    conn_by_page: dict[str, dict] = {
        Path(e["image"]).stem: e
        for e in connections
        if pdf_stem in e["image"]
    }

    doc = GlasanaDocument(source_pdf=pdf_stem)

    with pdfplumber.open(pdf_path) as pdf:
        for img_info in sorted(
            [img for img in coco["images"] if pdf_stem in img["file_name"]],
            key=lambda img: int(img["file_name"].rsplit("_", 1)[1].replace(".jpg", "")),
        ):
            page_no = int(img_info["file_name"].rsplit("_", 1)[1].replace(".jpg", ""))
            page_stem = f"{pdf_stem}_{page_no}"
            img_w, img_h = img_info["width"], img_info["height"]

            doc.pages[page_no] = PageInfo(
                page_no=page_no,
                width=img_w,
                height=img_h,
                source_file=img_info["file_name"],
            )

            if page_stem not in conn_by_page:
                # No reading order data for this page — skip
                continue

            conn_entry = conn_by_page[page_stem]
            regions = conn_entry["regions"]           # list in source order
            tgt_index = conn_entry["layoutreader"]["text"]["tgt_index"]
            # tgt_index[i] = source index of the i-th item in reading order
            ordered_regions = [regions[i] for i in tgt_index]

            # Extract text for all regions on this page at once
            texts = _extract_text_for_page(pdf, page_no, regions, img_w, img_h)

            # --- Article grouping: start a new article at each Headline ---
            current_article: Article | None = None

            for reading_pos, region in enumerate(ordered_regions):
                label = region["label"]
                bbox = BBox.from_norm_1000(region["bbox_norm_1000"], img_w, img_h)
                prov = Provenance.from_bbox(
                    page_no=page_no,
                    bbox=bbox,
                    source_region_id=region["region_id"],
                )
                text = texts.get(region["region_id"], "")
                item_cls = LABEL_TO_CLASS.get(label, ParagraphItem)

                # Build the item
                kwargs = dict(provenance=prov, reading_order=reading_pos)
                if issubclass(item_cls, FigureItem) and not issubclass(item_cls, TextItem):
                    item: AnyDocItem = item_cls(**kwargs)
                else:
                    item = item_cls(text=text, **kwargs)

                # Start a new article at each headline
                if isinstance(item, HeadlineItem):
                    current_article = Article(title=text)
                    doc.add_article(current_article)

                # Assign to current article
                if current_article is not None:
                    item.article_id = current_article.article_id
                    current_article.item_ids.append(item.item_id)

                doc.add_item(item)

    return doc


def main():
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not pdf_path:
        print("Usage: python example_build_doc.py <pdf_path>")
        sys.exit(1)

    print(f"Building GlasanaDocument for {pdf_path} ...")
    doc = build_document(pdf_path)

    print(f"\nDocument summary:")
    print(f"  Pages    : {len(doc.pages)}")
    print(f"  Items    : {len(doc.items)} total, {len(doc.body_order)} in body")
    print(f"  Articles : {len(doc.articles)}")
    for art in doc.articles.values():
        print(f"    - \"{art.title}\" ({len(art.item_ids)} items)")

    # Save JSON intermediate
    out_json = Path(pdf_path).stem + "_doc.json"
    Path(out_json).write_text(doc.model_dump_json(indent=2))
    print(f"\nSaved intermediate JSON → {out_json}")

    # Export Markdown
    out_md = Path(pdf_path).stem + ".md"
    Path(out_md).write_text(to_markdown(doc))
    print(f"Saved Markdown           → {out_md}")

    # Export HTML
    out_html = Path(pdf_path).stem + ".html"
    Path(out_html).write_text(to_html(doc))
    print(f"Saved HTML               → {out_html}")


if __name__ == "__main__":
    main()
