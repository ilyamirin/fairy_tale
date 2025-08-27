import os
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont


def _wrap_text(draw, text, font, max_width):
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        if not words:
            lines.append("")
            continue
        cur = ""
        for w in words:
            test = (cur + (" " if cur else "") + w)
            w_px = draw.textlength(test, font=font)
            if w_px <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
    return lines


def generate_pdf(timeline: List[Dict[str, Any]], video_files: List[str], chosen_plot: Dict[str, Any], output_path: str = "static/tale.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # A4 page in pixels (150 DPI): 1240x1754
    PAGE_W, PAGE_H = 1240, 1754
    MARGIN = 60
    CONTENT_W = PAGE_W - 2 * MARGIN

    # Fonts
    try:
        font_title = ImageFont.truetype("arial.ttf", 40)
        font_role = ImageFont.truetype("arial.ttf", 28)
        font_text = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        font_title = ImageFont.load_default()
        font_role = ImageFont.load_default()
        font_text = ImageFont.load_default()

    pages = []
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)
    y = MARGIN

    # Title with chosen plot
    title = f"ТЕМА СКАЗКИ: {chosen_plot['название']}"
    title_lines = _wrap_text(draw, title, font_title, CONTENT_W)
    for line in title_lines:
        draw.text((MARGIN, y), line, fill="black", font=font_title)
        y += int(font_title.size * 1.3)
    y += 10

    subtitle = f"Описание: {chosen_plot['описание']}"
    for line in _wrap_text(draw, subtitle, font_text, CONTENT_W):
        draw.text((MARGIN, y), line, fill="black", font=font_text)
        y += int(font_text.size * 1.4)

    y += 20

    def new_page():
        nonlocal page, draw, y
        pages.append(page)
        page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        draw = ImageDraw.Draw(page)
        y = MARGIN

    for item in timeline:
        itype = item.get("type")
        if itype == "header":
            block = item.get("text", "")
            block_lines = _wrap_text(draw, block, font_title, CONTENT_W)
            needed_h = len(block_lines) * int(font_title.size * 1.3) + 20
            if y + needed_h > PAGE_H - MARGIN:
                new_page()
            for line in block_lines:
                draw.text((MARGIN, y), line, fill="black", font=font_title)
                y += int(font_title.size * 1.3)
            y += 10
        elif itype == "agent":
            role = item.get("role", "")
            text = item.get("text", "")
            role_label = f"{role}:"
            role_h = int(font_role.size * 1.4)
            text_lines = _wrap_text(draw, text, font_text, CONTENT_W)
            text_h = len(text_lines) * int(font_text.size * 1.4)
            needed_h = role_h + text_h + 12
            if y + needed_h > PAGE_H - MARGIN:
                new_page()
            draw.text((MARGIN, y), role_label, fill="black", font=font_role)
            y += role_h
            for line in text_lines:
                draw.text((MARGIN, y), line, fill="black", font=font_text)
                y += int(font_text.size * 1.4)
            y += 12
        elif itype == "story":
            label = "ТЕКСТ СКАЗКИ:"
            label_h = int(font_role.size * 1.4)
            text = item.get("text", "")
            text_lines = _wrap_text(draw, text, font_text, CONTENT_W)
            text_h = len(text_lines) * int(font_text.size * 1.4)
            needed_h = label_h + text_h + 12
            if y + needed_h > PAGE_H - MARGIN:
                new_page()
            draw.text((MARGIN, y), label, fill="black", font=font_role)
            y += label_h
            for line in text_lines:
                draw.text((MARGIN, y), line, fill="black", font=font_text)
                y += int(font_text.size * 1.4)
            y += 12
        elif itype == "image":
            img_path = item.get("path")
            if not img_path or not os.path.exists(img_path):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            max_w = CONTENT_W
            max_h = int(PAGE_H * 0.45)
            w, h = img.size
            scale = min(max_w / w, max_h / h, 1.0)
            new_w = int(w * scale)
            new_h = int(h * scale)
            if y + new_h > PAGE_H - MARGIN:
                new_page()
            img_resized = img.resize((new_w, new_h))
            page.paste(img_resized, (MARGIN, y))
            y += new_h + 16
        elif itype == "video":
            note = f"Видео: {os.path.basename(item.get('path', ''))}"
            lines = _wrap_text(draw, note, font_text, CONTENT_W)
            needed_h = len(lines) * int(font_text.size * 1.4) + 8
            if y + needed_h > PAGE_H - MARGIN:
                new_page()
            for line in lines:
                draw.text((MARGIN, y), line, fill="black", font=font_text)
                y += int(font_text.size * 1.4)
            y += 8

    # Final page: list videos
    if video_files:
        if y + int(font_title.size * 1.3) + 20 > PAGE_H - MARGIN:
            new_page()
        draw.text((MARGIN, y), "Список видео:", fill="black", font=font_title)
        y += int(font_title.size * 1.3) + 10
        for v in video_files:
            entry = f"→ {os.path.basename(v)}"
            for line in _wrap_text(draw, entry, font_text, CONTENT_W):
                if y + int(font_text.size * 1.4) > PAGE_H - MARGIN:
                    new_page()
                draw.text((MARGIN, y), line, fill="black", font=font_text)
                y += int(font_text.size * 1.4)

    # Save PDF
    pages.append(page)
    first, rest = pages[0], pages[1:]
    first.save(output_path, "PDF", resolution=150.0, save_all=True, append_images=rest)
    print(f"📄 PDF сохранён: {output_path}")
