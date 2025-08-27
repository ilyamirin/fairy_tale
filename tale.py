import ollama
import argparse
import random
from PIL import Image, ImageDraw, ImageFont
import os
import torch
from diffusers import AutoPipelineForText2Image, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END

load_dotenv()

# === Аргументы ===
parser = argparse.ArgumentParser()
parser.add_argument('--generate-video', action='store_true', help='Включить генерацию видео (требует много VRAM)')
args = parser.parse_args()
# args.generate_video = True  # Раскомментируйте, чтобы включить

# === Цвета ===
COLORS = {
    'Рассказчик': '\033[94m',
    'Ребёнок': '\033[92m',
    'Редактор': '\033[93m',
    'Сказка': '\033[95m',
    'Этап': '\033[1;97m',
    'Видео': '\033[96m',
    'ENDC': '\033[0m'
}

# === Модели ===
print("🚀 Загрузка моделей...")
# Актуально в случае использования моделей из transformers


# Генерация изображений
print("🎨 Загрузка SDXL-Turbo...")
pipe_sdxl = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipe_sdxl.to("cuda")

# Генерация видео (XT)
pipe_svd = None
if args.generate_video:
    print("🎥 Загрузка Stable Video Diffusion XT 1.1...")
    pipe_svd = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        torch_dtype=torch.float16,
        variant="fp16",
        token=os.environ["HF_TOKEN"],
    )
    pipe_svd.to("cuda")


# === Диалоговые узлы будут реализованы через LangGraph ===


# === Тема и этапы ===
PLOT_TEMPLATES = [
    {
        "название": "Спасение похищенной царевны",
        "описание": "Змей унёс царевну. Герой отправляется в путь.",
        "промпт": "Царевну похитил змей, и добрый молодец отправляется её спасать."
    },
    {
        "название": "Поиски живой воды",
        "описание": "Царь болен — нужна вода за тридевять земель.",
        "промпт": "Царь заболел — нужна живая вода за тридевять земель."
    },
    {
        "название": "Превращение брата в лягушку",
        "описание": "Брат заколдован. Сестра ищет способ снять заклятие.",
        "промпт": "Брат превращён в лягушку — нужно снять заклятие."
    },
    {
        "название": "Три испытания Бабы-Яги",
        "описание": "Герой должен пройти три трудных испытания.",
        "промпт": "Герой приходит к Бабе-Яге и должен пройти три испытания."
    },
    {
        "название": "Сказочный брак как награда",
        "описание": "Младший брат женится на царевне, несмотря на насмешки.",
        "промпт": "Младший брат женится на царевне, несмотря на насмешки."
    },
    {
        "название": "Победа над ложным героем",
        "описание": "Настоящий герой разоблачён на свадьбе.",
        "промпт": "Ложный герой присваивает заслуги, но правда раскрывается."
    },
    {
        "название": "Герой из бедняков",
        "описание": "Сирота спасает королевство и становится королём.",
        "промпт": "Сирота спасает королевство и становится королём."
    },
    {
        "название": "Волшебное дерево",
        "описание": "Дерево даёт одно желание, но с условием.",
        "промпт": "Волшебное дерево даёт одно желание, но с условием."
    },
    {
        "название": "Ключ от подземелья",
        "описание": "Герой ищет ключ, спрятанный в сердце горы.",
        "промпт": "Герой ищет ключ, спрятанный в сердце горы."
    },
    {
        "название": "Песня, пробуждающая камни",
        "описание": "Только музыка может оживить зачарованный город.",
        "промпт": "Только музыка может оживить зачарованный город."
    },
    {
        "название": "Зеркало правды",
        "описание": "Зеркало показывает не отражение, а душу.",
        "промпт": "Зеркало показывает не отражение, а душу."
    },
    {
        "название": "Конь-огонёк",
        "описание": "Герой получает волшебного коня, который бежит по небу.",
        "промпт": "Герой получает волшебного коня, который бежит по небу."
    },
    {
        "название": "Огонь, вода и... математика",
        "описание": "Чтобы пройти, нужно решить загадку от старца.",
        "промпт": "Чтобы пройти, нужно решить загадку от старца."
    },
    {
        "название": "Дерево, которое помнит всё",
        "описание": "Герой спрашивает у дерева, как победить злодея.",
        "промпт": "Герой спрашивает у дерева, как победить злодея."
    },
    {
        "название": "Царство, потерянное во времени",
        "описание": "Город застыл в прошлом — нужно вернуть время.",
        "промпт": "Город застыл в прошлом — нужно вернуть время."
    }
]
chosen_plot = random.choice(PLOT_TEMPLATES)
print(f"\n{COLORS['Этап']}🎯 ТЕМА СКАЗКИ: {chosen_plot['название']}{COLORS['ENDC']}\n")

CAMPBELL_STAGES = [
    "1. Обычный мир",
    "2. Призыв к приключению",
    "3. Отказ от призыва",
    "4. Встреча с наставником",
    "5. Переход порога",
    "6. Испытания, союзники, враги",
    "7. Подземелье: приближение к пещере",
    "8. Кульминация: испытание огнём",
    "9. Вознаграждение",
    "10. Возвращение с даром"
]

# === Состояние и граф LangGraph ===
class TaleState(TypedDict, total=False):
    stage: str
    chosen_plot: Dict[str, Any]
    history: List[Dict[str, str]]
    full_tale: List[str]
    pdf_timeline: List[Dict[str, Any]]
    video_files: List[str]
    tale_fragment: Optional[str]
    child_resp: Optional[str]
    editor_resp: Optional[str]
    img_path: Optional[str]


def call_ollama(system_prompt: str, user_prompt: str, model: str = "qwen:14b-chat-q4_K_M") -> str:
    resp = ollama.generate(
        model=model,
        prompt=f"{system_prompt}\n\n{user_prompt}",
        options={
            'temperature': 0.6,
            'top_p': 0.95,
            'top_k': 30,
            'num_ctx': 8192
        }
    )
    return resp.get('response', '').strip()


def narrator_node(state: TaleState) -> TaleState:
    stage = state.get('stage', '')
    chosen = state.get('chosen_plot')
    recent = state.get('history', [])[-2:] if state.get('history') else 'начало'
    stage_prompt = f"Расскажи этап сказки: {stage}. Тема: {chosen}. Учти: {recent}"
    tale_fragment = call_ollama(
        "Ты — волшебный рассказчик. Говори на русском, в стиле народной сказки.",
        stage_prompt,
    )
    history = state.get('history', []) + [{"role": "Рассказчик", "content": tale_fragment}]
    full_tale = state.get('full_tale', []) + [f"**{stage}**\n{tale_fragment}"]
    pdf_timeline = state.get('pdf_timeline', []) + [
        {"type": "header", "text": stage.upper()},
        {"type": "agent", "role": "Рассказчик", "text": tale_fragment},
        {"type": "story", "text": tale_fragment},
    ]
    print(f"\n{'=' * 70}")
    print(f"{COLORS['Этап']}{stage.upper()}{COLORS['ENDC']}")
    print(f"{'=' * 70}")
    print(f"\n{COLORS['Рассказчик']}💬 {tale_fragment}{COLORS['ENDC']}")
    return {
        **state,
        'tale_fragment': tale_fragment,
        'history': history,
        'full_tale': full_tale,
        'pdf_timeline': pdf_timeline,
    }


def child_node(state: TaleState) -> TaleState:
    tale_fragment = state.get('tale_fragment', '')
    child_resp = call_ollama(
        "Ты — ребёнок 7 лет. Реагируй по-русски: задавай вопросы, проси изменить.",
        f"Ты услышал: {tale_fragment}...",
    )
    print(f"\n{COLORS['Ребёнок']}💬 {child_resp}{COLORS['ENDC']}")
    history = state.get('history', []) + [{"role": "Ребёнок", "content": child_resp}]
    pdf_timeline = state.get('pdf_timeline', []) + [{"type": "agent", "role": "Ребёнок", "text": child_resp}]
    return {**state, 'child_resp': child_resp, 'history': history, 'pdf_timeline': pdf_timeline}


def editor_node(state: TaleState) -> TaleState:
    tale_fragment = state.get('tale_fragment', '')
    editor_resp = call_ollama(
        "Ты — редактор. Проверяй логику сказки.",
        f"Проверь логику: {tale_fragment}",
    )
    print(f"\n{COLORS['Редактор']}💬 {editor_resp}{COLORS['ENDC']}")
    history = state.get('history', []) + [{"role": "Редактор", "content": editor_resp}]
    pdf_timeline = state.get('pdf_timeline', []) + [{"type": "agent", "role": "Редактор", "text": editor_resp}]
    return {**state, 'editor_resp': editor_resp, 'history': history, 'pdf_timeline': pdf_timeline}


def media_node(state: TaleState) -> TaleState:
    stage = state.get('stage', '')
    tale_fragment = state.get('tale_fragment', '')
    print(f"\n{COLORS['Сказка']}📖 ТЕКСТ СКАЗКИ:{COLORS['ENDC']}")
    print(f"{tale_fragment}\n")
    img_filename = f"stage_{int(stage.split('.')[0])}.png" if stage else "stage_image.png"
    img_path = generate_image(tale_fragment, img_filename)
    pdf_timeline = state.get('pdf_timeline', []) + [{"type": "image", "path": img_path}]
    video_files = state.get('video_files', [])
    if args.generate_video:
        video_filename = f"video_{img_filename.replace('.png', '.mp4')}"
        video_path = generate_video(img_path, tale_fragment, f"static/{video_filename}")
        if video_path:
            video_files = video_files + [video_path]
            pdf_timeline.append({"type": "video", "path": video_path})
    return {**state, 'img_path': img_path, 'pdf_timeline': pdf_timeline, 'video_files': video_files}


def build_graph():
    graph = StateGraph(TaleState)
    graph.add_node('narrator', narrator_node)
    graph.add_node('child', child_node)
    graph.add_node('editor', editor_node)
    graph.add_node('media', media_node)
    graph.set_entry_point('narrator')
    graph.add_edge('narrator', 'child')
    graph.add_edge('child', 'editor')
    graph.add_edge('editor', 'media')
    graph.add_edge('media', END)
    return graph.compile(debug=False)


# === Генерация изображения ===
def generate_image(prompt, filename="stage_image.png"):
    short_prompt = " ".join(prompt.split(" ")[:50])  # Первые 50 слов
    full_prompt = (
        f"Акварельная иллюстрация в стиле импрессионизма, русская народная сказка — {short_prompt}. "
        "Impressionist watercolor, dreamy light, Russian fairy tale, 4k"
    )
    image = pipe_sdxl(prompt=full_prompt, num_inference_steps=4, guidance_scale=0.0).images[0]

    os.makedirs("static", exist_ok=True)
    img_path = f"static/{filename}"
    image.save(img_path)

    print(f"🖼️ Иллюстрация сгенерирована: {img_path}")

    # Показ изображения в отдельном окне (консольный режим)
    try:
        image.show()
    except Exception as e:
        print(f"⚠️ Не удалось открыть окно просмотра изображения: {e}")

    return img_path


# === Генерация видео ===
def generate_video(image_path, prompt, output_path="video.mp4"):
    if not args.generate_video:
        return None

    short_prompt = " ".join(prompt.split()[:70])
    image = Image.open(image_path).convert("RGB")
    image = image.resize((1024, 576))

    print(f"🎥 Генерация видео: {short_prompt[:60]}...")

    frames = pipe_svd(
        image,
        decode_chunk_size=8,
        generator=torch.manual_seed(42),
        motion_bucket_id=120,
        noise_aug_strength=0.02,
        num_frames=25,
        frame_rate=6
    ).frames[0]

    export_to_video(frames, output_path, fps=6)
    print(f"✅ Видео сохранено: {output_path}")
    return output_path


# === Оркестрация через LangGraph ===
app = build_graph()
state: TaleState = {
    'chosen_plot': chosen_plot,
    'history': [],
    'full_tale': [],
    'pdf_timeline': [],
    'video_files': [],
}

for stage in CAMPBELL_STAGES:
    state['stage'] = stage
    state = app.invoke(state)

# === Финал ===
print(f"\n{'=' * 70}")
print(f"{COLORS['Сказка']}📜 ПОЛНАЯ СКАЗКА{COLORS['ENDC']}")
print(f"{'=' * 70}")

for part in state.get('full_tale', []):
    print(f"\n{part}\n")
    print(f"{'-' * 50}")

if args.generate_video and state.get('video_files'):
    print(f"\n{COLORS['Видео']}🎬 Видео сгенерированы:{COLORS['ENDC']}")
    for v in state.get('video_files', []):
        print(f"→ {v}")

# === Генерация PDF ===
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


def generate_pdf(timeline, video_files, output_path="static/tale.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Страница A4 в пикселях (150 DPI): 1240x1754
    PAGE_W, PAGE_H = 1240, 1754
    MARGIN = 60
    CONTENT_W = PAGE_W - 2 * MARGIN

    # Шрифты
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

    # Заглавие с темой сказки
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
            max_h = int(PAGE_H * 0.45)  # ограничить высоту на странице
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
            # Добавим заметку о видеофайле
            note = f"Видео: {os.path.basename(item.get('path', ''))}"
            lines = _wrap_text(draw, note, font_text, CONTENT_W)
            needed_h = len(lines) * int(font_text.size * 1.4) + 8
            if y + needed_h > PAGE_H - MARGIN:
                new_page()
            for line in lines:
                draw.text((MARGIN, y), line, fill="black", font=font_text)
                y += int(font_text.size * 1.4)
            y += 8

    # Заключительная страница: список видео (если есть)
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

    # Сохранение PDF
    pages.append(page)
    first, rest = pages[0], pages[1:]
    first.save(output_path, "PDF", resolution=150.0, save_all=True, append_images=rest)
    print(f"📄 PDF сохранён: {output_path}")

# Сохранить PDF после вывода полной сказки
try:
    generate_pdf(state.get('pdf_timeline', []), state.get('video_files', []), output_path="static/tale.pdf")
except Exception as e:
    print(f"⚠️ Не удалось создать PDF: {e}")