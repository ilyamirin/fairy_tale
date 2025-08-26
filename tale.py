import ollama
import argparse
import random
from PIL import Image
import os
import torch
from diffusers import AutoPipelineForText2Image, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from dotenv import load_dotenv

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


# === Агенты через Ollama ===
class Agent:
    def __init__(self, name, system_prompt, model="qwen:14b-chat-q4_K_M"):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model

    def respond(self, prompt):
        response = ollama.generate(
            model=self.model,
            prompt=f"{self.system_prompt}\n\n{prompt}",
            options={
                'temperature': 0.6,
                'top_p': 0.95,
                'top_k': 30,
                'num_ctx': 8192
            }
        )
        return response['response'].strip()


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
print(f"\n{COLORS['Этап']}🎯 ТЕМА СКАЗКИ: {chosen_plot["название"]}{COLORS['ENDC']}\n")

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

# === Инициализация агентов ===
narrator = Agent("Рассказчик", "Ты — волшебный рассказчик. Говори на русском, в стиле народной сказки.")
child = Agent("Ребёнок", "Ты — ребёнок 7 лет. Реагируй по-русски: задавай вопросы, проси изменить.")
editor = Agent("Редактор", "Ты — редактор. Проверяй логику сказки.")

# === Хранение ===
history = []
full_tale = []
video_files = []


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


# === Основной цикл ===
for stage in CAMPBELL_STAGES:
    print(f"\n{'=' * 70}")
    print(f"{COLORS['Этап']}{stage.upper()}{COLORS['ENDC']}")
    print(f"{'=' * 70}")

    stage_prompt = f"Расскажи этап сказки: {stage}. Тема: {chosen_plot}. Учти: {history[-2:] if history else 'начало'}"
    tale_fragment = narrator.respond(stage_prompt)
    history.append({"role": "Рассказчик", "content": tale_fragment})
    full_tale.append(f"**{stage}**\n{tale_fragment}")

    # Диалог
    print(f"\n{COLORS['Рассказчик']}💬 {tale_fragment}{COLORS['ENDC']}")

    child_resp = child.respond(f"Ты услышал: {tale_fragment}...")
    print(f"\n{COLORS['Ребёнок']}💬 {child_resp}{COLORS['ENDC']}")

    editor_resp = editor.respond(f"Проверь логику: {tale_fragment}")
    print(f"\n{COLORS['Редактор']}💬 {editor_resp}{COLORS['ENDC']}")

    # Вывод текста главы
    print(f"\n{COLORS['Сказка']}📖 ТЕКСТ СКАЗКИ:{COLORS['ENDC']}")
    print(f"{tale_fragment}\n")

    # Генерация и вывод картинки сразу
    img_filename = f"stage_{CAMPBELL_STAGES.index(stage) + 1}.png"
    img_path = generate_image(tale_fragment, img_filename)

    # Генерация видео (опционально)
    if args.generate_video:
        video_filename = f"video_stage_{CAMPBELL_STAGES.index(stage) + 1}.mp4"
        video_path = generate_video(img_path, tale_fragment, f"static/{video_filename}")
        if video_path:
            video_files.append(video_path)

# === Финал ===
print(f"\n{'=' * 70}")
print(f"{COLORS['Сказка']}📜 ПОЛНАЯ СКАЗКА{COLORS['ENDC']}")
print(f"{'=' * 70}")

for part in full_tale:
    print(f"\n{part}\n")
    print(f"{'-' * 50}")

if args.generate_video:
    print(f"\n{COLORS['Видео']}🎬 Видео сгенерированы:{COLORS['ENDC']}")
    for v in video_files:
        print(f"→ {v}")