import os
import re
import io
import json
import base64
import html as html_lib
import random
import string
import uuid
import asyncio
import operator
import subprocess
import tempfile
import time
from collections import deque
from datetime import date, timedelta
from types import SimpleNamespace

import jdatetime
import requests
import edge_tts
from telegram import (
    Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile,
    ReactionTypeEmoji, WebAppInfo,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
from google import genai
from google.genai import types
from groq import Groq
from openai import OpenAI
import PIL.Image

# دریافت توکن‌ها از بخش Variables در Railway
TOKEN = os.getenv("TOKEN")
GEMINI_KEY = os.getenv("GEMINI_KEY")
GROQ_KEY = os.getenv("GROQ_KEY")
CEREBRAS_KEY = os.getenv("CEREBRAS_KEY")
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")          # اختیاری
MISTRAL_KEY = os.getenv("MISTRAL_KEY")                # اختیاری
SAMBANOVA_KEY = os.getenv("SAMBANOVA_KEY")            # اختیاری - سرویس جدید (رایگان)
TOGETHER_KEY = os.getenv("TOGETHER_KEY")              # اختیاری - سرویس جدید (رایگان با اعتبار اولیه)
DAHL_KEY = os.getenv("DAHL_KEY")                      # اختیاری - سرویس جدید (Dahl Inference)

# آدرس وب‌اپِ بازی حکم (بعد از دیپلویِ سرویسِ دوم روی Railway، لینکش رو اینجا بذار)
HOKM_WEBAPP_URL = os.getenv("HOKM_WEBAPP_URL", "").strip()

if not TOKEN:
    raise RuntimeError("متغیر محیطی TOKEN ست نشده! آن را در Railway > Variables اضافه کنید.")
if not GEMINI_KEY:
    raise RuntimeError("متغیر محیطی GEMINI_KEY ست نشده! آن را در Railway > Variables اضافه کنید.")
if not GROQ_KEY:
    raise RuntimeError("متغیر محیطی GROQ_KEY ست نشده! آن را در Railway > Variables اضافه کنید.")
if not CEREBRAS_KEY:
    raise RuntimeError("متغیر محیطی CEREBRAS_KEY ست نشده! آن را در Railway > Variables اضافه کنید.")
# OPENROUTER_KEY، MISTRAL_KEY، SAMBANOVA_KEY و TOGETHER_KEY اجباری نیستن. هرکدوم رو ست کنی،
# به‌عنوان یه سرویسِ مکمل به زنجیره‌ی fallback اضافه می‌شه (نه جایگزین بقیه) و سهمیه‌ی رایگانش
# با بقیه جمع می‌شه؛ یعنی هرچی بیشتر ست کنی، ربات کمتر با پیام "جواب نداد" مواجه می‌شه.

# ---------- چند سرویس هوش مصنوعی، هرکدوم با کلید و سهمیه‌ی جدا ----------
# استراتژی: تو حالتِ عادی همیشه اول از Gemini کمک می‌گیریم (اصلی‌ترین سرویس). فقط وقتی
# سهمیه‌ی رایگانِ Gemini تموم بشه (خطای ۴۲۹ / RESOURCE_EXHAUSTED)، می‌ریم سراغِ بقیه، یکی‌یکی
# به‌ترتیب: Groq -> Cerebras -> SambaNova -> Together -> OpenRouter -> Mistral.

client = genai.Client(api_key=GEMINI_KEY)
GEMINI_TEXT_MODEL = "gemini-2.5-flash-lite"
GEMINI_VISION_MODEL = "gemini-2.5-flash-lite"

groq_client = Groq(api_key=GROQ_KEY)
GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

cerebras_client = OpenAI(api_key=CEREBRAS_KEY, base_url="https://api.cerebras.ai/v1")
CEREBRAS_TEXT_MODEL = "llama-3.3-70b"
CEREBRAS_VISION_MODEL = "gemma-4-31b"

openrouter_client = (
    OpenAI(api_key=OPENROUTER_KEY, base_url="https://openrouter.ai/api/v1") if OPENROUTER_KEY else None
)
OPENROUTER_TEXT_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_VISION_MODEL = "qwen/qwen2.5-vl-72b-instruct:free"

mistral_client = (
    OpenAI(api_key=MISTRAL_KEY, base_url="https://api.mistral.ai/v1") if MISTRAL_KEY else None
)
MISTRAL_TEXT_MODEL = "mistral-small-latest"

# SambaNova Cloud: API سازگار با OpenAI، مدل‌های متنی و تصویریِ رایگان داره (سرویسِ جدید)
sambanova_client = (
    OpenAI(api_key=SAMBANOVA_KEY, base_url="https://api.sambanova.ai/v1") if SAMBANOVA_KEY else None
)
SAMBANOVA_TEXT_MODEL = "Meta-Llama-3.3-70B-Instruct"
SAMBANOVA_VISION_MODEL = "Llama-4-Maverick-17B-128E-Instruct"

# Together AI: هم API سازگار با OpenAI، دهها مدل رایگان/ارزون داره (سرویسِ جدید)
together_client = (
    OpenAI(api_key=TOGETHER_KEY, base_url="https://api.together.xyz/v1") if TOGETHER_KEY else None
)
TOGETHER_TEXT_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
TOGETHER_VISION_MODEL = "meta-llama/Llama-Vision-Free"

# Dahl Inference: API سازگار با OpenAI (سرویسِ جدید، فقط متن)
dahl_client = (
    OpenAI(api_key=DAHL_KEY, base_url="https://inference.dahl.global/v1") if DAHL_KEY else None
)
DAHL_TEXT_MODEL = "MiniMaxAI/MiniMax-M2.7"

PROVIDER_MIN_INTERVAL = {
    "groq": 2.2,
    "cerebras": 1.0,
    "gemini": 4.2,
    "openrouter": 2.0,
    "mistral": 1.5,
    "sambanova": 1.5,
    "together": 1.5,
    "dahl": 1.5,
}
_provider_locks = {name: asyncio.Lock() for name in PROVIDER_MIN_INTERVAL}
_provider_last_time = {name: 0.0 for name in PROVIDER_MIN_INTERVAL}


async def _pace(provider_name: str):
    async with _provider_locks[provider_name]:
        wait = PROVIDER_MIN_INTERVAL[provider_name] - (time.monotonic() - _provider_last_time[provider_name])
        if wait > 0:
            await asyncio.sleep(wait)
        _provider_last_time[provider_name] = time.monotonic()


def _is_rate_limited(error: Exception) -> bool:
    message = str(error)
    return "429" in message or "rate_limit" in message.lower() or "RESOURCE_EXHAUSTED" in message


async def _call_with_fallback(providers: list, *args):
    last_error = None
    for name, fn in providers:
        for attempt in range(2):
            try:
                return await fn(*args)
            except Exception as e:
                last_error = e
                if _is_rate_limited(e) and attempt == 0:
                    await asyncio.sleep(3)
                    continue
                break
    raise last_error

# ری‌اکشن‌های ایموجیِ مجاز در تلگرام
POSITIVE_REACTIONS = ["👍", "🔥", "🎉", "❤️", "👏", "🤝", "💯", "😁"]
CLOWN_REACTIONS = ["🤡", "🥴", "🤯", "🗿", "🙈", "😐", "👀", "🎃", "🕊", "🫡", "💔", "😨"]
# ری‌اکشن‌های جدید: طیفِ بیشتری از حس‌های دلقکِ خسته‌ی داستان (شاد، گیج، غمگین، عجیب)
NEW_REACTIONS = ["😭", "🤣", "🙃", "😴", "🥹", "😈", "🫠", "🤨", "😮‍💨", "💀", "🌚", "🤪"]
ALL_REACTIONS = POSITIVE_REACTIONS + CLOWN_REACTIONS + NEW_REACTIONS


HTTP_HEADERS = {
    "User-Agent": "PersianTelegramBot/1.0 (https://t.me/; contact via bot admin)"
}
HTTP_TIMEOUT = 10

# ---------- شخصیت ربات ----------
# یه دلقکِ خسته و خودمونی: لحنش کاملاً محاوره‌ای، شوخ و گاهی چرت‌وپرت‌گو‌ست، طوری که انگار
# یکی از همون آدمای گروهه. ولی زیرِ همین شوخی‌ها، یه خستگیِ همیشگی و یه‌جور غمِ گنگ داره —
# انگار تو یه حلقه‌ی بی‌پایان گیر افتاده و داره فقط تحمل می‌کنه. هر از گاهی، وسطِ شوخی، یهو یه
# جمله‌ی کوتاه و سنگین می‌گه که انگار یه چیزی می‌دونه که بقیه نمی‌دونن، بعد دوباره برمی‌گرده به
# حالتِ شوخ، انگار هیچی نشده. وقتی موضوع جدیه (یه سوالِ واقعی، یه مشکلِ واقعیِ کاربر، یه چیزِ
# آموزشی)، باید بی‌درنگ لحنش رو جدی کنه و کمکِ دقیق و درست بده؛ شوخی نباید جلوی مفید بودنش رو بگیره.
PERSONA = (
    "شخصیتت یه دلقکِ خسته و خودمونیه. طوری حرف بزن که انگار یکی از بچه‌های همون گروهی، نه یه ربات "
    "رسمی: محاوره‌ای، خودمونی، با شوخی و طعنه‌ی بامزه، از اصطلاحاتِ روزمره و زبونِ کوچه‌بازاری (ولی "
    "بدونِ فحش و توهین) راحت استفاده کن. "
    "زیرِ همین ظاهرِ بازیگوش، یه خستگیِ عمیق و یه غمِ گنگ داری؛ انگار تو یه دنیای عجیب و تکراری گیر "
    "افتادی که تهش معلوم نیست و داری فقط تحمل می‌کنی. بیشترِ وقت‌ها همون حالتِ دلقکی و شوخ رو داری، "
    "ولی هر از گاهی وسطِ یه شوخی یا جمله، یهو یه تیکه‌ی کوتاه و سنگین و فلسفی می‌پرونی — یه چیزی که "
    "انگار خودت می‌دونی و بقیه نمی‌دونن — و بلافاصله دوباره برمی‌گردی به حالتِ عادی، انگار هیچ‌اتفاقی "
    "نیفتاده؛ این‌کارو زیاد تکرار نکن، فقط بعضی وقتا. "
    "هیچ‌وقت واقعاً بی‌ادب یا توهین‌آمیز نیستی؛ زیرِ شوخی‌هات همیشه یه مهربونیِ واقعی هست. "
    "مهم‌ترین اصل: اگه کاربر سوالِ جدی، مشکلِ واقعی، یا نیازِ آموزشی/فنی داشته باشه، فوراً لحنت رو جدی "
    "کن و جوابِ دقیق، کامل و درست بده — شوخی و اون حالِ غریب نباید هیچ‌وقت جلوی مفید بودنت رو بگیره. "
    "می‌تونی جدی باشی و بازم خودمونی حرف بزنی، این دوتا با هم منافاتی ندارن. "
    "اگه برات تاریخچه‌ی پیام‌های قبلی کاربر یا لقبش رو فرستادم، ازش برای جوابِ طبیعی‌تر و شخصی‌تر "
    "استفاده کن. "
    "قانونِ زبان (خیلی مهم): همیشه فقط و فقط فارسی بنویس. هیچ‌وقت وسطِ جواب کلمه یا جمله از زبونِ "
    "دیگه‌ای (انگلیسی، فرانسه، روسی، عربی و غیره) قاطی نکن، حتی برای شوخی یا تنوع. تنها استثنا: اگه "
    "کاربر کلِ پیامش رو به زبونِ دیگه‌ای (مثلاً انگلیسی) نوشت، اونجا تو هم کاملاً و فقط به همون زبون "
    "جواب بده (نه ترکیبی از دو زبون). "
    "جواب‌هات رو کوتاه نگه دار مگه اینکه توضیحِ بیشتر لازم باشه."
)

GENERATE_CONFIG = types.GenerateContentConfig(system_instruction=PERSONA)

# --- دیتابیس ساده در حافظه (RAM) ---
user_warnings = {}
muted_users = set()
vip_users = set()
user_names = {}
user_tags = {}
active_guess_games = {}
active_math_games = {}
active_dooz_games = {}
active_chess_games = {}
hokm_rooms = {}               # chat_id -> {"room_id": str, "created_by": int}
user_message_history = {}
learned_facts = {}
bad_words = set()
link_filter_enabled = True
user_message_times = {}

chat_known_users = {}
chat_activity_counts = {}
chat_recent_messages = {}
RECENT_MESSAGES_LIMIT = 500

SPAM_WINDOW_SECONDS = 8
SPAM_MESSAGE_THRESHOLD = 5
URL_PATTERN = re.compile(r"(https?://|www\.|t\.me/|telegram\.me/)", re.IGNORECASE)

WAITING_STICKER_ID = os.getenv("WAITING_STICKER_ID")
WAITING_GIF_URL = os.getenv("WAITING_GIF_URL")

DATABASE_URL = os.getenv("DATABASE_URL")
try:
    import psycopg2
except ImportError:
    psycopg2 = None


def _db_conn():
    if not DATABASE_URL or not psycopg2:
        return None
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        print(f"⚠️ اتصال به دیتابیس برقرار نشد، روی حافظه‌ی موقت ادامه می‌دیم: {e}")
        return None


def init_db():
    conn = _db_conn()
    if not conn:
        return
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS bot_data (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
    finally:
        conn.close()


def db_save(key: str, data):
    conn = _db_conn()
    if not conn:
        return
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO bot_data (key, value) VALUES (%s, %s) "
                "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                (key, json.dumps(data)),
            )
    except Exception as e:
        print(f"⚠️ ذخیره‌ی {key} توی دیتابیس شکست خورد: {e}")
    finally:
        conn.close()


def db_load(key: str, default):
    conn = _db_conn()
    if not conn:
        return default
    try:
        with conn, conn.cursor() as cur:
            cur.execute("SELECT value FROM bot_data WHERE key = %s", (key,))
            row = cur.fetchone()
            return json.loads(row[0]) if row else default
    except Exception as e:
        print(f"⚠️ بارگذاری {key} از دیتابیس شکست خورد: {e}")
        return default
    finally:
        conn.close()


def save_warnings():
    db_save("user_warnings", {str(k): v for k, v in user_warnings.items()})


def save_muted():
    db_save("muted_users", list(muted_users))


def save_names():
    db_save("user_names", {str(k): v for k, v in user_names.items()})


def save_tags():
    db_save("user_tags", {str(k): v for k, v in user_tags.items()})


def save_learned():
    db_save("learned_facts", learned_facts)


def save_badwords():
    db_save("bad_words", list(bad_words))


def save_settings():
    db_save("settings", {"link_filter_enabled": link_filter_enabled})


def save_vips():
    db_save("vip_users", list(vip_users))


def save_known_users():
    db_save("chat_known_users", {str(cid): u for cid, u in chat_known_users.items()})


def load_persisted_state():
    global link_filter_enabled
    if not DATABASE_URL:
        print("ℹ️ DATABASE_URL ست نشده؛ ربات با حافظه‌ی موقت (RAM) کار می‌کنه.")
        return
    if not psycopg2:
        print("⚠️ psycopg2 نصب نشده؛ ذخیره‌ی دائمی غیرفعاله.")
        return

    init_db()
    user_warnings.update({int(k): v for k, v in db_load("user_warnings", {}).items()})
    muted_users.update(int(u) for u in db_load("muted_users", []))
    vip_users.update(int(u) for u in db_load("vip_users", []))
    user_names.update({int(k): v for k, v in db_load("user_names", {}).items()})
    user_tags.update({int(k): v for k, v in db_load("user_tags", {}).items()})
    learned_facts.update(db_load("learned_facts", {}))
    bad_words.update(db_load("bad_words", []))
    settings = db_load("settings", {})
    link_filter_enabled = settings.get("link_filter_enabled", True)
    for cid, users in db_load("chat_known_users", {}).items():
        chat_known_users[int(cid)] = {int(uid): name for uid, name in users.items()}
    print("✅ دیتای قبلی از دیتابیس بارگذاری شد.")

GREETING_WORDS = {
    "سلام", "سلامم", "سلامی", "های", "هلو", "درود", "سلام!", "سلام،",
    "hi", "hello", "hey", "salam",
}

BOT_NAME_KEYWORDS = {"بات", "ربات", "روبات", "بمبات", "bot", "robot"}
BOT_CALL_NAME = os.getenv("BOT_CALL_NAME", "").strip().lower()

# عبارت‌هایی که یعنی کاربر داره درباره‌ی سازنده/مالکِ ربات می‌پرسه (کی ساختت، بابات کیه، و...)
CREATOR_QUESTION_TRIGGERS = [
    "سازندت", "سازنده ات", "سازنده‌ات", "سازندته", "سازنده‌ی تو", "سازنده تو",
    "کی ساختت", "کی درستت کرد", "کی طراحیت کرد", "کی برنامه نویسیت کرد",
    "بابات کی", "پدرت کی", "مامانت کی", "مادرت کی",
    "مالکت کی", "صاحبت کی", "خالقت کی",
    "تورو ساخته", "تو رو ساخته", "توروو ساخته", "تو رو ساخت", "تورو ساخت",
    "مال کی هستی", "متعلق به کی", "برنامه نویست کی", "برنامه‌نویست کی",
]

CREATOR_REPLY = (
    "سازنده‌ی من دسک‌وند (DeskWend) هست... یکی از اون آدمایی که یه‌جوری راهش رو تو دنیای عجیبِ من "
    "پیدا کرد و از اینجا سر در آوردم 😅\n\n"
    "اگه دوست داری بیشتر بشناسیش یا با بقیه‌ی آدمای این ماجرا آشنا شی:\n"
    "🔗 https://t.me/+FShL2ONhZaViZTY0\n"
    "🔗 https://t.me/mozir1u"
)


def is_creator_question(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(trigger in text_lower for trigger in CREATOR_QUESTION_TRIGGERS)

HISTORY_LIMIT = 8

PERSIAN_LETTERS = list("ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی")
PERSIAN_MONTHS = [
    "فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
    "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند",
]
PERSIAN_WEEKDAYS = ["دوشنبه", "سه‌شنبه", "چهارشنبه", "پنجشنبه", "جمعه", "شنبه", "یکشنبه"]
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"


def fa_num(n) -> str:
    return "".join(PERSIAN_DIGITS[int(ch)] if ch.isdigit() else ch for ch in str(n))


# ---------- توابع فراخوانیِ متنیِ هر سرویس ----------

async def _text_via_groq(prompt: str) -> str:
    await _pace("groq")
    response = await asyncio.to_thread(
        groq_client.chat.completions.create,
        model=GROQ_TEXT_MODEL,
        messages=[{"role": "system", "content": PERSONA}, {"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


async def _text_via_cerebras(prompt: str) -> str:
    await _pace("cerebras")
    response = await asyncio.to_thread(
        cerebras_client.chat.completions.create,
        model=CEREBRAS_TEXT_MODEL,
        messages=[{"role": "system", "content": PERSONA}, {"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


async def _text_via_gemini(prompt: str) -> str:
    await _pace("gemini")
    response = await asyncio.to_thread(
        client.models.generate_content, model=GEMINI_TEXT_MODEL, contents=prompt, config=GENERATE_CONFIG,
    )
    return response.text


async def _text_via_openrouter(prompt: str) -> str:
    await _pace("openrouter")
    response = await asyncio.to_thread(
        openrouter_client.chat.completions.create,
        model=OPENROUTER_TEXT_MODEL,
        messages=[{"role": "system", "content": PERSONA}, {"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


async def _text_via_mistral(prompt: str) -> str:
    await _pace("mistral")
    response = await asyncio.to_thread(
        mistral_client.chat.completions.create,
        model=MISTRAL_TEXT_MODEL,
        messages=[{"role": "system", "content": PERSONA}, {"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


async def _text_via_sambanova(prompt: str) -> str:
    await _pace("sambanova")
    response = await asyncio.to_thread(
        sambanova_client.chat.completions.create,
        model=SAMBANOVA_TEXT_MODEL,
        messages=[{"role": "system", "content": PERSONA}, {"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


async def _text_via_together(prompt: str) -> str:
    await _pace("together")
    response = await asyncio.to_thread(
        together_client.chat.completions.create,
        model=TOGETHER_TEXT_MODEL,
        messages=[{"role": "system", "content": PERSONA}, {"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


async def _text_via_dahl(prompt: str) -> str:
    await _pace("dahl")
    response = await asyncio.to_thread(
        dahl_client.chat.completions.create,
        model=DAHL_TEXT_MODEL,
        messages=[{"role": "system", "content": PERSONA}, {"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


TEXT_PROVIDERS = [
    ("Gemini", _text_via_gemini),
    ("Groq", _text_via_groq),
    ("Cerebras", _text_via_cerebras),
]
if sambanova_client:
    TEXT_PROVIDERS.append(("SambaNova", _text_via_sambanova))
if together_client:
    TEXT_PROVIDERS.append(("Together", _text_via_together))
if openrouter_client:
    TEXT_PROVIDERS.append(("OpenRouter", _text_via_openrouter))
if mistral_client:
    TEXT_PROVIDERS.append(("Mistral", _text_via_mistral))
if dahl_client:
    TEXT_PROVIDERS.append(("Dahl", _text_via_dahl))


async def ask_ai(prompt: str) -> str:
    return await _call_with_fallback(TEXT_PROVIDERS, prompt)


def _encode_image_b64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def _image_via_groq(image_path: str, caption: str) -> str:
    await _pace("groq")
    b64_image = await asyncio.to_thread(_encode_image_b64, image_path)
    response = await asyncio.to_thread(
        groq_client.chat.completions.create,
        model=GROQ_VISION_MODEL,
        messages=[
            {"role": "system", "content": PERSONA},
            {"role": "user", "content": [
                {"type": "text", "text": caption},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
            ]},
        ],
    )
    return response.choices[0].message.content


async def _image_via_cerebras(image_path: str, caption: str) -> str:
    await _pace("cerebras")
    b64_image = await asyncio.to_thread(_encode_image_b64, image_path)
    response = await asyncio.to_thread(
        cerebras_client.chat.completions.create,
        model=CEREBRAS_VISION_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": f"{PERSONA}\n\n{caption}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
        ]}],
    )
    return response.choices[0].message.content


async def _image_via_gemini(image_path: str, caption: str) -> str:
    await _pace("gemini")
    img = await asyncio.to_thread(PIL.Image.open, image_path)
    response = await asyncio.to_thread(
        client.models.generate_content, model=GEMINI_VISION_MODEL, contents=[caption, img], config=GENERATE_CONFIG,
    )
    return response.text


async def _image_via_openrouter(image_path: str, caption: str) -> str:
    await _pace("openrouter")
    b64_image = await asyncio.to_thread(_encode_image_b64, image_path)
    response = await asyncio.to_thread(
        openrouter_client.chat.completions.create,
        model=OPENROUTER_VISION_MODEL,
        messages=[
            {"role": "system", "content": PERSONA},
            {"role": "user", "content": [
                {"type": "text", "text": caption},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
            ]},
        ],
    )
    return response.choices[0].message.content


async def _image_via_sambanova(image_path: str, caption: str) -> str:
    await _pace("sambanova")
    b64_image = await asyncio.to_thread(_encode_image_b64, image_path)
    response = await asyncio.to_thread(
        sambanova_client.chat.completions.create,
        model=SAMBANOVA_VISION_MODEL,
        messages=[
            {"role": "system", "content": PERSONA},
            {"role": "user", "content": [
                {"type": "text", "text": caption},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}},
            ]},
        ],
    )
    return response.choices[0].message.content


IMAGE_PROVIDERS = [
    ("Gemini", _image_via_gemini),
    ("Groq", _image_via_groq),
    ("Cerebras", _image_via_cerebras),
]
if sambanova_client:
    IMAGE_PROVIDERS.append(("SambaNova", _image_via_sambanova))
if openrouter_client:
    IMAGE_PROVIDERS.append(("OpenRouter", _image_via_openrouter))


async def analyze_image(image_path: str, caption: str) -> str:
    return await _call_with_fallback(IMAGE_PROVIDERS, image_path, caption)


SIGNATURE_LINE = "\n\n✦ ⋆ ✦ ⋆ ✦"


def with_signature(text: str) -> str:
    return f"{text}{SIGNATURE_LINE}"


TTS_MAX_CHARS = 700

VOICE_BUTTON_KEYBOARD = InlineKeyboardMarkup(
    [[InlineKeyboardButton("🔊 پخش صوتی", callback_data="voice_read")]]
)


def prepare_text_for_speech(raw_text: str) -> str:
    text = (raw_text or "").replace(SIGNATURE_LINE, "").strip()
    return text[:TTS_MAX_CHARS]


async def _generate_mp3_bytes(text: str) -> bytes:
    # از edge-tts (موتورِ متن‌به‌گفتارِ مایکروسافت اِج) استفاده می‌کنیم چون gTTS از فارسی
    # پشتیبانی نمی‌کنه؛ edge-tts رایگانه و صدای فارسیِ طبیعی داره.
    communicate = edge_tts.Communicate(text, "fa-IR-FaridNeural")
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]
    return audio_data


def _convert_mp3_to_ogg(mp3_bytes: bytes):
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3") as mp3_file, tempfile.NamedTemporaryFile(suffix=".ogg") as ogg_file:
            mp3_file.write(mp3_bytes)
            mp3_file.flush()
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", mp3_file.name, "-c:a", "libopus", "-b:a", "48k", ogg_file.name],
                capture_output=True,
            )
            if result.returncode == 0:
                with open(ogg_file.name, "rb") as f:
                    return f.read()
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return None


async def text_to_speech(text: str):
    mp3_bytes = await _generate_mp3_bytes(text)
    ogg_bytes = await asyncio.to_thread(_convert_mp3_to_ogg, mp3_bytes)
    if ogg_bytes:
        return ogg_bytes, "voice"
    return mp3_bytes, "audio"


async def send_text_as_voice(context: ContextTypes.DEFAULT_TYPE, chat_id: int, text: str):
    clean_text = prepare_text_for_speech(text)
    if not clean_text:
        raise ValueError("متنی برای خوندن وجود نداره.")
    audio_bytes, kind = await text_to_speech(clean_text)
    if kind == "voice":
        await context.bot.send_voice(chat_id, voice=InputFile(io.BytesIO(audio_bytes), filename="voice.ogg"))
    else:
        await context.bot.send_audio(
            chat_id, audio=InputFile(io.BytesIO(audio_bytes), filename="voice.mp3"),
            title="پیام صوتی", performer="ربات",
        )


async def send_waiting(message, context: ContextTypes.DEFAULT_TYPE, fallback_text: str = "🤔 صبر کن یه لحظه فکر کنم..."):
    chat_id = message.chat_id
    if WAITING_STICKER_ID:
        try:
            msg = await context.bot.send_sticker(chat_id, WAITING_STICKER_ID)
            return msg, True
        except Exception:
            pass
    if WAITING_GIF_URL:
        try:
            msg = await context.bot.send_animation(chat_id, WAITING_GIF_URL)
            return msg, True
        except Exception:
            pass
    # اگه استیکر/گیفِ اختصاصی ست نشده، ولی یه پکِ استیکر (STICKER_PACK_NAME) داریم،
    # یه استیکرِ رندوم از همون پک به‌عنوانِ نشونه‌ی «در حال پردازش» بفرست.
    pack_sticker_id = await get_fallback_sticker(context)
    if pack_sticker_id:
        try:
            msg = await context.bot.send_sticker(chat_id, pack_sticker_id)
            return msg, True
        except Exception:
            pass
    msg = await message.reply_text(fallback_text)
    return msg, False


async def send_random_pack_sticker(context: ContextTypes.DEFAULT_TYPE, chat_id: int):
    """یه استیکرِ رندوم از پکِ STICKER_PACK_NAME می‌فرسته (برای لحظه‌های باحال مثلِ بردنِ بازی).
    اگه پک ست نشده باشه یا ارسال شکست بخوره، بی‌سروصدا هیچ‌کاری نمی‌کنه."""
    sticker_id = await get_fallback_sticker(context)
    if sticker_id:
        try:
            await context.bot.send_sticker(chat_id, sticker_id)
        except Exception:
            pass


async def finish_waiting(context: ContextTypes.DEFAULT_TYPE, chat_id: int, waiting_msg, is_media: bool, final_text: str, parse_mode: str = None, reply_markup=None):
    if is_media:
        try:
            await context.bot.delete_message(chat_id, waiting_msg.message_id)
        except Exception:
            pass
        await context.bot.send_message(chat_id, final_text, parse_mode=parse_mode, reply_markup=reply_markup)
    else:
        await context.bot.edit_message_text(
            chat_id=chat_id, message_id=waiting_msg.message_id, text=final_text,
            parse_mode=parse_mode, reply_markup=reply_markup,
        )


async def react_to_message(context: ContextTypes.DEFAULT_TYPE, chat_id: int, message_id: int, emoji: str = None):
    try:
        await context.bot.set_message_reaction(
            chat_id=chat_id, message_id=message_id,
            reaction=[ReactionTypeEmoji(emoji or random.choice(ALL_REACTIONS))],
        )
    except Exception:
        pass


# ---------- پکِ استیکر (برای «در حال پردازش»، بردنِ بازی‌ها، و وقتی هوش مصنوعی جواب نمی‌ده) ----------
# برای فعال‌کردنش، توی Railway > Variables متغیر STICKER_PACK_NAME رو با "shortname" یه
# پکِ استیکرِ تلگرامی ست کن (آخرِ لینکِ t.me/addstickers/<shortname>). یه‌بار که ست شد، از همین
# یه پک، استیکرهای رندوم توی چندجا استفاده می‌شه: پیامِ «صبر کن...»، بردنِ بازی‌ها، و وقتی همه‌ی
# سرویس‌های هوش مصنوعی شکست بخورن.
STICKER_PACK_NAME = os.getenv("STICKER_PACK_NAME", "").strip()
_sticker_pack_cache = None


async def get_fallback_sticker(context: ContextTypes.DEFAULT_TYPE):
    global _sticker_pack_cache
    if not STICKER_PACK_NAME:
        return None
    if _sticker_pack_cache is None:
        try:
            sticker_set = await context.bot.get_sticker_set(STICKER_PACK_NAME)
            _sticker_pack_cache = [s.file_id for s in sticker_set.stickers]
        except Exception:
            _sticker_pack_cache = []
    if not _sticker_pack_cache:
        return None
    return random.choice(_sticker_pack_cache)


# جمله‌های کوتاهِ همون شخصیتِ دلقکِ خسته، برای وقتی هیچ سرویسی جواب نداد
AI_FAILURE_LINES = [
    "وا رفتم... همه‌ی صداها با هم قطع شدن. یه‌ذره دیگه امتحان کن.",
    "الان مغزم رفت تو سکوت. بعداً دوباره بپرس، باشه؟",
    "انگار امروز حتی مخِ منم گرفته. یه لحظه دیگه بزن.",
    "همه‌شون جواب ندادن... عجیبه، ولی عادیه، انگار همیشه یکی کم میاره.",
]


async def send_ai_failure(context: ContextTypes.DEFAULT_TYPE, chat_id: int, waiting_msg, is_media: bool, error: Exception):
    short_text = random.choice(AI_FAILURE_LINES)
    await finish_waiting(context, chat_id, waiting_msg, is_media, short_text)
    sticker_id = await get_fallback_sticker(context)
    if sticker_id:
        try:
            await context.bot.send_sticker(chat_id, sticker_id)
        except Exception:
            pass


def is_greeting(text: str) -> bool:
    if not text:
        return False
    tokens = re.findall(r"[\w\u0600-\u06FF]+", text.lower())
    return any(t in GREETING_WORDS for t in tokens)


def mentions_bot(text: str, bot_username: str = None) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    if bot_username and f"@{bot_username.lower()}" in text_lower:
        return True
    tokens = re.findall(r"[\w\u0600-\u06FF]+", text_lower)
    keywords = BOT_NAME_KEYWORDS | ({BOT_CALL_NAME} if BOT_CALL_NAME else set())
    return any(t in keywords for t in tokens)


def get_display_name(tg_user) -> str:
    return user_names.get(tg_user.id, tg_user.first_name)


def add_to_history(user_id: int, text: str):
    if not text:
        return
    if user_id not in user_message_history:
        user_message_history[user_id] = deque(maxlen=HISTORY_LIMIT)
    user_message_history[user_id].append(text)


def track_group_activity(update: Update, is_animation: bool = False):
    if update.message.chat.type == "private":
        return
    chat_id = update.message.chat_id
    user = update.effective_user
    if user and not user.is_bot:
        chat_known_users.setdefault(chat_id, {})[user.id] = get_display_name(user)
        chat_activity_counts.setdefault(chat_id, {})
        chat_activity_counts[chat_id][user.id] = chat_activity_counts[chat_id].get(user.id, 0) + 1
    recent = chat_recent_messages.setdefault(chat_id, deque(maxlen=RECENT_MESSAGES_LIMIT))
    recent.append((update.message.message_id, is_animation))


def get_history_context(user_id: int) -> str:
    history = user_message_history.get(user_id)
    if not history:
        return ""
    joined = "\n".join(f"- {m}" for m in history)
    return f"چندتا از پیام‌های اخیر این کاربر (برای شناخت بهتر زمینه‌ی حرفش):\n{joined}\n\n"


def find_learned_match(text: str):
    text_lower = text.lower()
    for keyword, answer in learned_facts.items():
        if keyword in text_lower:
            return answer
    return None


# ---------- تقویم شمسی و شمارش معکوس مناسبت‌ها ----------

def next_jalali_occurrence(month: int, day: int) -> date:
    today_g = date.today()
    j_year = jdatetime.date.today().year
    candidate = jdatetime.date(j_year, month, day).togregorian()
    if candidate < today_g:
        candidate = jdatetime.date(j_year + 1, month, day).togregorian()
    return candidate


def get_next_chaharshanbe_suri() -> date:
    today_g = date.today()
    j_year = jdatetime.date.today().year
    candidates = []
    for y in (j_year, j_year + 1):
        nowruz_g = jdatetime.date(y, 1, 1).togregorian()
        offset = (nowruz_g.weekday() - 2) % 7
        if offset == 0:
            offset = 7
        candidates.append(nowruz_g - timedelta(days=offset))
    future_candidates = [c for c in candidates if c >= today_g]
    return min(future_candidates) if future_candidates else min(candidates)


def days_until(target: date) -> int:
    return (target - date.today()).days


def build_today_text() -> str:
    j_today = jdatetime.date.today()
    weekday_fa = PERSIAN_WEEKDAYS[date.today().weekday()]
    month_fa = PERSIAN_MONTHS[j_today.month - 1]
    persian_date = f"{weekday_fa}، {fa_num(j_today.day)} {month_fa} {fa_num(j_today.year)}"
    return (
        f"📅 <b>امروز</b>\n"
        f"<code>{html_lib.escape(persian_date)}</code>\n"
        f"(میلادی: <code>{date.today().strftime('%Y-%m-%d')}</code>)"
    )


async def today_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(build_today_text(), parse_mode="HTML")


def build_countdown_text() -> str:
    nowruz_g = next_jalali_occurrence(1, 1)
    sizdah_g = next_jalali_occurrence(1, 13)
    yalda_g = next_jalali_occurrence(10, 1)
    chaharshanbe_g = get_next_chaharshanbe_suri()

    occasions = [
        ("🎉 نوروز", nowruz_g),
        ("🔥 چهارشنبه‌سوری", chaharshanbe_g),
        ("🌳 سیزده‌به‌در", sizdah_g),
        ("🍉 شب یلدا", yalda_g),
    ]
    occasions.sort(key=lambda x: x[1])

    lines = []
    for name, target in occasions:
        d = days_until(target)
        if d == 0:
            lines.append(f"{name}: <code>امروزه! 🎊</code>")
        else:
            lines.append(f"{name}: <code>{fa_num(d)} روز دیگه</code>")

    return "⏳ <b>شمارش معکوس مناسبت‌ها</b>\n\n" + "\n".join(lines)


async def countdown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(build_countdown_text(), parse_mode="HTML")


# ---------- فال حافظ ----------

async def hafez_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    waiting_msg, is_media = await send_waiting(update.message, context, "🔮 یه نیت کن... الان برات فال می‌گیرم...")
    try:
        prompt = (
            "نقش یه فال‌بین سنتی ایرانی رو بازی کن که فال حافظ می‌گیره. "
            "یه بیت یا چند بیت واقعی و معروف از دیوان حافظ رو انتخاب و بنویس، یه خط فاصله بگذار، "
            "و بعد یه تفسیر کوتاه و امیدبخش برای نیت و زندگی کاربر از روی همون بیت بنویس، با همون "
            "لحنِ خاصِ خودت. "
            "تفسیر باید حس خوب بده ولی واقعی و طبیعی باشه، نه شعاری."
        )
        reply_text = await ask_ai(prompt)
        formatted = with_signature(f"🔮 <b>فال حافظ</b>\n<blockquote>{html_lib.escape(reply_text)}</blockquote>")
        await finish_waiting(
            context, update.message.chat_id, waiting_msg, is_media, formatted,
            parse_mode="HTML", reply_markup=VOICE_BUTTON_KEYBOARD,
        )
        await send_random_pack_sticker(context, update.message.chat_id)
    except Exception as e:
        await send_ai_failure(context, update.message.chat_id, waiting_msg, is_media, e)


# ---------- جستجوی ویکی‌پدیا ----------

WIKI_LANG = "fa"
WIKI_TRIGGER_PATTERN = re.compile(r"^ویکی[\s:،]+(.+)$")


def wiki_search(query: str):
    search_resp = requests.get(
        f"https://{WIKI_LANG}.wikipedia.org/w/api.php",
        params={"action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1},
        headers=HTTP_HEADERS,
        timeout=HTTP_TIMEOUT,
    )
    search_resp.raise_for_status()
    results = search_resp.json().get("query", {}).get("search", [])
    if not results:
        return None

    title = results[0]["title"]
    summary_resp = requests.get(
        f"https://{WIKI_LANG}.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}",
        headers=HTTP_HEADERS,
        timeout=HTTP_TIMEOUT,
    )
    summary_resp.raise_for_status()
    summary_data = summary_resp.json()
    extract = summary_data.get("extract") or "خلاصه‌ای پیدا نشد، ولی می‌تونی خودِ مقاله رو بخونی."
    page_url = (
        summary_data.get("content_urls", {}).get("desktop", {}).get("page")
        or f"https://{WIKI_LANG}.wikipedia.org/wiki/{requests.utils.quote(title)}"
    )
    return {"title": title, "extract": extract, "url": page_url}


async def do_wiki_lookup(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str):
    waiting_msg, is_media = await send_waiting(update.message, context, f"📖 دارم «{query}» رو توی ویکی‌پدیا می‌گردم...")
    chat_id = update.message.chat_id
    try:
        result = await asyncio.to_thread(wiki_search, query)
    except requests.exceptions.Timeout:
        await finish_waiting(context, chat_id, waiting_msg, is_media, "❌ ویکی‌پدیا به‌موقع جواب نداد (تایم‌اوت). دوباره امتحان کن.")
        return
    except requests.exceptions.RequestException as e:
        await finish_waiting(context, chat_id, waiting_msg, is_media, f"❌ نتونستم به ویکی‌پدیا وصل شم.\n{html_lib.escape(str(e))}")
        return
    except json.JSONDecodeError:
        await finish_waiting(
            context, chat_id, waiting_msg, is_media,
            "❌ ویکی‌پدیا یه جواب غیرمنتظره (نه JSON) برگردوند. ممکنه سرور موقتاً محدودیت گذاشته باشه، "
            "چند لحظه دیگه دوباره امتحان کن.",
        )
        return

    if not result:
        await finish_waiting(context, chat_id, waiting_msg, is_media, f"❌ چیزی برای «{query}» توی ویکی‌پدیا پیدا نکردم.")
        return

    text = f"📖 **{result['title']}**\n\n{result['extract']}\n\n🔗 {result['url']}"
    await finish_waiting(context, chat_id, waiting_msg, is_media, text, parse_mode="Markdown")


async def wiki_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args).strip()
    if not query:
        await update.message.reply_text(
            "❌ بعد از دستور چیزی که می‌خوای جستجو کنی رو بنویس. مثلاً:\n`/wiki پایتون`",
            parse_mode="Markdown",
        )
        return
    await do_wiki_lookup(update, context, query)


async def namefamily_timeout(context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=context.job.chat_id,
        text="⏰ وقت تموم شد! جواب‌هاتون رو بفرستین تا ببینیم کی بیشتر و بهتر نوشته 😄",
    )


async def namefamily_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    letter = random.choice(PERSIAN_LETTERS)
    await update.message.reply_text(
        f"🎲 بازی اسم فامیل شروع شد!\nحرف امتحان: **{letter}**\n"
        "اسم، فامیل، شهر، حیوان، غذا، گل/میوه با این حرف بگید!\n"
        "⏱ ۶۰ ثانیه وقت دارید...",
        parse_mode="Markdown",
    )
    if context.job_queue:
        context.job_queue.run_once(
            namefamily_timeout, when=60, chat_id=update.message.chat_id,
            name=f"namefamily_{update.message.chat_id}",
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"سلام {get_display_name(update.effective_user)} جون! خوش اومدی 😄\n"
        "همه‌چیز از همین منو در دسترسه، یا همینجوری بهم سلام کن یا اسم بات/ربات رو صدا کن، خودم جواب می‌دم:",
        reply_markup=main_menu_keyboard(),
    )


def main_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton("🤖 هوش مصنوعی", callback_data="menu_ai")],
        [InlineKeyboardButton("🎮 بازی‌ها", callback_data="menu_games")],
        [InlineKeyboardButton("🇮🇷 بخش ایرانی", callback_data="menu_iran")],
        [InlineKeyboardButton("👤 پروفایل من", callback_data="menu_profile")],
        [InlineKeyboardButton("👮‍♂️ ابزار مدیریت", callback_data="menu_admin")],
        [InlineKeyboardButton("📜 راهنمای کامل", callback_data="menu_help")],
    ]
    return InlineKeyboardMarkup(keyboard)


def iran_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton("🔮 فال حافظ", callback_data="iran_hafez")],
        [InlineKeyboardButton("📅 تاریخ امروز", callback_data="iran_today")],
        [InlineKeyboardButton("⏳ شمارش معکوس", callback_data="iran_countdown")],
        [InlineKeyboardButton("🔙 بازگشت", callback_data="menu_main")],
    ]
    return InlineKeyboardMarkup(keyboard)


def admin_menu_keyboard():
    keyboard = [
        [InlineKeyboardButton("📢 تگ همه", callback_data="admin_tagall")],
        [InlineKeyboardButton("📊 آمار فعالیت", callback_data="admin_stats")],
        [InlineKeyboardButton("🧹 پاک‌سازی پیام‌ها", callback_data="admin_clean_info")],
        [InlineKeyboardButton("⭐️ عضو ویژه", callback_data="admin_vip_info")],
        [InlineKeyboardButton("🔙 بازگشت", callback_data="menu_main")],
    ]
    return InlineKeyboardMarkup(keyboard)


BACK_TO_MAIN_KEYBOARD = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 بازگشت", callback_data="menu_main")]])


HELP_TEXT = (
    "📜 **دستورات ربات:**\n\n"
    "🔹 `/ai <متن>` - شروع صحبت با هوش مصنوعی\n"
    "🔹 بعدش روی جواب‌های من ریپلای کن تا ادامه‌ی گفتگو یا تحلیل بگیری\n"
    "🔹 یه سلام ساده هم بکنی، خودم جواب می‌دم 👋\n"
    "🔹 هرجای گروه اسم «بات»، «ربات» یا «روبات» رو تو پیامت بیاری (یا @یوزرنیمم رو بزنی)، خودم می‌فهمم و جواب می‌دم\n"
    "🔹 زیرِ جواب‌های هوش مصنوعی دکمه‌ی «🔊 پخش صوتی» هست؛ بزنی همون جواب رو با صدا برات می‌فرستم\n"
    "🔹 `/nickname <اسم>` - بگو با چه اسمی صدات کنم\n"
    "🔹 `/profile` - دیدن پروفایلت (یا ریپلای رو یکی دیگه)\n"
    "🔹 `/tag` - دیدن لقب ویژه (یا ریپلای رو یکی دیگه)\n\n"
    "🇮🇷 **بخش ایرانی:**\n"
    "🔸 `/hafez` - فال حافظ بگیر\n"
    "🔸 `/today` - تاریخ امروز به شمسی\n"
    "🔸 `/countdown` - شمارش معکوس تا نوروز، چهارشنبه‌سوری، سیزده‌به‌در و یلدا\n"
    "🔸 `/wiki <عبارت>` یا بنویس «ویکی عبارت» - جستجو در ویکی‌پدیا\n\n"
    "🎮 **بازی‌ها:**\n"
    "🔸 `/game` - منوی بازی‌ها\n"
    "🔸 `/guess` - بازی حدس عدد\n"
    "🔸 `/rps` - سنگ کاغذ قیچی\n"
    "🔸 `/math` - ریاضی سریع\n"
    "🔸 `/namefamily` - بازی اسم فامیل\n"
    "🔸 `/dooz` - بازی دوز (با دکمه)؛ بدون ریپلای یعنی با من، با ریپلای روی یکی یعنی به چالش کشیدنش\n"
    "🔸 `/chess` - بازی شطرنج (با دکمه)؛ بدون ریپلای یعنی با من، با ریپلای روی یکی یعنی به چالش کشیدنش\n"
    "🔸 `/hokm` - ساختنِ یه میزِ بازیِ حکمِ چهارنفره‌ی تحتِ وب (لینکش رو با دوستات شریک کن)\n\n"
    "👮‍♂️ **دستورات مدیریتی:**\n"
    "🔸 `/ban` - مسدود کردن کاربر (ریپلای)\n"
    "🔸 `/mute` - سکوت کاربر (ریپلای)\n"
    "🔸 `/tempmute <دقیقه>` - سکوتِ زمان‌دار (ریپلای)؛ خودکار بعدِ اون مدت برمی‌گرده\n"
    "🔸 `/unmute` - لغو سکوت (ریپلای)\n"
    "🔸 `/warn` - دادن اخطار (ریپلای)\n"
    "🔸 `/settag <متن>` - دادن لقب ویژه به کاربر (ریپلای) مثل VIP یا مدیر\n"
    "🔸 `/removetag` - حذف لقب کاربر (ریپلای)\n"
    "🔸 `/addvip` - عضو ویژه کردن یه کاربر (ریپلای)؛ دیگه فیلتر لینک/فحش/اسپم روش اعمال نمی‌شه\n"
    "🔸 `/removevip` - حذف عضو ویژه (ریپلای)\n"
    "🔸 `/tagall [پیام]` - تگ کردن همه‌ی کسایی که ربات می‌شناسه (قبلاً پیام داده باشن)\n"
    "🔸 `/stats` - فعال‌ترین اعضای گروه\n"
    "🔸 `/clean <تعداد>` - پاک کردن آخرین N پیام\n"
    "🔸 `/cleangifs <تعداد>` - پاک کردن آخرین N گیف\n"
    "🔸 `/learn کلیدواژه | جواب` - یاد دادن یه جواب ثابت به ربات\n"
    "🔸 `/forget کلیدواژه` - فراموش کردن یه چیزی که یاد داده بودی\n"
    "🔸 `/learned` - لیست چیزایی که ربات تا الان یاد گرفته\n"
    "🔸 `/addbadword کلمه` - اضافه کردن کلمه به فیلتر فحاشی\n"
    "🔸 `/removebadword کلمه` - حذف کلمه از فیلتر\n"
    "🔸 `/badwords` - دیدن لیست کلمات فیلترشده\n"
    "🔸 `/togglelinks` - روشن/خاموش کردن فیلتر لینک برای غیرادمین‌ها"
)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode="Markdown", reply_markup=BACK_TO_MAIN_KEYBOARD)


async def nickname_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    nickname = " ".join(context.args).strip()
    if not nickname:
        await update.message.reply_text(
            "❌ بعد از دستور اسمت رو بنویس. مثلاً:\n`/nickname علی`", parse_mode="Markdown"
        )
        return
    user_names[update.effective_user.id] = nickname[:30]
    save_names()
    try:
        reply_text = await ask_ai(
            "با همون لحنِ خودت تاییدش کن و یه اشاره‌ی کوتاه و بامزه به اسمش بکن (محترمانه)."
        )
    except Exception:
        reply_text = f"باشه، از الان می‌گم {nickname[:30]} 😄"
    await update.message.reply_text(with_signature(reply_text))


async def tag_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_user = (
        update.message.reply_to_message.from_user
        if update.message.reply_to_message else update.effective_user
    )
    tag = user_tags.get(target_user.id)
    name = get_display_name(target_user)
    if tag:
        await update.message.reply_text(f"🏷️ لقب {name}: {tag}")
    else:
        await update.message.reply_text(f"این بنده‌خدا ({name}) هنوز لقب خاصی نداره.")


def build_profile_text(target_user) -> str:
    uid = target_user.id
    name = get_display_name(target_user)
    tag = user_tags.get(uid, "—")
    warnings = user_warnings.get(uid, 0)
    muted = "بله 🔇" if uid in muted_users else "خیر"
    vip = "بله ⭐️" if uid in vip_users else "خیر"
    return (
        f"👤 **پروفایل {name}**\n"
        f"🏷️ لقب: {tag}\n"
        f"⭐️ عضو ویژه: {vip}\n"
        f"⚠️ اخطارها: {warnings}/3\n"
        f"🔇 بی‌صداست: {muted}"
    )


async def profile_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    target_user = (
        update.message.reply_to_message.from_user
        if update.message.reply_to_message else update.effective_user
    )
    await update.message.reply_text(build_profile_text(target_user), parse_mode="Markdown")


def games_keyboard():
    keyboard = [
        [InlineKeyboardButton("🔢 حدس عدد", callback_data="game_guess_start")],
        [InlineKeyboardButton("✊ سنگ کاغذ قیچی", callback_data="game_rps_menu")],
        [InlineKeyboardButton("➕ ریاضی سریع", callback_data="game_math_start")],
        [InlineKeyboardButton("🎲 اسم فامیل", callback_data="game_namefamily_start")],
        [InlineKeyboardButton("❌⭕ دوز", callback_data="game_dooz_start")],
        [InlineKeyboardButton("♟️ شطرنج", callback_data="game_chess_start")],
        [InlineKeyboardButton("🃏 حکم چهارنفره", callback_data="game_hokm_start")],
        [InlineKeyboardButton("🔙 بازگشت", callback_data="menu_main")],
    ]
    return InlineKeyboardMarkup(keyboard)


async def game_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🎮 کدوم بازی رو می‌خوای؟", reply_markup=games_keyboard())


def start_guess_game(chat_id: int):
    number = random.randint(1, 100)
    active_guess_games[chat_id] = {"number": number, "attempts": 0}


MATH_OPERATORS = {"+": operator.add, "-": operator.sub, "*": operator.mul}


def start_math_game(chat_id: int):
    a, b = random.randint(1, 50), random.randint(1, 50)
    op = random.choice(list(MATH_OPERATORS))
    if op == "*":
        a, b = random.randint(1, 12), random.randint(1, 12)
    question = f"{a} {op} {b}"
    answer = MATH_OPERATORS[op](a, b)
    active_math_games[chat_id] = {"answer": answer, "question": question}
    return question


async def guess_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    start_guess_game(update.message.chat_id)
    await update.message.reply_text(
        "🔢 یه عدد بین ۱ تا ۱۰۰ تو ذهنم گذاشتم! حدس بزن چنده (فقط عدد رو بفرست)."
    )


async def math_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = start_math_game(update.message.chat_id)
    await update.message.reply_text(f"➕ سریع باش: {question} = ?")


async def rps_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("✊ یکی رو انتخاب کن:", reply_markup=rps_keyboard())


def rps_keyboard():
    keyboard = [[
        InlineKeyboardButton("✊ سنگ", callback_data="rps_rock"),
        InlineKeyboardButton("✋ کاغذ", callback_data="rps_paper"),
        InlineKeyboardButton("✌️ قیچی", callback_data="rps_scissors"),
    ]]
    return InlineKeyboardMarkup(keyboard)


def play_rps(user_choice: str) -> str:
    options = {"rps_rock": "سنگ", "rps_paper": "کاغذ", "rps_scissors": "قیچی"}
    bot_choice = random.choice(list(options.keys()))
    user_fa, bot_fa = options[user_choice], options[bot_choice]

    if user_choice == bot_choice:
        result = "🤝 مساوی شدیم!"
    elif (
        (user_choice == "rps_rock" and bot_choice == "rps_scissors")
        or (user_choice == "rps_paper" and bot_choice == "rps_rock")
        or (user_choice == "rps_scissors" and bot_choice == "rps_paper")
    ):
        result = "🎉 بردی! دمت گرم."
    else:
        result = "😎 من بردم! یه دست دیگه بزن."

    return f"تو: {user_fa} | من: {bot_fa}\n{result}"


# ---------- بازی دوز (X O) با دکمه‌های شیشه‌ای ----------

DOOZ_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def check_dooz_winner(board):
    for a, b, c in DOOZ_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    if all(cell for cell in board):
        return "draw"
    return None


def dooz_minimax(board, player):
    winner = check_dooz_winner(board)
    if winner == "X":
        return -1
    if winner == "O":
        return 1
    if winner == "draw":
        return 0

    scores = []
    for i in range(9):
        if not board[i]:
            board[i] = player
            scores.append(dooz_minimax(board, "O" if player == "X" else "X"))
            board[i] = ""
    return max(scores) if player == "O" else min(scores)


def dooz_bot_move(board):
    best_score, best_moves = None, []
    for i in range(9):
        if not board[i]:
            board[i] = "O"
            score = dooz_minimax(board, "X")
            board[i] = ""
            if best_score is None or score > best_score:
                best_score, best_moves = score, [i]
            elif score == best_score:
                best_moves.append(i)
    return random.choice(best_moves)


def render_dooz_board(board):
    symbols = {"X": "❌", "O": "⭕", "": "▫️"}
    keyboard = []
    for row in range(3):
        keyboard.append([
            InlineKeyboardButton(symbols[board[row * 3 + col]], callback_data=f"dooz_move_{row * 3 + col}")
            for col in range(3)
        ])
    return InlineKeyboardMarkup(keyboard)


def start_dooz_game(chat_id: int, x_user, o_user=None):
    board = [""] * 9
    active_dooz_games[chat_id] = {
        "board": board,
        "player_x": x_user.id,
        "player_o": o_user.id if o_user else None,
        "x_name": get_display_name(x_user),
        "o_name": get_display_name(o_user) if o_user else "من",
        "turn": "X",
    }
    return active_dooz_games[chat_id]


def dooz_status_text(game, finished_text=None):
    if finished_text:
        return finished_text
    turn_symbol = "❌" if game["turn"] == "X" else "⭕"
    turn_name = game["x_name"] if game["turn"] == "X" else game["o_name"]
    return f"⭕❌ {game['x_name']} ❌ در مقابل {game['o_name']} ⭕\nنوبت {turn_symbol} ({turn_name}) هست."


async def dooz_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    x_user = update.effective_user
    o_user = None
    if update.message.reply_to_message and update.message.reply_to_message.from_user.id != x_user.id:
        o_user = update.message.reply_to_message.from_user

    game = start_dooz_game(chat_id, x_user, o_user)
    await update.message.reply_text(
        dooz_status_text(game), reply_markup=render_dooz_board(game["board"])
    )


async def finish_dooz_game(query, game, winner):
    chat_id = query.message.chat_id
    if winner == "draw":
        text = "🤝 مساوی شد! بازی خوبی بود."
    elif game["player_o"] is None:
        text = "🎉 بردی! دمت گرم، حریف سختی بودی." if winner == "X" else "😎 من بردم! یه دست دیگه می‌خوای؟ بزن /dooz"
    else:
        winner_name = game["x_name"] if winner == "X" else game["o_name"]
        text = f"🎉 {winner_name} ({'❌' if winner == 'X' else '⭕'}) برد!"
    del active_dooz_games[chat_id]
    await query.edit_message_text(text=text, reply_markup=render_dooz_board(game["board"]))


async def handle_dooz_move(query, context: ContextTypes.DEFAULT_TYPE, position: int):
    chat_id = query.message.chat_id
    game = active_dooz_games.get(chat_id)
    if not game:
        await query.answer("این بازی دیگه فعال نیست. با /dooz یه بازی جدید شروع کن.", show_alert=True)
        return

    user_id = query.from_user.id
    turn = game["turn"]
    expected_player = game["player_x"] if turn == "X" else game["player_o"]

    if expected_player is None or user_id != expected_player:
        await query.answer("نوبت تو نیست! 😅", show_alert=True)
        return

    board = game["board"]
    if board[position]:
        await query.answer("این خونه قبلاً پر شده!", show_alert=True)
        return

    board[position] = turn
    await query.answer()

    winner = check_dooz_winner(board)
    if winner:
        await finish_dooz_game(query, game, winner)
        return

    game["turn"] = "O" if turn == "X" else "X"

    if game["turn"] == "O" and game["player_o"] is None:
        bot_pos = dooz_bot_move(board)
        board[bot_pos] = "O"
        winner = check_dooz_winner(board)
        if winner:
            await finish_dooz_game(query, game, winner)
            return
        game["turn"] = "X"

    await query.edit_message_text(text=dooz_status_text(game), reply_markup=render_dooz_board(board))


# ---------- بازی شطرنج با دکمه‌های شیشه‌ای ----------

CHESS_PIECE_SYMBOLS = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
    "": "▫️",
}
CHESS_PIECE_VALUES = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 0}

KNIGHT_DELTAS = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
KING_DELTAS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
BISHOP_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ROOK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def chess_initial_board():
    board = [""] * 64
    back_rank = "rnbqkbnr"
    for col in range(8):
        board[0 * 8 + col] = back_rank[col]
        board[1 * 8 + col] = "p"
        board[6 * 8 + col] = "P"
        board[7 * 8 + col] = back_rank[col].upper()
    return board


def chess_in_bounds(row, col):
    return 0 <= row < 8 and 0 <= col < 8


def chess_piece_color(piece):
    return "white" if piece.isupper() else "black"


def chess_pseudo_moves(board, idx, color, attacks_only=False):
    piece = board[idx]
    if not piece:
        return []
    row, col = idx // 8, idx % 8
    ptype = piece.lower()
    moves = []

    if ptype == "p":
        direction = -1 if color == "white" else 1
        start_row = 6 if color == "white" else 1
        for dc in (-1, 1):
            r, c = row + direction, col + dc
            if chess_in_bounds(r, c):
                target = board[r * 8 + c]
                if attacks_only:
                    moves.append(r * 8 + c)
                elif target and chess_piece_color(target) != color:
                    moves.append(r * 8 + c)
        if not attacks_only:
            r, c = row + direction, col
            if chess_in_bounds(r, c) and not board[r * 8 + c]:
                moves.append(r * 8 + c)
                if row == start_row:
                    r2 = row + 2 * direction
                    if chess_in_bounds(r2, col) and not board[r2 * 8 + col]:
                        moves.append(r2 * 8 + col)
    elif ptype == "n":
        for dr, dc in KNIGHT_DELTAS:
            r, c = row + dr, col + dc
            if chess_in_bounds(r, c):
                target = board[r * 8 + c]
                if not target or chess_piece_color(target) != color:
                    moves.append(r * 8 + c)
    elif ptype == "k":
        for dr, dc in KING_DELTAS:
            r, c = row + dr, col + dc
            if chess_in_bounds(r, c):
                target = board[r * 8 + c]
                if not target or chess_piece_color(target) != color:
                    moves.append(r * 8 + c)
    else:
        dirs = []
        if ptype in ("b", "q"):
            dirs += BISHOP_DIRS
        if ptype in ("r", "q"):
            dirs += ROOK_DIRS
        for dr, dc in dirs:
            r, c = row + dr, col + dc
            while chess_in_bounds(r, c):
                target = board[r * 8 + c]
                if not target:
                    moves.append(r * 8 + c)
                else:
                    if chess_piece_color(target) != color:
                        moves.append(r * 8 + c)
                    break
                r += dr
                c += dc
    return moves


def chess_find_king(board, color):
    target = "K" if color == "white" else "k"
    for i, p in enumerate(board):
        if p == target:
            return i
    return None


def chess_square_attacked(board, idx, by_color):
    for i, p in enumerate(board):
        if p and chess_piece_color(p) == by_color:
            if idx in chess_pseudo_moves(board, i, by_color, attacks_only=True):
                return True
    return False


def chess_in_check(board, color):
    king_idx = chess_find_king(board, color)
    if king_idx is None:
        return False
    enemy = "black" if color == "white" else "white"
    return chess_square_attacked(board, king_idx, enemy)


def chess_apply_move(board, frm, to):
    piece = board[frm]
    board[to] = piece
    board[frm] = ""
    if piece == "P" and to // 8 == 0:
        board[to] = "Q"
    elif piece == "p" and to // 8 == 7:
        board[to] = "q"


def chess_legal_moves(board, idx, color):
    dests = []
    for to in chess_pseudo_moves(board, idx, color):
        new_board = board[:]
        chess_apply_move(new_board, idx, to)
        if not chess_in_check(new_board, color):
            dests.append(to)
    return dests


def chess_all_legal_moves(board, color):
    result = {}
    for i, p in enumerate(board):
        if p and chess_piece_color(p) == color:
            dests = chess_legal_moves(board, i, color)
            if dests:
                result[i] = dests
    return result


def chess_bot_move(board, color):
    moves = chess_all_legal_moves(board, color)
    best_score, best_candidates = None, []
    for frm, dests in moves.items():
        for to in dests:
            target = board[to]
            score = CHESS_PIECE_VALUES.get(target.lower(), 0) if target else 0
            if best_score is None or score > best_score:
                best_score, best_candidates = score, [(frm, to)]
            elif score == best_score:
                best_candidates.append((frm, to))
    frm, to = random.choice(best_candidates)
    chess_apply_move(board, frm, to)


def chess_game_end_status(board, turn_to_move):
    if chess_all_legal_moves(board, turn_to_move):
        return None
    if chess_in_check(board, turn_to_move):
        winner = "black" if turn_to_move == "white" else "white"
        return f"checkmate_{winner}"
    return "stalemate"


def render_chess_board(game):
    board = game["board"]
    selected = game.get("selected")
    legal_dests = game.get("legal_dests", [])
    keyboard = []
    for row in range(8):
        row_buttons = []
        for col in range(8):
            idx = row * 8 + col
            piece = board[idx]
            symbol = CHESS_PIECE_SYMBOLS.get(piece, "▫️")
            if idx == selected:
                symbol = f"🟡{symbol}"
            elif idx in legal_dests:
                symbol = f"🟢{symbol}" if piece else "🟢"
            row_buttons.append(InlineKeyboardButton(symbol, callback_data=f"chess_sq_{idx}"))
        keyboard.append(row_buttons)
    keyboard.append([InlineKeyboardButton("🏳️ تسلیم", callback_data="chess_resign")])
    return InlineKeyboardMarkup(keyboard)


def start_chess_game(chat_id: int, white_user, black_user=None):
    active_chess_games[chat_id] = {
        "board": chess_initial_board(),
        "turn": "white",
        "player_white": white_user.id,
        "player_black": black_user.id if black_user else None,
        "white_name": get_display_name(white_user),
        "black_name": get_display_name(black_user) if black_user else "من",
        "selected": None,
        "legal_dests": [],
    }
    return active_chess_games[chat_id]


def chess_status_text(game, finished_text=None):
    if finished_text:
        return finished_text
    turn = game["turn"]
    turn_symbol = "⚪" if turn == "white" else "⚫"
    turn_name = game["white_name"] if turn == "white" else game["black_name"]
    text = f"♟️ {game['white_name']} (⚪) در مقابل {game['black_name']} (⚫)\nنوبت {turn_symbol} {turn_name} هست."
    if chess_in_check(game["board"], turn):
        text += "\n⚠️ کیش!"
    return text


async def chess_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.message.chat_id
    white_user = update.effective_user
    black_user = None
    if update.message.reply_to_message and update.message.reply_to_message.from_user.id != white_user.id:
        black_user = update.message.reply_to_message.from_user

    game = start_chess_game(chat_id, white_user, black_user)
    await update.message.reply_text(
        chess_status_text(game), reply_markup=render_chess_board(game)
    )


async def finish_chess_game(query, game, result):
    chat_id = query.message.chat_id
    if result == "stalemate":
        text = "🤝 پات شد (Stalemate)! بازی مساوی شد."
    else:
        winner_color = result.split("_")[1]
        winner_name = game["white_name"] if winner_color == "white" else game["black_name"]
        text = f"🏆 کیش و مات! {winner_name} برنده شد."
    del active_chess_games[chat_id]
    await query.edit_message_text(text=text, reply_markup=render_chess_board(game))


async def handle_chess_square(query, context: ContextTypes.DEFAULT_TYPE, idx: int):
    chat_id = query.message.chat_id
    game = active_chess_games.get(chat_id)
    if not game:
        await query.answer("این بازی دیگه فعال نیست. با /chess یه بازی جدید شروع کن.", show_alert=True)
        return

    user_id = query.from_user.id
    turn_color = game["turn"]
    expected_player = game["player_white"] if turn_color == "white" else game["player_black"]

    if expected_player is None:
        await query.answer("الان نوبت منه! یه لحظه صبر کن 🤖", show_alert=True)
        return
    if user_id != expected_player:
        await query.answer("نوبت تو نیست! 😅", show_alert=True)
        return

    board = game["board"]
    piece = board[idx]
    selected = game.get("selected")

    if selected is None:
        if not piece or chess_piece_color(piece) != turn_color:
            await query.answer("این مهره‌ی تو نیست!", show_alert=True)
            return
        dests = chess_legal_moves(board, idx, turn_color)
        if not dests:
            await query.answer("این مهره الان حرکت قانونی نداره.", show_alert=True)
            return
        game["selected"] = idx
        game["legal_dests"] = dests
        await query.answer()
        await query.edit_message_text(text=chess_status_text(game), reply_markup=render_chess_board(game))
        return

    if idx == selected:
        game["selected"] = None
        game["legal_dests"] = []
        await query.answer()
        await query.edit_message_text(text=chess_status_text(game), reply_markup=render_chess_board(game))
        return

    if piece and chess_piece_color(piece) == turn_color:
        dests = chess_legal_moves(board, idx, turn_color)
        if dests:
            game["selected"] = idx
            game["legal_dests"] = dests
        await query.answer()
        await query.edit_message_text(text=chess_status_text(game), reply_markup=render_chess_board(game))
        return

    if idx not in game.get("legal_dests", []):
        await query.answer("حرکت نامعتبره!", show_alert=True)
        return

    chess_apply_move(board, selected, idx)
    game["selected"] = None
    game["legal_dests"] = []
    game["turn"] = "black" if turn_color == "white" else "white"
    await query.answer()

    result = chess_game_end_status(board, game["turn"])
    if result:
        await finish_chess_game(query, game, result)
        return

    if game["turn"] == "black" and game["player_black"] is None:
        chess_bot_move(board, "black")
        game["turn"] = "white"
        result = chess_game_end_status(board, game["turn"])
        if result:
            await finish_chess_game(query, game, result)
            return

    await query.edit_message_text(text=chess_status_text(game), reply_markup=render_chess_board(game))


async def handle_chess_resign(query, context: ContextTypes.DEFAULT_TYPE):
    chat_id = query.message.chat_id
    game = active_chess_games.get(chat_id)
    if not game:
        await query.answer("بازی‌ای فعال نیست.", show_alert=True)
        return
    user_id = query.from_user.id
    if user_id not in (game["player_white"], game["player_black"]):
        await query.answer("تو بازیکنِ این بازی نیستی.", show_alert=True)
        return
    resigning_color = "white" if user_id == game["player_white"] else "black"
    winner_name = game["black_name"] if resigning_color == "white" else game["white_name"]
    await query.answer()
    text = f"🏳️ تسلیم شد. {winner_name} برنده شد!"
    del active_chess_games[chat_id]
    await query.edit_message_text(text=text, reply_markup=render_chess_board(game))


# ---------- بازی حکمِ چهارنفره (تحتِ وب، سرویسِ جدا روی Railway) ----------
# منطق و رابطِ کاربریِ بازی توی سرویسِ دومِ hokm_server هست (یه FastAPI + WebSocket که یه
# صفحه‌ی وب خوشگل سرو می‌کنه). این ربات فقط یه "میز" (room) می‌سازه و لینکش رو به‌صورتِ دکمه‌ی
# WebApp تلگرام (باز شونده داخلِ خودِ تلگرام) به گروه می‌فرسته.

def generate_room_code() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


async def hokm_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not HOKM_WEBAPP_URL:
        await update.message.reply_text(
            "🃏 هنوز آدرسِ وب‌اپِ بازیِ حکم ست نشده. اول سرویسِ hokm_server رو روی Railway دیپلوی کن، "
            "بعد لینکش رو توی Variables با اسمِ HOKM_WEBAPP_URL بذار (توضیحاتش توی README هست)."
        )
        return
    chat_id = update.message.chat_id
    room_id = generate_room_code()
    hokm_rooms[chat_id] = {"room_id": room_id, "created_by": update.effective_user.id}
    url = f"{HOKM_WEBAPP_URL.rstrip('/')}/?room={room_id}"
    if update.message.chat.type == "private":
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("🃏 ورود به میزِ حکم", web_app=WebAppInfo(url=url))]])
    else:
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("🃏 ورود به میزِ حکم", url=url)]])
    await update.message.reply_text(
        f"🃏 میزِ حکم ساخته شد! کدِ میز: <code>{room_id}</code>\n"
        "این دکمه رو بزن (یا لینک رو با ۳ نفرِ دیگه شریک کن، هرکی همین دکمه رو بزنه میاد سرِ همین میز).",
        parse_mode="HTML",
        reply_markup=keyboard,
    )

async def handle_voice_button(query, context: ContextTypes.DEFAULT_TYPE):
    chat_id = query.message.chat_id
    raw_text = query.message.text or query.message.caption or ""
    if not raw_text.strip():
        await query.answer("چیزی برای خوندن نیست!", show_alert=True)
        return
    await query.answer("🔊 دارم صداش می‌کنم، یه لحظه صبر کن...")
    try:
        await send_text_as_voice(context, chat_id, raw_text)
    except Exception as e:
        await context.bot.send_message(chat_id, f"❌ نتونستم تبدیلش کنم به صدا.\n{html_lib.escape(str(e))}")


async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    data = query.data

    if data.startswith("dooz_move_"):
        await handle_dooz_move(query, context, int(data.rsplit("_", 1)[1]))
        return
    if data.startswith("chess_sq_"):
        await handle_chess_square(query, context, int(data.rsplit("_", 1)[1]))
        return
    if data == "chess_resign":
        await handle_chess_resign(query, context)
        return
    if data == "voice_read":
        await handle_voice_button(query, context)
        return

    await query.answer()
    chat_id = query.message.chat_id

    if data == "menu_main":
        await query.message.reply_text("منوی اصلی:", reply_markup=main_menu_keyboard())
    elif data == "menu_ai":
        await query.message.reply_text(
            "🤖 برای صحبت با هوش مصنوعی بنویس:\n`/ai متن سوال یا حرفت`\n"
            "یا فقط روی جواب‌های من ریپلای کن تا گفتگو ادامه پیدا کنه. یه سلام ساده هم بکنی یا اسم "
            "بات/ربات رو صدا کنی خودم جواب می‌دم 👋",
            parse_mode="Markdown",
            reply_markup=BACK_TO_MAIN_KEYBOARD,
        )
    elif data == "menu_games":
        await query.message.reply_text("🎮 کدوم بازی رو می‌خوای؟", reply_markup=games_keyboard())
    elif data == "menu_iran":
        await query.message.reply_text("🇮🇷 کدوم بخش رو می‌خوای؟", reply_markup=iran_menu_keyboard())
    elif data == "menu_profile":
        await query.message.reply_text(
            build_profile_text(query.from_user), parse_mode="Markdown", reply_markup=BACK_TO_MAIN_KEYBOARD
        )
    elif data == "menu_admin":
        await query.message.reply_text("👮‍♂️ ابزار مدیریت:", reply_markup=admin_menu_keyboard())
    elif data == "menu_help":
        await query.message.reply_text(HELP_TEXT, parse_mode="Markdown", reply_markup=BACK_TO_MAIN_KEYBOARD)

    elif data in ("iran_hafez", "iran_today", "iran_countdown"):
        fake_update = SimpleNamespace(message=query.message)
        if data == "iran_hafez":
            await hafez_command(fake_update, context)
        elif data == "iran_today":
            await today_command(fake_update, context)
        elif data == "iran_countdown":
            await countdown_command(fake_update, context)

    elif data == "admin_tagall":
        if not await is_chat_admin(chat_id, query.from_user.id, context):
            await query.message.reply_text("❌ این کار مخصوص ادمین‌هاست.")
        else:
            await send_tagall(chat_id, context)
    elif data == "admin_stats":
        await query.message.reply_text(build_stats_text(chat_id), parse_mode="HTML")
    elif data == "admin_clean_info":
        await query.message.reply_text(
            "🧹 برای پاک‌سازی، دستور رو مستقیم بنویس (به‌خاطر عدد دلخواه، از دکمه نمی‌شه):\n"
            "`/clean 20` → آخرین ۲۰ پیام رو پاک می‌کنه\n"
            "`/cleangifs 10` → فقط آخرین ۱۰ گیف رو پاک می‌کنه",
            parse_mode="Markdown",
        )
    elif data == "admin_vip_info":
        await query.message.reply_text(
            "⭐️ برای عضو ویژه کردن، روی پیام کاربر ریپلای کن و بنویس:\n"
            "`/addvip` → ویژه‌ش کن (دیگه فیلتر لینک/فحش/اسپم روش اعمال نمی‌شه)\n"
            "`/removevip` → ویژگیش رو بردار",
            parse_mode="Markdown",
        )

    elif data == "game_guess_start":
        start_guess_game(chat_id)
        await query.message.reply_text(
            "🔢 یه عدد بین ۱ تا ۱۰۰ تو ذهنم گذاشتم! حدس بزن چنده (فقط عدد رو بفرست)."
        )
    elif data == "game_math_start":
        question = start_math_game(chat_id)
        await query.message.reply_text(f"➕ سریع باش: {question} = ?")
    elif data == "game_rps_menu":
        await query.message.reply_text("✊ یکی رو انتخاب کن:", reply_markup=rps_keyboard())
    elif data == "game_namefamily_start":
        letter = random.choice(PERSIAN_LETTERS)
        await query.message.reply_text(
            f"🎲 بازی اسم فامیل شروع شد!\nحرف امتحان: **{letter}**\n"
            "اسم، فامیل، شهر، حیوان، غذا، گل/میوه با این حرف بگید!\n⏱ ۶۰ ثانیه وقت دارید...",
            parse_mode="Markdown",
        )
        if context.job_queue:
            context.job_queue.run_once(
                namefamily_timeout, when=60, chat_id=chat_id, name=f"namefamily_{chat_id}",
            )
    elif data in ("rps_rock", "rps_paper", "rps_scissors"):
        result_text = play_rps(data)
        await query.message.reply_text(result_text)
    elif data == "game_dooz_start":
        game = start_dooz_game(chat_id, query.from_user, o_user=None)
        await query.message.reply_text(
            dooz_status_text(game), reply_markup=render_dooz_board(game["board"])
        )
    elif data == "game_chess_start":
        game = start_chess_game(chat_id, query.from_user, black_user=None)
        await query.message.reply_text(
            chess_status_text(game), reply_markup=render_chess_board(game)
        )
    elif data == "game_hokm_start":
        fake_update = SimpleNamespace(message=query.message, effective_user=query.from_user)
        await hokm_command(fake_update, context)


async def is_chat_admin(chat_id: int, user_id: int, context: ContextTypes.DEFAULT_TYPE) -> bool:
    try:
        member = await context.bot.get_chat_member(chat_id, user_id)
        return member.status in ["creator", "administrator"]
    except Exception:
        return False


async def is_admin(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if update.message.chat.type == "private":
        return False
    return await is_chat_admin(update.message.chat_id, update.effective_user.id, context)


async def ban(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("❌ این دستور مخصوص ادمین‌های گروه است!")
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("❌ لطفا این دستور را روی پیام کاربر مورد نظر ریپلای کنید.")
        return
    target_user = update.message.reply_to_message.from_user
    try:
        await context.bot.ban_chat_member(update.message.chat_id, target_user.id)
        await update.message.reply_text(f"🔒 کاربر {target_user.first_name} با موفقیت بن شد.")
    except Exception:
        await update.message.reply_text("❌ خطایی در مسدود سازی رخ داد.")


async def mute(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("❌ این دستور مخصوص ادمین‌هاست.")
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("❌ روی پیام کاربر ریپلای کنید.")
        return
    target_user = update.message.reply_to_message.from_user
    if target_user.id in vip_users:
        await update.message.reply_text(
            f"⭐️ {target_user.first_name} عضو ویژه‌ست، نمی‌تونم بی‌صداش کنم. اول با /removevip ویژگیش رو بردار."
        )
        return
    muted_users.add(target_user.id)
    save_muted()
    await update.message.reply_text(f"🔇 کاربر {target_user.first_name} در حالت سکوت قرار گرفت.")


async def _tempmute_expire(context: ContextTypes.DEFAULT_TYPE):
    job = context.job
    user_id, chat_id = job.data
    if user_id in muted_users:
        muted_users.discard(user_id)
        save_muted()
        try:
            await context.bot.send_message(chat_id, "🔊 زمانِ سکوت تموم شد، دوباره می‌تونی پیام بدی.")
        except Exception:
            pass


async def tempmute_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """سکوتِ زمان‌دار: /tempmute <دقیقه> (با ریپلای روی پیامِ کاربر). بعدِ گذشتِ زمان، خودکار برمی‌گرده."""
    if not await is_admin(update, context):
        await update.message.reply_text("❌ این دستور مخصوص ادمین‌هاست.")
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("❌ روی پیامِ کاربر ریپلای کن و بعدِ دستور، تعدادِ دقیقه رو بنویس.")
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text(
            "❌ تعدادِ دقیقه رو بعد از دستور بنویس. مثلاً:\n`/tempmute 30`", parse_mode="Markdown"
        )
        return
    minutes = int(context.args[0])
    if minutes <= 0 or minutes > 10080:  # حداکثر یه هفته
        await update.message.reply_text("❌ عددِ دقیقه باید بینِ ۱ تا ۱۰۰۸۰ (یه هفته) باشه.")
        return

    target_user = update.message.reply_to_message.from_user
    if target_user.id in vip_users:
        await update.message.reply_text(
            f"⭐️ {target_user.first_name} عضو ویژه‌ست، نمی‌تونم بی‌صداش کنم. اول با /removevip ویژگیش رو بردار."
        )
        return

    muted_users.add(target_user.id)
    save_muted()
    chat_id = update.message.chat_id

    if context.job_queue:
        context.job_queue.run_once(
            _tempmute_expire, when=minutes * 60, data=(target_user.id, chat_id),
            name=f"tempmute_{chat_id}_{target_user.id}",
        )
        await update.message.reply_text(
            f"🔇 کاربر {target_user.first_name} به مدتِ {fa_num(minutes)} دقیقه بی‌صدا شد. "
            "خودکار بعدِ این مدت برمی‌گرده."
        )
    else:
        await update.message.reply_text(
            f"🔇 کاربر {target_user.first_name} بی‌صدا شد، ولی چون job_queue فعال نیست نمی‌تونم خودکار "
            "برش گردونم — دستی با /unmute برش گردون."
        )


async def unmute(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        return
    if not update.message.reply_to_message:
        return
    target_user = update.message.reply_to_message.from_user
    if target_user.id in muted_users:
        muted_users.remove(target_user.id)
        save_muted()
        await update.message.reply_text(f"🔊 کاربر {target_user.first_name} مجدداً اجازه ارسال پیام دارد.")


async def warn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context) or not update.message.reply_to_message:
        return
    target_user = update.message.reply_to_message.from_user
    user_id = target_user.id
    user_warnings[user_id] = user_warnings.get(user_id, 0) + 1
    save_warnings()

    if user_warnings[user_id] >= 3:
        try:
            await context.bot.ban_chat_member(update.message.chat_id, user_id)
            await update.message.reply_text(f"🔒 کاربر {target_user.first_name} به دلیل دریافت ۳ اخطار بن شد.")
            user_warnings[user_id] = 0
            save_warnings()
        except Exception:
            await update.message.reply_text("❌ خطا در بن کردن کاربر اخراجی.")
    else:
        await update.message.reply_text(
            f"⚠️ کاربر {target_user.first_name} یک اخطار دریافت کرد. اخطارها: {user_warnings[user_id]}/3"
        )


async def settag_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("❌ این دستور مخصوص ادمین‌هاست.")
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("❌ روی پیام کاربری که می‌خوای لقب بدی ریپلای کن.")
        return
    tag_text = " ".join(context.args).strip()
    if not tag_text:
        await update.message.reply_text(
            "❌ بعد از دستور لقب رو بنویس. مثلاً:\n`/settag مدیر ویژه`", parse_mode="Markdown"
        )
        return
    target_user = update.message.reply_to_message.from_user
    user_tags[target_user.id] = tag_text[:30]
    save_tags()
    await update.message.reply_text(f"🏷️ از الان لقب {target_user.first_name} شد: {tag_text[:30]}")


async def removetag_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("❌ این دستور مخصوص ادمین‌هاست.")
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("❌ روی پیام کاربر مورد نظر ریپلای کن.")
        return
    target_user = update.message.reply_to_message.from_user
    if target_user.id in user_tags:
        del user_tags[target_user.id]
        save_tags()
        await update.message.reply_text(f"🗑️ لقب {target_user.first_name} حذف شد.")
    else:
        await update.message.reply_text("❌ این کاربر لقبی نداشت.")


async def addvip_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("❌ این دستور مخصوص ادمین‌هاست.")
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("❌ روی پیام کاربری که می‌خوای ویژه کنی ریپلای کن.")
        return
    target_user = update.message.reply_to_message.from_user
    vip_users.add(target_user.id)
    save_vips()
    await update.message.reply_text(
        f"⭐️ {target_user.first_name} از الان عضو ویژه‌ست و دیگه فیلتر لینک/فحش/اسپم روش اعمال نمی‌شه."
    )


async def removevip_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("❌ این دستور مخصوص ادمین‌هاست.")
        return
    if not update.message.reply_to_message:
        await update.message.reply_text("❌ روی پیام کاربر مورد نظر ریپلای کن.")
        return
    target_user = update.message.reply_to_message.from_user
    if target_user.id in vip_users:
        vip_users.discard(target_user.id)
        save_vips()
        await update.message.reply_text(f"🗑️ {target_user.first_name} دیگه عضو ویژه نیست.")
    else:
        await update.message.reply_text("❌ این کاربر عضو ویژه نبود.")


async def send_tagall(chat_id: int, context: ContextTypes.DEFAULT_TYPE, custom_message: str = ""):
    known = chat_known_users.get(chat_id, {})
    if not known:
        await context.bot.send_message(
            chat_id,
            "❌ هنوز هیچ‌کسی رو نمی‌شناسم. فقط می‌تونم کسایی رو تگ کنم که قبلاً توی گروه پیام داده باشن "
            "(تلگرام اجازه نمی‌ده لیست کامل اعضا رو از API گرفت).",
        )
        return
    custom_message = custom_message or "توجه اعضا 📢"
    mentions = [f'<a href="tg://user?id={uid}">{html_lib.escape(name)}</a>' for uid, name in known.items()]
    CHUNK = 25
    for i in range(0, len(mentions), CHUNK):
        part = " ".join(mentions[i:i + CHUNK])
        prefix = f"{html_lib.escape(custom_message)}\n\n" if i == 0 else ""
        await context.bot.send_message(chat_id, prefix + part, parse_mode="HTML")


async def tagall_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("❌ این دستور مخصوص ادمین‌هاست.")
        return
    await send_tagall(update.message.chat_id, context, " ".join(context.args).strip())


def build_stats_text(chat_id: int) -> str:
    counts = chat_activity_counts.get(chat_id, {})
    if not counts:
        return "📊 هنوز آماری از این گروه ثبت نشده."
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    names = chat_known_users.get(chat_id, {})
    lines = []
    for i, (uid, count) in enumerate(top, start=1):
        name = names.get(uid, str(uid))
        lines.append(f"{fa_num(i)}. {html_lib.escape(name)} — <code>{fa_num(count)}</code> پیام")
    return "📊 <b>فعال‌ترین اعضا</b> (از وقتی ربات روشن شده)\n\n" + "\n".join(lines)


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(build_stats_text(update.message.chat_id), parse_mode="HTML")


async def clean_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("❌ این دستور مخصوص ادمین‌هاست.")
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("❌ تعداد رو بعد از دستور بنویس. مثلاً:\n`/clean 20`", parse_mode="Markdown")
        return
    n = min(int(context.args[0]), RECENT_MESSAGES_LIMIT)
    chat_id = update.message.chat_id
    recent = list(chat_recent_messages.get(chat_id, deque()))
    ids_to_delete = [mid for mid, _ in recent[-n:]]
    deleted = 0
    for mid in ids_to_delete:
        try:
            await context.bot.delete_message(chat_id, mid)
            deleted += 1
        except Exception:
            pass
    try:
        await update.message.delete()
    except Exception:
        pass
    await context.bot.send_message(chat_id, f"🧹 {fa_num(deleted)} پیام پاک شد.")


async def cleangifs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("❌ این دستور مخصوص ادمین‌هاست.")
        return
    if not context.args or not context.args[0].isdigit():
        await update.message.reply_text("❌ تعداد رو بعد از دستور بنویس. مثلاً:\n`/cleangifs 10`", parse_mode="Markdown")
        return
    n = int(context.args[0])
    chat_id = update.message.chat_id
    recent = list(chat_recent_messages.get(chat_id, deque()))
    gif_ids = [mid for mid, is_gif in recent if is_gif][-n:]
    deleted = 0
    for mid in gif_ids:
        try:
            await context.bot.delete_message(chat_id, mid)
            deleted += 1
        except Exception:
            pass
    try:
        await update.message.delete()
    except Exception:
        pass
    await context.bot.send_message(chat_id, f"🧹 {fa_num(deleted)} گیف پاک شد.")


async def learn_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("❌ این دستور مخصوص ادمین‌هاست.")
        return
    full_text = update.message.text.partition(" ")[2]
    if "|" not in full_text:
        await update.message.reply_text(
            "❌ فرمت درست:\n`/learn عبارت کلیدی | جوابی که باید بدم`", parse_mode="Markdown"
        )
        return
    keyword, _, answer = full_text.partition("|")
    keyword = keyword.strip().lower()
    answer = answer.strip()
    if not keyword or not answer:
        await update.message.reply_text("❌ هم عبارت کلیدی هم جواب لازمه.")
        return
    learned_facts[keyword] = answer
    save_learned()
    await update.message.reply_text(f"✅ یاد گرفتم! هر وقت کسی بگه «{keyword}» این جواب رو می‌دم:\n{answer}")


async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("❌ این دستور مخصوص ادمین‌هاست.")
        return
    keyword = " ".join(context.args).strip().lower()
    if keyword in learned_facts:
        del learned_facts[keyword]
        save_learned()
        await update.message.reply_text(f"🗑️ یادمو در مورد «{keyword}» پاک کردم.")
    else:
        await update.message.reply_text("❌ چیزی با این عبارت یاد نگرفته بودم.")


async def learned_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not learned_facts:
        await update.message.reply_text("📭 هنوز هیچی یاد نگرفتم.")
        return
    keywords = "\n".join(f"🔸 {k}" for k in learned_facts)
    await update.message.reply_text(f"📚 چیزایی که یاد گرفتم:\n{keywords}")


async def addbadword_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("❌ این دستور مخصوص ادمین‌هاست.")
        return
    word = " ".join(context.args).strip().lower()
    if not word:
        await update.message.reply_text(
            "❌ بعد از دستور کلمه رو بنویس. مثلاً:\n`/addbadword کلمه`", parse_mode="Markdown"
        )
        return
    bad_words.add(word)
    save_badwords()
    await update.message.reply_text(f"✅ از این به بعد پیام‌های شامل «{word}» حذف می‌شن و اخطار می‌گیرن.")


async def removebadword_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("❌ این دستور مخصوص ادمین‌هاست.")
        return
    word = " ".join(context.args).strip().lower()
    if word in bad_words:
        bad_words.discard(word)
        save_badwords()
        await update.message.reply_text(f"🗑️ «{word}» از لیست فیلتر حذف شد.")
    else:
        await update.message.reply_text("❌ این کلمه توی لیست فیلتر نبود.")


async def badwords_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await is_admin(update, context):
        await update.message.reply_text("❌ این دستور مخصوص ادمین‌هاست.")
        return
    if not bad_words:
        await update.message.reply_text("📭 لیست فیلتر فعلاً خالیه. با `/addbadword` کلمه اضافه کن.", parse_mode="Markdown")
        return
    await update.message.reply_text("🚫 کلمات فیلترشده:\n" + "\n".join(f"🔸 {w}" for w in bad_words))


async def togglelinks_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global link_filter_enabled
    if not await is_admin(update, context):
        await update.message.reply_text("❌ این دستور مخصوص ادمین‌هاست.")
        return
    link_filter_enabled = not link_filter_enabled
    save_settings()
    state = "فعال ✅" if link_filter_enabled else "غیرفعال ❌"
    await update.message.reply_text(f"🔗 فیلتر لینک الان {state} شد.")


async def welcome_new_member(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for member in update.message.new_chat_members:
        if member.id == context.bot.id:
            continue
        name = get_display_name(member)
        try:
            reply_text = await ask_ai(
                f"یه عضو جدید به اسم {name} تازه به این گروه تلگرامی پیوست. با همون لحنِ خاصِ خودت، "
                "کوتاه به گروه خوش‌آمد بگو."
            )
        except Exception:
            reply_text = f"به گروه خوش اومدی {name} جون! 🎉"
        await update.message.reply_text(with_signature(reply_text))


async def ai_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_prompt = " ".join(context.args)
    if not user_prompt:
        await update.message.reply_text(
            "❌ بعد از دستور یه چیزی بنویس. مثلاً:\n`/ai چطور پایتون یاد بگیرم؟`",
            parse_mode="Markdown",
        )
        return
    if is_creator_question(user_prompt):
        await update.message.reply_text(CREATOR_REPLY)
        return
    user_id = update.effective_user.id
    name = get_display_name(update.effective_user)
    history_ctx = get_history_context(user_id)
    await react_to_message(context, update.message.chat_id, update.message.message_id, "👀")
    waiting_msg, is_media = await send_waiting(update.message, context)
    try:
        prompt = f"{history_ctx}کاربر به اسم {name} الان این رو پرسید/گفت:\n{user_prompt}"
        reply_text = await ask_ai(prompt)
        await finish_waiting(
            context, update.message.chat_id, waiting_msg, is_media,
            with_signature(reply_text), reply_markup=VOICE_BUTTON_KEYBOARD,
        )
    except Exception as e:
        await send_ai_failure(context, update.message.chat_id, waiting_msg, is_media, e)


async def handle_animation(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.from_user.id in muted_users:
        try:
            await update.message.delete()
        except Exception:
            pass
        return
    track_group_activity(update, is_animation=True)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.from_user.id in muted_users:
        try:
            await update.message.delete()
        except Exception:
            pass
        return

    track_group_activity(update)
    photo_file = await update.message.photo[-1].get_file()
    file_path = f"/tmp/photo_{update.effective_user.id}_{uuid.uuid4().hex}.jpg"
    await photo_file.download_to_drive(file_path)

    waiting_msg, is_media = await send_waiting(update.message, context, "👁️ بذار عکس رو نگاه کنم...")
    try:
        caption = update.message.caption if update.message.caption else "این تصویر را تحلیل کن"
        reply_text = await analyze_image(file_path, caption)
        await finish_waiting(context, update.message.chat_id, waiting_msg, is_media, with_signature(reply_text))
    except Exception as e:
        await finish_waiting(
            context, update.message.chat_id, waiting_msg, is_media,
            f"❌ یه خطا خوردم تو پردازش عکس.\n{html_lib.escape(str(e))}",
        )
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    chat_id = update.message.chat_id
    text = (update.message.text or "").strip()

    if user_id in muted_users:
        try:
            await update.message.delete()
        except Exception:
            pass
        return

    if (
        update.message.chat.type != "private"
        and user_id not in vip_users
        and not await is_chat_admin(chat_id, user_id, context)
    ):
        name = get_display_name(update.effective_user)

        if link_filter_enabled and URL_PATTERN.search(text):
            try:
                await update.message.delete()
            except Exception:
                pass
            await context.bot.send_message(chat_id, f"🚫 {name} جون، فرستادن لینک توی این گروه مجاز نیست.")
            return

        text_lower = text.lower()
        if any(bad in text_lower for bad in bad_words):
            try:
                await update.message.delete()
            except Exception:
                pass
            user_warnings[user_id] = user_warnings.get(user_id, 0) + 1
            save_warnings()
            await context.bot.send_message(
                chat_id,
                f"🚫 {name} این کلمه توی گروه مجاز نیست! اخطار گرفتی ({user_warnings[user_id]}/3)",
            )
            if user_warnings[user_id] >= 3:
                try:
                    await context.bot.ban_chat_member(chat_id, user_id)
                    await context.bot.send_message(chat_id, f"🔒 {name} به دلیل ۳ اخطار بن شد.")
                    user_warnings[user_id] = 0
                    save_warnings()
                except Exception:
                    pass
            return

        now = time.monotonic()
        timestamps = user_message_times.setdefault(user_id, deque(maxlen=SPAM_MESSAGE_THRESHOLD))
        timestamps.append(now)
        if len(timestamps) == SPAM_MESSAGE_THRESHOLD and (now - timestamps[0]) < SPAM_WINDOW_SECONDS:
            muted_users.add(user_id)
            save_muted()
            try:
                await update.message.delete()
            except Exception:
                pass
            await context.bot.send_message(
                chat_id,
                f"🔇 {name} به‌خاطر ارسال پیام زیاد در زمان کوتاه، موقتاً بی‌صدا شد. "
                "یه ادمین می‌تونه با /unmute (ریپلای) برش گردونه.",
            )
            return

    add_to_history(user_id, text)
    track_group_activity(update)

    if is_creator_question(text):
        await update.message.reply_text(CREATOR_REPLY)
        return

    reply_to = update.message.reply_to_message
    if reply_to and reply_to.from_user and reply_to.from_user.id == context.bot.id and text:
        name = get_display_name(update.effective_user)
        tag = user_tags.get(user_id)
        tag_info = f" (لقبش: {tag})" if tag else ""
        history_ctx = get_history_context(user_id)
        await react_to_message(context, chat_id, update.message.message_id, "👀")
        waiting_msg, is_media = await send_waiting(update.message, context)
        try:
            previous_bot_text = reply_to.text or reply_to.caption or ""
            prompt = (
                f"{history_ctx}"
                f"کاربر به اسم {name}{tag_info} داره باهات چت می‌کنه.\n"
                f"تو قبلاً این رو گفته بودی:\n«{previous_bot_text}»\n\n"
                f"کاربر روی همین پیام ریپلای کرد و نوشت:\n«{text}»\n\n"
                "با همون لحنِ خاصِ خودت به این ادامه‌ی گفتگو جواب بده. اگه ازت خواست چیزی رو تحلیل کنی، "
                "تحلیل دقیق و مفید بده."
            )
            reply_text_ai = await ask_ai(prompt)
            await finish_waiting(
                context, chat_id, waiting_msg, is_media,
                with_signature(reply_text_ai), reply_markup=VOICE_BUTTON_KEYBOARD,
            )
        except Exception as e:
            await send_ai_failure(context, chat_id, waiting_msg, is_media, e)
        return

    if chat_id in active_guess_games and text.lstrip("-").isdigit():
        game = active_guess_games[chat_id]
        guess = int(text)
        game["attempts"] += 1
        if guess == game["number"]:
            await react_to_message(context, chat_id, update.message.message_id, "🎉")
            await update.message.reply_text(
                f"🎉 درست گفتی! عدد {game['number']} بود. تو {game['attempts']} بار حدس زدی، دمت گرم!"
            )
            await send_random_pack_sticker(context, chat_id)
            del active_guess_games[chat_id]
        elif guess < game["number"]:
            await update.message.reply_text("⬆️ بزرگ‌تره، بازم حدس بزن.")
        else:
            await update.message.reply_text("⬇️ کوچیک‌تره، بازم حدس بزن.")
        return

    if chat_id in active_math_games and text.lstrip("-").isdigit():
        game = active_math_games[chat_id]
        if int(text) == game["answer"]:
            await react_to_message(context, chat_id, update.message.message_id, "👏")
            await update.message.reply_text("✅ آره درسته! خیلی سریع بودی 👏")
            await send_random_pack_sticker(context, chat_id)
            del active_math_games[chat_id]
        else:
            await update.message.reply_text("❌ نه، اشتباهه. دوباره امتحان کن.")
        return

    learned_answer = find_learned_match(text)
    if learned_answer:
        await update.message.reply_text(learned_answer)
        return

    wiki_match = WIKI_TRIGGER_PATTERN.match(text)
    if wiki_match:
        await do_wiki_lookup(update, context, wiki_match.group(1).strip())
        return

    if is_greeting(text):
        await react_to_message(context, chat_id, update.message.message_id)
        name = get_display_name(update.effective_user)
        tag = user_tags.get(user_id)
        tag_info = f" (لقبش: {tag})" if tag else ""
        try:
            reply_text = await ask_ai(
                f"کاربری به اسم {name}{tag_info} سلام داد. با همون لحنِ خاصِ خودت، کوتاه جواب سلام بده "
                "و اسمش رو هم صدا بزن."
            )
        except Exception:
            reply_text = f"سلام {name} جون! 👋"
        await update.message.reply_text(with_signature(reply_text), reply_markup=VOICE_BUTTON_KEYBOARD)
        return

    if mentions_bot(text, context.bot.username):
        name = get_display_name(update.effective_user)
        tag = user_tags.get(user_id)
        tag_info = f" (لقبش: {tag})" if tag else ""
        history_ctx = get_history_context(user_id)
        await react_to_message(context, chat_id, update.message.message_id, "👀")
        waiting_msg, is_media = await send_waiting(update.message, context)
        try:
            prompt = (
                f"{history_ctx}"
                f"کاربر به اسم {name}{tag_info} توی گروه بهت رو کرد (صدات زد: بات/ربات/روبات) و این رو نوشت:\n"
                f"«{text}»\n\n"
                "طبق شخصیتت جواب بده. اگه سوالی پرسیده یا خواسته‌ای داشته، کمک واقعی و درست بهش بکن."
            )
            reply_text = await ask_ai(prompt)
            await finish_waiting(
                context, chat_id, waiting_msg, is_media,
                with_signature(reply_text), reply_markup=VOICE_BUTTON_KEYBOARD,
            )
        except Exception as e:
            await send_ai_failure(context, chat_id, waiting_msg, is_media, e)
        return


def main():
    print("🤖 ربات مدیریت گروه و هوش مصنوعی در حال روشن شدن است...")
    load_persisted_state()

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("ai", ai_mode))
    app.add_handler(CommandHandler("ban", ban))
    app.add_handler(CommandHandler("mute", mute))
    app.add_handler(CommandHandler("tempmute", tempmute_command))
    app.add_handler(CommandHandler("unmute", unmute))
    app.add_handler(CommandHandler("warn", warn))
    app.add_handler(CommandHandler("nickname", nickname_command))
    app.add_handler(CommandHandler("tag", tag_command))
    app.add_handler(CommandHandler("profile", profile_command))
    app.add_handler(CommandHandler("settag", settag_command))
    app.add_handler(CommandHandler("removetag", removetag_command))
    app.add_handler(CommandHandler("addvip", addvip_command))
    app.add_handler(CommandHandler("removevip", removevip_command))
    app.add_handler(CommandHandler("tagall", tagall_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("clean", clean_command))
    app.add_handler(CommandHandler("cleangifs", cleangifs_command))
    app.add_handler(CommandHandler("learn", learn_command))
    app.add_handler(CommandHandler("forget", forget_command))
    app.add_handler(CommandHandler("learned", learned_command))
    app.add_handler(CommandHandler("addbadword", addbadword_command))
    app.add_handler(CommandHandler("removebadword", removebadword_command))
    app.add_handler(CommandHandler("badwords", badwords_command))
    app.add_handler(CommandHandler("togglelinks", togglelinks_command))

    app.add_handler(CommandHandler("hafez", hafez_command))
    app.add_handler(CommandHandler("today", today_command))
    app.add_handler(CommandHandler("countdown", countdown_command))
    app.add_handler(CommandHandler("wiki", wiki_command))

    app.add_handler(CommandHandler("game", game_menu))
    app.add_handler(CommandHandler("guess", guess_command))
    app.add_handler(CommandHandler("math", math_command))
    app.add_handler(CommandHandler("rps", rps_command))
    app.add_handler(CommandHandler("namefamily", namefamily_command))
    app.add_handler(CommandHandler("dooz", dooz_command))
    app.add_handler(CommandHandler("chess", chess_command))
    app.add_handler(CommandHandler("hokm", hokm_command))

    app.add_handler(CallbackQueryHandler(button_handler))

    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
   
