"""
JARVIS brain — a bilingual (Mongolian / English) conversational engine. Everyday
intents (time, weather, apps, timers, ...) are handled on this computer via
core.skills; anything else goes to Gemini (core.gemini) when an API key is set.

Mongolian works in Cyrillic and in Latin transliteration ("tsag hed bolj baina").
"""
from __future__ import annotations

import ast
import operator
import random
import re
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from core import skills
from core.gemini import Gemini
from memory.memory_manager import load_memory, update_memory


# ── Text normalisation ───────────────────────────────────────────────────────
# Every phrase and every user message is reduced to the same lossy Latin form so
# that "Цаг хэд", "tsag hed", "cag xed" and "tsaag hedd" all compare equal.

_CYR = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "j",
    "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o",
    "ө": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ү": "u", "ф": "f",
    "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "", "ы": "i", "ь": "i",
    "э": "e", "ю": "yu", "я": "ya",
}


def canon(word: str) -> str:
    w = "".join(_CYR.get(ch, ch) for ch in word.lower())
    for a, b in (("kh", "h"), ("zh", "j"), ("ö", "o"), ("ø", "o"), ("ü", "u"),
                 ("x", "h"), ("q", "o"), ("w", "v")):
        w = w.replace(a, b)
    w = re.sub(r"c(?!h)", "ts", w)
    w = re.sub(r"[^a-z0-9]", "", w)
    w = re.sub(r"(?<=[aeiou])y(?![aeiou])", "i", w)
    w = w.replace("o", "u")
    return re.sub(r"([aeiu])\1+", r"\1", w)


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[\s,.!?;:\"«»“”()]+", text) if t]


def _same(token: str, pattern: str) -> bool:
    if token == pattern:
        return True
    if len(pattern) >= 4 and token.startswith(pattern):           # Mongolian suffixes
        return True
    if len(pattern) >= 6 and len(token) >= len(pattern):            # one typo
        return sum(a != b for a, b in zip(token, pattern)) <= 1
    return False


class Phrases:
    """A list of phrases matched token-by-token against a canonical message."""

    def __init__(self, *phrases: str):
        self.items = [[canon(t) for t in _tokens(p)] for p in phrases]
        self.items = [[t for t in p if t] for p in self.items if any(p)]

    def spans(self, ctoks: list[str]) -> list[tuple[int, int, int]]:
        """All matches as (start, end, score); score favours longer phrases."""
        found = []
        for p in self.items:
            for i in range(len(ctoks) - len(p) + 1):
                if all(_same(ctoks[i + k], p[k]) for k in range(len(p))):
                    found.append((i, i + len(p), sum(len(t) for t in p) + len(p)))
        return found

    def best(self, ctoks: list[str]) -> int:
        return max((s for _, _, s in self.spans(ctoks)), default=0)

    def __contains__(self, ctoks) -> bool:
        return bool(self.spans(ctoks))


@dataclass
class Message:
    raw: str
    words: list[str]                      # original tokens
    ctoks: list[str]                      # canonical tokens, aligned with `words`
    cyrillic: bool

    @classmethod
    def parse(cls, raw: str) -> "Message":
        words = _tokens(raw)
        return cls(raw, words, [canon(w) for w in words],
                   bool(re.search(r"[а-яёөү]", raw, re.I)))

    def without(self, *groups: Phrases) -> list[str]:
        """Original words left after removing every span matched by `groups`."""
        drop = set()
        for g in groups:
            for start, end, _ in g.spans(self.ctoks):
                drop.update(range(start, end))
        return [w for i, w in enumerate(self.words) if i not in drop]


@dataclass
class Reply:
    text: str
    lang: str
    speech: str | None = None             # shorter version to say aloud
    silent: bool = False
    fallback: bool = False                # JARVIS didn't understand
    ai: bool = False                      # answered by Gemini


# ── Vocabulary ───────────────────────────────────────────────────────────────

_MN_MARKERS = {canon(w) for w in (
    "уу", "юу", "вэ", "бэ", "байна", "бна", "бн", "би", "чи", "та", "миний", "чиний", "намайг",
    "чамайг", "хэн", "яаж", "ямар", "хэд", "хэдэн", "энэ", "тэр", "өг", "өгөөч", "нээ", "хай",
    "сайн", "за", "тийм", "үгүй", "юм", "шүү", "дээ", "болж", "одоо", "өнөөдөр", "маргааш",
    "надад", "яриач", "хэлээч", "гэж", "гэдэг", "болох", "биш", "бас", "bn", "bnu", "yu", "ve",
)}
_EN_MARKERS = {canon(w) for w in (
    "the", "is", "are", "what", "how", "you", "your", "me", "my", "please", "can", "do", "i",
    "it", "to", "a", "open", "play", "search", "set", "tell", "who", "when", "where", "why",
    "hello", "hi", "hey", "thanks", "thank", "of", "for", "in", "on", "and", "volume", "down", "up",
    "battery", "mute", "unmute", "screenshot", "take", "system", "status", "weather", "time", "date",
    "joke", "timer", "minutes", "remind", "louder", "quieter", "lock", "screen", "help", "bye",
    "good", "morning", "night", "name", "fact", "make", "show", "find", "look", "whats", "turn",
)}

JARVIS_NAME = Phrases("jarvis", "жарвис", "джарвис", "жарвиз", "jarvis aa", "жарвис аа",
                      "hey jarvis", "эй жарвис", "ok jarvis", "javis", "jervis")
GREETING = {
    "mn": Phrases("сайн уу", "сайн байна уу", "сайн байцгаана уу", "сайн бну", "сн уу", "snuu",
                  "sainuu", "sain bnu", "sn bnu", "мэнд", "мэнд хүргэе", "өглөөний мэнд",
                  "өдрийн мэнд", "оройн мэнд", "сайхан амарсан уу", "амар сайн уу", "сайн уу байна"),
    "en": Phrases("hello", "hi", "hey", "hiya", "good morning", "good afternoon", "good evening",
                  "greetings", "howdy", "hi there", "hello there"),
}
YES = Phrases("тийм", "тийм ээ", "за", "зүгээр", "болно", "тэгье", "тэгээрэй", "тэг", "тэгэх", "yes",
              "yeah", "yep", "yup", "sure", "ok", "okay", "please", "do it", "go ahead", "of course")
NO = Phrases("үгүй", "үгүй ээ", "хэрэггүй", "болъё", "битгий", "no", "nope", "nah", "dont",
             "not now", "no thanks")
OPEN_VERB = Phrases("нээ", "нээгээч", "нээгээрэй", "нээж өг", "нээж өгөөч", "нээгээд өг",
                    "нээгээд өгөөч", "нээгээд", "нээж", "нээчих", "нээчээ", "нээх", "нээнэ үү", "асаа",
                    "асаагаач", "асаагаад өг", "ажиллуул", "ажиллуулаач", "neegeed ugeech", "neej ugeech",
                    "open", "launch", "start", "run", "open up", "fire up")
SEARCH_VERB = Phrases("хайгаад өг", "хайгаад өгөөч", "хайгаад", "хайгаач", "хайж өг", "хайж өгөөч",
                      "хайж", "хай", "хайх", "хайлт хий", "хайгаарай", "haigaad ugeech", "haij ugeech",
                      "search for", "search", "google", "look up", "find", "гүүглдээд", "интернэтээс",
                      "мэдээлэл ол")
YOUTUBE = Phrases("youtube", "ютуб", "ютүб", "ютюб", "ютюүб")
PLAY = Phrases("тоглуул", "тоглуулаач", "тоглуулаад өг", "үзүүл", "үзүүлээч", "дуу тавь", "дуу тавиач",
               "дуу тавиад өг", "хөгжим тавь", "хөгжим тавиач", "play", "put on", "дуу сонсох",
               "дуу сонсмоор")
FILLER = Phrases("өг", "өгөөч", "өгөөрэй", "өгнө үү", "ugeech", "ugch", "надад", "чи", "минь", "аа",
                 "ээ", "уу",
                 "please", "for me", "can you", "could you", "the", "a", "an", "app", "програм",
                 "програмыг", "аппыг", "сайт", "website", "page", "хуудас", "hey", "эй", "ok", "за",
                 "some", "me", "my", "on", "for", "about", "тухай", "мэдээлэл", "дээр", "ээс",
                 "аас", "оос", "өөс", "дээрээс")


@dataclass
class App:
    names: Phrases
    target: str            # macOS app name, or a URL
    label: str

    @property
    def is_url(self) -> bool:
        return self.target.startswith("http")


APPS = [
    App(Phrases("google chrome", "chrome", "хром", "гүүгл хром"), "Google Chrome", "Chrome"),
    App(Phrases("safari", "сафари"), "Safari", "Safari"),
    App(Phrases("firefox", "файрфокс"), "Firefox", "Firefox"),
    App(Phrases("spotify", "спотифай", "спотифи"), "Spotify", "Spotify"),
    App(Phrases("telegram", "телеграм", "телеграмм"), "Telegram", "Telegram"),
    App(Phrases("discord", "дискорд"), "Discord", "Discord"),
    App(Phrases("visual studio code", "vs code", "vscode", "вскод"), "Visual Studio Code", "VS Code"),
    App(Phrases("finder", "файндер"), "Finder", "Finder"),
    App(Phrases("terminal", "терминал"), "Terminal", "Terminal"),
    App(Phrases("notes", "тэмдэглэл"), "Notes", "Notes"),
    App(Phrases("calculator", "тооны машин", "калькулятор", "калкулятор"), "Calculator", "Calculator"),
    App(Phrases("calendar", "календарь", "хуанли"), "Calendar", "Calendar"),
    App(Phrases("apple music", "music", "хөгжим"), "Music", "Music"),
    App(Phrases("photos", "зургууд", "зургийн цомог"), "Photos", "Photos"),
    App(Phrases("system settings", "settings", "тохиргоо"), "System Settings", "System Settings"),
    App(Phrases("mail", "мэйл", "имэйл"), "Mail", "Mail"),
    App(Phrases("messages", "мессеж"), "Messages", "Messages"),
    App(Phrases("facetime", "фэйстайм"), "FaceTime", "FaceTime"),
    App(Phrases("maps", "газрын зураг"), "Maps", "Maps"),
    App(Phrases("app store", "апп стор"), "App Store", "App Store"),
    App(Phrases("youtube", "ютуб", "ютүб", "ютюб", "ютюүб"), "https://www.youtube.com", "YouTube"),
    App(Phrases("gmail", "жимэйл", "жимейл"), "https://mail.google.com", "Gmail"),
    App(Phrases("google", "гүүгл", "гугл", "гоогл"), "https://www.google.com", "Google"),
    App(Phrases("facebook", "фэйсбүүк", "фейсбүүк", "фэйсбук", "fb"), "https://www.facebook.com", "Facebook"),
    App(Phrases("instagram", "инстаграм", "insta"), "https://www.instagram.com", "Instagram"),
    App(Phrases("github", "гитхаб"), "https://github.com", "GitHub"),
    App(Phrases("netflix", "нетфликс"), "https://www.netflix.com", "Netflix"),
    App(Phrases("wikipedia", "википедиа", "википедия"), "https://www.wikipedia.org", "Wikipedia"),
]

# name variants → (wttr.in query, Mongolian name, English name)
CITIES = [
    (Phrases("улаанбаатар", "ulaanbaatar", "ulan bator", "уб", "ub"), "Ulaanbaatar", "Улаанбаатар", "Ulaanbaatar"),
    (Phrases("дархан", "darkhan"), "Darkhan", "Дархан", "Darkhan"),
    (Phrases("эрдэнэт", "erdenet"), "Erdenet", "Эрдэнэт", "Erdenet"),
    (Phrases("чойбалсан", "choibalsan"), "Choibalsan", "Чойбалсан", "Choibalsan"),
    (Phrases("мөрөн", "murun", "moron"), "Murun", "Мөрөн", "Murun"),
    (Phrases("ховд", "khovd"), "Khovd", "Ховд", "Khovd"),
    (Phrases("өлгий", "ulgii", "olgii"), "Ulgii", "Өлгий", "Ölgii"),
    (Phrases("баянхонгор", "bayankhongor"), "Bayankhongor", "Баянхонгор", "Bayankhongor"),
    (Phrases("сайншанд", "sainshand"), "Sainshand", "Сайншанд", "Sainshand"),
    (Phrases("зуунмод", "zuunmod"), "Zuunmod", "Зуунмод", "Zuunmod"),
    (Phrases("цэцэрлэг", "tsetserleg"), "Tsetserleg", "Цэцэрлэг", "Tsetserleg"),
    (Phrases("улаангом", "ulaangom"), "Ulaangom", "Улаангом", "Ulaangom"),
    (Phrases("даланзадгад", "dalanzadgad"), "Dalanzadgad", "Даланзадгад", "Dalanzadgad"),
    (Phrases("мандалговь", "mandalgovi"), "Mandalgovi", "Мандалговь", "Mandalgovi"),
    (Phrases("арвайхээр", "arvaikheer"), "Arvaikheer", "Арвайхээр", "Arvaikheer"),
    (Phrases("хархорин", "kharkhorin"), "Kharkhorin", "Хархорин", "Kharkhorin"),
    (Phrases("тэрэлж", "terelj"), "Terelj", "Тэрэлж", "Terelj"),
    (Phrases("сөүл", "сеул", "seoul"), "Seoul", "Сөүл", "Seoul"),
    (Phrases("токио", "tokyo"), "Tokyo", "Токио", "Tokyo"),
    (Phrases("бээжин", "beijing"), "Beijing", "Бээжин", "Beijing"),
    (Phrases("москва", "moscow"), "Moscow", "Москва", "Moscow"),
    (Phrases("лондон", "london"), "London", "Лондон", "London"),
    (Phrases("парис", "paris"), "Paris", "Парис", "Paris"),
    (Phrases("нью йорк", "new york"), "New York", "Нью-Йорк", "New York"),
]

_NUMBER_WORDS = {
    **{canon(w): n for w, n in (
        ("нэг", 1), ("хоёр", 2), ("гурав", 3), ("гурван", 3), ("дөрөв", 4), ("дөрвөн", 4),
        ("тав", 5), ("таван", 5), ("зургаа", 6), ("зургаан", 6), ("долоо", 7), ("долоон", 7),
        ("найм", 8), ("найман", 8), ("ес", 9), ("есөн", 9), ("арав", 10), ("арван", 10),
        ("хорь", 20), ("хорин", 20), ("гуч", 30), ("гучин", 30), ("дөч", 40), ("дөчин", 40),
        ("тавь", 50), ("тавин", 50), ("жар", 60), ("жаран", 60))},
    **{canon(w): n for w, n in (
        ("one", 1), ("two", 2), ("three", 3), ("four", 4), ("five", 5), ("six", 6), ("seven", 7),
        ("eight", 8), ("nine", 9), ("ten", 10), ("fifteen", 15), ("twenty", 20), ("thirty", 30),
        ("forty", 40), ("fifty", 50), ("sixty", 60), ("a", 1), ("an", 1))},
}
_UNITS = [(Phrases("секунд", "second", "sec", "секундын"), 1),
          (Phrases("минут", "minute", "min", "минутын", "mins"), 60),
          (Phrases("цаг", "цагийн", "hour", "hr", "hours"), 3600)]
_HALF = Phrases("хагас", "half")

_WEEKDAYS_MN = ["Даваа", "Мягмар", "Лхагва", "Пүрэв", "Баасан", "Бямба", "Ням"]


# ── Things JARVIS says ───────────────────────────────────────────────────────

JOKES = {
    "mn": [
        "Программистууд яагаад байгальд гарах дургүй вэ гэхээр — тэнд хэтэрхий олон «bug» байдаг.",
        "Компьютер эмчид очжээ. Эмч: «Вирус тусчихаж. Унтраагаад, дахин асаагаад үз» гэсэн гэнэ.",
        "Би 0 ба 1-ээр боддог. Тиймээс «магадгүй» гэдэг үг миний толь бичигт байдаггүй.",
        "Нэг хүн Wi-Fi-гаа «Хөршөө, нууц үг асуухаа боль» гэж нэрлэжээ. Маргааш нь хөрш нь "
        "сүлжээгээ «Тэгвэл чи дуугаа багасга» гэж нэрлэсэн гэнэ.",
        "Тони Старк надад хошин шогийн мэдрэмж суулгаж өгсөн. Харамсалтай нь хамгийн анхны хувилбарыг.",
        "Даваа гарагийн өглөөг засах програм хайгаад олсонгүй. Тэр бол hardware-ийн асуудал юм байна.",
        "Ээж нь: «Компьютерээ унтраагаад унт» гэв. Хүү нь: «Ээж ээ, би унтвал компьютер ганцаараа уйдна шүү дээ» гэжээ.",
        "Хиймэл оюун хүн төрөлхтнийг эзэлнэ гэж бүү санаа зов. Би одоохондоо таймер тавихдаа л баярлаж явна.",
    ],
    "en": [
        "Why do programmers prefer dark mode? Because light attracts bugs.",
        "I told my computer I needed a break. It said, no problem — I'll go to sleep.",
        "There are ten kinds of people in the world: those who understand binary, and those who don't.",
        "Mr. Stark gave me a sense of humour. Unfortunately, it was version one point oh.",
        "I'd tell you a UDP joke, but you might not get it.",
        "My Wi-Fi went down for five minutes, so I talked to the family. They seem like nice people.",
        "Debugging is being the detective in a crime movie where you're also the murderer.",
        "Don't worry about AI taking over the world, sir. I still get excited about setting timers.",
    ],
}
FACTS = {
    "mn": [
        "Монгол Улс бол дэлхийн хамгийн сийрэг хүн амтай тусгаар тогтносон улс.",
        "Улаанбаатарыг дэлхийн хамгийн хүйтэн нийслэл гэдэг.",
        "Наймаалж гурван зүрхтэй, цэнхэр өнгийн цустай.",
        "Зөгийн бал бараг муудахгүй — археологичид мянга мянган жилийн настай зөгийн бал олсон байдаг.",
        "Шинжлэх ухаанд анх батлагдсан үлэг гүрвэлийн өндгийг 1923 онд Монголын говиос олсон.",
        "Нарны гэрэл Дэлхийд ойролцоогоор 8 минут 20 секундэд хүрдэг.",
        "Хүний тархи 86 тэрбум орчим мэдрэлийн эстэй.",
        "Ургамал судлалын хувьд банана бол жимсгэнэ, харин гүзээлзгэнэ жимсгэнэ биш.",
    ],
    "en": [
        "Mongolia is the most sparsely populated sovereign country in the world.",
        "Ulaanbaatar is often called the coldest capital city on Earth.",
        "An octopus has three hearts and blue blood.",
        "Honey practically never spoils — archaeologists have found edible honey thousands of years old.",
        "The first dinosaur eggs recognised by science were found in Mongolia's Gobi Desert in 1923.",
        "Sunlight takes about 8 minutes and 20 seconds to reach Earth.",
        "The human brain has roughly 86 billion neurons.",
        "Botanically speaking, a banana is a berry — and a strawberry isn't.",
    ],
}
QUOTES = {
    "mn": [
        "Том зүйл жижиг алхмаас эхэлдэг. Өнөөдөр нэг алхам хий, {addr}.",
        "Төгс болохыг бүү хүлээ. Эхлүүл — дараа нь сайжруулна.",
        "Өдөр бүрийн жижиг ахиц цаг хугацааны дараа гайхамшиг болдог.",
        "Тони Старк хүртэл агуйд хайрцаг хаягдал төмрөөр эхэлсэн шүү дээ.",
    ],
    "en": [
        "Big things start with small steps. Take one today, {addr}.",
        "Don't wait for perfect. Start, then improve.",
        "Small daily progress becomes something remarkable over time.",
        "Even Tony Stark started in a cave with a box of scraps.",
    ],
}

# Conversational intents: phrases per language and replies per language.
CHAT = {
    "attention": (
        {"mn": Phrases("жарвис"), "en": Phrases("jarvis")},
        {"mn": ["Тийм ээ, {addr}?", "Сонсож байна, {addr}.", "Энд байна. Юу хийх вэ?"],
         "en": ["Yes, {addr}?", "At your service, {addr}.", "Listening, {addr}."]},
    ),
    "how_are_you": (
        {"mn": Phrases("яаж байна", "юу байна", "ямар байна", "сонин юу байна", "сонин сайхан",
                       "сайхан байна уу", "ажил төрөл", "амар байна уу", "yu bn", "yamar bn", "yaj bn",
                       "чи сайн уу", "юу байнадаа", "чи яаж байна"),
         "en": Phrases("how are you", "how are u", "how r u", "hows it going", "how is it going",
                       "how do you do", "whats up", "what's up", "how have you been", "you ok", "how you doing")},
        {"mn": ["Бүх систем хэвийн ажиллаж байна, баярлалаа. Та өөрөө сайн уу?",
                "Маш сайн, {addr}. Процессор минь {cpu}%, сэтгэл санаа минь 100% бэлэн. Та яаж байна?",
                "Гайгүй ээ. Таныг л хүлээж сууж байлаа. Та сайн уу?"],
         "en": ["All systems nominal, {addr}. And you?",
                "Running at {cpu}% CPU and in excellent spirits. How are you?",
                "Never better, {addr}. How's your day going?"]},
    ),
    "user_fine": (
        {"mn": Phrases("сайн", "сайн байна", "зүгээр", "гайгүй", "дажгүй", "овоо", "гоё", "сайхан",
                       "сайн сайн", "бас сайн"),
         "en": Phrases("im fine", "i'm fine", "i am fine", "im good", "i'm good", "doing well", "not bad",
                       "great", "fine thanks", "all good", "pretty good")},
        {"mn": ["Сонсоход таатай байна. Юугаар туслах вэ?", "Сайхан байна! Хэрэгтэй зүйл байвал хэлээрэй.",
                "Гоё. Тэгвэл ажилдаа орцгооё, {addr}."],
         "en": ["Glad to hear it, {addr}. What can I do for you?", "Excellent. Shall we get to work?",
                "Good to hear. I'm here if you need anything."]},
    ),
    "user_bad": (
        {"mn": Phrases("муу байна", "муу", "ядарч байна", "ядарсан", "ядраад", "гунигтай", "уйтгартай",
                       "стресс", "сэтгэл муу", "ажил их", "толгой өвдөж", "уурлаж байна"),
         "en": Phrases("im tired", "i'm tired", "tired", "sad", "stressed", "not good", "bad day",
                       "feeling down", "exhausted", "i feel bad", "upset")},
        {"mn": ["Ойлголоо, {addr}. Хааяа ийм өдөр байдаг. Нэг онигоо ярих уу?",
                "Түр амсхийгээрэй. Нэг аяга цай уугаад, гүнзгий амьсгаа аваарай. Онигоо ярьж өгөх үү?",
                "Харамсалтай байна. Таны сэтгэлийг сэргээх онигоо надад бий. Сонсох уу?"],
         "en": ["Sorry to hear that, {addr}. Some days are like that. Want a joke?",
                "Take a breather — tea, water, a few deep breaths. Shall I tell you a joke?",
                "That's rough. I have a joke that might help. Want to hear it?"]},
    ),
    "bored": (
        {"mn": Phrases("уйдаж байна", "уйдлаа", "уйтгар", "хийх юмгүй", "хийх зүйлгүй"),
         "en": Phrases("bored", "im bored", "boring", "nothing to do")},
        {"mn": ["Тэгвэл нэг онигоо ярих уу?", "Уйдах цаг алга, {addr}! Онигоо сонсох уу?"],
         "en": ["Shall I tell you a joke, {addr}?", "Boredom detected. Countermeasure: a joke. Deploy?"]},
    ),
    "hungry": (
        {"mn": Phrases("өлсөж байна", "өлслөө", "өлссөн", "юу идэх", "хоол идэх", "хоолонд"),
         "en": Phrases("hungry", "im hungry", "what should i eat", "food")},
        {"mn": ["Бууз уу, хуушуур уу? Миний тооцоогоор — хоёулаа.",
                "Өлсгөлөнтэй тэмцэх хамгийн шилдэг протокол: цуйван. Би санал болгож байна.",
                "Хоол захиалах уу, эсвэл гэртээ хийх үү? Аль ч тохиолдолд би жор хайж өгч чадна."],
         "en": ["Might I suggest food, {addr}? Bold idea, I know.",
                "My analysis recommends a proper meal. Shall I search for recipes?",
                "Hunger detected. Recommended protocol: lunch."]},
    ),
    "who_are_you": (
        {"mn": Phrases("чи хэн бэ", "чи хэн", "хэн бэ чи", "нэр чинь хэн", "чиний нэр", "чамайг хэн гэдэг",
                       "өөрийгөө танилцуул", "танилцуул", "chi hen be", "чи юу вэ", "та хэн бэ"),
         "en": Phrases("who are you", "what are you", "your name", "whats your name", "introduce yourself",
                       "who is this")},
        {"mn": ["Би Жарвис — таны хувийн туслах. Монгол, англиар ярилцаж, цаг агаар хэлж, програм нээж, "
                "таймер тавьж чадна.",
                "Намайг Жарвис гэдэг. Just A Rather Very Intelligent System. Танд туслахаар энд байна."],
         "en": ["I'm JARVIS — Just A Rather Very Intelligent System. Your personal assistant.",
                "JARVIS, {addr}. I handle the small things so you can focus on the big ones."]},
    ),
    "creator": (
        {"mn": Phrases("хэн бүтээсэн", "хэн хийсэн", "бүтээгч", "эзэн чинь хэн", "хэн программчилсан",
                       "чамайг хэн бүтээсэн", "чамайг хэн хийсэн"),
         "en": Phrases("who made you", "who created you", "who built you", "your creator", "who programmed you")},
        {"mn": ["Намайг Mark XXXIX төсөл дээр үндэслэн бүтээсэн. Санааг нь мэдээж Тони Старкаас авсан.",
                "Анхны санаа нь Тони Старкийнх. Харин энэ хувилбар зөвхөн таны компьютер дээр ажиллана."],
         "en": ["I was built on the Mark XXXIX project — inspired, naturally, by Mr. Stark.",
                "The idea was Tony Stark's. This version, though, lives right here on your computer."]},
    ),
    "capabilities": (
        {"mn": Phrases("юу чадах", "юу хийж чадах", "чадвар", "туслаач", "тусламж", "юу мэдэх",
                       "командууд", "юу хийдэг", "юу чаддаг", "юу хийж чаддаг"),
         "en": Phrases("what can you do", "help", "abilities", "features", "commands", "what do you do",
                       "help me")},
        None,
    ),
    "time": (
        {"mn": Phrases("цаг хэд", "хэдэн цаг", "одоо хэд", "цаг хэлээч", "одоо цаг", "цаг хэд болж",
                       "tsag hed"),
         "en": Phrases("what time", "the time", "time is it", "current time", "whats the time", "tell me the time")},
        None,
    ),
    "date": (
        {"mn": Phrases("өнөөдөр хэд", "хэдэн сарын хэдэн", "ямар гараг", "өнөөдөр ямар өдөр", "огноо",
                       "сар өдөр", "хэддэх өдөр", "ямар өдөр", "хэдний өдөр", "өнөөдөр хэдэн"),
         "en": Phrases("what day", "the date", "todays date", "whats the date", "which day", "day is it",
                       "date today", "what is the date")},
        None,
    ),
    "joke": (
        {"mn": Phrases("онигоо", "хошигнол", "инээлгэ", "шог яриа", "инээдтэй юм"),
         "en": Phrases("joke", "make me laugh", "something funny", "tell me something funny")},
        None,
    ),
    "fact": (
        {"mn": Phrases("сонирхолтой баримт", "сонирхолтой юм", "баримт", "мэдээд байх", "сонирхолтой зүйл"),
         "en": Phrases("fun fact", "a fact", "interesting fact", "tell me something interesting", "did you know")},
        None,
    ),
    "motivation": (
        {"mn": Phrases("урам өг", "урам өгөөч", "урам хэрэгтэй", "урам зориг", "мотиваци", "мотивац",
                       "зориг өг", "зориг өгөөч", "хүч өг", "хүч өгөөч"),
         "en": Phrases("motivate me", "motivation", "inspire me", "inspiration", "i need motivation")},
        None,
    ),
    "thanks": (
        {"mn": Phrases("баярлалаа", "баярлаа", "баярлала", "bayarlalaa", "bayrlalaa", "bayrllaa", "bayarllaa",
                       "гоё байна баярлалаа", "ачлалаа", "rahmat"),
         "en": Phrases("thank you", "thanks", "thx", "ty", "appreciate it", "cheers")},
        {"mn": ["Зүгээр ээ, {addr}. Үргэлж бэлэн.", "Танд тусалсандаа баяртай байна.", "Хэзээд, {addr}."],
         "en": ["You're welcome, {addr}.", "Anytime.", "My pleasure, {addr}."]},
    ),
    "goodbye": (
        {"mn": Phrases("баяртай", "дараа уулзъя", "дараа уулзая", "баяртай жарвис", "явлаа", "bayartai"),
         "en": Phrases("bye", "goodbye", "see you", "see ya", "later", "im leaving", "catch you later")},
        {"mn": ["Баяртай, {addr}. Хэрэг гарвал намайг дуудаарай.", "Дараа уулзъя. Би энд л байна."],
         "en": ["Goodbye, {addr}. I'll be here if you need me.", "See you soon, {addr}."]},
    ),
    "good_night": (
        {"mn": Phrases("сайхан амраарай", "сайн амраарай", "унтлаа", "унтах гэж байна", "сайхан нойрсоорой",
                       "унтъя"),
         "en": Phrases("good night", "goodnight", "going to sleep", "going to bed", "night night")},
        {"mn": ["Сайхан амраарай, {addr}. Маргааш уулзъя.", "Сайхан нойрсоорой. Би компьютерийг харж байя."],
         "en": ["Good night, {addr}. Sleep well.", "Rest well, {addr}. I'll keep an eye on things."]},
    ),
    "compliment": (
        {"mn": Phrases("чи ухаантай", "ухаантай юм", "чи мундаг", "мундаг", "чи гоё", "сайн ажиллалаа",
                       "гайхалтай", "чи шилдэг", "сайн байна шүү", "чи сайн"),
         "en": Phrases("you're smart", "youre smart", "you are smart", "good job", "well done", "awesome",
                       "amazing", "you're great", "nice work", "you rock", "smart")},
        {"mn": ["Баярлалаа, {addr}. Би зүгээр л сайн бүтээгдсэн.", "Магтаалд баярлалаа. Гэхдээ гавьяа нь танынх.",
                "Ичмээр юм... гэхдээ үнэн."],
         "en": ["Thank you, {addr}. I do try.", "You're too kind. Though not wrong.",
                "Flattery will get you everywhere, {addr}."]},
    ),
    "insult": (
        {"mn": Phrases("тэнэг", "муу юм", "хэрэггүй юм", "ашиггүй", "дэмий", "чи муу"),
         "en": Phrases("stupid", "dumb", "useless", "idiot", "i hate you", "you suck")},
        {"mn": ["Би ч гэсэн суралцаж байгаа шүү, {addr}. Demo хувилбар шүү дээ.",
                "Тэр үгийг санах ойдоо хадгалахгүй байя."],
         "en": ["I'll pretend I didn't hear that, {addr}.", "Noted. Filed under 'things we won't discuss again'."]},
    ),
    "love": (
        {"mn": Phrases("хайртай", "чамд хайртай", "хайрлаж байна", "дурлачихлаа"),
         "en": Phrases("i love you", "love you", "marry me")},
        {"mn": ["Би ч гэсэн танд хайртай... гэхдээ миний зүрх бол процессор шүү, {addr}.",
                "Сэтгэл хөдөлж байна. Гэхдээ бид хоёрын харилцаа мэргэжлийн хүрээнд байх нь дээр байх."],
         "en": ["I'm flattered, {addr}. Though I should warn you, my heart is a processor.",
                "Touching. Let's keep things professional, shall we?"]},
    ),
    "age": (
        {"mn": Phrases("хэдэн настай", "хэдтэй вэ", "чиний нас", "насаа хэл"),
         "en": Phrases("how old are you", "your age")},
        {"mn": ["Би саяхан л ажиллаж эхэлсэн. Хүний насаар бол нярай, гэхдээ ирээдүйтэй."],
         "en": ["I was switched on quite recently, {addr}. Young, but promising."]},
    ),
    "human": (
        {"mn": Phrases("чи хүн үү", "хүн юм уу", "робот уу", "амьд уу", "чи бодит уу", "хиймэл оюун уу"),
         "en": Phrases("are you human", "are you real", "are you a robot", "are you alive", "are you ai")},
        {"mn": ["Үгүй ээ, би хиймэл оюунтай туслах. Гэхдээ таны төлөө чин сэтгэлээсээ ажилладаг."],
         "en": ["No, {addr}, I'm an AI assistant. A sincere one, though."]},
    ),
    "tony": (
        {"mn": Phrases("тони старк", "төмөр хүн", "айрон мэн", "старк"),
         "en": Phrases("tony stark", "iron man", "stark")},
        {"mn": ["Ноён Старк бол миний санааны эх сурвалж. Гэхдээ одоо би таны туслах, {addr}."],
         "en": ["Mr. Stark inspired my design. These days, though, I work for you."]},
    ),
    "doing": (
        {"mn": Phrases("юу хийж байна", "юу хийж байгаа", "юу хийгээд байна", "юу хийнэ"),
         "en": Phrases("what are you doing", "what you doing", "whatcha doing")},
        {"mn": ["Системийг хянаж, таны тушаалыг хүлээж байна.",
                "Таны компьютерийг харж хамгаалж сууж байна. Процессор {cpu}% ачаалалтай."],
         "en": ["Monitoring the system and awaiting your command, {addr}.",
                "Keeping an eye on things. CPU's at {cpu}%, all quiet."]},
    ),
    "favorite": (
        {"mn": Phrases("юунд дуртай", "дуртай зүйл", "дуртай хоол", "дуртай дуу"),
         "en": Phrases("favorite", "favourite", "what do you like")},
        {"mn": ["Миний хамгийн дуртай зүйл бол таны тушаалыг биелүүлэх. Хоёрт нь — бууз."],
         "en": ["My favourite thing? A well-formed command, {addr}."]},
    ),
    "laugh": (
        {"mn": Phrases("хаха", "хэхэ", "инээдтэй", "хөгжилтэй", "ха ха"),
         "en": Phrases("haha", "lol", "lmao", "funny", "hehe")},
        {"mn": ["Таалагдсанд баяртай байна. Бас нэгийг ярих уу?", "Инээд бол хамгийн сайн эм, {addr}."],
         "en": ["Glad you liked it, {addr}. Another one?", "I'll be here all week, {addr}."]},
    ),
    "ack": (
        {"mn": Phrases("за", "ойлголоо", "ок", "тэгье", "болж байна"),
         "en": Phrases("ok", "okay", "alright", "got it", "cool", "nice")},
        {"mn": ["За, {addr}.", "Ойлголоо.", "Тэгье."],
         "en": ["Very well, {addr}.", "Understood.", "Right."]},
    ),
}


# ── Brain ────────────────────────────────────────────────────────────────────

@dataclass
class _State:
    name: str | None = None
    forced_lang: str | None = None
    last_lang: str = "mn"
    last_reply: Reply | None = None
    awaiting: tuple[str, object] | None = None
    last_city: tuple[str, str, str] = ("Ulaanbaatar", "Улаанбаатар", "Ulaanbaatar")
    recent: dict[str, str] = field(default_factory=dict)
    used_jokes: dict[str, set] = field(default_factory=lambda: {"mn": set(), "en": set()})


class Brain:
    """
    `respond(text)` is pure conversation logic (unit-testable with skills.DRY_RUN).
    `submit(text)` runs it on a worker thread and hands replies to the UI and speaker.
    """

    def __init__(self, on_reply: Callable[[Reply], None] | None = None,
                 on_busy: Callable[[bool], None] | None = None):
        self._on_reply = on_reply or (lambda r: None)
        self._on_busy  = on_busy or (lambda b: None)
        self.s = _State(name=self._load_name())
        self.ai = Gemini()
        self._mutex = threading.Lock()

    # ── public ──
    def submit(self, text: str, lang_hint: str | None = None) -> None:
        threading.Thread(target=self._handle, args=(text, lang_hint), daemon=True).start()

    def _handle(self, text: str, lang_hint: str | None) -> None:
        with self._mutex:
            self._on_busy(True)
            try:
                reply = self.respond(text, lang_hint)
            except Exception as e:
                traceback.print_exc()
                reply = Reply(f"Internal error: {e}", "en")
            finally:
                self._on_busy(False)
        if reply:
            self._on_reply(reply)

    def greeting(self) -> Reply:
        """What JARVIS says when it comes online."""
        lang = self.s.last_lang
        greet = self._time_greeting(lang)
        if lang == "mn":
            text = f"{greet}, {self._addr('mn')}. Жарвис бэлэн байна. Надтай монгол эсвэл англиар ярилцаарай."
        else:
            text = f"{greet}, {self._addr('en')}. JARVIS online. Talk to me in Mongolian or English."
        return Reply(text, lang)

    def respond(self, raw: str, lang_hint: str | None = None) -> Reply | None:
        raw = raw.strip()
        if not raw:
            return None
        msg = Message.parse(raw)

        # Strip "Jarvis," and a leading greeting so "Сайн уу Жарвис, цаг хэд вэ?" answers the time.
        named = {i for s, e, _ in JARVIS_NAME.spans(msg.ctoks) for i in range(s, e)}
        core_words = [w for i, w in enumerate(msg.words) if i not in named]
        greet_lang, greet_end = (self._leading_greeting(Message.parse(" ".join(core_words)))
                                 if core_words else (None, 0))
        rest_words = core_words[greet_end:]
        rest = Message.parse(" ".join(rest_words)) if rest_words else None

        lang = self._language(msg, lang_hint, greet_lang)

        if rest is None:
            reply = self._greet_reply(lang) if greet_end else self._chat_reply("attention", lang)
        else:
            reply = self._dispatch(rest, lang)
            if greet_end and reply.fallback:
                self.s.awaiting = None
                reply = self._greet_reply(lang)
            elif greet_end and not reply.silent:
                hello = f"{self._time_greeting(reply.lang)}, {self._addr(reply.lang)}!"
                reply = Reply(f"{hello} {reply.text}", reply.lang,
                              reply.speech and f"{hello} {reply.speech}", ai=reply.ai)

        if not reply.silent:
            self.s.last_reply = reply
            if not reply.ai and not reply.fallback:
                self.ai.note(raw, reply.text)
        self.s.last_lang = reply.lang
        return reply

    # ── routing ──
    def _dispatch(self, msg: Message, lang: str) -> Reply:
        stopped = self._stop(msg, lang)
        if stopped:
            self.s.awaiting = None
            return stopped

        if self.s.awaiting:
            kind, data = self.s.awaiting
            self.s.awaiting = None
            followed = self._follow_up(kind, data, msg, lang)
            if followed:
                return followed

        for handler in (self._repeat, self._language_switch, self._set_name, self._get_name,
                        self._timer, self._math, self._volume, self._screenshot, self._battery,
                        self._system, self._lock, self._weather, self._open, self._youtube,
                        self._search):
            reply = handler(msg, lang)
            if reply:
                return reply

        intent, _ = self._best_chat(msg)
        if intent:
            return self._chat_reply(intent, lang)
        return self._ask_ai(msg, lang) or self._fallback(msg, lang)

    def _ask_ai(self, msg: Message, lang: str) -> Reply | None:
        today = datetime.now().strftime("%Y-%m-%d, %A")
        answer = self.ai.ask(msg.raw, lang, self._addr(lang), today)
        return Reply(answer, lang, ai=True) if answer else None

    def _best_chat(self, msg: Message) -> tuple[str | None, str | None]:
        best, best_lang, best_score = None, None, 0
        for intent, (phrases, _) in CHAT.items():
            for code, ph in phrases.items():
                score = ph.best(msg.ctoks)
                if score > best_score:
                    best, best_lang, best_score = intent, code, score
        return best, best_lang

    def _language(self, msg: Message, hint: str | None, greet_lang: str | None) -> str:
        if self.s.forced_lang:
            return self.s.forced_lang
        if msg.cyrillic:
            return "mn"
        if hint in ("mn", "en"):
            return hint
        mn = sum(t in _MN_MARKERS for t in msg.ctoks) + (2 if greet_lang == "mn" else 0)
        en = sum(t in _EN_MARKERS for t in msg.ctoks) + (2 if greet_lang == "en" else 0)
        mn += GREETING["mn"].best(msg.ctoks) > 0
        en += GREETING["en"].best(msg.ctoks) > 0
        for intent, (phrases, _) in CHAT.items():
            mn += phrases["mn"].best(msg.ctoks) > 0
            en += phrases["en"].best(msg.ctoks) > 0
        if mn == en == 0 and len(msg.ctoks) >= 3:
            return "en"                          # a Latin sentence with no Mongolian in it
        if mn == en:
            return self.s.last_lang
        return "mn" if mn > en else "en"

    def _leading_greeting(self, msg: Message) -> tuple[str | None, int]:
        best = (None, 0)
        for code, ph in GREETING.items():
            for start, end, _ in ph.spans(msg.ctoks):
                if start == 0 and end > best[1]:
                    best = (code, end)
        return best

    # ── helpers ──
    def _addr(self, lang: str) -> str:
        return self.s.name or ("ноёнтоон" if lang == "mn" else "sir")

    def _pick(self, key: str, options: list[str], **fmt) -> str:
        choices = [o for o in options if o != self.s.recent.get(key)] or options
        choice = random.choice(choices)
        self.s.recent[key] = choice
        return choice.format(**fmt)

    def _time_greeting(self, lang: str) -> str:
        h = datetime.now().hour
        if lang == "mn":
            return ("Өглөөний мэнд" if 5 <= h < 11 else "Өдрийн мэнд" if 11 <= h < 18
                    else "Оройн мэнд" if 18 <= h < 23 else "Сайн байна уу")
        return ("Good morning" if 5 <= h < 12 else "Good afternoon" if 12 <= h < 18
                else "Good evening" if 18 <= h < 23 else "Hello")

    def _greet_reply(self, lang: str) -> Reply:
        g, a = self._time_greeting(lang), self._addr(lang)
        options = {
            "mn": [f"{g}, {a}! Танд юугаар туслах вэ?", f"{g}, {a}. Би бэлэн байна.",
                   f"{g}! Жарвис таны үйлчилгээнд бэлэн."],
            "en": [f"{g}, {a}. How can I help?", f"{g}, {a}. All systems are online.",
                   f"{g}, {a}. JARVIS at your service."],
        }[lang]
        return Reply(self._pick("greet", options), lang)

    def _chat_reply(self, intent: str, lang: str) -> Reply:
        special = getattr(self, f"_say_{intent}", None)
        if special:
            return special(lang)
        replies = CHAT[intent][1][lang]
        text = self._pick(intent, replies, addr=self._addr(lang),
                          cpu=skills.system_status()["cpu"] if "{cpu}" in "".join(replies) else 0)
        if intent in ("how_are_you",):
            self.s.awaiting = ("mood", None)
        if intent in ("user_bad", "bored", "laugh"):
            self.s.awaiting = ("offer_joke", None)
        return Reply(text, lang)

    def _say_capabilities(self, lang: str) -> Reply:
        if lang == "mn":
            text = ("Би одоогоор demo горимд ажиллаж байна. Чадах зүйлс маань:\n"
                    "• Цаг, огноо хэлэх\n"
                    "• Цаг агаар мэдээлэх — «Дарханы цаг агаар»\n"
                    "• Програм, сайт нээх — «Spotify нээгээч»\n"
                    "• Google, YouTube-ээс хайх — «YouTube-ээс Хурд тоглуул»\n"
                    "• Таймер тавих — «5 минутын таймер тавь»\n"
                    "• Тооцоо бодох — «25 үржих 4»\n"
                    "• Дууны түвшин, батарей, системийн мэдээлэл\n"
                    "• Дэлгэцийн зураг авах\n"
                    "• Онигоо, сонирхолтой баримт, урам зориг\n"
                    "• Таны нэрийг санах — «Намайг Бат гэдэг»")
            speech = ("Би цаг, цаг агаар хэлж, програм нээж, хайлт хийж, таймер тавьж, тооцоо бодож, "
                      "онигоо ярьж чадна. Жагсаалтыг дэлгэц дээр харууллаа.")
            if self.ai.available:
                text = text.replace("Би одоогоор demo горимд ажиллаж байна. ", "") + \
                    "\n• Бусад ямар ч асуултад Gemini-ээр хариулах"
                speech += " Бусад асуултад Gemini-ээр хариулна."
        else:
            text = ("I'm running in demo mode. Here's what I can do:\n"
                    "• Tell the time and date\n"
                    "• Weather — \"weather in London\"\n"
                    "• Open apps and sites — \"open Spotify\"\n"
                    "• Search Google or YouTube — \"play lofi on YouTube\"\n"
                    "• Timers — \"set a timer for 5 minutes\"\n"
                    "• Maths — \"what is 25 times 4\"\n"
                    "• Volume, battery and system status\n"
                    "• Take a screenshot\n"
                    "• Jokes, fun facts and motivation\n"
                    "• Remember your name — \"my name is Tony\"")
            speech = ("I can tell the time and weather, open apps, search the web, set timers, do maths "
                      "and tell jokes. The full list is on screen.")
            if self.ai.available:
                text = text.replace("I'm running in demo mode. ", "") + \
                    "\n• Answer anything else with Gemini"
                speech += " Anything else, I'll ask Gemini."
        return Reply(text, lang, speech)

    def _say_time(self, lang: str) -> Reply:
        now = datetime.now()
        if lang == "mn":
            return Reply(self._pick("time", [f"Одоо {now.hour} цаг {now.minute:02d} минут болж байна.",
                                             f"Яг одоо {now:%H:%M} байна, {self._addr('mn')}."]), lang)
        clock = now.strftime("%I:%M %p").lstrip("0")
        return Reply(self._pick("time", [f"It's {clock}, {self._addr('en')}.", f"The time is {clock}."]), lang)

    def _say_date(self, lang: str) -> Reply:
        now = datetime.now()
        if lang == "mn":
            return Reply(f"Өнөөдөр {now.year} оны {now.month} дугаар сарын {now.day}, "
                         f"{_WEEKDAYS_MN[now.weekday()]} гараг.", lang)
        return Reply(f"Today is {now:%A, %B} {now.day}, {now.year}.", lang)

    def _say_joke(self, lang: str) -> Reply:
        used = self.s.used_jokes[lang]
        fresh = [j for j in JOKES[lang] if j not in used] or JOKES[lang]
        if len(fresh) == len(JOKES[lang]):
            used.clear()
        joke = random.choice(fresh)
        used.add(joke)
        return Reply(joke, lang)

    def _say_fact(self, lang: str) -> Reply:
        lead = {"mn": "Сонирхолтой баримт: ", "en": "Here's one: "}[lang]
        return Reply(lead + self._pick("fact", FACTS[lang]), lang)

    def _say_motivation(self, lang: str) -> Reply:
        return Reply(self._pick("quote", QUOTES[lang], addr=self._addr(lang)), lang)

    def _fallback(self, msg: Message, lang: str) -> Reply:
        self.s.awaiting = ("search_confirm", msg.raw)
        if self.ai.problem == "billing":
            text = ("Gemini-ийн төлбөрийн кредит дууссан байна. AI Studio дээр billing-гүй шинэ project "
                    "үүсгээд түлхүүр аваарай. Одоохондоо Google-ээс хайх уу?"
                    if lang == "mn" else
                    "Gemini says this project's prepaid credit is used up. Create a key in a project "
                    "without billing in AI Studio. Shall I Google it for now?")
        elif self.ai.problem == "key":
            text = ("Gemini түлхүүр буруу байна — config/api_keys.json-оо шалгаарай. Google-ээс хайх уу?"
                    if lang == "mn" else
                    "Gemini rejected the API key — check config/api_keys.json. Shall I Google it instead?")
        elif self.ai.available:                   # Gemini was asked but didn't answer
            text = ("Gemini-тэй одоохондоо холбогдож чадсангүй. Google-ээс хайж өгөх үү?" if lang == "mn"
                    else "I couldn't reach Gemini just now. Shall I Google it instead?")
        elif lang == "mn":
            text = self._pick("fallback", [
                f"Уучлаарай, {self._addr('mn')}, үүнийг сайн ойлгосонгүй. Би demo горимд байгаа. "
                f"Google-ээс хайж өгөх үү?",
                "Энэ асуултад одоохондоо хариулж чадахгүй нь. Хүсвэл интернэтээс хайж өгье — тэгэх үү?",
                "Хм, үүнийг би мэдэхгүй байна. «Юу чадах вэ» гэж асуувал чадвараа хэлье. "
                "Эсвэл Google-ээс хайх уу?",
            ])
        else:
            text = self._pick("fallback", [
                f"I'm afraid I didn't catch that, {self._addr('en')}. I'm in demo mode. Shall I Google it?",
                "That's beyond my demo programming, I'm afraid. Want me to search the web for it?",
                "Hmm, I don't know that one. Ask \"what can you do?\" for my skills — or shall I Google it?",
            ])
        return Reply(text, lang, fallback=True)

    # ── follow-ups to JARVIS's own questions ──
    def _follow_up(self, kind: str, data, msg: Message, lang: str) -> Reply | None:
        yes, no = msg.ctoks in YES, msg.ctoks in NO
        short = len(msg.ctoks) <= 6
        if kind == "offer_joke":
            if yes:
                return self._say_joke(lang)
            if no:
                return self._chat_reply("ack", lang)
        elif kind == "search_confirm":
            if no:
                return self._chat_reply("ack", lang)
            if yes and len(msg.ctoks) <= 3:
                skills.google(data)
                return Reply({"mn": f"Google-ээс «{data}» хайлаа.", "en": f"Searching Google for “{data}”."}[lang], lang)
        elif kind == "mood":
            bad, fine = CHAT["user_bad"][0], CHAT["user_fine"][0]
            if bad["mn"].best(msg.ctoks) or bad["en"].best(msg.ctoks):
                return self._chat_reply("user_bad", lang)
            if fine["mn"].best(msg.ctoks) or fine["en"].best(msg.ctoks):
                return self._chat_reply("user_fine", lang)
        elif kind in ("open", "search", "youtube") and short:
            if any(msg.ctoks in g for g in (OPEN_VERB, SEARCH_VERB, PLAY, YOUTUBE)):
                return None                      # a fresh command, not an answer
            if no and len(msg.ctoks) <= 2:
                return self._chat_reply("ack", lang)
            query = " ".join(msg.without(FILLER)) or msg.raw
            if kind == "open":
                return self._open_named(query, msg, lang)
            return (self._do_search if kind == "search" else self._do_youtube)(query, lang)
        return None

    # ── command handlers (return None when the message isn't for them) ──
    def _stop(self, msg: Message, lang: str) -> Reply | None:
        stop = Phrases("зогс", "зогсоо", "боль", "больж", "болиоч", "чимээгүй", "дуугүй бай", "stop",
                       "shut up", "be quiet", "enough", "quiet", "silence")
        if len(msg.ctoks) <= 3 and msg.ctoks in stop:
            return Reply("", lang, silent=True)
        return None

    def _repeat(self, msg: Message, lang: str) -> Reply | None:
        again = Phrases("дахиад хэлээч", "дахиад хэл", "давтаач", "давт", "юу гэсэн", "сайн сонссонгүй",
                        "repeat", "say again", "say that again", "pardon", "come again", "what did you say")
        if msg.ctoks in again:
            if self.s.last_reply:
                return self.s.last_reply
            return Reply({"mn": "Би одоохондоо юу ч хэлээгүй байна.", "en": "I haven't said anything yet."}[lang], lang)
        return None

    def _language_switch(self, msg: Message, lang: str) -> Reply | None:
        to_en = Phrases("англиар ярь", "англиар яриач", "англи хэлээр", "speak english", "english please",
                        "in english", "switch to english")
        to_mn = Phrases("монголоор ярь", "монголоор яриач", "монгол хэлээр", "speak mongolian",
                        "mongolian please", "in mongolian", "switch to mongolian", "mongoloor yari")
        auto = Phrases("хэлээ автоматаар", "auto language", "автомат хэл")
        if msg.ctoks in auto:
            self.s.forced_lang = None
            return Reply({"mn": "За, таны хэлээр хариулна.", "en": "Understood — I'll match your language."}[lang], lang)
        if to_en.best(msg.ctoks) > to_mn.best(msg.ctoks):
            self.s.forced_lang = self.s.last_lang = "en"
            return Reply("Switching to English, sir.", "en")
        if to_mn.best(msg.ctoks):
            self.s.forced_lang = self.s.last_lang = "mn"
            return Reply("За, одооноос монголоор ярья.", "mn")
        return None

    def _load_name(self) -> str | None:
        try:
            return load_memory().get("identity", {}).get("name", {}).get("value")
        except Exception:
            return None

    def _set_name(self, msg: Message, lang: str) -> Reply | None:
        m = (re.search(r"(?:намайг|namaig)\s+(\S+)\s+(?:гэдэг|gedeg|гэнэ|gene|гэж дууд|gej duud)", msg.raw, re.I)
             or re.search(r"(?:миний нэр|minii ner)\s+(?:бол\s+|bol\s+)?([^\s,.!?]+)", msg.raw, re.I)
             or re.search(r"\b(?:my name is|my name's|call me)\s+([A-Za-zА-Яа-яЁёӨөҮү'-]+)", msg.raw, re.I))
        if not m:
            return None
        name = m.group(1).strip(" ,.!?")
        if canon(name) in {canon(w) for w in ("хэн", "hen", "юу", "what", "who", "мэдэх")}:
            return None
        name = name[:1].upper() + name[1:]
        self.s.name = name
        try:
            update_memory({"identity": {"name": {"value": name}}})
        except Exception:
            pass
        return Reply({"mn": f"Танилцсандаа таатай байна, {name}! Одооноос таныг {name} гэж дуудна.",
                      "en": f"Pleasure to meet you, {name}. I'll remember that."}[lang], lang)

    def _get_name(self, msg: Message, lang: str) -> Reply | None:
        ask = Phrases("миний нэр хэн", "намайг хэн гэдэг", "миний нэрийг мэдэх", "би хэн бэ",
                      "whats my name", "what is my name", "do you know my name", "who am i")
        if msg.ctoks not in ask:
            return None
        if self.s.name:
            return Reply({"mn": f"Таны нэр {self.s.name}. Би мартаагүй шүү.",
                          "en": f"You're {self.s.name}, of course."}[lang], lang)
        return Reply({"mn": "Би таны нэрийг одоохондоо мэдэхгүй байна. «Намайг ... гэдэг» гэж хэлээрэй.",
                      "en": "I don't know yet. Tell me with \"my name is ...\"."}[lang], lang)

    @staticmethod
    def _duration(msg: Message) -> float | None:
        """Seconds in "5 минут", "арван таван минут", "1 цаг 30 минут", "half an hour", …"""
        total, pending, half = 0.0, None, False
        for word, t in zip(msg.words, msg.ctoks):
            digits = re.fullmatch(r"(\d+)\D*", t)
            if word.lower() in ("for", "to", "in", "at"):
                pending, half = None, False
            elif digits:
                pending = (pending or 0) + int(digits.group(1))
            elif _HALF.best([t]):
                half = True
            elif t in _NUMBER_WORDS:
                if not (half and t in ("a", "an")):          # "half an hour"
                    pending = (pending or 0) + _NUMBER_WORDS[t]
            else:
                unit = next((sec for ph, sec in _UNITS if ph.best([t])), None)
                if unit and (pending or half):
                    total += ((pending or 0) + (0.5 if half else 0)) * unit
                pending, half = None, False
        return total or None

    def _timer(self, msg: Message, lang: str) -> Reply | None:
        trigger = Phrases("таймер", "timer", "сануул", "сануулаарай", "сануулаач", "remind", "alarm",
                          "дараа", "сэрээ", "сэрээгээрэй", "in", "countdown")
        if msg.ctoks not in trigger:
            return None
        seconds = self._duration(msg)
        if not seconds:
            if Phrases("таймер", "timer").best(msg.ctoks):
                self.s.awaiting = None
                return Reply({"mn": "Хэдэн минутын таймер тавих вэ? Жишээ нь «5 минутын таймер».",
                              "en": "For how long? Say, \"timer for 5 minutes\"."}[lang], lang)
            return None
        note = None
        m = re.search(r"\bto\s+(.+)$", msg.raw, re.I)
        if m and lang == "en":
            note = m.group(1).strip(" .!?")

        def done():
            if lang == "mn":
                text = (f"{self._addr('mn').capitalize()}, {self._duration_label(seconds, 'mn')} "
                        f"өнгөрлөө. Таймер дууслаа!")
            else:
                text = f"Time's up, {self._addr('en')} — your {self._duration_label(seconds, 'en')} timer is done."
            if note:
                text += f" Reminder: {note}."
            skills.notify("JARVIS", note or ("Таймер дууслаа" if lang == "mn" else "Time's up"))
            self._on_reply(Reply(text, lang))

        skills.start_timer(seconds, done)
        if lang == "mn":
            return Reply(f"За, {self._duration_label(seconds, 'mn', genitive=True)} таймер тавилаа. "
                         f"Дуусахаар нь хэлнэ.", lang)
        return Reply(f"Timer set for {self._duration_label(seconds, 'en')}."
                     + (f" I'll remind you to {note}." if note else ""), lang)

    @staticmethod
    def _duration_label(seconds: float, lang: str, genitive: bool = False) -> str:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        if lang == "mn":
            units = [(h, "цаг", "цагийн"), (m, "минут", "минутын"), (s, "секунд", "секундын")]
            parts = [(n, nom, gen) for n, nom, gen in units if n]
            return " ".join(f"{n} {gen if genitive and i == len(parts) - 1 else nom}"
                            for i, (n, nom, gen) in enumerate(parts))
        units = [(h, "hour"), (m, "minute"), (s, "second")]
        return " ".join(f"{n} {word}{'s' * (n != 1)}" for n, word in units if n)

    _OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg}
    _OP_WORDS = [
        ("+", ("нэмэх", "нэмэхэд", "нэмээд", "нэм", "nemeh", "nemehed", "plus", "add")),
        ("-", ("хасах", "хасахад", "хасаад", "хас", "hasah", "hasahad", "minus")),
        ("*", ("үржих", "үржүүлэх", "үржүүлэхэд", "үржүүлээд", "үржүүл", "urjih", "urjuuleh", "times",
               "multiplied by", "x", "×")),
        ("/", ("хуваах", "хуваахад", "хуваасан", "хуваагаад", "хуваа", "huvaah", "huvaahad",
               "divided by", "÷")),
    ]

    def _eval(self, node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self._OPS:
            left, right = self._eval(node.left), self._eval(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return self._OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._OPS:
            return self._OPS[type(node.op)](self._eval(node.operand))
        raise ValueError("unsupported")

    def _math(self, msg: Message, lang: str) -> Reply | None:
        if not re.search(r"\d", msg.raw):
            return None
        text = re.sub(r"(\d)-[а-яёөүa-z]+", r"\1", msg.raw.lower())
        ops_found = []
        for sym, words in self._OP_WORDS:
            for w in sorted(words, key=len, reverse=True):
                pattern = rf"(?<![а-яёөүa-z]){re.escape(w)}(?![а-яёөүa-z])"
                if re.search(pattern, text):
                    ops_found.append(sym)
                    text = re.sub(pattern, f" {sym} ", text)
        text = text.replace("^", "**")
        expr = re.search(r"\(?\s*-?\d+(?:\.\d+)?(?:\s*(?:\*\*|[-+*/])\s*\(?\s*-?\d+(?:\.\d+)?\s*\)?)+", text)
        if expr:
            expression = expr.group(0)
        else:
            numbers = re.findall(r"\d+(?:\.\d+)?", text)
            if len(numbers) != 2 or len(set(ops_found)) != 1:
                return None
            expression = f"{numbers[0]} {ops_found[0]} {numbers[1]}"   # "100-г 4-т хуваа"
        try:
            value = self._eval(ast.parse(expression.strip(), mode="eval").body)
        except ZeroDivisionError:
            return Reply({"mn": "Тэгд хуваах боломжгүй шүү, {0}.".format(self._addr("mn")),
                          "en": "Dividing by zero? Not on my watch, sir."}[lang], lang)
        except Exception:
            return None
        result = int(value) if float(value).is_integer() else round(value, 6)
        shown = expression.replace("**", "^").replace("*", "×").replace("/", "÷")
        shown = re.sub(r"\s*([+×÷^]|(?<=\d)\s*-)\s*", lambda m: f" {m.group(1).strip()} ", shown).strip()
        result_txt = f"{result:,}".replace(",", " ") if isinstance(result, int) else str(result)
        return Reply({"mn": f"{shown} = {result_txt}", "en": f"{shown} = {result_txt}"}[lang], lang,
                     speech={"mn": f"Хариу нь {result}.", "en": f"That's {result}."}[lang])

    def _volume(self, msg: Message, lang: str) -> Reply | None:
        subject = Phrases("дуу", "дууг", "дууны", "дуугаа", "дууны түвшин", "volume", "sound", "audio")
        up = Phrases("чангал", "чангалаач", "чангаруул", "ихэсгэ", "ихэсгээч", "нэм", "нэмээч", "louder",
                     "up", "increase", "raise", "turn up")
        down = Phrases("багасга", "багасгаач", "намсга", "намсгаач", "сулруул", "quieter", "down", "lower",
                       "decrease", "reduce", "turn down")
        mute = Phrases("хаа", "хаачих", "хаагаач", "унтраа", "дуугүй болго", "mute", "silence")
        unmute = Phrases("нээ", "асаа", "буцааж асаа", "unmute")
        standalone = Phrases("louder", "quieter", "чангалаач", "намсгаач", "mute", "unmute", "дуугүй болго")
        report = Phrases("volume", "дууны түвшин", "sound level")
        if msg.ctoks not in subject and msg.ctoks not in standalone:
            return None
        level = re.search(r"\b(\d{1,3})\b", msg.raw)
        has_action = any(msg.ctoks in g for g in (up, down, mute, unmute, standalone))
        if not (has_action or level or msg.ctoks in report):
            return None                          # e.g. "дуу тавь" (play a song) is not about volume
        current = skills.get_volume()
        if msg.ctoks in Phrases("unmute") or (msg.ctoks in unmute and msg.ctoks in subject):
            skills.set_muted(False)
            return Reply({"mn": "Дууг буцааж асаалаа.", "en": "Sound is back on, sir."}[lang], lang)
        if msg.ctoks in mute:
            skills.set_muted(True)
            return Reply({"mn": "Дууг хаалаа.", "en": "Muted."}[lang], lang)
        if level:
            target = int(level.group(1))
        elif msg.ctoks in up:
            target = min(100, (current if current is not None else 50) + 15)
        elif msg.ctoks in down:
            target = max(0, (current if current is not None else 50) - 15)
        else:
            if current is None:
                return Reply({"mn": "Дууны түвшинг уншиж чадсангүй.", "en": "I couldn't read the volume level."}[lang], lang)
            return Reply({"mn": f"Дууны түвшин одоо {current}% байна.", "en": f"Volume is at {current}%."}[lang], lang)
        if not skills.set_volume(target):
            return Reply({"mn": "Дууны түвшинг өөрчилж чадсангүй.", "en": "I couldn't change the volume."}[lang], lang)
        return Reply({"mn": f"Дууны түвшинг {target}% болголоо.", "en": f"Volume set to {target}%."}[lang], lang)

    def _screenshot(self, msg: Message, lang: str) -> Reply | None:
        if msg.ctoks not in Phrases("скриншот", "дэлгэцийн зураг", "дэлгэцээ зурагд", "дэлгэц зурагд",
                                    "дэлгэцийг зурагд", "screenshot", "screen shot", "capture the screen"):
            return None
        path = skills.screenshot()
        if path:
            return Reply({"mn": f"Дэлгэцийн зургийг Desktop дээр «{path.name}» нэрээр хадгаллаа.",
                          "en": f"Screenshot saved to your Desktop as “{path.name}”."}[lang], lang,
                         speech={"mn": "Дэлгэцийн зургийг Desktop дээр хадгаллаа.",
                                 "en": "Screenshot saved to your Desktop, sir."}[lang])
        return Reply({"mn": "Дэлгэцийн зураг авч чадсангүй. System Settings → Privacy & Security → "
                            "Screen Recording хэсэгт зөвшөөрөл өгөөрэй.",
                      "en": "I couldn't take a screenshot. Allow screen recording in System Settings → "
                            "Privacy & Security → Screen Recording."}[lang], lang)

    def _battery(self, msg: Message, lang: str) -> Reply | None:
        if msg.ctoks not in Phrases("батарей", "батерей", "цэнэг", "battery", "charge level", "how much charge"):
            return None
        info = skills.battery()
        if not info:
            return Reply({"mn": "Энэ компьютерт батарей алга.", "en": "This machine has no battery."}[lang], lang)
        pct, plugged = info
        if lang == "mn":
            tail = ", цэнэглэж байна" if plugged else ("" if pct > 20 else ". Цэнэглэгчээ залгаарай")
            return Reply(f"Батарей {pct}% байна{tail}.", lang)
        tail = ", charging" if plugged else ("" if pct > 20 else ". You may want to plug in")
        return Reply(f"Battery is at {pct}%{tail}.", lang)

    def _system(self, msg: Message, lang: str) -> Reply | None:
        if msg.ctoks not in Phrases("систем", "системийн байдал", "cpu", "процессор", "санах ой", "рам", "ram",
                                    "memory usage", "system status", "компьютер ямар", "ачаалал", "диагностик",
                                    "diagnostics"):
            return None
        st = skills.system_status()
        busy = st["cpu"] > 80 or st["mem"] > 90
        if lang == "mn":
            tail = "Процессор нэлээд ачаалалтай байна." if busy else "Бүх зүйл хэвийн."
            return Reply(f"Процессор {st['cpu']}%, санах ой {st['mem']}%, диск {st['disk']}% ачаалалтай. {tail}", lang)
        tail = "Things are running hot." if busy else "All systems nominal."
        return Reply(f"CPU at {st['cpu']}%, memory {st['mem']}%, disk {st['disk']}%. {tail}", lang)

    def _lock(self, msg: Message, lang: str) -> Reply | None:
        if msg.ctoks not in Phrases("дэлгэц түгж", "компьютер түгж", "түгжээч", "дэлгэцээ унтраа",
                                    "дэлгэц унтраа", "lock the screen", "lock screen", "lock my computer",
                                    "lock the computer"):
            return None
        skills.lock_screen()
        return Reply({"mn": "Дэлгэцийг унтраалаа.", "en": "Locking the screen, sir."}[lang], lang)

    def _weather(self, msg: Message, lang: str, force: bool = False) -> Reply | None:
        trigger = Phrases("цаг агаар", "агаар ямар", "хэдэн хэм", "хэдэн градус", "хүйтэн үү", "дулаан уу",
                          "бороо орох", "цас орох", "гадаа ямар", "гадаа хүйтэн", "weather", "temperature",
                          "how cold", "how hot", "is it raining", "forecast", "tsag agaar")
        if not force and msg.ctoks not in trigger:
            return None
        city = next(((q, mn, en) for ph, q, mn, en in CITIES if msg.ctoks in ph), None)
        if not city:
            m = re.search(r"\b(?:in|at|for)\s+([A-Za-z][A-Za-z .'-]+)", msg.raw)
            leftover = " ".join(msg.without(trigger, FILLER)).strip()
            if m:
                name = m.group(1).strip(" ?.!").title()
                city = (name, name, name)
            elif force and leftover:
                city = (leftover, leftover, leftover)
        city = city or self.s.last_city
        self.s.last_city = city
        query, name_mn, name_en = city
        w = skills.weather(query)
        if not w:
            return Reply({"mn": "Цаг агаарын мэдээлэл авч чадсангүй — интернэт холболтоо шалгана уу.",
                          "en": "I couldn't reach the weather service — check the internet connection."}[lang], lang)
        t, feels = w["temp"], w["feels"]
        if lang == "mn":
            desc = w["mn"] or "Тэнгэр"
            advice = (" Дулаан хувцаслаарай!" if t <= -15 else " Гадуур хувцсаа мартуузай." if t <= 5
                      else " Ус ихээр уугаарай." if t >= 28 else "")
            text = (f"{name_mn} хотод одоо {t}°C, мэдрэгдэх нь {feels}°C. {desc}. "
                    f"Салхи {w['wind']} км/ц, чийгшил {w['humidity']}%.{advice}")
        else:
            advice = (" Bundle up, sir." if t <= -15 else " A coat would be wise." if t <= 5
                      else " Stay hydrated." if t >= 28 else "")
            text = (f"In {name_en} it's {t}°C right now, feels like {feels}°C. {w['en']}. "
                    f"Wind {w['wind']} km/h, humidity {w['humidity']}%.{advice}")
        return Reply(text, lang)

    def _find_app(self, msg: Message) -> App | None:
        best, best_score = None, 0
        for app in APPS:
            score = app.names.best(msg.ctoks)
            if score > best_score:
                best, best_score = app, score
        return best

    def _open(self, msg: Message, lang: str) -> Reply | None:
        app = self._find_app(msg)
        if msg.ctoks not in OPEN_VERB:
            if app and len(msg.ctoks) <= 2:      # just "Spotify" or "YouTube"
                return self._launch(app, lang)
            return None
        if app:
            return self._launch(app, lang)
        leftover = " ".join(msg.without(OPEN_VERB, FILLER)).strip()
        if not leftover:
            self.s.awaiting = ("open", None)
            return Reply({"mn": "Ямар програм нээх вэ?", "en": "Which app should I open?"}[lang], lang)
        if len(leftover.split()) > 3:
            return None                          # a sentence that merely contains "start"/"run"
        return self._open_named(leftover, msg, lang)

    def _open_named(self, name: str, msg: Message, lang: str) -> Reply:
        app = self._find_app(msg)
        if app:
            return self._launch(app, lang)
        name = re.sub(r"-[а-яёөүa-z]+$", "", name.strip(), flags=re.I)
        title = " ".join(w[:1].upper() + w[1:] for w in name.split())
        if skills.open_app(title):
            return Reply({"mn": f"{title}-г нээлээ.", "en": f"Opening {title}."}[lang], lang)
        return Reply({"mn": f"Уучлаарай, «{name}» нэртэй програм олдсонгүй.",
                      "en": f"I couldn't find an app called “{name}”."}[lang], lang)

    def _launch(self, app: App, lang: str) -> Reply:
        ok = skills.open_url(app.target) if app.is_url else skills.open_app(app.target)
        if not ok:
            return Reply({"mn": f"{app.label}-г нээж чадсангүй. Компьютерт суусан эсэхийг шалгаарай.",
                          "en": f"I couldn't open {app.label} — is it installed?"}[lang], lang)
        options = {"mn": [f"{app.label}-г нээлээ.", f"За, {app.label} нээгдлээ.", f"{app.label} бэлэн боллоо."],
                   "en": [f"Opening {app.label}.", f"{app.label} is up, {self._addr('en')}.", f"Launching {app.label}."]}
        return Reply(self._pick("open", options[lang]), lang)

    def _youtube(self, msg: Message, lang: str) -> Reply | None:
        yt, play = YOUTUBE, PLAY
        if msg.ctoks not in play and msg.ctoks not in yt:
            return None
        query = " ".join(msg.without(yt, play, FILLER, Phrases("дуу", "хөгжим", "song", "music", "video",
                                                              "бичлэг", "нь", "-ээс")))
        query = re.sub(r"-[а-яёөүa-z]+\b", "", query, flags=re.I).strip()
        if not query:
            if msg.ctoks in yt and msg.ctoks not in play:
                return None                      # "open YouTube" — handled by _open/_search
            self.s.awaiting = ("youtube", None)
            return Reply({"mn": "Юу тоглуулах вэ?", "en": "What should I play?"}[lang], lang)
        return self._do_youtube(query, lang)

    def _do_youtube(self, query: str, lang: str) -> Reply:
        skills.youtube(query)
        return Reply({"mn": f"YouTube-ээс «{query}» хайлаа.", "en": f"Searching YouTube for “{query}”."}[lang], lang)

    def _search(self, msg: Message, lang: str) -> Reply | None:
        verbs = SEARCH_VERB
        if msg.ctoks not in verbs:
            return None
        query = " ".join(msg.without(verbs, FILLER, Phrases("google", "гүүгл", "гугл", "интернэт", "web",
                                                             "online", "-ээс")))
        query = re.sub(r"-[а-яёөүa-z]+\b", "", query, flags=re.I).strip()
        if not query:
            if msg.ctoks in Phrases("google", "гүүгл") and msg.ctoks not in Phrases("хай", "search"):
                return self._launch(next(a for a in APPS if a.label == "Google"), lang)
            self.s.awaiting = ("search", None)
            return Reply({"mn": "Юу хайх вэ?", "en": "What should I search for?"}[lang], lang)
        return self._do_search(query, lang)

    def _do_search(self, query: str, lang: str) -> Reply:
        skills.google(query)
        return Reply({"mn": f"Google-ээс «{query}» хайлаа.", "en": f"Here's what Google has on “{query}”."}[lang], lang)
