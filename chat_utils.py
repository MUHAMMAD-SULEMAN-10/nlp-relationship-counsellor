"""WhatsApp chat parsing, stats, chunking and sampling (no heavy dependencies)."""
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

# Android: 12/31/23, 9:41 PM - Name: text
_ANDROID = re.compile(
    r"^(?P<date>\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap]\.?[Mm]\.?)?)\s+-\s+"
    r"(?P<sender>[^:]+?):\s(?P<text>.*)$"
)
# iOS: [31/12/2023, 21:41:05] Name: text
_IOS = re.compile(
    r"^\[(?P<date>\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap]\.?[Mm]\.?)?)\]\s+"
    r"(?P<sender>[^:]+?):\s(?P<text>.*)$"
)
_PATTERNS = (_ANDROID, _IOS)

_SKIP_TEXT = {
    "<media omitted>",
    "image omitted",
    "video omitted",
    "audio omitted",
    "sticker omitted",
    "gif omitted",
    "document omitted",
    "this message was deleted",
    "you deleted this message",
    "null",
}
_INVISIBLE = "‎‏‪‬﻿"


@dataclass
class Message:
    stamp: str
    sender: str
    text: str


def parse_chat(raw: str) -> list[Message]:
    """Parse an exported WhatsApp .txt into messages. Multi-line messages are merged."""
    messages: list[Message] = []
    for line in raw.splitlines():
        line = line.strip(_INVISIBLE + " \t")
        if not line:
            continue
        match = next((m for p in _PATTERNS if (m := p.match(line))), None)
        if match:
            text = match["text"].strip()
            if text.lower() in _SKIP_TEXT:
                continue
            messages.append(
                Message(f"{match['date']} {match['time']}", match["sender"].strip(), text)
            )
        elif messages:
            messages[-1].text += " " + line  # continuation of previous message
    return messages


def sender_stats(messages: list[Message]) -> dict[str, dict]:
    """Message count, share and average length per sender."""
    counts: Counter = Counter()
    words: dict[str, int] = defaultdict(int)
    for m in messages:
        counts[m.sender] += 1
        words[m.sender] += len(m.text.split())
    total = sum(counts.values()) or 1
    return {
        s: {
            "messages": c,
            "share_pct": round(100 * c / total, 1),
            "avg_words": round(words[s] / c, 1),
        }
        for s, c in counts.most_common()
    }


def format_line(m: Message) -> str:
    return f"{m.sender}: {m.text}"


def make_chunks(messages: list[Message], per_chunk: int = 12, overlap: int = 3) -> list[str]:
    """Group consecutive messages into overlapping chunks for retrieval."""
    step = max(1, per_chunk - overlap)
    chunks = []
    for i in range(0, len(messages), step):
        window = messages[i : i + per_chunk]
        if window:
            chunks.append("\n".join(format_line(m) for m in window))
        if i + per_chunk >= len(messages):
            break
    return chunks


def sample_transcript(messages: list[Message], max_chars: int = 12000, blocks: int = 8) -> str:
    """Whole chat if it fits; otherwise evenly spaced blocks covering start to end."""
    lines = [format_line(m) for m in messages]
    if sum(len(l) + 1 for l in lines) <= max_chars:
        return "\n".join(lines)

    per_block = max_chars // blocks
    n = len(lines)
    parts = []
    for b in range(blocks):
        start = b * n // blocks
        end = (b + 1) * n // blocks
        used, picked = 0, []
        for line in lines[start:end]:
            if used + len(line) + 1 > per_block:
                break
            picked.append(line)
            used += len(line) + 1
        if picked:
            parts.append("\n".join(picked))
    return "\n[...]\n".join(parts)
