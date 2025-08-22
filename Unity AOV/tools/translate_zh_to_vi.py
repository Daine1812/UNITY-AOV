import os
import re
import sys
import time
import json
import ast
from typing import List, Tuple, Dict, Optional

try:
    # deep-translator is more stable than googletrans
    from deep_translator import GoogleTranslator
except Exception as import_error:  # pragma: no cover
    print("[ERROR] deep-translator is not installed. Run: pip install deep-translator", file=sys.stderr)
    raise


def unquote_po_line(line: str) -> Optional[str]:
    line = line.strip()
    if not (len(line) >= 2 and line.startswith('"') and line.endswith('"')):
        return None
    try:
        return ast.literal_eval(line)
    except Exception:
        # Fallback: strip enclosing quotes without escape handling
        return line[1:-1]


def protect_placeholders(text: str) -> Tuple[str, Dict[str, str]]:
    token_map: Dict[str, str] = {}
    token_counter = 0

    def _add_token(original: str) -> str:
        nonlocal token_counter
        token = f"__PH_{token_counter}__"
        token_counter += 1
        token_map[token] = original
        return token

    # Protect HTML/XML tags
    def replace_tag(match):
        return _add_token(match.group(0))

    protected = re.sub(r"<[^>]+>", replace_tag, text)

    # Protect placeholders like %1, %2, %d, %s, %1s, %\d+, etc.
    def replace_percent(match):
        return _add_token(match.group(0))

    protected = re.sub(r"%(?:\d+\$)?[\ds]", replace_percent, protected)
    protected = re.sub(r"%\d+", replace_percent, protected)

    # Protect brace placeholders like {X}, {name}
    def replace_brace(match):
        return _add_token(match.group(0))

    protected = re.sub(r"\{[^\}]+\}", replace_brace, protected)

    # Protect style-like fragments that shouldn't be translated inside quotes attributes
    # e.g., href='statement', color:#00c1cd, etc.
    protected = re.sub(r"[a-zA-Z_-]+:\s*#[0-9a-fA-F]{3,6}", replace_brace, protected)

    return protected, token_map


def unprotect_placeholders(text: str, token_map: Dict[str, str]) -> str:
    for token, original in token_map.items():
        text = text.replace(token, original)
    return text


def translate_texts(texts: List[str], translator: GoogleTranslator, src: str = "zh-CN", dest: str = "vi") -> List[str]:
    if not texts:
        return []
    results: List[str] = []
    batch_size = 50  # conservative to reduce throttling
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        protected_batch: List[str] = []
        token_maps: List[Dict[str, str]] = []
        for t in batch:
            protected, token_map = protect_placeholders(t)
            protected_batch.append(protected)
            token_maps.append(token_map)

        tries = 0
        while True:
            tries += 1
            try:
                # deep-translator translates one text at a time
                translated_texts = []
                for protected_text in protected_batch:
                    translated = translator.translate(protected_text)
                    translated_texts.append(translated)
                
                # Unprotect
                for i in range(len(translated_texts)):
                    translated_texts[i] = unprotect_placeholders(translated_texts[i], token_maps[i])
                results.extend(translated_texts)
                break
            except Exception as translate_error:
                # Backoff and retry a few times
                if tries >= 5:
                    # On persistent failure, fall back to original text
                    for i in range(len(batch)):
                        results.append(batch[i])
                    print(f"[WARN] Translation failed after retries: {translate_error}")
                    break
                sleep_time = min(2 ** tries, 30)
                time.sleep(sleep_time)
    return results


class PoEntry:
    def __init__(self):
        self.msgid_lines: List[str] = []  # quoted lines
        self.msgstr_lines: List[str] = []  # quoted lines

    def get_msgid(self) -> str:
        return "".join([unquote_po_line(l) or "" for l in self.msgid_lines])

    def get_msgstr(self) -> str:
        return "".join([unquote_po_line(l) or "" for l in self.msgstr_lines])

    def set_msgstr_from_text(self, text: str) -> None:
        # Re-split the text by \n while preserving trailing \n in PO format
        parts = text.split("\n")
        self.msgstr_lines = []
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                line = json.dumps(part + "\n", ensure_ascii=False)
            else:
                # last part without trailing newline
                line = json.dumps(part, ensure_ascii=False)
            # json.dumps uses double quotes suitable for PO format
            self.msgstr_lines.append(line)


def parse_po(filepath: str) -> List[Tuple[str, PoEntry, int, int]]:
    """
    Parse a PO-like file into a list of entries with their line index ranges.
    Returns list of tuples: (state, entry, start_index, end_index)
    state is one of 'header' | 'entry'.
    start_index and end_index are inclusive indices in the original lines list.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    entries: List[Tuple[str, PoEntry, int, int]] = []

    i = 0
    n = len(lines)
    # Detect header: starts with msgid "" then msgstr "" and subsequent quoted lines
    if i < n and lines[i].startswith("msgid \"\""):
        header_entry = PoEntry()
        header_entry.msgid_lines.append(lines[i].split("msgid ", 1)[1].strip())
        i += 1
        if i < n and lines[i].startswith("msgstr "):
            header_entry.msgstr_lines.append(lines[i].split("msgstr ", 1)[1].strip())
            i += 1
            start_idx = 0
            # consume subsequent quoted lines as part of header msgstr
            while i < n and lines[i].lstrip().startswith('"'):
                header_entry.msgstr_lines.append(lines[i].strip())
                i += 1
            end_idx = i - 1
            entries.append(("header", header_entry, start_idx, end_idx))

    # Parse remaining entries
    while i < n:
        line = lines[i]
        if line.startswith("msgid "):
            entry = PoEntry()
            start_idx = i
            entry.msgid_lines.append(line.split("msgid ", 1)[1].strip())
            i += 1
            # Multiline msgid
            while i < n and lines[i].lstrip().startswith('"'):
                entry.msgid_lines.append(lines[i].strip())
                i += 1
            # msgstr
            if i < n and lines[i].startswith("msgstr "):
                entry.msgstr_lines.append(lines[i].split("msgstr ", 1)[1].strip())
                i += 1
                while i < n and lines[i].lstrip().startswith('"'):
                    entry.msgstr_lines.append(lines[i].strip())
                    i += 1
            end_idx = i - 1
            entries.append(("entry", entry, start_idx, end_idx))
        else:
            i += 1

    return entries


def rebuild_po(filepath: str, entries: List[Tuple[str, PoEntry, int, int]], original_lines: List[str]) -> List[str]:
    lines = original_lines[:]
    for state, entry, start, end in entries:
        if state == "header":
            # rebuild header msgstr only; keep msgid as-is
            idx = start
            # msgid line stays
            idx += 1
            # msgstr line
            lines[idx] = "msgstr \"\"\n"
            idx += 1
            # remove old header quoted lines
            while idx <= end:
                lines[idx] = ""
                idx += 1
            # insert new header msgstr lines after start+1
            insertion_point = start + 2
            new_lines = [l + ("\n" if not l.endswith("\n") else "") for l in entry.msgstr_lines]
            lines[insertion_point:insertion_point] = new_lines
        else:
            # Replace msgstr block lines with new ones
            # Find the line index where msgstr starts
            msgid_start = start
            idx = msgid_start + 1
            while idx <= end and original_lines[idx].lstrip().startswith('"'):
                idx += 1
            if idx <= end and original_lines[idx].startswith("msgstr "):
                # Overwrite the first msgstr line
                lines[idx] = "msgstr " + (entry.msgstr_lines[0] if entry.msgstr_lines else '""') + "\n"
                idx2 = idx + 1
                # Clear existing continuation lines
                while idx2 <= end and original_lines[idx2].lstrip().startswith('"'):
                    lines[idx2] = ""
                    idx2 += 1
                # Insert new continuation lines after idx
                insertion_point = idx + 1
                continuation = entry.msgstr_lines[1:]
                new_lines = [l + ("\n" if not l.endswith("\n") else "") for l in continuation]
                lines[insertion_point:insertion_point] = new_lines

    # Remove empty placeholders introduced by our in-place strategy
    final_lines = [ln for ln in lines if ln != ""]
    return final_lines


def main():
    src_file = os.path.join(os.getcwd(), "zh-Hans.txt")
    dst_file = os.path.join(os.getcwd(), "vi.txt")

    if not os.path.exists(src_file):
        print(f"[ERROR] Source file not found: {src_file}", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Parsing source file...")
    with open(src_file, "r", encoding="utf-8") as f:
        original_lines = f.readlines()
    entries = parse_po(src_file)
    print(f"[INFO] Found {len(entries)} entries (including header).")

    translator = GoogleTranslator(source="zh-CN", target="vi")

    # Prepare texts for translation (skip header; handle separately)
    translatable_indices: List[int] = []
    translatable_texts: List[str] = []
    for idx, (state, entry, _s, _e) in enumerate(entries):
        if state == "header":
            continue
        text = entry.get_msgstr()
        # Only translate non-empty Chinese strings
        if text.strip() == "":
            continue
        translatable_indices.append(idx)
        translatable_texts.append(text)

    print(f"[INFO] Translating {len(translatable_texts)} msgstr strings to Vietnamese...")
    translated_texts = translate_texts(translatable_texts, translator, src="zh-CN", dest="vi")

    # Apply translations back to entries
    for j, idx in enumerate(translatable_indices):
        entries[idx][1].set_msgstr_from_text(translated_texts[j])

    # Handle header: copy and modify language fields
    if entries and entries[0][0] == "header":
        header_entry = entries[0][1]
        header_text = header_entry.get_msgstr()
        header_lines = header_text.split("\n")
        new_header_lines: List[str] = []
        for line in header_lines:
            if line.startswith("Language: "):
                new_header_lines.append("Language: vi")
            elif line.startswith("x-poedit-language: "):
                new_header_lines.append("x-poedit-language: vi")
            else:
                new_header_lines.append(line)
        new_header_text = "\n".join(new_header_lines)
        header_entry.set_msgstr_from_text(new_header_text)

    print("[INFO] Rebuilding destination file...")
    final_lines = rebuild_po(src_file, entries, original_lines)

    with open(dst_file, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(final_lines)

    print(f"[DONE] Wrote Vietnamese translation to: {dst_file}")


if __name__ == "__main__":
    main()


