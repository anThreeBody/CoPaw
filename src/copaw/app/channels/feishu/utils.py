# -*- coding: utf-8 -*-
"""Feishu channel pure helpers (session id, sender display, markdown)."""

import json
import re
from typing import Optional

from .constants import FEISHU_SESSION_ID_SUFFIX_LEN


def short_session_id_from_full_id(full_id: str) -> str:
    """Use last N chars of full_id (chat_id or open_id) as session_id."""
    n = FEISHU_SESSION_ID_SUFFIX_LEN
    return full_id[-n:] if len(full_id) >= n else full_id


def sender_display_string(
    nickname: Optional[str],
    sender_id: str,
) -> str:
    """Build sender display as nickname#last4(sender_id), like DingTalk."""
    nick = (nickname or "").strip() if isinstance(nickname, str) else ""
    sid = (sender_id or "").strip()
    suffix = sid[-4:] if len(sid) >= 4 else (sid or "????")
    return f"{(nick or 'unknown')}#{suffix}"


def extract_json_key(content: Optional[str], *keys: str) -> Optional[str]:
    """Parse JSON content and return first present key."""
    if not content:
        return None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return None
    for k in keys:
        v = data.get(k) or data.get(k.replace("_", "").lower())
        if v:
            return str(v).strip()
    return None


def normalize_feishu_md(text: str) -> str:
    """
    Light markdown normalization for Feishu post (avoid broken rendering).
    """
    if not text or not text.strip():
        return text
    # Ensure newline before code fence so Feishu parses it
    text = re.sub(r"([^\n])(```)", r"\1\n\2", text)
    # Convert markdown tables to aligned text (Feishu doesn't support markdown tables)
    text = markdown_table_to_text(text)
    return text


def markdown_table_to_text(text: str) -> str:
    """
    Convert markdown tables to aligned plain text for Feishu compatibility.

    Example input:
        | Header1 | Header2 |
        |---------|---------|
        | Cell1   | Cell2   |

    Example output:
        Header1    Header2
        --------   --------
        Cell1      Cell2
    """
    lines = text.split("\n")
    result = []
    i = 0
    in_table = False

    while i < len(lines):
        line = lines[i]

        # Detect table start: line contains | and ---
        if "|" in line and i + 1 < len(lines) and "|" in lines[i + 1]:
            # Check if next line is separator (contains ---, :--, :-:, --:)
            next_line = lines[i + 1].strip()
            if re.match(r"^\|?[\s:-]*\|[\s:-]*\|", next_line):
                # This is a table
                in_table = True
                table_lines = []

                # Collect table rows
                while i < len(lines) and "|" in lines[i]:
                    row = lines[i].strip()
                    # Remove leading/trailing |
                    if row.startswith("|"):
                        row = row[1:]
                    if row.endswith("|"):
                        row = row[:-1]
                    # Skip separator line (contains only -, :, |)
                    if re.match(r"^[\s|:-]+$", row):
                        i += 1
                        continue
                    if row:  # Skip empty rows
                        table_lines.append([cell.strip() for cell in row.split("|")])
                    i += 1

                # Convert to aligned text
                if table_lines:
                    # Calculate column widths
                    num_cols = len(table_lines[0]) if table_lines else 0
                    if num_cols > 0:
                        col_widths = []
                        for col_idx in range(num_cols):
                            max_width = max(
                                len(row[col_idx]) if col_idx < len(row) else 0
                                for row in table_lines
                            )
                            col_widths.append(max_width)

                        # Format each row
                        for row in table_lines:
                            formatted_cells = []
                            for col_idx, cell in enumerate(row):
                                if col_idx < len(col_widths):
                                    # Left-align with padding
                                    padded = cell.ljust(col_widths[col_idx])
                                    formatted_cells.append(padded)
                            result.append("  ".join(formatted_cells))
                in_table = False
                continue
        else:
            if in_table and not line.strip():
                in_table = False
            result.append(line)
        i += 1

    return "\n".join(result)
