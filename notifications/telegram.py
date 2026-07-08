"""Telegram alerts for value opportunities."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.token = (bot_token or TELEGRAM_BOT_TOKEN or "").strip()
        self.chat_id = (chat_id or TELEGRAM_CHAT_ID or "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str) -> bool:
        if not self.enabled:
            logger.info("Telegram disabled — set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID")
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            response = requests.post(
                url,
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                timeout=15,
            )
            response.raise_for_status()
            return True
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False

    def send_value_scan(self, hits: List[Dict], tour: str = "atp") -> bool:
        if not hits:
            return self.send(f"🎾 <b>{tour.upper()}</b> scan: geen value boven drempel.")
        lines = [f"🎾 <b>{tour.upper()} Value Scan</b> — {len(hits)} hits\n"]
        for hit in hits[:10]:
            lines.append(
                f"• <b>{hit['value_side']}</b> @ {hit['odds']:.2f}\n"
                f"  {hit['match']} | edge {hit['edge']:+.1%}\n"
                f"  {hit.get('action', 'VALUE')} | {hit.get('tournament', '')}"
            )
        return self.send("\n".join(lines))

    def send_report(self, title: str, body: str) -> bool:
        return self.send(f"📊 <b>{title}</b>\n\n{body}")