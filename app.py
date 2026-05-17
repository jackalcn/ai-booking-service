import json
import logging
import os
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from random import randint
from time import monotonic
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# 基本設定
# -----------------------------
FAQ_FILE_PATH = Path("faq.json")
FAQ_MATCH_THRESHOLD = 0.6
COMPANY_NAME = "NovaCare 企業客服中心"
OPENAI_TIMEOUT_SECONDS = 20
AI_CONNECT_TIMEOUT_SECONDS = 6
AI_READ_TIMEOUT_SECONDS = 12
AI_TOTAL_TIMEOUT_SECONDS = 25

AGENT_PROFILE = {
    "name": "李美雅",
    "title": "資深客服顧問",
    "avatar": "李",
}

SYSTEM_PROMPT = """
你是一位專業、親切、有耐心的企業客服助理。
你只能回答與公司產品、訂單、付款、退換貨、保固、物流、發票、技術支援、客服聯絡方式相關的問題。
請使用繁體中文回答。
回答要清楚、簡潔、有禮貌。
如果資料不足，請說明目前資料不足，建議聯繫人工客服確認。
不可以亂編公司政策、價格、保固期限或承諾。
""".strip()

LOGGER = logging.getLogger(__name__)

GEMINI_MODEL_FALLBACKS = ["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash-latest"]

AI_ERROR_HINTS = {
    "missing_api_key": "尚未設定可用的 API Key，請先在 Secrets 或 .env 補上金鑰。",
    "ai_timeout": "AI 回覆逾時，可能是網路或服務壅塞，請稍後重試或簡化問題。",
    "openai_api_error": "OpenAI 呼叫失敗，請檢查金鑰、模型名稱與額度是否正常。",
    "gemini_invalid_api_key": "Google API Key 無效或已失效，請到 Google AI Studio 重新產生後更新 Secrets。",
    "gemini_permission_denied": "目前的 Google API Key 權限不足，請確認是否開啟 Generative Language API 並允許伺服器端呼叫。",
    "gemini_api_not_enabled": "此 Google 專案尚未啟用 Generative Language API，請先在 Google Cloud / AI Studio 啟用後再試。",
    "gemini_key_restricted": "目前 API Key 設有 HTTP referrer/IP 限制，Streamlit 雲端伺服器無法使用；請改用可供伺服器端呼叫的金鑰。",
    "gemini_quota_exceeded": "Google Gemini 配額已達上限（免費配額為 15 RPM）。請到 Google AI Studio 重新產生一組新的 API Key 後更新 Secrets，或等待配額重置後再試。",
    "gemini_model_not_found": "目前設定的 GEMINI_MODEL 不可用，建議改為 gemini-2.0-flash。",
    "gemini_bad_request": "Gemini 請求格式或模型設定不被接受，請檢查 GEMINI_MODEL 與 API 設定。",
    "gemini_safety_block": "本次提問被安全政策攔截，請改寫提問內容後再試。",
    "gemini_server_error": "Google Gemini 服務暫時異常，請稍後重試。",
    "gemini_api_error": "Google Gemini 呼叫失敗，請確認網路、API 設定與服務狀態。",
    "empty_response": "AI 回傳內容為空，請稍後重試或改寫問題。",
}

AI_RETRY_HINTS = {
    "ai_timeout": "建議 20 到 30 秒後重試 1 次；若連續失敗請改由人工客服接手。",
    "openai_api_error": "可能是供應商暫時壅塞，建議稍後重試；若連續 2 次失敗請轉人工客服。",
    "gemini_server_error": "Google 服務暫時異常，建議 30 秒後重試；若仍失敗請轉人工客服。",
    "gemini_api_error": "網路或 API 服務不穩，建議稍後重試 1 次。",
    "empty_response": "請將問題改短且更明確後重試，若仍無回覆建議轉人工客服。",
    "gemini_quota_exceeded": "此錯誤通常重試無效，請等待配額重置或更換可用 API Key。",
    "gemini_invalid_api_key": "此錯誤通常重試無效，請先更新有效 API Key。",
    "gemini_key_restricted": "此錯誤通常重試無效，請先解除 API Key 的來源限制。",
    "gemini_api_not_enabled": "此錯誤通常重試無效，請先啟用 Generative Language API。",
    "gemini_model_not_found": "請先調整 GEMINI_MODEL（建議 gemini-2.0-flash-lite）後再重試。",
    "gemini_bad_request": "請先檢查模型與設定格式，修正後再重試。",
}

HUMAN_HANDOFF_KEYWORDS = [
    "人工客服",
    "真人",
    "專人",
    "電話",
    "客訴",
    "申訴",
    "緊急",
    "退款",
    "退費",
    "取消訂單",
    "改地址",
    "扣款",
    "發票",
    "統編",
    "個資",
    "信用卡",
]

HUMAN_HANDOFF_CATEGORY_KEYWORDS = {
    "訂單查詢": ["訂單", "單號", "地址", "取消"],
    "付款方式": ["扣款", "刷卡", "付款失敗", "退款"],
    "退換貨": ["退款", "退貨", "瑕疵", "換貨"],
    "發票問題": ["發票", "統編", "抬頭"],
}


def get_runtime_setting(key: str, default_value: str = "") -> str:
    """優先讀取環境變數，若無則讀取 Streamlit secrets。"""
    env_value = os.getenv(key, "").strip()
    if env_value:
        return env_value

    try:
        secret_value = st.secrets.get(key, "")
        if isinstance(secret_value, str):
            return secret_value.strip() or default_value
        return str(secret_value).strip() or default_value
    except Exception:
        return default_value


def resolve_ai_config() -> Dict[str, str]:
    """解析 AI 供應商設定，支援 OpenAI 與 Google Gemini。"""
    provider_raw = get_runtime_setting("AI_PROVIDER", "auto").strip().lower()
    openai_key = get_runtime_setting("OPENAI_API_KEY", "")
    google_key = get_runtime_setting("GOOGLE_API_KEY", "")

    # 兼容另一種常見命名，避免使用者更動變數名後無法讀取。
    if not google_key:
        google_key = get_runtime_setting("GEMINI_API_KEY", "")

    openai_model = get_runtime_setting("OPENAI_MODEL", "gpt-4o-mini")
    gemini_model = get_runtime_setting("GEMINI_MODEL", "gemini-1.5-flash")

    if provider_raw == "openai":
        return {
            "provider": "openai",
            "api_key": openai_key,
            "model": openai_model,
            "provider_label": "OpenAI",
            "source_label": "OpenAI",
        }

    if provider_raw in ("gemini", "google"):
        return {
            "provider": "gemini",
            "api_key": google_key,
            "model": gemini_model,
            "provider_label": "Google Gemini",
            "source_label": "Google Gemini",
        }

    # AI_PROVIDER=auto 或未設定時，自動選擇目前可用金鑰。
    if openai_key:
        return {
            "provider": "openai",
            "api_key": openai_key,
            "model": openai_model,
            "provider_label": "自動偵測（OpenAI）",
            "source_label": "OpenAI",
        }

    if google_key:
        return {
            "provider": "gemini",
            "api_key": google_key,
            "model": gemini_model,
            "provider_label": "自動偵測（Google Gemini）",
            "source_label": "Google Gemini",
        }

    return {
        "provider": "openai",
        "api_key": "",
        "model": openai_model,
        "provider_label": "自動偵測（未設定金鑰）",
        "source_label": "AI",
    }


def build_ai_user_prompt(user_question: str, category: str) -> str:
    """統一組裝 AI 問答 prompt，確保不同供應商語氣一致。"""
    return (
        f"使用者問題分類：{category}\n"
        f"使用者問題：{user_question}\n\n"
        "請以專業親切型企業客服口吻回答，"
        "若資料不足請明確表示並建議聯繫人工客服。"
    )


def map_gemini_http_error(response: requests.Response) -> str:
    """將 Gemini HTTP 錯誤映射為可判讀的錯誤碼。"""
    status_code = response.status_code
    status_text = ""
    message_text = ""

    try:
        payload = response.json()
        error_obj = payload.get("error", {}) if isinstance(payload, dict) else {}
        status_text = str(error_obj.get("status", "")).upper()
        message_text = str(error_obj.get("message", ""))
    except Exception:
        message_text = response.text or ""

    message_lower = message_text.lower()

    if "api key" in message_lower and ("invalid" in message_lower or "not valid" in message_lower):
        return "gemini_invalid_api_key"

    if "api has not been used" in message_lower or "it is disabled" in message_lower:
        return "gemini_api_not_enabled"

    if "referer restrictions" in message_lower or "referrer restrictions" in message_lower:
        return "gemini_key_restricted"

    if "ip address restrictions" in message_lower:
        return "gemini_key_restricted"

    if status_code == 404 or "not found for api version" in message_lower:
        return "gemini_model_not_found"

    if "model" in message_lower and "not found" in message_lower:
        return "gemini_model_not_found"

    if status_code == 429 or "quota" in message_lower or "rate limit" in message_lower:
        return "gemini_quota_exceeded"

    if status_code == 403 or status_text == "PERMISSION_DENIED":
        if "quota" in message_lower:
            return "gemini_quota_exceeded"
        return "gemini_permission_denied"

    if status_code == 400:
        return "gemini_bad_request"

    if status_code >= 500:
        return "gemini_server_error"

    return "gemini_api_error"


def build_ai_error_hint(error_code: Optional[str], model_name: str) -> str:
    """將錯誤碼轉為使用者可採取行動的提示。"""
    if not error_code:
        return ""

    base_hint = AI_ERROR_HINTS.get(error_code, "")
    if not base_hint:
        return ""

    if error_code == "gemini_model_not_found":
        return f"{base_hint}（目前設定：{model_name}）"

    return base_hint


def build_retry_hint(error_code: Optional[str]) -> str:
    """依錯誤碼提供可執行的重試建議。"""
    if not error_code:
        return ""

    return AI_RETRY_HINTS.get(error_code, "建議稍後再重試 1 次，若仍失敗請改由人工客服接手。")


def should_suggest_human_transfer(
    user_question: str,
    category: str,
    answer_text: str = "",
    error_code: Optional[str] = None,
) -> bool:
    """根據提問內容、分類與回覆語意，判斷是否應建議人工客服接手。"""
    if error_code:
        return True

    normalized_question = normalize_text(user_question)

    if any(keyword in normalized_question for keyword in HUMAN_HANDOFF_KEYWORDS):
        return True

    category_keywords = HUMAN_HANDOFF_CATEGORY_KEYWORDS.get(category, [])
    if category_keywords and any(keyword in normalized_question for keyword in category_keywords):
        return True

    if answer_text and any(
        keyword in answer_text for keyword in ["資料不足", "無法確認", "建議聯繫人工客服", "需由人工客服"]
    ):
        return True

    return False

CATEGORY_KEYWORDS = {
    "產品介紹": ["產品", "方案", "規格", "功能", "比較", "介紹"],
    "訂單查詢": ["訂單", "下單", "出貨", "取消", "地址", "編號"],
    "付款方式": ["付款", "刷卡", "轉帳", "分期", "貨到付款", "發票支付"],
    "退換貨": ["退貨", "換貨", "退款", "鑑賞期", "瑕疵", "退回"],
    "保固服務": ["保固", "送修", "維修", "保修", "故障", "RMA"],
    "技術支援": ["無法", "錯誤", "登入", "閃退", "更新", "安裝", "密碼"],
    "聯絡客服": ["客服", "聯絡", "電話", "信箱", "人工", "真人"],
    "營業時間": ["營業時間", "上班", "服務時間", "幾點", "假日", "週末"],
    "發票問題": ["發票", "統編", "抬頭", "電子發票", "載具", "報帳"],
    "物流配送": ["物流", "配送", "運費", "宅配", "超商", "到貨", "離島"],
}


# -----------------------------
# 資料處理函式
# -----------------------------
def normalize_text(text: str) -> str:
    """將文字做基本清理，讓比對結果更穩定。"""
    return " ".join(text.strip().lower().split())


def calculate_similarity(text_a: str, text_b: str) -> float:
    """使用 difflib 計算兩段文字的相似度。"""
    return SequenceMatcher(None, normalize_text(text_a), normalize_text(text_b)).ratio()


def keyword_similarity(user_question: str, keywords: List[str]) -> float:
    """計算使用者問題與 FAQ 關鍵字的相似度。"""
    if not keywords:
        return 0.0

    question_text = normalize_text(user_question)
    scores: List[float] = []

    for keyword in keywords:
        keyword_text = normalize_text(str(keyword))
        if not keyword_text:
            continue

        # 相似度分數 + 是否包含關鍵字分數，取較高值。
        ratio_score = SequenceMatcher(None, question_text, keyword_text).ratio()
        contains_score = 1.0 if keyword_text in question_text else 0.0
        scores.append(max(ratio_score, contains_score))

    return max(scores) if scores else 0.0


def load_faq_data(file_path: Path) -> List[Dict]:
    """讀取 FAQ JSON 檔案，若有問題則回傳空陣列並顯示友善提示。"""
    if not file_path.exists():
        st.error("找不到 faq.json，請確認檔案存在於專案根目錄。")
        return []

    try:
        with file_path.open("r", encoding="utf-8") as f:
            faq_data = json.load(f)

        if not isinstance(faq_data, list):
            st.error("faq.json 格式不正確，請確認最外層為陣列。")
            return []

        return faq_data

    except json.JSONDecodeError:
        st.error("faq.json 解析失敗，請檢查 JSON 格式是否正確。")
        return []
    except Exception:
        st.error("讀取 FAQ 資料時發生問題，請稍後再試。")
        return []


def find_best_faq(user_question: str, faq_data: List[Dict]) -> Tuple[Optional[Dict], float]:
    """從 FAQ 中找出最接近的問題，回傳 FAQ 項目與相似度。"""
    best_item: Optional[Dict] = None
    best_score = 0.0

    for item in faq_data:
        faq_question = str(item.get("question", ""))
        faq_keywords = item.get("keywords", [])

        question_score = calculate_similarity(user_question, faq_question)
        keyword_score = keyword_similarity(user_question, faq_keywords)

        # 若問題中直接包含任何關鍵字，額外給一點加分。
        normalized_question = normalize_text(user_question)
        keyword_hit = 1.0 if any(normalize_text(str(k)) in normalized_question for k in faq_keywords) else 0.0

        total_score = (question_score * 0.55) + (keyword_score * 0.35) + (keyword_hit * 0.10)

        if total_score > best_score:
            best_score = total_score
            best_item = item

    return best_item, best_score


def classify_question(user_question: str, faq_item: Optional[Dict] = None) -> str:
    """先使用 FAQ 分類，若沒有再用關鍵字規則分類。"""
    if faq_item and faq_item.get("category"):
        return str(faq_item.get("category"))

    normalized_question = normalize_text(user_question)

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in normalized_question for keyword in keywords):
            return category

    return "其他問題"


def current_timestamp() -> str:
    """回傳目前時間字串，作為訊息時間戳記。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_case_id() -> str:
    """建立案件編號，格式示例：CS-20260518-8342。"""
    date_part = datetime.now().strftime("%Y%m%d")
    serial_part = randint(1000, 9999)
    return f"CS-{date_part}-{serial_part}"


def get_case_status() -> str:
    """根據目前對話內容推估案件狀態。"""
    messages = st.session_state.get("messages", [])
    user_count = sum(1 for msg in messages if msg.get("role") == "user")

    if user_count == 0:
        return "待提問"

    if any(msg.get("role") == "assistant" and msg.get("suggest_human") for msg in messages):
        return "建議人工接手"

    return "AI 協助中"


def build_chat_transcript() -> str:
    """整理可下載的對話紀錄文字。"""
    lines = [
        f"{COMPANY_NAME} 對話紀錄",
        f"案件編號：{st.session_state.get('case_id', '-')}",
        f"建立時間：{st.session_state.get('chat_started_at', '-')}",
        "",
    ]

    for msg in st.session_state.get("messages", []):
        role = msg.get("role", "assistant")
        timestamp = msg.get("timestamp", "未記錄時間")

        if role == "user":
            lines.append(f"[{timestamp}] 使用者提問")
            lines.append(msg.get("content", ""))
            lines.append("")
            continue

        lines.append(f"[{timestamp}] 客服回覆（{AGENT_PROFILE['name']} / {AGENT_PROFILE['title']}）")
        lines.append(f"問題分類：{msg.get('category', '其他問題')}")
        lines.append(f"客服回覆：{msg.get('content', '')}")
        lines.append(f"資料來源：{msg.get('source', '未知')}")
        transfer_text = "是" if msg.get("suggest_human", False) else "否"
        lines.append(f"是否建議轉人工客服：{transfer_text}")
        lines.append("")

    return "\n".join(lines)


# -----------------------------
# AI 回覆函式
# -----------------------------
def generate_openai_response(
    api_key: str,
    user_question: str,
    category: str,
    model_name: str,
) -> Tuple[Optional[str], Optional[str]]:
    """呼叫 OpenAI API 產生客服回答。"""
    if not api_key:
        return None, "missing_api_key"

    try:
        client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_SECONDS)

        response = client.chat.completions.create(
            model=model_name,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_ai_user_prompt(user_question, category),
                },
            ],
        )

        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            return None, "empty_response"

        return answer, None

    except Exception as exc:
        if "timeout" in str(exc).lower():
            return None, "ai_timeout"
        # 這裡不回傳技術細節，避免將複雜錯誤直接顯示給一般使用者。
        return None, "openai_api_error"


def _build_gemini_payload(user_question: str, category: str, api_version: str) -> dict:
    """依 API 版本組裝 payload；v1 不支援 systemInstruction，改嵌入 user 訊息。"""
    user_text = build_ai_user_prompt(user_question, category)
    if api_version == "v1beta":
        return {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {"temperature": 0.3},
        }
    # v1: embed system prompt into user message
    return {
        "contents": [{"role": "user", "parts": [{"text": f"{SYSTEM_PROMPT}\n\n{user_text}"}]}],
        "generationConfig": {"temperature": 0.3},
    }


def generate_gemini_response(
    api_key: str,
    user_question: str,
    category: str,
    model_name: str,
) -> Tuple[Optional[str], Optional[str]]:
    """呼叫 Google Gemini API 產生客服回答。"""
    if not api_key:
        return None, "missing_api_key"

    candidate_models = [model_name]
    for fallback_model in GEMINI_MODEL_FALLBACKS:
        if fallback_model != model_name:
            candidate_models.append(fallback_model)

    # v1beta 優先（完整支援 systemInstruction）；v1 作為相容備援
    api_versions = ["v1beta", "v1"]
    last_error_code = "gemini_api_error"
    started_at = monotonic()

    try:
        for candidate_model in candidate_models:
            for api_version in api_versions:
                if monotonic() - started_at >= AI_TOTAL_TIMEOUT_SECONDS:
                    LOGGER.warning(
                        "Gemini request timed out by total budget: %ss",
                        AI_TOTAL_TIMEOUT_SECONDS,
                    )
                    return None, "ai_timeout"

                endpoint = (
                    f"https://generativelanguage.googleapis.com/{api_version}/"
                    f"models/{candidate_model}:generateContent"
                )
                payload = _build_gemini_payload(user_question, category, api_version)

                response = requests.post(
                    endpoint,
                    params={"key": api_key},
                    json=payload,
                    timeout=(AI_CONNECT_TIMEOUT_SECONDS, AI_READ_TIMEOUT_SECONDS),
                )

                if response.status_code >= 400:
                    last_error_code = map_gemini_http_error(response)
                    LOGGER.warning(
                        "Gemini API failed: status=%s code=%s version=%s model=%s",
                        response.status_code,
                        last_error_code,
                        api_version,
                        candidate_model,
                    )

                    # 遇到模型不存在或參數不相容時，繼續嘗試下一個版本/模型。
                    if last_error_code in {"gemini_model_not_found", "gemini_bad_request"}:
                        continue

                    return None, last_error_code

                data = response.json()
                candidates = data.get("candidates", []) if isinstance(data, dict) else []
                if not candidates:
                    block_reason = (
                        data.get("promptFeedback", {}).get("blockReason")
                        if isinstance(data, dict)
                        else None
                    )
                    if block_reason:
                        LOGGER.warning(
                            "Gemini blocked by safety policy: version=%s model=%s reason=%s",
                            api_version,
                            candidate_model,
                            block_reason,
                        )
                        return None, "gemini_safety_block"

                    last_error_code = "empty_response"
                    continue

                parts = candidates[0].get("content", {}).get("parts", [])
                text_parts = [
                    str(part.get("text", "")).strip()
                    for part in parts
                    if str(part.get("text", "")).strip()
                ]

                answer = "\n".join(text_parts).strip()
                if not answer:
                    last_error_code = "empty_response"
                    continue

                return answer, None

        return None, last_error_code

    except requests.Timeout:
        LOGGER.exception("Gemini request timeout")
        return None, "ai_timeout"
    except requests.RequestException:
        LOGGER.exception("Gemini request exception")
        return None, "gemini_api_error"
    except Exception:
        LOGGER.exception("Unexpected Gemini error")
        return None, "gemini_api_error"


def generate_ai_response(
    ai_provider: str,
    api_key: str,
    user_question: str,
    category: str,
    model_name: str,
) -> Tuple[Optional[str], Optional[str]]:
    """依供應商路由至對應 AI API。"""
    if ai_provider == "gemini":
        return generate_gemini_response(api_key, user_question, category, model_name)

    return generate_openai_response(api_key, user_question, category, model_name)


# -----------------------------
# Streamlit 畫面與互動邏輯
# -----------------------------
def init_session_state() -> None:
    """初始化 session_state，確保聊天紀錄與狀態可持續保存。"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "case_id" not in st.session_state:
        st.session_state.case_id = generate_case_id()

    if "chat_started_at" not in st.session_state:
        st.session_state.chat_started_at = current_timestamp()

    if "auto_scroll_to_latest" not in st.session_state:
        st.session_state.auto_scroll_to_latest = False

    if "scroll_to_top_on_load" not in st.session_state:
        st.session_state.scroll_to_top_on_load = True


def build_sidebar() -> None:
    """建立側邊欄客服資訊與系統說明。"""
    with st.sidebar:
        st.markdown(
            f"""
            <div class="side-brand">
                <div class="side-logo">NC</div>
                <div>
                    <div class="side-brand-title">{COMPANY_NAME}</div>
                    <div class="side-brand-sub">專業親切型企業客服</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="side-agent-card">
                <div class="side-agent-avatar">{AGENT_PROFILE['avatar']}</div>
                <div>
                    <div class="side-agent-name">{AGENT_PROFILE['name']}</div>
                    <div class="side-agent-title">{AGENT_PROFILE['title']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"**案件編號：** {st.session_state.case_id}")
        st.markdown(f"**案件建立時間：** {st.session_state.chat_started_at}")

        st.header("企業客服中心")
        st.subheader("聯絡資訊")
        st.write("客服時間：週一至週五 09:00-18:00")
        st.write("客服信箱：service@example.com")
        st.write("客服電話：0800-000-000")

        st.divider()
        st.subheader("系統流程")
        st.markdown(
            "1. 優先比對 FAQ 知識庫。\n"
            "2. FAQ 相似度低於 0.6 時改由 AI 回答。\n"
            "3. 資料不足時會提醒轉人工客服。"
        )
        st.caption("SLA：一般問題預計 5 分鐘內提供初步回覆。")
        st.info("提醒：若涉及帳務爭議或個資驗證，建議直接聯繫人工客服。")

        st.divider()
        if st.session_state.messages:
            st.download_button(
                label="下載對話紀錄（TXT）",
                data=build_chat_transcript(),
                file_name=f"{st.session_state.case_id}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        if st.button("清除對話紀錄", use_container_width=True):
            st.session_state.messages = []
            st.session_state.case_id = generate_case_id()
            st.session_state.chat_started_at = current_timestamp()
            st.session_state.auto_scroll_to_latest = False
            st.session_state.scroll_to_top_on_load = True
            st.rerun()


def render_service_overview(ai_enabled: bool, mode_text: str, faq_count: int) -> None:
    """顯示服務狀態卡，讓畫面更像正式客服儀表板。"""
    user_message_count = sum(1 for m in st.session_state.messages if m.get("role") == "user")
    mode_class = "status-ok" if ai_enabled else "status-warn"
    case_status = get_case_status()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="info-tile">
                <div class="tile-title">服務模式</div>
                <div class="{mode_class}">{mode_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="info-tile">
                <div class="tile-title">案件狀態</div>
                <div class="tile-value">{case_status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="info-tile">
                <div class="tile-title">FAQ 知識庫</div>
                <div class="tile-value">{faq_count} 筆</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="info-tile">
                <div class="tile-title">本次對話問題數</div>
                <div class="tile-value">{user_message_count} 題</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def scroll_to_latest_message() -> None:
    """在訊息更新後自動捲到最新內容，避免使用者看不到回覆。"""
    components.html(
        """
        <script>
        let attempts = 0;
        const maxAttempts = 28;

        const runScroll = () => {
            const rootDoc = window.parent?.document;
            if (!rootDoc) return false;

            const scroller = rootDoc.querySelector('[data-testid="stAppScrollToBottomContainer"]');
            if (!scroller) return false;

            scroller.scrollTo({ top: scroller.scrollHeight, behavior: 'auto' });
            const distance = scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop;
            return distance <= 12;
        };

        const timer = setInterval(() => {
            attempts += 1;
            const done = runScroll();
            if (done || attempts >= maxAttempts) {
                clearInterval(timer);
            }
        }, 120);
        </script>
        """,
        height=0,
    )


def scroll_to_page_top() -> None:
    """在初次載入時將頁面定位到最上方。"""
    components.html(
        """
        <script>
        let attempts = 0;
        const maxAttempts = 20;

        const runScroll = () => {
            const rootDoc = window.parent?.document;
            if (!rootDoc) return false;

            const scroller = rootDoc.querySelector('[data-testid="stAppScrollToBottomContainer"]');
            if (!scroller) return false;

            scroller.scrollTo({ top: 0, behavior: 'auto' });
            return scroller.scrollTop <= 1;
        };

        const timer = setInterval(() => {
            attempts += 1;
            const done = runScroll();
            if (done || attempts >= maxAttempts) {
                clearInterval(timer);
            }
        }, 120);
        </script>
        """,
        height=0,
    )


def show_quick_buttons() -> Optional[str]:
    """顯示常見問題快捷按鈕，回傳被點選的問題文字。"""
    st.markdown("### 常見問題快捷提問")
    quick_questions = [
        "營業時間",
        "付款方式",
        "退換貨",
        "保固服務",
        "物流配送",
        "聯絡客服",
    ]

    selected_question: Optional[str] = None
    with st.container(border=True):
        st.caption("可快速點選以下常見主題，立即取得標準客服回覆。")
        columns = st.columns(3)

        for index, question in enumerate(quick_questions):
            if columns[index % 3].button(question, key=f"quick_btn_{index}", use_container_width=True):
                selected_question = question

    return selected_question


def render_assistant_message(message: Dict, index: int) -> None:
    """以固定格式顯示客服回答，並提供滿意度回饋。"""
    message_time = message.get("timestamp", "未記錄時間")
    st.caption(
        f"客服人員：{AGENT_PROFILE['name']}（{AGENT_PROFILE['title']}） ｜ "
        f"回覆時間：{message_time} ｜ 案件編號：{st.session_state.case_id}"
    )
    st.markdown(f"**問題分類：** {message.get('category', '其他問題')}")
    with st.container(border=True):
        st.markdown("**客服回覆：**")
        st.write(message.get("content", ""))

    st.markdown(f"**資料來源：** {message.get('source', '未知')}")
    transfer_text = "是" if message.get("suggest_human", False) else "否"
    st.markdown(f"**是否建議轉人工客服：** {transfer_text}")

    col1, col2, col3 = st.columns([1, 1, 4])
    if col1.button("👍 有幫助", key=f"feedback_up_{index}"):
        message["feedback"] = "helpful"

    if col2.button("👎 沒有幫助", key=f"feedback_down_{index}"):
        message["feedback"] = "not_helpful"

    feedback_status = message.get("feedback")
    if feedback_status == "helpful":
        st.caption("感謝您的回饋，我們會持續優化客服品質。")
    elif feedback_status == "not_helpful":
        st.warning("很抱歉沒有解決您的問題，建議您留下聯絡方式或改由人工客服協助。")


def build_answer(
    user_question: str,
    faq_data: List[Dict],
    ai_provider: str,
    api_key: str,
    ai_model: str,
    ai_source_label: str,
) -> Dict:
    """依照規則產生回答：先 FAQ、再 AI、最後人工客服建議。"""
    best_faq, faq_score = find_best_faq(user_question, faq_data)
    category = classify_question(user_question, best_faq)

    # 預設值
    source = "FAQ 知識庫"
    suggest_human = False

    # 規則 1：FAQ 相似度高於門檻，直接使用 FAQ 答案。
    if best_faq and faq_score >= FAQ_MATCH_THRESHOLD:
        answer_text = str(best_faq.get("answer", "目前資料不足，建議您聯繫人工客服。"))
        category = str(best_faq.get("category", category))
        suggest_human = should_suggest_human_transfer(
            user_question=user_question,
            category=category,
            answer_text=answer_text,
            error_code=None,
        )

        return {
            "role": "assistant",
            "category": category,
            "content": answer_text,
            "suggest_human": suggest_human,
            "source": source,
            "feedback": None,
        }

    # 規則 2：FAQ 未命中且沒有 API Key，改用 FAQ-only 模式。
    if not api_key:
        fallback_text = (
            "目前尚未設定可用的 AI API Key，因此只能使用 FAQ 知識庫回答。\n\n"
            "此問題在現有 FAQ 中資料不足，建議您聯繫人工客服進一步確認。\n"
            "還有其他需要我協助的地方嗎？"
        )

        return {
            "role": "assistant",
            "category": category,
            "content": fallback_text,
            "suggest_human": True,
            "source": "FAQ 知識庫（AI 未啟用）",
            "feedback": None,
        }

    # 規則 3：FAQ 未命中且有 API Key，呼叫 AI 產生回答。
    ai_answer, error_code = generate_ai_response(ai_provider, api_key, user_question, category, ai_model)
    source = f"AI 智慧客服（{ai_source_label}）"

    if error_code is None and ai_answer:
        suggest_human = should_suggest_human_transfer(
            user_question=user_question,
            category=category,
            answer_text=ai_answer,
            error_code=None,
        )

        return {
            "role": "assistant",
            "category": category,
            "content": ai_answer,
            "suggest_human": suggest_human,
            "source": source,
            "feedback": None,
        }

    # 規則 4：AI API 呼叫失敗時，顯示友善訊息，不顯示技術細節。
    fail_text = (
        "抱歉，系統目前暫時無法完成 AI 回覆。\n"
        "建議您稍後再試，或改由人工客服協助處理。\n"
        "還有其他需要我協助的地方嗎？"
    )

    error_hint = build_ai_error_hint(error_code, ai_model)
    if error_hint:
        fail_text = f"{fail_text}\n\n系統診斷建議：{error_hint}"
    retry_hint = build_retry_hint(error_code)
    if retry_hint:
        fail_text = f"{fail_text}\n錯誤重試提示：{retry_hint}"
    if error_code:
        fail_text = f"{fail_text}\n系統診斷代碼：{error_code}"

    return {
        "role": "assistant",
        "category": category,
        "content": fail_text,
        "suggest_human": True,
        "source": f"AI 智慧客服（{ai_source_label} 暫時不可用）",
        "feedback": None,
    }


def main() -> None:
    """主程式入口。"""
    # 載入 .env 並解析供應商設定（OpenAI / Google Gemini）。
    load_dotenv()
    ai_config = resolve_ai_config()
    ai_provider = ai_config.get("provider", "openai")
    api_key = ai_config.get("api_key", "")
    ai_model = ai_config.get("model", "gpt-4o-mini")
    ai_mode_label = ai_config.get("provider_label", "未啟用")
    ai_source_label = ai_config.get("source_label", "AI")
    ai_enabled = bool(api_key)

    # 設定頁面樣式，讓整體看起來像企業客服系統。
    st.set_page_config(page_title="AI 智慧客服問答系統", page_icon="💬", layout="wide")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
        }

        .stApp {
            background: linear-gradient(180deg, #edf3fb 0%, #f9fbff 40%, #ffffff 100%);
        }

        :root {
            --streamlit-header-height: 0px;
            --main-top-safe-gap: 1rem;
        }

        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        button[kind="header"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        .block-container {
            max-width: 1180px;
            padding-top: var(--main-top-safe-gap);
            padding-bottom: 2rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            overflow: visible;
        }

        [data-testid="stAppScrollToBottomContainer"] {
            overflow-y: auto;
            overflow-x: hidden;
            -webkit-overflow-scrolling: touch;
        }

        [data-testid="stVerticalBlock"] {
            overflow: visible;
        }

        [data-testid="stColumn"] {
            overflow: visible;
        }

        .header-panel {
            background: #ffffff;
            border: 1px solid #d5e0ef;
            border-left: 6px solid #0b4aa6;
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: 0 8px 20px rgba(12, 42, 84, 0.08);
            margin-bottom: 10px;
        }

        .brand-row {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .brand-logo {
            width: 54px;
            height: 54px;
            border-radius: 14px;
            background: linear-gradient(135deg, #11438a 0%, #0d63c8 100%);
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            font-weight: 700;
            letter-spacing: 0.8px;
            box-shadow: 0 6px 14px rgba(10, 63, 135, 0.28);
        }

        .brand-name {
            margin: 0;
            color: #0c2d57;
            font-size: 1.45rem;
            font-weight: 700;
        }

        .brand-subtitle {
            margin: 2px 0 0 0;
            color: #3c567f;
            font-size: 0.95rem;
        }

        .meta-row {
            margin-top: 12px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .meta-chip {
            background: #eef4ff;
            border: 1px solid #d0def5;
            color: #244978;
            border-radius: 999px;
            padding: 5px 11px;
            font-size: 0.86rem;
            font-weight: 500;
        }

        .header-panel p {
            margin: 0;
            color: #334e72;
            font-size: 0.98rem;
            line-height: 1.55;
        }

        .agent-panel {
            background: #ffffff;
            border: 1px solid #d5e2f5;
            border-radius: 12px;
            box-shadow: 0 8px 18px rgba(15, 45, 90, 0.08);
            padding: 14px;
            min-height: 130px;
        }

        .agent-row {
            display: flex;
            gap: 12px;
            align-items: center;
        }

        .agent-avatar {
            width: 52px;
            height: 52px;
            border-radius: 50%;
            background: linear-gradient(140deg, #f7fbff 0%, #d8e8ff 100%);
            border: 1px solid #a9c7ee;
            color: #103d74;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            font-weight: 700;
        }

        .agent-name {
            margin: 0;
            color: #0f2d59;
            font-weight: 700;
            font-size: 1rem;
        }

        .agent-title {
            margin: 2px 0 0 0;
            color: #426084;
            font-size: 0.86rem;
        }

        .agent-online {
            margin-top: 10px;
            color: #0f6848;
            background: #e8f6ee;
            border: 1px solid #b7e2cb;
            border-radius: 999px;
            display: inline-block;
            padding: 4px 10px;
            font-weight: 700;
            font-size: 0.85rem;
        }

        .info-tile {
            background: #ffffff;
            border: 1px solid #d8e3f2;
            border-radius: 12px;
            padding: 12px 14px;
            box-shadow: 0 4px 12px rgba(17, 40, 77, 0.05);
            min-height: 90px;
        }

        .side-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            background: #ffffff;
            border: 1px solid #d8e4f5;
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 10px;
        }

        .side-logo {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            background: linear-gradient(135deg, #124b95 0%, #0e63c8 100%);
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 0.9rem;
        }

        .side-brand-title {
            color: #103566;
            font-size: 0.96rem;
            font-weight: 700;
        }

        .side-brand-sub {
            color: #4a648b;
            font-size: 0.78rem;
            margin-top: 2px;
        }

        .side-agent-card {
            display: flex;
            gap: 10px;
            align-items: center;
            background: #ffffff;
            border: 1px solid #d8e4f5;
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 12px;
        }

        .side-agent-avatar {
            width: 38px;
            height: 38px;
            border-radius: 50%;
            border: 1px solid #adc9ee;
            background: #e9f2ff;
            color: #103d74;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
        }

        .side-agent-name {
            color: #183f73;
            font-weight: 700;
            font-size: 0.9rem;
        }

        .side-agent-title {
            color: #4f678e;
            font-size: 0.78rem;
            margin-top: 2px;
        }

        .tile-title {
            color: #4a6287;
            font-size: 0.9rem;
            margin-bottom: 6px;
        }

        .tile-value {
            color: #102f5c;
            font-size: 1.15rem;
            font-weight: 700;
        }

        .status-ok {
            color: #0f6848;
            background: #e8f6ee;
            border: 1px solid #b7e2cb;
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.9rem;
        }

        .status-warn {
            color: #805b00;
            background: #fff6dc;
            border: 1px solid #f4df9a;
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.9rem;
        }

        [data-testid="stChatMessage"] {
            border: 1px solid #dce5f3;
            border-radius: 12px;
            padding: 8px 12px;
            background: #ffffff;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fbff 0%, #eef4fd 100%);
            border-right: 1px solid #d9e4f3;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()
    build_sidebar()

    header_col1, header_col2 = st.columns([2.15, 1], gap="medium")
    with header_col1:
        st.markdown(
            f"""
            <div class="header-panel">
                <div class="brand-row">
                    <div class="brand-logo">NC</div>
                    <div>
                        <p class="brand-name">AI 智慧客服問答系統</p>
                        <p class="brand-subtitle">{COMPANY_NAME}</p>
                    </div>
                </div>
                <div class="meta-row">
                    <span class="meta-chip">案件編號：{st.session_state.case_id}</span>
                    <span class="meta-chip">建立時間：{st.session_state.chat_started_at}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with header_col2:
        st.markdown(
            f"""
            <div class="agent-panel">
                <div class="agent-row">
                    <div class="agent-avatar">{AGENT_PROFILE['avatar']}</div>
                    <div>
                        <p class="agent-name">{AGENT_PROFILE['name']}</p>
                        <p class="agent-title">{AGENT_PROFILE['title']}</p>
                    </div>
                </div>
                <div class="agent-online">目前狀態：線上服務中</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("歡迎使用企業客服網站。系統會先提供 FAQ 標準答案，再由 AI 客服補充說明。")

    # 若尚未設定 API Key，主畫面顯示提醒，但系統仍可使用 FAQ 模式。
    if not ai_enabled:
        st.info("目前尚未設定可用的 AI API Key，因此只能使用 FAQ 知識庫回答。")
    else:
        st.caption(f"目前 AI 供應商：{ai_mode_label} ｜ 模型：{ai_model}")

    faq_data = load_faq_data(FAQ_FILE_PATH)
    mode_text = f"AI + FAQ 雙模式（{ai_mode_label}）" if ai_enabled else "FAQ 模式（AI 未啟用）"
    render_service_overview(ai_enabled, mode_text, len(faq_data))

    selected_quick_question = show_quick_buttons()
    st.markdown("### 客服對話區")
    st.caption(
        "回覆內容固定顯示：問題分類、客服回覆、資料來源、是否建議轉人工客服。"
        " 每則訊息皆附上時間戳記與案件編號。"
    )

    # 先顯示既有聊天紀錄。
    for index, message in enumerate(st.session_state.messages):
        role = message.get("role", "assistant")
        if role == "user":
            with st.chat_message("user", avatar="🧑"):
                st.write(message.get("content", ""))
                st.caption(
                    f"提問時間：{message.get('timestamp', '未記錄時間')} ｜ "
                    f"案件編號：{st.session_state.case_id}"
                )
        else:
            with st.chat_message("assistant", avatar="👩‍💼"):
                render_assistant_message(message, index)

    if st.session_state.get("auto_scroll_to_latest"):
        scroll_to_latest_message()
        st.session_state.auto_scroll_to_latest = False
    elif st.session_state.get("scroll_to_top_on_load"):
        scroll_to_page_top()
        st.session_state.scroll_to_top_on_load = False

    user_input = st.chat_input("請輸入您想詢問的內容，例如：如何申請退貨？")
    final_question = selected_quick_question if selected_quick_question else user_input

    # 有收到新問題才觸發回覆流程。
    if final_question:
        final_question = final_question.strip()
        if not final_question:
            return

        # 記錄使用者問題。
        question_timestamp = current_timestamp()
        st.session_state.messages.append(
            {
                "role": "user",
                "content": final_question,
                "timestamp": question_timestamp,
            }
        )

        # 產生客服回覆並存入聊天紀錄。
        with st.spinner("客服系統正在整理回覆，請稍候..."):
            try:
                assistant_message = build_answer(
                    final_question,
                    faq_data,
                    ai_provider,
                    api_key,
                    ai_model,
                    ai_source_label,
                )
            except Exception:
                LOGGER.exception("Unhandled error during answer generation")
                assistant_message = {
                    "role": "assistant",
                    "category": classify_question(final_question),
                    "content": (
                        "抱歉，系統目前發生暫時性問題，已自動停止此次請求以避免卡住。\n"
                        "建議您稍後再試，或改由人工客服協助處理。\n"
                        "系統診斷代碼：internal_error"
                    ),
                    "suggest_human": True,
                    "source": "系統保護機制",
                    "feedback": None,
                }
        assistant_message["timestamp"] = current_timestamp()
        assistant_message["case_id"] = st.session_state.case_id
        assistant_message["agent_name"] = AGENT_PROFILE["name"]
        assistant_message["agent_title"] = AGENT_PROFILE["title"]
        st.session_state.messages.append(assistant_message)
        st.session_state.scroll_to_top_on_load = False
        st.session_state.auto_scroll_to_latest = True

        # 重新整理畫面，讓新訊息立即顯示。
        st.rerun()


if __name__ == "__main__":
    main()
