import json
import logging
import os
import re
import smtplib
import sqlite3
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from email.message import EmailMessage
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
DB_FILE_PATH = Path("booking_system.db")
FAQ_MATCH_THRESHOLD = 0.58
COMPANY_NAME = "家登精密工業股份有限公司"
APP_NAME = "AI 智慧訂房及客服系統"
APP_SUBTITLE = "家登精密內部差旅宿舍訂房與客服平台"
INTERNAL_USE_NOTICE = "僅限家登精密員工及授權同仁使用"
SYSTEM_OWNER = "吳佩綺"
OPENAI_TIMEOUT_SECONDS = 20
AI_CONNECT_TIMEOUT_SECONDS = 6
AI_READ_TIMEOUT_SECONDS = 12
AI_TOTAL_TIMEOUT_SECONDS = 25
BOOKING_STATUS_OPTIONS = ["待審核", "已核准", "已拒絕", "已取消", "已完成"]
BOOKING_ID_PATTERN = re.compile(r"BK-\d{8}-\d{4}")

AGENT_PROFILE = {
    "name": "林芷涵",
    "title": "行政總務智慧助理",
    "avatar": "GD",
}

SYSTEM_PROMPT = """
你是家登精密內部使用的 AI 智慧訂房與客服助理。
你只能回答以下主題：公司出差宿舍訂房申請、入住退房規範、訂房異動、宿舍設備報修、差旅核銷與聯絡窗口。
請使用繁體中文回答。
回答要清楚、簡潔、有禮貌，並優先給可執行步驟。
你不可捏造房況、核准結果、費用金額或人事政策；若資料不足，必須明確說明並建議轉人工窗口確認。
""".strip()

LOGGER = logging.getLogger(__name__)

GEMINI_MODEL_FALLBACKS = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash-latest"]

AI_ERROR_HINTS = {
    "missing_api_key": "尚未設定可用的 API Key，請先在 Secrets 或 .env 補上金鑰。",
    "ai_timeout": "AI 回覆逾時，可能是網路或服務壅塞，請稍後重試或簡化問題。",
    "openai_api_error": "OpenAI 呼叫失敗，請檢查金鑰、模型名稱與額度是否正常。",
    "gemini_invalid_api_key": "Google API Key 無效或已失效，請到 Google AI Studio 重新產生後更新設定。",
    "gemini_permission_denied": "目前的 Google API Key 權限不足，請確認已開啟 Generative Language API。",
    "gemini_api_not_enabled": "此 Google 專案尚未啟用 Generative Language API，請先啟用後再試。",
    "gemini_key_restricted": "目前 API Key 設有 HTTP referrer/IP 限制，伺服器端無法使用；請改用可供伺服器端呼叫的金鑰。",
    "gemini_quota_exceeded": "Google Gemini 配額已達上限，請等待配額重置或更換可用 API Key。",
    "gemini_model_not_found": "目前設定的 GEMINI_MODEL 不可用，建議改為 gemini-2.0-flash。",
    "gemini_bad_request": "Gemini 請求格式或模型設定不被接受，請檢查 GEMINI_MODEL 與 API 設定。",
    "gemini_safety_block": "本次提問被安全政策攔截，請改寫提問內容後再試。",
    "gemini_server_error": "Google Gemini 服務暫時異常，請稍後重試。",
    "gemini_api_error": "Google Gemini 呼叫失敗，請確認網路、API 設定與服務狀態。",
    "empty_response": "AI 回傳內容為空，請稍後重試或改寫問題。",
}

AI_RETRY_HINTS = {
    "ai_timeout": "建議 20 到 30 秒後重試 1 次；若連續失敗請改由行政窗口接手。",
    "openai_api_error": "可能是供應商暫時壅塞，建議稍後重試；若連續失敗請轉人工窗口。",
    "gemini_server_error": "Google 服務暫時異常，建議 30 秒後重試；若仍失敗請轉人工窗口。",
    "gemini_api_error": "網路或 API 服務不穩，建議稍後重試 1 次。",
    "empty_response": "請將問題改短且更明確後重試，若仍無回覆建議轉人工窗口。",
    "gemini_quota_exceeded": "此錯誤通常重試無效，請等待配額重置或更換可用 API Key。",
    "gemini_invalid_api_key": "此錯誤通常重試無效，請先更新有效 API Key。",
    "gemini_key_restricted": "此錯誤通常重試無效，請先解除 API Key 的來源限制。",
    "gemini_api_not_enabled": "此錯誤通常重試無效，請先啟用 Generative Language API。",
    "gemini_model_not_found": "請先調整 GEMINI_MODEL（建議 gemini-2.0-flash）後再重試。",
    "gemini_bad_request": "請先檢查模型與設定格式，修正後再重試。",
}

HUMAN_HANDOFF_KEYWORDS = [
    "緊急",
    "保全",
    "受傷",
    "門禁",
    "遺失",
    "騷擾",
    "客訴",
    "申訴",
    "個資",
    "異常扣款",
    "報帳爭議",
    "退款",
]

HUMAN_HANDOFF_CATEGORY_KEYWORDS = {
    "緊急事件": ["緊急", "保全", "醫療", "受傷", "消防"],
    "費用與核銷": ["核銷", "扣款", "退款", "爭議"],
    "訂房異動": ["取消", "改期", "改入住", "改退房"],
}

CATEGORY_KEYWORDS = {
    "訂房申請": ["訂房", "申請", "住宿", "出差", "房型", "床位"],
    "訂房異動": ["改期", "取消", "修改", "延住", "提前退房"],
    "入住與退房": ["入住", "退房", "check in", "check out", "門禁", "鑰匙"],
    "宿舍規範": ["規範", "門禁", "清潔", "禁菸", "宵禁", "訪客"],
    "設備報修": ["報修", "故障", "空調", "熱水", "網路", "設備"],
    "費用與核銷": ["核銷", "費用", "發票", "報帳", "補助", "扣款"],
    "系統操作": ["登入", "帳號", "權限", "看不到", "送出失敗"],
    "聯絡窗口": ["窗口", "分機", "電話", "信箱", "人工"],
    "緊急事件": ["緊急", "保全", "受傷", "遺失", "危險"],
}

DEPARTMENT_OPTIONS = [
    "製造中心",
    "研發中心",
    "品保中心",
    "供應鏈管理",
    "資訊部",
    "財務部",
    "管理部",
    "其他",
]

TRIP_CITY_OPTIONS = ["新北土城", "新竹", "台中", "台南樹谷", "高雄路竹", "其他"]
DORM_OPTIONS = ["土城宿舍 A 棟", "土城宿舍 B 棟", "樹谷宿舍", "高雄宿舍", "待分配"]
ROOM_TYPE_OPTIONS = ["單人房", "雙人房", "女性樓層", "無障礙需求", "待安排"]


# -----------------------------
# 通用工具
# -----------------------------
def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def split_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def first_non_empty(*values: str) -> str:
    for value in values:
        text = str(value).strip()
        if text:
            return text
    return ""


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


def current_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_case_id(prefix: str = "CS") -> str:
    date_part = datetime.now().strftime("%Y%m%d")
    serial_part = randint(1000, 9999)
    return f"{prefix}-{date_part}-{serial_part}"


def get_query_param_value(key: str) -> str:
    try:
        value = st.query_params.get(key, "")
    except Exception:
        return ""

    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    return str(value).strip()


def get_context_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    try:
        raw_headers = getattr(st.context, "headers", None)
        if raw_headers:
            for key, value in raw_headers.items():
                headers[str(key).lower()] = str(value)
    except Exception:
        return headers

    return headers


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def calculate_similarity(text_a: str, text_b: str) -> float:
    return SequenceMatcher(None, normalize_text(text_a), normalize_text(text_b)).ratio()


def keyword_similarity(user_question: str, keywords: List[str]) -> float:
    if not keywords:
        return 0.0

    question_text = normalize_text(user_question)
    scores: List[float] = []

    for keyword in keywords:
        keyword_text = normalize_text(str(keyword))
        if not keyword_text:
            continue

        ratio_score = SequenceMatcher(None, question_text, keyword_text).ratio()
        contains_score = 1.0 if keyword_text in question_text else 0.0
        scores.append(max(ratio_score, contains_score))

    return max(scores) if scores else 0.0


# -----------------------------
# SSO / 身份
# -----------------------------
def resolve_sso_profile() -> Optional[Dict[str, str]]:
    """透過 Query/Header/Secrets 取得 SSO Claim，供內網反向代理整合。"""
    sso_enabled = parse_bool(get_runtime_setting("ENABLE_SSO", "false"))
    if not sso_enabled:
        return None

    headers = get_context_headers()

    employee_id = first_non_empty(
        get_query_param_value("emp_id"),
        headers.get("x-employee-id", ""),
        get_runtime_setting("SSO_EMPLOYEE_ID", ""),
    )
    display_name = first_non_empty(
        get_query_param_value("name"),
        headers.get("x-user-name", ""),
        get_runtime_setting("SSO_USER_NAME", ""),
    )
    department = first_non_empty(
        get_query_param_value("dept"),
        headers.get("x-user-dept", ""),
        get_runtime_setting("SSO_USER_DEPT", ""),
    )
    company_email = first_non_empty(
        get_query_param_value("email"),
        headers.get("x-user-email", ""),
        get_runtime_setting("SSO_USER_EMAIL", ""),
    )
    contact_ext = first_non_empty(
        get_query_param_value("ext"),
        headers.get("x-user-ext", ""),
        get_runtime_setting("SSO_USER_EXT", ""),
    )

    roles_text = first_non_empty(
        get_query_param_value("roles"),
        headers.get("x-user-roles", ""),
        get_runtime_setting("SSO_USER_ROLES", ""),
    )
    roles = split_csv(roles_text)

    if not employee_id and not display_name and not company_email:
        return None

    return {
        "employee_id": employee_id or "",
        "name": display_name or "未命名同仁",
        "department": department or "其他",
        "company_email": company_email or "",
        "contact_ext": contact_ext or "",
        "roles": roles,
        "source": "SSO Claim",
    }


def resolve_manual_profile() -> Optional[Dict[str, str]]:
    manual_profile = st.session_state.get("manual_profile", {})
    if not isinstance(manual_profile, dict):
        return None

    employee_id = str(manual_profile.get("employee_id", "")).strip()
    name = str(manual_profile.get("name", "")).strip()
    company_email = str(manual_profile.get("company_email", "")).strip()

    if not employee_id and not name and not company_email:
        return None

    roles = split_csv(str(manual_profile.get("roles", "")))

    return {
        "employee_id": employee_id,
        "name": name or "未命名同仁",
        "department": str(manual_profile.get("department", "其他")).strip() or "其他",
        "company_email": company_email,
        "contact_ext": str(manual_profile.get("contact_ext", "")).strip(),
        "roles": roles,
        "source": "手動身份",
    }


def get_active_profile() -> Optional[Dict[str, str]]:
    sso_profile = resolve_sso_profile()
    if sso_profile:
        st.session_state.user_profile = sso_profile
        return sso_profile

    manual_profile = resolve_manual_profile()
    if manual_profile:
        st.session_state.user_profile = manual_profile
        return manual_profile

    profile = st.session_state.get("user_profile")
    if isinstance(profile, dict) and profile:
        return profile

    return None


def is_admin_user(profile: Optional[Dict[str, str]]) -> bool:
    admin_ids = split_csv(get_runtime_setting("ADMIN_EMPLOYEE_IDS", ""))
    admin_names = split_csv(get_runtime_setting("ADMIN_NAMES", SYSTEM_OWNER))

    if profile:
        employee_id = str(profile.get("employee_id", "")).strip()
        name = str(profile.get("name", "")).strip()
        roles = [role.lower() for role in profile.get("roles", [])]

        if employee_id and employee_id in admin_ids:
            return True
        if name and name in admin_names:
            return True
        if any(role in {"admin", "reviewer", "manager", "hr", "it"} for role in roles):
            return True

    return bool(st.session_state.get("admin_unlocked", False))


# -----------------------------
# 資料庫
# -----------------------------
def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database() -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bookings (
                booking_id TEXT PRIMARY KEY,
                employee_id TEXT NOT NULL,
                traveler_name TEXT NOT NULL,
                department TEXT,
                company_email TEXT,
                trip_city TEXT,
                dormitory TEXT,
                room_type TEXT,
                check_in TEXT,
                check_out TEXT,
                nights INTEGER,
                contact_ext TEXT,
                need_parking INTEGER DEFAULT 0,
                need_reimbursement_doc INTEGER DEFAULT 0,
                late_arrival INTEGER DEFAULT 0,
                special_note TEXT,
                status TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                created_by TEXT,
                reviewer TEXT,
                reviewer_comment TEXT,
                assigned_room TEXT,
                updated_at TEXT,
                updated_by TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notification_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                booking_id TEXT,
                event_type TEXT,
                channel TEXT,
                recipient TEXT,
                title TEXT,
                message TEXT,
                success INTEGER,
                error_detail TEXT,
                created_at TEXT
            )
            """
        )
        conn.commit()


def row_to_booking(row: sqlite3.Row) -> Dict:
    return {
        "booking_id": row["booking_id"],
        "employee_id": row["employee_id"],
        "traveler_name": row["traveler_name"],
        "department": row["department"] or "",
        "company_email": row["company_email"] or "",
        "trip_city": row["trip_city"] or "",
        "dormitory": row["dormitory"] or "",
        "room_type": row["room_type"] or "",
        "check_in": row["check_in"] or "",
        "check_out": row["check_out"] or "",
        "nights": int(row["nights"] or 0),
        "contact_ext": row["contact_ext"] or "",
        "need_parking": bool(row["need_parking"]),
        "need_reimbursement_doc": bool(row["need_reimbursement_doc"]),
        "late_arrival": bool(row["late_arrival"]),
        "special_note": row["special_note"] or "",
        "status": row["status"] or "待審核",
        "submitted_at": row["submitted_at"] or "",
        "created_by": row["created_by"] or "",
        "reviewer": row["reviewer"] or "",
        "reviewer_comment": row["reviewer_comment"] or "",
        "assigned_room": row["assigned_room"] or "",
        "updated_at": row["updated_at"] or "",
        "updated_by": row["updated_by"] or "",
    }


def insert_booking_record(record: Dict) -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO bookings (
                booking_id, employee_id, traveler_name, department, company_email,
                trip_city, dormitory, room_type, check_in, check_out, nights,
                contact_ext, need_parking, need_reimbursement_doc, late_arrival,
                special_note, status, submitted_at, created_by, reviewer,
                reviewer_comment, assigned_room, updated_at, updated_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("booking_id"),
                record.get("employee_id"),
                record.get("traveler_name"),
                record.get("department"),
                record.get("company_email"),
                record.get("trip_city"),
                record.get("dormitory"),
                record.get("room_type"),
                record.get("check_in"),
                record.get("check_out"),
                int(record.get("nights", 0)),
                record.get("contact_ext"),
                1 if record.get("need_parking") else 0,
                1 if record.get("need_reimbursement_doc") else 0,
                1 if record.get("late_arrival") else 0,
                record.get("special_note"),
                record.get("status"),
                record.get("submitted_at"),
                record.get("created_by"),
                record.get("reviewer", ""),
                record.get("reviewer_comment", ""),
                record.get("assigned_room", ""),
                record.get("updated_at", ""),
                record.get("updated_by", ""),
            ),
        )
        conn.commit()


def fetch_booking_records(
    limit: int = 200,
    status_filter: str = "全部",
    employee_id: str = "",
) -> List[Dict]:
    clauses: List[str] = []
    values: List[str] = []

    if status_filter != "全部":
        clauses.append("status = ?")
        values.append(status_filter)

    employee_id = employee_id.strip()
    if employee_id:
        clauses.append("employee_id = ?")
        values.append(employee_id)

    where_clause = ""
    if clauses:
        where_clause = "WHERE " + " AND ".join(clauses)

    query = (
        "SELECT * FROM bookings "
        f"{where_clause} "
        "ORDER BY submitted_at DESC "
        "LIMIT ?"
    )
    values.append(str(limit))

    with get_db_connection() as conn:
        rows = conn.execute(query, values).fetchall()

    return [row_to_booking(row) for row in rows]


def fetch_booking_by_id(booking_id: str) -> Optional[Dict]:
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM bookings WHERE booking_id = ?", (booking_id,)).fetchone()

    if not row:
        return None

    return row_to_booking(row)


def update_booking_review(
    booking_id: str,
    new_status: str,
    reviewer: str,
    reviewer_comment: str,
    assigned_room: str,
    updated_by: str,
) -> bool:
    if new_status not in BOOKING_STATUS_OPTIONS:
        return False

    with get_db_connection() as conn:
        current = conn.execute("SELECT booking_id FROM bookings WHERE booking_id = ?", (booking_id,)).fetchone()
        if not current:
            return False

        conn.execute(
            """
            UPDATE bookings
            SET status = ?, reviewer = ?, reviewer_comment = ?, assigned_room = ?,
                updated_at = ?, updated_by = ?
            WHERE booking_id = ?
            """,
            (
                new_status,
                reviewer.strip(),
                reviewer_comment.strip(),
                assigned_room.strip(),
                current_timestamp(),
                updated_by.strip(),
                booking_id,
            ),
        )
        conn.commit()

    return True


def log_notification_event(
    booking_id: str,
    event_type: str,
    channel: str,
    recipient: str,
    title: str,
    message: str,
    success: bool,
    error_detail: str = "",
) -> None:
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO notification_logs (
                booking_id, event_type, channel, recipient, title, message,
                success, error_detail, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                booking_id,
                event_type,
                channel,
                recipient,
                title,
                message,
                1 if success else 0,
                error_detail,
                current_timestamp(),
            ),
        )
        conn.commit()


def fetch_notification_logs(booking_id: str, limit: int = 30) -> List[Dict]:
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT log_id, booking_id, event_type, channel, recipient, title,
                   success, error_detail, created_at
            FROM notification_logs
            WHERE booking_id = ?
            ORDER BY log_id DESC
            LIMIT ?
            """,
            (booking_id, limit),
        ).fetchall()

    logs: List[Dict] = []
    for row in rows:
        logs.append(
            {
                "時間": row["created_at"],
                "事件": row["event_type"],
                "頻道": row["channel"],
                "收件者": row["recipient"],
                "標題": row["title"],
                "結果": "成功" if row["success"] else "失敗",
                "錯誤": row["error_detail"] or "-",
            }
        )
    return logs


# -----------------------------
# 通知
# -----------------------------
def resolve_notification_config() -> Dict[str, str]:
    return {
        "email_enabled": parse_bool(get_runtime_setting("NOTIFY_EMAIL_ENABLED", "false")),
        "smtp_host": get_runtime_setting("SMTP_HOST", ""),
        "smtp_port": get_runtime_setting("SMTP_PORT", "587"),
        "smtp_username": get_runtime_setting("SMTP_USERNAME", ""),
        "smtp_password": get_runtime_setting("SMTP_PASSWORD", ""),
        "smtp_from": get_runtime_setting("SMTP_FROM", ""),
        "smtp_use_tls": parse_bool(get_runtime_setting("SMTP_USE_TLS", "true")),
        "booking_notify_to_email": get_runtime_setting("BOOKING_NOTIFY_TO_EMAIL", ""),
        "teams_enabled": parse_bool(get_runtime_setting("NOTIFY_TEAMS_ENABLED", "false")),
        "teams_webhook_url": get_runtime_setting("TEAMS_WEBHOOK_URL", ""),
    }


def send_email_notification(
    cfg: Dict[str, str],
    recipients: List[str],
    subject: str,
    body: str,
) -> Tuple[bool, str]:
    if not cfg.get("email_enabled"):
        return False, "email_disabled"

    if not recipients:
        return False, "no_recipient"

    smtp_host = str(cfg.get("smtp_host", "")).strip()
    smtp_from = str(cfg.get("smtp_from", "")).strip()
    if not smtp_host or not smtp_from:
        return False, "smtp_incomplete"

    try:
        smtp_port = int(str(cfg.get("smtp_port", "587")))
    except Exception:
        smtp_port = 587

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = smtp_from
    message["To"] = ", ".join(recipients)
    message.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as smtp:
            if cfg.get("smtp_use_tls"):
                smtp.starttls()
            username = str(cfg.get("smtp_username", "")).strip()
            password = str(cfg.get("smtp_password", "")).strip()
            if username:
                smtp.login(username, password)
            smtp.send_message(message)
        return True, "ok"
    except Exception as exc:
        LOGGER.exception("Email notify failed")
        return False, str(exc)


def send_teams_notification(cfg: Dict[str, str], title: str, body: str) -> Tuple[bool, str]:
    if not cfg.get("teams_enabled"):
        return False, "teams_disabled"

    webhook_url = str(cfg.get("teams_webhook_url", "")).strip()
    if not webhook_url:
        return False, "teams_webhook_missing"

    payload = {
        "text": f"{title}\n\n{body}",
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=12)
        if response.status_code >= 400:
            return False, f"http_{response.status_code}"
        return True, "ok"
    except Exception as exc:
        LOGGER.exception("Teams notify failed")
        return False, str(exc)


def notify_booking_submission(record: Dict, cfg: Dict[str, str]) -> str:
    booking_id = str(record.get("booking_id", ""))
    title = f"[新訂房申請] {booking_id}"
    body = (
        f"入住人：{record.get('traveler_name', '-')}（{record.get('employee_id', '-')}）\n"
        f"部門：{record.get('department', '-')}\n"
        f"入住：{record.get('check_in', '-')} 至 {record.get('check_out', '-')}\n"
        f"宿舍：{record.get('dormitory', '-')} / 房型：{record.get('room_type', '-')}\n"
        f"備註：{record.get('special_note', '-') }"
    )

    results: List[str] = []

    recipients = split_csv(cfg.get("booking_notify_to_email", ""))
    if recipients:
        ok, detail = send_email_notification(cfg, recipients, title, body)
        log_notification_event(
            booking_id,
            "booking_submitted",
            "email",
            ", ".join(recipients),
            title,
            body,
            ok,
            "" if ok else detail,
        )
        results.append(f"Email:{'成功' if ok else '失敗'}")

    teams_ok, teams_detail = send_teams_notification(cfg, title, body)
    if cfg.get("teams_enabled"):
        log_notification_event(
            booking_id,
            "booking_submitted",
            "teams",
            "booking_channel",
            title,
            body,
            teams_ok,
            "" if teams_ok else teams_detail,
        )
        results.append(f"Teams:{'成功' if teams_ok else '失敗'}")

    return "、".join(results) if results else "未啟用通知"


def notify_booking_status_change(record: Dict, cfg: Dict[str, str], reviewer: str) -> str:
    booking_id = str(record.get("booking_id", ""))
    title = f"[訂房狀態更新] {booking_id} → {record.get('status', '-') }"
    body = (
        f"入住人：{record.get('traveler_name', '-')}\n"
        f"狀態：{record.get('status', '-')}\n"
        f"審核者：{reviewer}\n"
        f"房號：{record.get('assigned_room', '-') or '-'}\n"
        f"審核備註：{record.get('reviewer_comment', '-') or '-'}"
    )

    results: List[str] = []

    employee_email = str(record.get("company_email", "")).strip()
    if employee_email:
        ok, detail = send_email_notification(cfg, [employee_email], title, body)
        log_notification_event(
            booking_id,
            "status_changed",
            "email",
            employee_email,
            title,
            body,
            ok,
            "" if ok else detail,
        )
        results.append(f"Email:{'成功' if ok else '失敗'}")

    teams_ok, teams_detail = send_teams_notification(cfg, title, body)
    if cfg.get("teams_enabled"):
        log_notification_event(
            booking_id,
            "status_changed",
            "teams",
            "booking_channel",
            title,
            body,
            teams_ok,
            "" if teams_ok else teams_detail,
        )
        results.append(f"Teams:{'成功' if teams_ok else '失敗'}")

    return "、".join(results) if results else "未啟用通知"


# -----------------------------
# FAQ 與 AI
# -----------------------------
def load_faq_data(file_path: Path) -> List[Dict]:
    if not file_path.exists():
        st.error("找不到 faq.json，請確認檔案存在於專案根目錄。")
        return []

    try:
        with file_path.open("r", encoding="utf-8") as file:
            faq_data = json.load(file)

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
    best_item: Optional[Dict] = None
    best_score = 0.0

    for item in faq_data:
        faq_question = str(item.get("question", ""))
        faq_keywords = item.get("keywords", [])

        question_score = calculate_similarity(user_question, faq_question)
        keyword_score = keyword_similarity(user_question, faq_keywords)

        normalized_question = normalize_text(user_question)
        keyword_hit = 1.0 if any(normalize_text(str(k)) in normalized_question for k in faq_keywords) else 0.0

        total_score = (question_score * 0.55) + (keyword_score * 0.35) + (keyword_hit * 0.10)

        if total_score > best_score:
            best_score = total_score
            best_item = item

    return best_item, best_score


def classify_question(user_question: str, faq_item: Optional[Dict] = None) -> str:
    if faq_item and faq_item.get("category"):
        return str(faq_item.get("category"))

    normalized_question = normalize_text(user_question)
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in normalized_question for keyword in keywords):
            return category

    return "其他問題"


def resolve_ai_config() -> Dict[str, str]:
    provider_raw = get_runtime_setting("AI_PROVIDER", "auto").strip().lower()
    openai_key = get_runtime_setting("OPENAI_API_KEY", "")
    google_key = get_runtime_setting("GOOGLE_API_KEY", "")

    if not google_key:
        google_key = get_runtime_setting("GEMINI_API_KEY", "")

    openai_model = get_runtime_setting("OPENAI_MODEL", "gpt-4o-mini")
    gemini_model = get_runtime_setting("GEMINI_MODEL", "gemini-2.0-flash")

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
    return (
        f"問題分類：{category}\n"
        f"使用者問題：{user_question}\n\n"
        "請用家登精密內部客服口吻回覆。"
        "若問題涉及訂房異動，提醒使用者提供訂房編號與入住日期。"
        "若涉及核銷或人事政策且資料不足，請建議聯繫行政總務或人資窗口。"
    )


def map_gemini_http_error(response: requests.Response) -> str:
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
    if not error_code:
        return ""

    base_hint = AI_ERROR_HINTS.get(error_code, "")
    if not base_hint:
        return ""

    if error_code == "gemini_model_not_found":
        return f"{base_hint}（目前設定：{model_name}）"

    return base_hint


def build_retry_hint(error_code: Optional[str]) -> str:
    if not error_code:
        return ""

    return AI_RETRY_HINTS.get(error_code, "建議稍後再重試 1 次，若仍失敗請改由人工窗口接手。")


def should_suggest_human_transfer(
    user_question: str,
    category: str,
    answer_text: str = "",
    error_code: Optional[str] = None,
) -> bool:
    if error_code:
        return True

    normalized_question = normalize_text(user_question)

    if any(keyword in normalized_question for keyword in HUMAN_HANDOFF_KEYWORDS):
        return True

    category_keywords = HUMAN_HANDOFF_CATEGORY_KEYWORDS.get(category, [])
    if category_keywords and any(keyword in normalized_question for keyword in category_keywords):
        return True

    if answer_text and any(
        keyword in answer_text for keyword in ["資料不足", "無法確認", "建議聯繫", "請轉人工", "需由人工"]
    ):
        return True

    return False


def generate_openai_response(
    api_key: str,
    user_question: str,
    category: str,
    model_name: str,
) -> Tuple[Optional[str], Optional[str]]:
    if not api_key:
        return None, "missing_api_key"

    try:
        client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_SECONDS)

        response = client.chat.completions.create(
            model=model_name,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_ai_user_prompt(user_question, category)},
            ],
        )

        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            return None, "empty_response"

        return answer, None

    except Exception as exc:
        if "timeout" in str(exc).lower():
            return None, "ai_timeout"
        return None, "openai_api_error"


def _build_gemini_payload(user_question: str, category: str, api_version: str) -> dict:
    user_text = build_ai_user_prompt(user_question, category)
    if api_version == "v1beta":
        return {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "generationConfig": {"temperature": 0.3},
        }

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
    if not api_key:
        return None, "missing_api_key"

    candidate_models = [model_name]
    for fallback_model in GEMINI_MODEL_FALLBACKS:
        if fallback_model != model_name:
            candidate_models.append(fallback_model)

    api_versions = ["v1beta", "v1"]
    last_error_code = "gemini_api_error"
    started_at = monotonic()

    try:
        for candidate_model in candidate_models:
            for api_version in api_versions:
                if monotonic() - started_at >= AI_TOTAL_TIMEOUT_SECONDS:
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
        return None, "ai_timeout"
    except requests.RequestException:
        return None, "gemini_api_error"
    except Exception:
        return None, "gemini_api_error"


def generate_ai_response(
    ai_provider: str,
    api_key: str,
    user_question: str,
    category: str,
    model_name: str,
) -> Tuple[Optional[str], Optional[str]]:
    if ai_provider == "gemini":
        return generate_gemini_response(api_key, user_question, category, model_name)

    return generate_openai_response(api_key, user_question, category, model_name)


# -----------------------------
# 互動資料
# -----------------------------
def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "booking_requests" not in st.session_state:
        st.session_state.booking_requests = []

    if "case_id" not in st.session_state:
        st.session_state.case_id = generate_case_id("CS")

    if "chat_started_at" not in st.session_state:
        st.session_state.chat_started_at = current_timestamp()

    if "auto_scroll_to_latest" not in st.session_state:
        st.session_state.auto_scroll_to_latest = False

    if "scroll_to_top_on_load" not in st.session_state:
        st.session_state.scroll_to_top_on_load = True

    if "manual_profile" not in st.session_state:
        st.session_state.manual_profile = {}

    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {}

    if "admin_unlocked" not in st.session_state:
        st.session_state.admin_unlocked = False


def get_case_status() -> str:
    messages = st.session_state.get("messages", [])
    user_count = sum(1 for message in messages if message.get("role") == "user")
    bookings = st.session_state.get("booking_requests", [])

    if user_count == 0 and not bookings:
        return "待提問"

    if any(message.get("role") == "assistant" and message.get("suggest_human") for message in messages):
        return "建議人工接手"

    pending_count = sum(1 for item in bookings if item.get("status") == "待審核")
    if pending_count:
        return "待審核案件處理中"

    return "流程運作中"


def refresh_booking_cache(profile: Optional[Dict[str, str]], is_admin: bool) -> None:
    if is_admin:
        st.session_state.booking_requests = fetch_booking_records(limit=300)
        return

    employee_id = ""
    if profile:
        employee_id = str(profile.get("employee_id", "")).strip()

    st.session_state.booking_requests = fetch_booking_records(limit=120, employee_id=employee_id)


def build_chat_transcript() -> str:
    lines = [
        f"{APP_NAME} 對話紀錄",
        f"公司：{COMPANY_NAME}",
        f"系統負責人：{SYSTEM_OWNER}",
        f"案件編號：{st.session_state.get('case_id', '-')}",
        f"建立時間：{st.session_state.get('chat_started_at', '-')}",
        "",
    ]

    booking_requests = st.session_state.get("booking_requests", [])
    if booking_requests:
        lines.append("[訂房申請紀錄]")
        for booking in booking_requests:
            lines.append(
                f"- {booking.get('booking_id', '-')}: {booking.get('traveler_name', '-')}, "
                f"{booking.get('check_in', '-')} 至 {booking.get('check_out', '-')}, "
                f"{booking.get('dormitory', '-')}, 狀態={booking.get('status', '-') }"
            )
        lines.append("")

    for message in st.session_state.get("messages", []):
        role = message.get("role", "assistant")
        timestamp = message.get("timestamp", "未記錄時間")

        if role == "user":
            lines.append(f"[{timestamp}] 使用者提問")
            lines.append(message.get("content", ""))
            lines.append("")
            continue

        lines.append(f"[{timestamp}] 客服回覆（{AGENT_PROFILE['name']} / {AGENT_PROFILE['title']}）")
        lines.append(f"問題分類：{message.get('category', '其他問題')}")
        lines.append(f"客服回覆：{message.get('content', '')}")
        lines.append(f"資料來源：{message.get('source', '未知')}")
        transfer_text = "是" if message.get("suggest_human", False) else "否"
        lines.append(f"是否建議轉人工窗口：{transfer_text}")
        lines.append("")

    return "\n".join(lines)


def build_booking_export_json() -> str:
    return json.dumps(st.session_state.get("booking_requests", []), ensure_ascii=False, indent=2)


# -----------------------------
# 畫面
# -----------------------------
def build_sidebar(notification_cfg: Dict[str, str], sso_enabled: bool) -> Tuple[Optional[Dict[str, str]], bool]:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="side-brand">
                <div class="side-logo">GD<span class="side-dot"></span></div>
                <div>
                    <div class="side-brand-title">{APP_NAME}</div>
                    <div class="side-brand-sub">{INTERNAL_USE_NOTICE}</div>
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

        profile = get_active_profile()

        st.subheader("身份與權限")
        if profile:
            st.success(
                f"已登入：{profile.get('name', '-') }（{profile.get('employee_id', '-') }）\n"
                f"來源：{profile.get('source', '-') }"
            )
            st.caption(f"部門：{profile.get('department', '-') }｜信箱：{profile.get('company_email', '-') or '-'}")
        elif sso_enabled:
            st.warning("SSO 已啟用，但目前尚未收到身份 Claim。可使用下方手動身份欄位進行測試。")
        else:
            st.caption("目前以一般模式執行，未啟用 SSO。")

        with st.expander("手動身份（開發測試）", expanded=not bool(profile)):
            manual_emp = st.text_input("員工編號", key="manual_emp")
            manual_name = st.text_input("姓名", key="manual_name")
            manual_dept = st.text_input("部門", key="manual_dept")
            manual_email = st.text_input("公司信箱", key="manual_email")
            manual_ext = st.text_input("分機", key="manual_ext")
            manual_roles = st.text_input("角色（逗號分隔）", key="manual_roles")

            col_a, col_b = st.columns(2)
            if col_a.button("套用手動身份", use_container_width=True):
                st.session_state.manual_profile = {
                    "employee_id": manual_emp,
                    "name": manual_name,
                    "department": manual_dept,
                    "company_email": manual_email,
                    "contact_ext": manual_ext,
                    "roles": manual_roles,
                }
                st.rerun()

            if col_b.button("清除手動身份", use_container_width=True):
                st.session_state.manual_profile = {}
                st.session_state.user_profile = {}
                st.rerun()

        admin = is_admin_user(profile)
        passcode = get_runtime_setting("ADMIN_REVIEW_PASSCODE", "")

        if not admin and passcode:
            st.caption("審核後台需管理權限")
            admin_code = st.text_input("審核解鎖碼", type="password", key="admin_unlock_code")
            if st.button("解鎖審核後台", use_container_width=True):
                if admin_code.strip() == passcode.strip():
                    st.session_state.admin_unlocked = True
                    st.success("審核權限已解鎖。")
                    st.rerun()
                else:
                    st.error("解鎖碼錯誤。")

        admin = is_admin_user(profile)
        if admin:
            st.success("目前具備審核後台權限")

        st.markdown(f"**案件編號：** {st.session_state.case_id}")
        st.markdown(f"**案件建立時間：** {st.session_state.chat_started_at}")

        st.divider()
        st.subheader("內部聯絡窗口")
        st.write(f"系統負責人：{SYSTEM_OWNER}")
        st.write("行政總務分機：1608")
        st.write("服務信箱：sales@gudeng.com")
        st.write("服務時段：週一至週五 08:30 - 17:30")

        st.divider()
        st.subheader("整合狀態")
        st.caption(f"SSO：{'啟用' if sso_enabled else '未啟用'}（Query/Header/Secrets Claim）")
        st.caption(f"資料庫：{DB_FILE_PATH.name}")
        st.caption(f"Email 通知：{'啟用' if notification_cfg.get('email_enabled') else '未啟用'}")
        st.caption(f"Teams 通知：{'啟用' if notification_cfg.get('teams_enabled') else '未啟用'}")

        st.divider()
        if st.session_state.messages:
            st.download_button(
                label="下載對話紀錄（TXT）",
                data=build_chat_transcript(),
                file_name=f"{st.session_state.case_id}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        if st.session_state.booking_requests:
            st.download_button(
                label="下載訂房申請（JSON）",
                data=build_booking_export_json(),
                file_name=f"booking-{st.session_state.case_id}.json",
                mime="application/json",
                use_container_width=True,
            )

        if st.button("清除對話紀錄", use_container_width=True):
            st.session_state.messages = []
            st.session_state.case_id = generate_case_id("CS")
            st.session_state.chat_started_at = current_timestamp()
            st.session_state.auto_scroll_to_latest = False
            st.session_state.scroll_to_top_on_load = True
            st.rerun()

    return profile, admin


def render_service_overview(ai_enabled: bool, mode_text: str, faq_count: int) -> None:
    user_message_count = sum(1 for message in st.session_state.messages if message.get("role") == "user")
    booking_count = len(st.session_state.booking_requests)
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
                <div class="tile-title">本次提問數</div>
                <div class="tile-value">{user_message_count} 題</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="info-tile">
                <div class="tile-title">訂房申請數</div>
                <div class="tile-value">{booking_count} 件（{case_status}）</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def scroll_to_latest_message() -> None:
    components.html(
        """
        <script>
        let attempts = 0;
        const maxAttempts = 24;

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
    st.markdown("### 快速提問")
    quick_questions = [
        "我要申請出差宿舍",
        "如何修改入住日期",
        "可以取消已送出的訂房嗎",
        "宿舍設備壞了怎麼報修",
        "差旅住宿怎麼核銷",
        "緊急狀況聯絡窗口",
    ]

    selected_question: Optional[str] = None
    with st.container(border=True):
        st.caption("點選常見主題可快速取得標準回答。")
        columns = st.columns(3)

        for index, question in enumerate(quick_questions):
            if columns[index % 3].button(question, key=f"quick_btn_{index}", use_container_width=True):
                selected_question = question

    return selected_question


def render_assistant_message(message: Dict, index: int) -> None:
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
    st.markdown(f"**是否建議轉人工窗口：** {transfer_text}")

    col1, col2, _ = st.columns([1, 1, 4])
    if col1.button("👍 有幫助", key=f"feedback_up_{index}"):
        message["feedback"] = "helpful"

    if col2.button("👎 需人工協助", key=f"feedback_down_{index}"):
        message["feedback"] = "not_helpful"

    feedback_status = message.get("feedback")
    if feedback_status == "helpful":
        st.caption("感謝回饋，我們會持續優化內部客服品質。")
    elif feedback_status == "not_helpful":
        st.warning("已收到回饋，建議改由行政總務窗口接手。")


# -----------------------------
# 訂房
# -----------------------------
def validate_booking_form(
    employee_id: str,
    traveler_name: str,
    check_in: date,
    check_out: date,
    company_email: str,
) -> Optional[str]:
    if not employee_id.strip():
        return "請輸入員工編號。"

    if not traveler_name.strip():
        return "請輸入入住人姓名。"

    if company_email.strip() and "@" not in company_email:
        return "公司信箱格式不正確。"

    if check_out <= check_in:
        return "退房日期需晚於入住日期。"

    if check_in < date.today():
        return "入住日期不可早於今天。"

    if (check_out - check_in).days > 30:
        return "單次住宿天數不可超過 30 天，請拆單或聯繫行政窗口。"

    return None


def build_booking_record(
    employee_id: str,
    traveler_name: str,
    department: str,
    company_email: str,
    trip_city: str,
    dormitory: str,
    room_type: str,
    check_in: date,
    check_out: date,
    contact_ext: str,
    need_parking: bool,
    need_reimbursement_doc: bool,
    late_arrival: bool,
    special_note: str,
    created_by: str,
) -> Dict:
    return {
        "booking_id": generate_case_id("BK"),
        "employee_id": employee_id.strip(),
        "traveler_name": traveler_name.strip(),
        "department": department,
        "company_email": company_email.strip(),
        "trip_city": trip_city,
        "dormitory": dormitory,
        "room_type": room_type,
        "check_in": check_in.strftime("%Y-%m-%d"),
        "check_out": check_out.strftime("%Y-%m-%d"),
        "nights": (check_out - check_in).days,
        "contact_ext": contact_ext.strip() or "未填寫",
        "need_parking": need_parking,
        "need_reimbursement_doc": need_reimbursement_doc,
        "late_arrival": late_arrival,
        "special_note": special_note.strip() or "無",
        "status": "待審核",
        "submitted_at": current_timestamp(),
        "created_by": created_by,
        "reviewer": "",
        "reviewer_comment": "",
        "assigned_room": "",
        "updated_at": "",
        "updated_by": "",
    }


def build_booking_assistant_reply(record: Dict, notify_result: str) -> str:
    parking_text = "需要" if record.get("need_parking") else "不需要"
    reimburse_text = "需要" if record.get("need_reimbursement_doc") else "不需要"
    late_text = "是" if record.get("late_arrival") else "否"

    return (
        "已收到您的出差宿舍訂房申請，以下為摘要：\n\n"
        f"- 訂房編號：{record.get('booking_id')}\n"
        f"- 入住人：{record.get('traveler_name')}（員編 {record.get('employee_id')}）\n"
        f"- 部門：{record.get('department')}\n"
        f"- 宿舍：{record.get('dormitory')} / 房型需求：{record.get('room_type')}\n"
        f"- 入住期間：{record.get('check_in')} 至 {record.get('check_out')}（{record.get('nights')} 晚）\n"
        f"- 停車需求：{parking_text}｜核銷文件：{reimburse_text}｜晚到：{late_text}\n"
        "- 目前狀態：待審核\n"
        f"- 通知結果：{notify_result}\n\n"
        "行政總務將依房況與差旅優先順序安排。若需緊急改期，請直接聯繫分機 1608。"
    )


def render_booking_workspace(
    profile: Optional[Dict[str, str]],
    is_admin: bool,
    notification_cfg: Dict[str, str],
    sso_enabled: bool,
) -> None:
    st.markdown("### 出差宿舍訂房申請")

    lock_identity_fields = bool(profile and profile.get("source") == "SSO Claim")

    default_emp = str(profile.get("employee_id", "")).strip() if profile else ""
    default_name = str(profile.get("name", "")).strip() if profile else ""
    default_department = str(profile.get("department", "")).strip() if profile else ""
    default_email = str(profile.get("company_email", "")).strip() if profile else ""
    default_ext = str(profile.get("contact_ext", "")).strip() if profile else ""

    department_options = DEPARTMENT_OPTIONS.copy()
    if default_department and default_department not in department_options:
        department_options.insert(0, default_department)

    dept_index = 0
    if default_department and default_department in department_options:
        dept_index = department_options.index(default_department)

    with st.container(border=True):
        st.caption("流程：員工提交申請 → 行政總務審核房況 → 通知入住安排。")
        if sso_enabled:
            st.caption("目前支援 SSO Claim 自動帶入（Query/Header/Secrets）。")

        with st.form("booking_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                employee_id = st.text_input("員工編號", value=default_emp, placeholder="例如：GD10258", disabled=lock_identity_fields)
                department = st.selectbox("部門", department_options, index=dept_index, disabled=lock_identity_fields)
                contact_ext = st.text_input("聯絡分機", value=default_ext, placeholder="例如：1688")

            with col2:
                traveler_name = st.text_input("入住人姓名", value=default_name, placeholder="請填寫中文姓名", disabled=lock_identity_fields)
                company_email = st.text_input("公司信箱", value=default_email, placeholder="name@gudeng.com", disabled=lock_identity_fields)
                trip_city = st.selectbox("出差地點", TRIP_CITY_OPTIONS)

            with col3:
                dormitory = st.selectbox("宿舍據點", DORM_OPTIONS)
                room_type = st.selectbox("房型需求", ROOM_TYPE_OPTIONS)
                check_in = st.date_input("入住日期", value=date.today() + timedelta(days=1))
                check_out = st.date_input("退房日期", value=date.today() + timedelta(days=2))

            col4, col5, col6 = st.columns(3)
            with col4:
                need_parking = st.checkbox("需停車位")
            with col5:
                need_reimbursement_doc = st.checkbox("需核銷證明")
            with col6:
                late_arrival = st.checkbox("預計 22:00 後入住")

            special_note = st.text_area("備註", placeholder="例如：同住人、交通需求、設備需求")
            submit_booking = st.form_submit_button("送出訂房申請", use_container_width=True)

        if submit_booking:
            validation_error = validate_booking_form(employee_id, traveler_name, check_in, check_out, company_email)
            if validation_error:
                st.error(validation_error)
            else:
                created_by = profile.get("name", employee_id) if profile else employee_id
                booking_record = build_booking_record(
                    employee_id,
                    traveler_name,
                    department,
                    company_email,
                    trip_city,
                    dormitory,
                    room_type,
                    check_in,
                    check_out,
                    contact_ext,
                    need_parking,
                    need_reimbursement_doc,
                    late_arrival,
                    special_note,
                    created_by,
                )

                try:
                    insert_booking_record(booking_record)
                except sqlite3.IntegrityError:
                    st.error("訂房編號重複，請重新送出。")
                    return
                except Exception:
                    LOGGER.exception("Insert booking failed")
                    st.error("資料庫寫入失敗，請稍後重試。")
                    return

                notify_result = notify_booking_submission(booking_record, notification_cfg)
                refresh_booking_cache(profile, is_admin)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "category": "訂房申請",
                        "content": build_booking_assistant_reply(booking_record, notify_result),
                        "suggest_human": False,
                        "source": "內部訂房流程（含資料庫與通知）",
                        "feedback": None,
                        "timestamp": current_timestamp(),
                    }
                )

                st.session_state.auto_scroll_to_latest = True
                st.session_state.scroll_to_top_on_load = False
                st.success(f"訂房申請已送出：{booking_record.get('booking_id')}（通知：{notify_result}）")
                st.rerun()

    st.markdown("#### 近期訂房申請")
    booking_requests = st.session_state.get("booking_requests", [])

    if not booking_requests:
        st.caption("目前查無訂房紀錄。")
        return

    recent_items = list(reversed(booking_requests[:10]))
    table_rows = []
    for item in recent_items:
        table_rows.append(
            {
                "訂房編號": item.get("booking_id", "-"),
                "入住人": item.get("traveler_name", "-"),
                "部門": item.get("department", "-"),
                "宿舍": item.get("dormitory", "-"),
                "入住": item.get("check_in", "-"),
                "退房": item.get("check_out", "-"),
                "狀態": item.get("status", "-"),
                "審核者": item.get("reviewer", "-"),
                "房號": item.get("assigned_room", "-"),
            }
        )

    st.dataframe(table_rows, hide_index=True, use_container_width=True)


def render_admin_review_workspace(
    profile: Optional[Dict[str, str]],
    is_admin: bool,
    notification_cfg: Dict[str, str],
) -> None:
    st.markdown("### 訂房審核後台")

    if not is_admin:
        st.info("目前未具備審核後台權限。若需臨時操作，可請管理員提供解鎖碼。")
        return

    filter_col1, filter_col2 = st.columns([1, 1])
    with filter_col1:
        status_filter = st.selectbox("狀態篩選", ["全部"] + BOOKING_STATUS_OPTIONS, key="admin_status_filter")
    with filter_col2:
        keyword = st.text_input("關鍵字（訂房編號/員編/姓名）", key="admin_keyword")

    records = fetch_booking_records(limit=400, status_filter=status_filter)
    if keyword.strip():
        keyword_text = keyword.strip().lower()
        records = [
            record
            for record in records
            if keyword_text in str(record.get("booking_id", "")).lower()
            or keyword_text in str(record.get("employee_id", "")).lower()
            or keyword_text in str(record.get("traveler_name", "")).lower()
        ]

    if not records:
        st.caption("目前無符合條件的訂房紀錄。")
        return

    table_rows = []
    for item in records:
        table_rows.append(
            {
                "訂房編號": item.get("booking_id", "-"),
                "員工": item.get("employee_id", "-"),
                "入住人": item.get("traveler_name", "-"),
                "入住": item.get("check_in", "-"),
                "退房": item.get("check_out", "-"),
                "狀態": item.get("status", "-"),
                "審核者": item.get("reviewer", "-"),
                "更新時間": item.get("updated_at", "-") or "-",
            }
        )
    st.dataframe(table_rows, hide_index=True, use_container_width=True)

    booking_ids = [item.get("booking_id", "") for item in records]
    selected_booking_id = st.selectbox("選擇要審核的訂房編號", booking_ids, key="admin_selected_booking")
    selected = fetch_booking_by_id(selected_booking_id)
    if not selected:
        return

    st.caption(
        f"目前狀態：{selected.get('status', '-') } ｜ "
        f"入住期間：{selected.get('check_in', '-') } 至 {selected.get('check_out', '-') } ｜ "
        f"宿舍：{selected.get('dormitory', '-') }"
    )

    current_status = selected.get("status", "待審核")
    status_index = BOOKING_STATUS_OPTIONS.index(current_status) if current_status in BOOKING_STATUS_OPTIONS else 0

    with st.form("admin_review_form"):
        new_status = st.selectbox("更新狀態", BOOKING_STATUS_OPTIONS, index=status_index)
        assigned_room = st.text_input("指派房號", value=selected.get("assigned_room", ""), placeholder="例如：A-1205")
        reviewer_comment = st.text_area("審核備註", value=selected.get("reviewer_comment", ""))
        send_notice = st.checkbox("更新後立即通知入住人", value=True)
        submit_review = st.form_submit_button("送出審核結果", use_container_width=True)

    if submit_review:
        reviewer_name = profile.get("name", SYSTEM_OWNER) if profile else SYSTEM_OWNER
        updater = profile.get("employee_id", reviewer_name) if profile else reviewer_name

        success = update_booking_review(
            selected_booking_id,
            new_status,
            reviewer_name,
            reviewer_comment,
            assigned_room,
            updater,
        )

        if not success:
            st.error("更新失敗，請重新整理後再試。")
            return

        updated = fetch_booking_by_id(selected_booking_id)
        notify_result = "未通知"
        if updated and send_notice:
            notify_result = notify_booking_status_change(updated, notification_cfg, reviewer_name)

        refresh_booking_cache(profile, is_admin)
        st.success(f"審核結果已更新（通知：{notify_result}）。")
        st.rerun()

    logs = fetch_notification_logs(selected_booking_id, limit=15)
    if logs:
        st.markdown("#### 通知紀錄")
        st.dataframe(logs, hide_index=True, use_container_width=True)


# -----------------------------
# 客服回答
# -----------------------------
def build_booking_status_answer(user_question: str) -> Optional[Dict]:
    matched = BOOKING_ID_PATTERN.search(user_question.upper())
    if not matched:
        return None

    booking_id = matched.group(0)
    record = fetch_booking_by_id(booking_id)
    if not record:
        return {
            "role": "assistant",
            "category": "訂房查詢",
            "content": f"查詢不到訂房編號 {booking_id}，請確認是否輸入正確。",
            "suggest_human": True,
            "source": "資料庫查詢",
            "feedback": None,
        }

    content = (
        f"已查到訂房編號 {booking_id}：\n"
        f"- 入住人：{record.get('traveler_name', '-')}\n"
        f"- 入住期間：{record.get('check_in', '-')} 至 {record.get('check_out', '-')}\n"
        f"- 宿舍：{record.get('dormitory', '-')}\n"
        f"- 目前狀態：{record.get('status', '-')}\n"
        f"- 審核者：{record.get('reviewer', '-') or '-'}\n"
        f"- 備註：{record.get('reviewer_comment', '-') or '-'}"
    )

    return {
        "role": "assistant",
        "category": "訂房查詢",
        "content": content,
        "suggest_human": False,
        "source": "資料庫查詢",
        "feedback": None,
    }


def build_answer(
    user_question: str,
    faq_data: List[Dict],
    ai_provider: str,
    api_key: str,
    ai_model: str,
    ai_source_label: str,
) -> Dict:
    booking_status_answer = build_booking_status_answer(user_question)
    if booking_status_answer:
        return booking_status_answer

    best_faq, faq_score = find_best_faq(user_question, faq_data)
    category = classify_question(user_question, best_faq)

    source = "FAQ 內部知識庫"

    if best_faq and faq_score >= FAQ_MATCH_THRESHOLD:
        answer_text = str(best_faq.get("answer", "目前資料不足，建議您聯繫行政窗口確認。"))
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

    if not api_key:
        fallback_text = (
            "目前尚未設定可用的 AI API Key，因此僅能提供 FAQ 內部知識庫回答。\n\n"
            "此問題在現有 FAQ 中資料不足，建議您聯繫行政總務分機 1608 進一步確認。"
        )

        return {
            "role": "assistant",
            "category": category,
            "content": fallback_text,
            "suggest_human": True,
            "source": "FAQ 內部知識庫（AI 未啟用）",
            "feedback": None,
        }

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

    fail_text = (
        "抱歉，系統目前暫時無法完成 AI 回覆。\n"
        "建議您稍後再試，或改由行政總務窗口協助處理。"
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


# -----------------------------
# 主程式
# -----------------------------
def main() -> None:
    load_dotenv()
    ai_config = resolve_ai_config()
    ai_provider = ai_config.get("provider", "openai")
    api_key = ai_config.get("api_key", "")
    ai_model = ai_config.get("model", "gpt-4o-mini")
    ai_mode_label = ai_config.get("provider_label", "未啟用")
    ai_source_label = ai_config.get("source_label", "AI")
    ai_enabled = bool(api_key)
    notification_cfg = resolve_notification_config()
    sso_enabled = parse_bool(get_runtime_setting("ENABLE_SSO", "false"))

    st.set_page_config(page_title=APP_NAME, page_icon="🏢", layout="wide")

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&family=Rajdhani:wght@600;700&display=swap');

        :root {
            --gd-blue: #0c5ca8;
            --gd-cyan: #22b8d8;
            --gd-magenta: #c32694;
            --gd-text: #163a59;
            --gd-border: #c6deef;
        }

        html, body, [class*="css"] {
            font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
            color: var(--gd-text);
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 16%, rgba(34, 184, 216, 0.16), transparent 30%),
                radial-gradient(circle at 88% 8%, rgba(195, 38, 148, 0.12), transparent 26%),
                radial-gradient(circle at 78% 34%, rgba(12, 92, 168, 0.08), transparent 28%),
                linear-gradient(180deg, #f5fbff 0%, #ffffff 44%);
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
            padding-top: 1rem;
            padding-bottom: 2.2rem;
            padding-left: 1.2rem;
            padding-right: 1.2rem;
        }

        .header-panel {
            background: #ffffff;
            border: 1px solid var(--gd-border);
            border-top: 4px solid var(--gd-cyan);
            border-left: 6px solid var(--gd-blue);
            border-radius: 14px;
            padding: 18px 20px;
            box-shadow: 0 10px 20px rgba(9, 72, 132, 0.09);
        }

        .brand-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-logo {
            width: 58px;
            height: 58px;
            border-radius: 14px;
            background: linear-gradient(135deg, #18b2d4 0%, #0c5ca8 78%);
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Rajdhani', sans-serif;
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: 1px;
            position: relative;
            box-shadow: 0 8px 16px rgba(9, 75, 138, 0.28);
        }

        .brand-logo::after {
            content: '';
            width: 9px;
            height: 9px;
            border-radius: 50%;
            background: var(--gd-magenta);
            position: absolute;
            top: 9px;
            right: 8px;
        }

        .brand-name {
            margin: 0;
            font-size: 1.5rem;
            font-weight: 700;
            color: #0f4379;
        }

        .brand-subtitle {
            margin: 2px 0 0 0;
            color: #4a6684;
            font-size: 0.95rem;
        }

        .meta-row {
            margin-top: 10px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .meta-chip {
            border-radius: 999px;
            padding: 5px 11px;
            font-size: 0.84rem;
            font-weight: 600;
            border: 1px solid #bed9ec;
            background: #eef8ff;
            color: #265785;
        }

        .agent-panel {
            background: #ffffff;
            border: 1px solid var(--gd-border);
            border-radius: 14px;
            box-shadow: 0 10px 18px rgba(10, 76, 140, 0.08);
            padding: 14px;
            min-height: 130px;
        }

        .agent-row {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .agent-avatar {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            border: 1px solid #9fc7e3;
            background: linear-gradient(150deg, #ffffff 0%, #ddf4fb 100%);
            color: #0e4f8f;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Rajdhani', sans-serif;
            font-size: 1rem;
            font-weight: 700;
            position: relative;
        }

        .agent-avatar::after {
            content: '';
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--gd-magenta);
            position: absolute;
            right: 2px;
            top: 2px;
        }

        .agent-name {
            margin: 0;
            color: #0f3966;
            font-weight: 700;
            font-size: 1rem;
        }

        .agent-title {
            margin: 2px 0 0 0;
            color: #4a6786;
            font-size: 0.85rem;
        }

        .agent-online {
            margin-top: 10px;
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid #88d9d0;
            background: #e9fcf8;
            color: #0d6e5b;
            font-size: 0.84rem;
            font-weight: 700;
        }

        .info-tile {
            background: #ffffff;
            border: 1px solid var(--gd-border);
            border-radius: 12px;
            padding: 12px 14px;
            box-shadow: 0 4px 12px rgba(15, 67, 120, 0.07);
            min-height: 90px;
        }

        .tile-title {
            color: #4a6484;
            font-size: 0.88rem;
            margin-bottom: 6px;
        }

        .tile-value {
            color: #0e3f73;
            font-weight: 700;
            font-size: 1.07rem;
        }

        .status-ok {
            color: #0f6c5c;
            background: #e8fbf5;
            border: 1px solid #9be2d7;
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.86rem;
        }

        .status-warn {
            color: #7f5c01;
            background: #fff7df;
            border: 1px solid #efd58d;
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.86rem;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7fcff 0%, #ebf7ff 100%);
            border-right: 1px solid #d4e5f3;
        }

        .side-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            background: #ffffff;
            border: 1px solid #d1e4f2;
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 10px;
        }

        .side-logo {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            background: linear-gradient(140deg, #18b2d4 0%, #0d5ca8 100%);
            color: #ffffff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Rajdhani', sans-serif;
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: 0.8px;
            position: relative;
        }

        .side-logo .side-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--gd-magenta);
            position: absolute;
            right: 5px;
            top: 6px;
        }

        .side-brand-title {
            color: #104171;
            font-weight: 700;
            font-size: 0.93rem;
        }

        .side-brand-sub {
            color: #4a6888;
            font-size: 0.76rem;
            margin-top: 2px;
        }

        .side-agent-card {
            display: flex;
            gap: 10px;
            align-items: center;
            background: #ffffff;
            border: 1px solid #d1e4f2;
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 12px;
        }

        .side-agent-avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            border: 1px solid #a8cde8;
            background: #e6f7ff;
            color: #0e4f8f;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Rajdhani', sans-serif;
            font-weight: 700;
            position: relative;
        }

        .side-agent-name {
            color: #123f6f;
            font-weight: 700;
            font-size: 0.9rem;
        }

        .side-agent-title {
            color: #4b6787;
            font-size: 0.78rem;
            margin-top: 2px;
        }

        [data-testid="stChatMessage"] {
            border: 1px solid #d9e8f4;
            border-radius: 12px;
            padding: 8px 12px;
            background: #ffffff;
        }

        [data-testid="stButton"] > button,
        [data-testid="stDownloadButton"] > button {
            border-radius: 10px;
            border: 1px solid #8abbe2;
            background: linear-gradient(135deg, #ffffff 0%, #ecf8ff 100%);
            color: #125084;
            font-weight: 600;
        }

        [data-testid="stButton"] > button:hover,
        [data-testid="stDownloadButton"] > button:hover {
            border-color: #2caed1;
            color: #0d3f73;
            box-shadow: 0 4px 12px rgba(22, 124, 186, 0.18);
        }

        [data-testid="stForm"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fcff 100%);
            border: 1px solid #d8e8f4;
            border-radius: 12px;
            padding: 10px;
        }

        @media (max-width: 960px) {
            .block-container {
                padding-left: 0.9rem;
                padding-right: 0.9rem;
            }

            .brand-name {
                font-size: 1.2rem;
            }

            .brand-logo {
                width: 50px;
                height: 50px;
                font-size: 1.1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()
    init_database()

    profile, admin_mode = build_sidebar(notification_cfg, sso_enabled)
    refresh_booking_cache(profile, admin_mode)

    header_col1, header_col2 = st.columns([2.2, 1], gap="medium")

    with header_col1:
        login_name = profile.get("name", "未登入") if profile else "未登入"
        st.markdown(
            f"""
            <div class="header-panel">
                <div class="brand-row">
                    <div class="brand-logo">GD</div>
                    <div>
                        <p class="brand-name">{APP_NAME}</p>
                        <p class="brand-subtitle">{APP_SUBTITLE}</p>
                    </div>
                </div>
                <div class="meta-row">
                    <span class="meta-chip">案件編號：{st.session_state.case_id}</span>
                    <span class="meta-chip">建立時間：{st.session_state.chat_started_at}</span>
                    <span class="meta-chip">公司：{COMPANY_NAME}</span>
                    <span class="meta-chip">系統負責人：{SYSTEM_OWNER}</span>
                    <span class="meta-chip">登入者：{login_name}</span>
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
                <div class="agent-online">目前狀態：內部服務中</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("設計風格參照家登精密官網視覺語言：藍綠主色、白底清晰資訊層次、洋紅點綴識別。")
    if sso_enabled:
        st.caption("SSO 已啟用：系統會優先讀取 Query/Header/Secrets 的員工 Claim 並自動帶入訂房欄位。")

    if not ai_enabled:
        st.info("目前尚未設定可用的 AI API Key，因此系統將以 FAQ 內部知識庫回覆為主。")
    else:
        st.caption(f"目前 AI 供應商：{ai_mode_label} ｜ 模型：{ai_model}")

    faq_data = load_faq_data(FAQ_FILE_PATH)
    mode_text = f"AI + FAQ 雙模式（{ai_mode_label}）" if ai_enabled else "FAQ 模式（AI 未啟用）"
    render_service_overview(ai_enabled, mode_text, len(faq_data))

    render_booking_workspace(profile, admin_mode, notification_cfg, sso_enabled)
    render_admin_review_workspace(profile, admin_mode, notification_cfg)

    selected_quick_question = show_quick_buttons()

    st.markdown("### 客服對話區")
    st.caption("回覆內容固定顯示：問題分類、客服回覆、資料來源、是否建議轉人工窗口。")

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
            with st.chat_message("assistant", avatar="🏢"):
                render_assistant_message(message, index)

    if st.session_state.get("auto_scroll_to_latest"):
        scroll_to_latest_message()
        st.session_state.auto_scroll_to_latest = False
    elif st.session_state.get("scroll_to_top_on_load"):
        scroll_to_page_top()
        st.session_state.scroll_to_top_on_load = False

    user_input = st.chat_input("請輸入您想詢問的內容，例如：BK-20260527-1234 狀態如何？")
    final_question = selected_quick_question if selected_quick_question else user_input

    if final_question:
        final_question = final_question.strip()
        if not final_question:
            return

        st.session_state.messages.append(
            {
                "role": "user",
                "content": final_question,
                "timestamp": current_timestamp(),
            }
        )

        with st.spinner("AI 訂房客服正在整理回覆，請稍候..."):
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
                        "建議您稍後再試，或改由行政總務窗口協助。\n"
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
        st.rerun()


if __name__ == "__main__":
    main()
