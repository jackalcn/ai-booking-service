import json
import os
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from random import randint
from typing import Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# 基本設定
# -----------------------------
FAQ_FILE_PATH = Path("faq.json")
FAQ_MATCH_THRESHOLD = 0.6
COMPANY_NAME = "NovaCare 企業客服中心"

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
def generate_ai_response(
    api_key: str,
    user_question: str,
    category: str,
    model_name: str,
) -> Tuple[Optional[str], Optional[str]]:
    """呼叫 OpenAI API 產生客服回答。"""
    if not api_key:
        return None, "missing_api_key"

    try:
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model_name,
            temperature=0.3,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"使用者問題分類：{category}\n"
                        f"使用者問題：{user_question}\n\n"
                        "請以專業親切型企業客服口吻回答，"
                        "若資料不足請明確表示並建議聯繫人工客服。"
                    ),
                },
            ],
        )

        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            return None, "empty_response"

        return answer, None

    except Exception:
        # 這裡不回傳技術細節，避免將複雜錯誤直接顯示給一般使用者。
        return None, "api_error"


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
            st.rerun()


def render_service_overview(api_key: str, faq_count: int) -> None:
    """顯示服務狀態卡，讓畫面更像正式客服儀表板。"""
    user_message_count = sum(1 for m in st.session_state.messages if m.get("role") == "user")
    mode_text = "AI + FAQ 雙模式" if api_key else "FAQ 模式（AI 未啟用）"
    mode_class = "status-ok" if api_key else "status-warn"
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


def build_answer(user_question: str, faq_data: List[Dict], api_key: str, ai_model: str) -> Dict:
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
            "目前尚未設定 AI API Key，因此只能使用 FAQ 知識庫回答。\n\n"
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
    ai_answer, error_code = generate_ai_response(api_key, user_question, category, ai_model)
    source = "AI 智慧客服"

    if error_code is None and ai_answer:
        # 若 AI 回覆中明確提到資料不足或人工客服，系統同步標示建議轉人工客服。
        suggest_human = any(keyword in ai_answer for keyword in ["資料不足", "人工客服", "聯繫客服", "無法確認"])

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

    return {
        "role": "assistant",
        "category": category,
        "content": fail_text,
        "suggest_human": True,
        "source": "AI 智慧客服（暫時不可用）",
        "feedback": None,
    }


def main() -> None:
    """主程式入口。"""
    # 載入 .env 讓 OPENAI_API_KEY 可由環境變數讀取。
    load_dotenv()
    api_key = get_runtime_setting("OPENAI_API_KEY", "")
    ai_model = get_runtime_setting("OPENAI_MODEL", "gpt-4o-mini")

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

        .block-container {
            max-width: 1180px;
            padding-top: 1.4rem;
            padding-bottom: 2rem;
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
    if not api_key:
        st.info("目前尚未設定 AI API Key，因此只能使用 FAQ 知識庫回答。")

    faq_data = load_faq_data(FAQ_FILE_PATH)
    render_service_overview(api_key, len(faq_data))

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
            assistant_message = build_answer(final_question, faq_data, api_key, ai_model)
        assistant_message["timestamp"] = current_timestamp()
        assistant_message["case_id"] = st.session_state.case_id
        assistant_message["agent_name"] = AGENT_PROFILE["name"]
        assistant_message["agent_title"] = AGENT_PROFILE["title"]
        st.session_state.messages.append(assistant_message)

        # 重新整理畫面，讓新訊息立即顯示。
        st.rerun()


if __name__ == "__main__":
    main()
