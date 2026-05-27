# AI 智慧訂房及客服系統（家登精密內部使用）

## 1. 專案定位
本專案為家登精密內部使用的「AI 智慧訂房及客服系統」，聚焦：
- 公司出差宿舍訂房申請
- 訂房異動與入住退房問題
- 宿舍規範與設備報修
- 差旅住宿核銷 FAQ 與 AI 客服

系統負責人：吳佩綺

## 2. 已完成功能
- SSO Claim 自動帶入員工資料（Query/Header/Secrets）
- 訂房申請資料落地 SQLite（booking_system.db）
- 訂房審核後台（待審核、已核准、已拒絕、已取消、已完成）
- 狀態更新通知（Email / Teams Webhook）
- 通知發送歷程記錄（notification_logs）
- FAQ 優先回答，未命中時改由 AI 回覆（OpenAI / Gemini）
- 支援以訂房編號快速查詢狀態（例如 BK-20260527-1234）
- 對話紀錄與訂房資料下載（TXT / JSON）

## 3. 視覺主題
介面參照家登精密官網視覺語言：
- 藍綠主色
- 白底高可讀資訊卡
- 洋紅色小面積點綴
- 內部儀表板式流程布局

## 4. 專案檔案結構
```text
.
├─ .streamlit/
│  ├─ config.toml
│  └─ secrets.toml.example
├─ .gitignore
├─ app.py
├─ faq.json
├─ requirements.txt
├─ runtime.txt
├─ .env.example
└─ README.md
```

## 5. 安裝方式
建議使用 Python 3.11。

```bash
pip install -r requirements.txt
```

## 6. 本機執行
```bash
streamlit run app.py
```

若找不到 `streamlit` 指令，改用：

```bash
python -m streamlit run app.py
```

## 7. AI 設定（OpenAI / Gemini）
```env
AI_PROVIDER=auto

OPENAI_API_KEY=你的 OpenAI API Key
OPENAI_MODEL=gpt-4o-mini

GOOGLE_API_KEY=你的 Google Gemini API Key
GEMINI_MODEL=gemini-2.0-flash
```

說明：
- `AI_PROVIDER=auto`：自動偵測可用金鑰（優先 OpenAI）
- `AI_PROVIDER=openai|gemini`：強制使用指定供應商
- 若未設定 API Key，系統仍可用 FAQ 模式

## 8. SSO 自動帶入設定
### 8.1 啟用
```env
ENABLE_SSO=true
```

### 8.2 Claim 來源
系統依序讀取：
1. URL Query（emp_id, name, dept, email, ext, roles）
2. HTTP Header（x-employee-id, x-user-name, x-user-dept, x-user-email, x-user-ext, x-user-roles）
3. .env / Secrets 預設值（SSO_EMPLOYEE_ID 等）

### 8.3 本機測試範例
```text
http://localhost:8501/?emp_id=GD10258&name=王小明&dept=資訊部&email=wang@gudeng.com&ext=1688&roles=admin
```

## 9. 審核後台設定
```env
ADMIN_EMPLOYEE_IDS=GD10258,GD20001
ADMIN_NAMES=吳佩綺
ADMIN_REVIEW_PASSCODE=你的臨時解鎖碼
```

說明：
- 若 SSO 角色或員編命中管理名單，可直接進入審核後台
- 若未命中，可用 `ADMIN_REVIEW_PASSCODE` 臨時解鎖

## 10. 通知設定（Email / Teams）
### 10.1 Email
```env
NOTIFY_EMAIL_ENABLED=true
SMTP_HOST=smtp.office365.com
SMTP_PORT=587
SMTP_USERNAME=your_account
SMTP_PASSWORD=your_password
SMTP_FROM=noreply@gudeng.com
SMTP_USE_TLS=true
BOOKING_NOTIFY_TO_EMAIL=admin1@gudeng.com,admin2@gudeng.com
```

### 10.2 Teams Webhook
```env
NOTIFY_TEAMS_ENABLED=true
TEAMS_WEBHOOK_URL=你的TeamsWebhookURL
```

## 11. 資料庫說明
- 檔案：`booking_system.db`
- 主要資料表：
1. `bookings`：訂房申請主檔與審核資訊
2. `notification_logs`：通知發送結果

## 12. FAQ 維護格式
`faq.json` 每筆格式如下：

```json
{
  "question": "常見問題",
  "answer": "標準回覆",
  "keywords": ["關鍵字1", "關鍵字2"],
  "category": "分類"
}
```

## 13. 部署到 Streamlit Community Cloud
1. 推送程式到 GitHub
2. 建立 Streamlit App，主程式選 `app.py`
3. 於 Secrets 填入 AI / SSO / 通知設定
4. 部署後驗證：
- SSO 欄位自動帶入
- 訂房寫入資料庫
- 審核狀態更新
- Email / Teams 通知

---
建議正式上線前再補上：公司 SSO Gateway、DB 備份策略、通知重送機制、權限稽核報表。
