from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

# 把一段 HTML 字符串包装成 HTMLResponse 顺手加上“不要缓存”的响应头
# 因为这些 demo 页面经常是调试用的，如果浏览器缓存了旧页面，你改完代码后刷新可能看不到最新效果。
def html_page(content: str) -> HTMLResponse:
    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )

# Demo 总入口页 / 导航页
@router.get("/demo", response_class=HTMLResponse)
async def demo_page() -> HTMLResponse:
    html = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agent Demo Hub</title>
  <style>
    :root {
      --bg: #f3eee6;
      --panel: rgba(255, 255, 255, 0.84);
      --text: #18212f;
      --muted: #5c6776;
      --line: rgba(24, 33, 47, 0.12);
      --accent: #163a5f;
      --accent-2: #c07d2d;
      --accent-3: #0f766e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(192, 125, 45, 0.16), transparent 30%),
        radial-gradient(circle at top right, rgba(22, 58, 95, 0.14), transparent 28%),
        linear-gradient(135deg, #f7f3eb 0%, #efe7dc 100%);
    }
    .shell {
      max-width: 1320px;
      margin: 0 auto;
      padding: 30px;
      display: grid;
      gap: 22px;
    }
    .hero, .section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: 0 24px 60px rgba(24, 33, 47, 0.12);
      backdrop-filter: blur(16px);
    }
    .hero {
      padding: 30px;
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 24px;
      align-items: center;
    }
    .kicker {
      text-transform: uppercase;
      letter-spacing: 0.22em;
      color: var(--accent-2);
      font-size: 12px;
      font-weight: 700;
    }
    h1 {
      margin: 12px 0 16px;
      font-size: 46px;
      line-height: 1.02;
    }
    .lede {
      margin: 0;
      color: var(--muted);
      line-height: 1.8;
      font-size: 16px;
      max-width: 680px;
    }
    .hero-card {
      border: 1px solid var(--line);
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.78);
      padding: 18px 20px;
      display: grid;
      gap: 12px;
    }
    .hero-card label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .hero-card code {
      font-size: 13px;
      word-break: break-word;
    }
    .hero-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin-top: 12px;
    }
    .toggle {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(22, 58, 95, 0.08);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }
    .toggle input {
      accent-color: var(--accent-3);
    }
    .chips {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .chip {
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(22, 58, 95, 0.08);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }
    .section {
      padding: 24px;
      display: grid;
      gap: 18px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 20px;
      background: rgba(255, 255, 255, 0.82);
      padding: 18px;
      display: grid;
      gap: 8px;
    }
    .stat label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .stat strong {
      font-size: 28px;
      line-height: 1;
    }
    .stat span {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    .section-head {
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 12px;
      flex-wrap: wrap;
    }
    .section-head h2 {
      margin: 0;
      font-size: 24px;
    }
    .section-head p {
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
      font-size: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 18px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.88);
      padding: 20px;
      display: grid;
      gap: 14px;
      min-height: 250px;
    }
    .card strong {
      font-size: 20px;
    }
    .card p {
      margin: 0;
      color: var(--muted);
      line-height: 1.75;
      font-size: 14px;
    }
    .card ul {
      margin: 0;
      padding-left: 18px;
      color: var(--text);
      line-height: 1.7;
      font-size: 14px;
    }
    .card a {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: fit-content;
      padding: 12px 16px;
      border-radius: 14px;
      text-decoration: none;
      font-weight: 800;
      color: white;
      background: linear-gradient(135deg, var(--accent), #2a5b86);
      box-shadow: 0 12px 28px rgba(22, 58, 95, 0.22);
    }
    .card.alt a {
      background: linear-gradient(135deg, var(--accent-3), #0b8f83);
      box-shadow: 0 12px 28px rgba(15, 118, 110, 0.22);
    }
    .card.warn a {
      background: linear-gradient(135deg, var(--accent-2), #d49441);
      box-shadow: 0 12px 28px rgba(192, 125, 45, 0.22);
    }
    .quick-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }
    .quick {
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.82);
      padding: 16px;
      text-decoration: none;
      color: var(--text);
      display: grid;
      gap: 8px;
    }
    .quick strong {
      font-size: 15px;
    }
    .quick span {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    .preview-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }
    .preview {
      border: 1px solid var(--line);
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.86);
      overflow: hidden;
    }
    .preview-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(22, 58, 95, 0.04);
    }
    .preview-head strong {
      font-size: 17px;
    }
    .preview-body {
      display: grid;
      gap: 12px;
      padding: 16px 18px 18px;
    }
    .preview-item {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.82);
      display: grid;
      gap: 6px;
      text-decoration: none;
      color: inherit;
    }
    .preview-item strong {
      font-size: 14px;
    }
    .preview-item span {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    .preview-item:hover {
      border-color: rgba(22, 58, 95, 0.24);
      transform: translateY(-1px);
      box-shadow: 0 10px 24px rgba(22, 58, 95, 0.08);
    }
    .badge {
      display: inline-flex;
      width: fit-content;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
    }
    .badge.ok {
      background: rgba(15, 118, 110, 0.12);
      color: var(--accent-3);
    }
    .badge.warn {
      background: rgba(192, 125, 45, 0.14);
      color: var(--accent-2);
    }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 18px;
      padding: 24px;
      text-align: center;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.6);
    }
    @media (max-width: 1120px) {
      .hero { grid-template-columns: 1fr; }
      .stats { grid-template-columns: 1fr 1fr; }
      .grid { grid-template-columns: 1fr; }
      .preview-grid { grid-template-columns: 1fr; }
      .quick-grid { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 720px) {
      .shell { padding: 18px; }
      h1 { font-size: 36px; }
      .stats { grid-template-columns: 1fr; }
      .quick-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div>
        <div class="kicker">Enterprise Debug Portal</div>
        <h1>Schedule Reminder Agent<br/>Demo Hub</h1>
        <p class="lede">
          这个页面把当前项目里最重要的调试入口统一收口，方便你从自然语言交互、会话历史追踪、
          AI 审计，到 API 文档和健康检查，形成一套更像企业内部运维后台的浏览体验。
        </p>
        <div class="chips">
          <span class="chip">Agent Chat</span>
          <span class="chip">Session Timeline</span>
          <span class="chip">LLM Audit</span>
          <span class="chip">OpenAPI</span>
        </div>
        <div class="hero-actions">
          <label class="toggle">
            <input id="autoRefreshToggle" type="checkbox" checked />
            自动刷新 dashboard
          </label>
          <span class="chip" id="refreshState">每 15 秒刷新一次</span>
        </div>
        <div class="hero-actions">
          <input id="adminTokenInput" class="field" style="max-width:320px;" placeholder="可选：admin token" />
          <button id="saveAdminTokenBtn" class="chip" style="border:0;cursor:pointer;">保存 Admin Token</button>
        </div>
      </div>
      <div class="hero-card">
        <div>
          <label>推荐路径</label>
          <code>1. 先去 Chat Demo 发起会话 → 2. 再到 History 看轨迹 → 3. 如需排查解析问题，跳到 LLM Audit</code>
        </div>
        <div>
          <label>当前项目风格</label>
          <code>后端主导 + 自带运维页面 + 企业级分层结构 + 可观察性优先</code>
        </div>
        <a class="quick" href="/docs">
          <strong>OpenAPI Docs</strong>
          <span>查看接口文档和调试参数。</span>
        </a>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>System Snapshot</h2>
          <p>首页直接读取现有接口，快速给你一个“当前系统是否正常、最近有没有数据”的整体感。</p>
        </div>
      </div>
      <div class="stats">
        <div class="stat">
          <label>Health</label>
          <strong id="healthStat">...</strong>
          <span id="healthMeta">正在检查服务状态</span>
        </div>
        <div class="stat">
          <label>Recent Sessions</label>
          <strong id="sessionStat">0</strong>
          <span>读取最近 Agent 会话快照列表</span>
        </div>
        <div class="stat">
          <label>Recent History</label>
          <strong id="historyStat">0</strong>
          <span>读取最近会话历史条数</span>
        </div>
        <div class="stat">
          <label>Recent Audit</label>
          <strong id="auditStat">0</strong>
          <span>读取最近 LLM 审计条数</span>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Core Consoles</h2>
          <p>从用户交互、会话轨迹到 AI 审计，按调试链路组织入口。</p>
        </div>
      </div>
      <div class="grid">
        <article class="card">
          <strong>Chat Demo</strong>
          <p>用于直接体验 <code>/api/v1/agent/chat</code> 的会话确认与执行流程，最适合做第一步手工验收。</p>
          <ul>
            <li>支持新建 session</li>
            <li>支持确认执行</li>
            <li>适合观察基础交互状态</li>
          </ul>
          <a href="/api/v1/demo/chat">Open Chat Demo</a>
        </article>

        <article class="card alt">
          <strong>Agent History</strong>
          <p>用于查看完整会话轨迹，适合回放某个 session 从 <code>confirm</code> 到 <code>reply</code> 的全过程。</p>
          <ul>
            <li>支持筛选和分页</li>
            <li>支持查看单个 session 时间线</li>
            <li>支持联动关联 schedule 详情</li>
          </ul>
          <a href="/api/v1/demo/agent-history">Open History Console</a>
        </article>

        <article class="card warn">
          <strong>LLM Audit</strong>
          <p>用于排查规则解析失败后的 LLM 兜底链路，适合查看 request/response、repair 阶段和错误原因。</p>
          <ul>
            <li>支持按 session_id 精确过滤</li>
            <li>支持按 success / parser_stage 排查</li>
            <li>适合 AI 接入联调</li>
          </ul>
          <a href="/api/v1/demo/llm-audit">Open LLM Audit</a>
        </article>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Quick Links</h2>
          <p>常用 API 和文档入口，方便你在浏览器和命令行之间来回切换。</p>
        </div>
      </div>
      <div class="quick-grid">
        <a class="quick" href="/docs">
          <strong>OpenAPI Docs</strong>
          <span>查看所有接口定义、入参、返回结构和调试表单。</span>
        </a>
        <a class="quick" href="/api/v1/health">
          <strong>Health Check</strong>
          <span>快速确认当前服务是否存活。</span>
        </a>
        <a class="quick" href="/api/v1/demo/agent-history">
          <strong>History API</strong>
          <span>直接访问会话历史分页接口。</span>
        </a>
        <a class="quick" href="/api/v1/demo/llm-audit">
          <strong>LLM Audit API</strong>
          <span>直接访问 LLM 审计分页接口。</span>
        </a>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Recent Activity</h2>
          <p>不离开首页也能先扫一眼最近会话和最近审计，有问题再跳转到对应页面深挖。</p>
        </div>
      </div>
      <div class="preview-grid">
        <section class="preview">
          <div class="preview-head">
            <strong>Recent Session History</strong>
            <a href="/api/v1/demo/agent-history">Open History</a>
          </div>
          <div class="preview-body" id="historyPreview">
            <div class="empty">正在加载最近会话历史...</div>
          </div>
        </section>
        <section class="preview">
          <div class="preview-head">
            <strong>Recent LLM Audit</strong>
            <a href="/api/v1/demo/llm-audit">Open Audit</a>
          </div>
          <div class="preview-body" id="auditPreview">
            <div class="empty">正在加载最近 AI 审计...</div>
          </div>
        </section>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Recent Schedules</h2>
          <p>首页顺手预览最近日程，方便把 Agent 结果和业务实体快速对上。</p>
        </div>
      </div>
      <div class="preview">
        <div class="preview-head">
          <strong>Recent Schedule Records</strong>
          <a href="/docs">Open API Docs</a>
        </div>
        <div class="preview-body" id="schedulePreview">
          <div class="empty">正在加载最近日程...</div>
        </div>
      </div>
    </section>
  </div>

  <script>
    const autoRefreshToggle = document.getElementById("autoRefreshToggle");
    const refreshState = document.getElementById("refreshState");
    const adminTokenInput = document.getElementById("adminTokenInput");
    const saveAdminTokenBtn = document.getElementById("saveAdminTokenBtn");
    const adminTokenStatus = document.getElementById("adminTokenStatus");
    const adminTokenStatus = document.getElementById("adminTokenStatus");
    const adminTokenStatus = document.getElementById("adminTokenStatus");
    const healthStat = document.getElementById("healthStat");
    const healthMeta = document.getElementById("healthMeta");
    const sessionStat = document.getElementById("sessionStat");
    const historyStat = document.getElementById("historyStat");
    const auditStat = document.getElementById("auditStat");
    const historyPreview = document.getElementById("historyPreview");
    const auditPreview = document.getElementById("auditPreview");
    const schedulePreview = document.getElementById("schedulePreview");
    let dashboardTimer = null;

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function renderEmpty(target, text) {
      target.innerHTML = `<div class="empty">${escapeHtml(text)}</div>`;
    }

    function getAdminToken() {
      return window.localStorage.getItem("demo_admin_token") || "";
    }

    function buildAdminHeaders() {
      const token = getAdminToken();
      return token ? { "X-Admin-Token": token } : {};
    }

    async function fetchAdminJson(url) {
      const response = await fetch(url, { headers: buildAdminHeaders() });
      return response.json();
    }

    function initializeAdminTokenInput() {
      adminTokenInput.value = getAdminToken();
      adminTokenStatus.textContent = getAdminToken() ? "Loaded saved admin token" : "No saved admin token";
    }

    function renderHistoryPreview(items) {
      if (!items.length) {
        renderEmpty(historyPreview, "当前没有最近会话历史。");
        return;
      }
      historyPreview.innerHTML = items.map(item => `
        <a class="preview-item" href="/api/v1/demo/agent-history">
          <strong>${escapeHtml(item.session_id)}</strong>
          <span>${escapeHtml(item.agent_state)} · parser=${escapeHtml(item.parser_source || "-")} · intent=${escapeHtml(item.intent)}</span>
          <span>${escapeHtml(item.user_input)}</span>
          ${item.suggested_inputs && item.suggested_inputs.length
            ? `<span>建议下一句：${escapeHtml(item.suggested_inputs.join(" | "))}</span>`
            : ""}
        </a>
      `).join("");
    }

    function renderAuditPreview(items) {
      if (!items.length) {
        renderEmpty(auditPreview, "当前没有最近 AI 审计。");
        return;
      }
      auditPreview.innerHTML = items.map(item => `
        <a class="preview-item" href="/api/v1/demo/llm-audit">
          <strong>${escapeHtml(item.session_id)}</strong>
          <span class="badge ${item.success ? "ok" : "warn"}">${item.success ? "SUCCESS" : "FAILED"}</span>
          <span>${escapeHtml(item.parser_stage)} · model=${escapeHtml(item.model_name || "-")}</span>
          <span>${escapeHtml(item.user_input)}</span>
        </a>
      `).join("");
    }

    function renderSchedulePreview(items) {
      if (!items.length) {
        renderEmpty(schedulePreview, "当前没有最近日程。");
        return;
      }
      schedulePreview.innerHTML = items.map(item => `
        <a class="preview-item" href="/api/v1/demo/agent-history">
          <strong>#${escapeHtml(item.id)} · ${escapeHtml(item.content)}</strong>
          <span>${escapeHtml(item.status)} · ${escapeHtml(item.cycle_rule)}</span>
          <span>${escapeHtml(item.schedule_date || "-")} ${escapeHtml(item.schedule_time || "")}</span>
        </a>
      `).join("");
    }

    function updateAutoRefreshState() {
      refreshState.textContent = autoRefreshToggle.checked
        ? "每 15 秒刷新一次"
        : "已关闭自动刷新";
    }

    function startAutoRefresh() {
      stopAutoRefresh();
      if (!autoRefreshToggle.checked) return;
      dashboardTimer = window.setInterval(() => {
        loadDashboard();
      }, 15000);
    }

    function stopAutoRefresh() {
      if (dashboardTimer != null) {
        window.clearInterval(dashboardTimer);
        dashboardTimer = null;
      }
    }

    function handleAdminUnauthorizedOnHub(scheduleItems) {
      stopAutoRefresh();
      autoRefreshToggle.checked = false;
      updateAutoRefreshState();
      healthStat.textContent = "AUTH";
      healthMeta.textContent = "Admin token is required. Please save a valid token.";
      historyStat.textContent = "0";
      auditStat.textContent = "0";
      renderEmpty(historyPreview, "Admin access denied. Please save a valid admin token.");
      renderEmpty(auditPreview, "Admin access denied. Please save a valid admin token.");
      renderSchedulePreview(scheduleItems || []);
    }

    async function loadDashboard() {
      try {
        const [healthResponse, sessionResponse, historyResponse, auditResponse, scheduleResponse] = await Promise.all([
          fetch("/api/v1/health"),
          fetch("/api/v1/agent/sessions?limit=5&offset=0"),
          fetch("/api/v1/admin/agent/sessions/history?limit=5&offset=0", { headers: buildAdminHeaders() }),
          fetch("/api/v1/admin/agent/llm-audit?limit=5&offset=0", { headers: buildAdminHeaders() }),
          fetch("/api/v1/schedule?limit=5&offset=0"),
        ]);

        const healthBody = await healthResponse.json();
        const sessionBody = await sessionResponse.json();
        const historyBody = await historyResponse.json();
        const auditBody = await auditResponse.json();
        const scheduleBody = await scheduleResponse.json();

        if (historyResponse.status === 401 || auditResponse.status === 401) {
          handleAdminUnauthorizedOnHub(scheduleBody?.data?.items || []);
          return;
        }

        healthStat.textContent = healthBody?.data?.status === "ok" ? "OK" : "DOWN";
        healthMeta.textContent = healthBody?.data?.status === "ok"
          ? "服务健康检查通过"
          : "服务健康检查异常";

        sessionStat.textContent = String(sessionBody?.data?.total ?? 0);
        historyStat.textContent = String(historyBody?.data?.total ?? 0);
        auditStat.textContent = String(auditBody?.data?.total ?? 0);

        renderHistoryPreview(historyBody?.data?.items || []);
        renderAuditPreview(auditBody?.data?.items || []);
        renderSchedulePreview(scheduleBody?.data?.items || []);
      } catch (error) {
        healthStat.textContent = "ERR";
        healthMeta.textContent = "首页状态加载失败";
        renderEmpty(historyPreview, "最近会话历史加载失败。");
        renderEmpty(auditPreview, "最近 AI 审计加载失败。");
        renderEmpty(schedulePreview, "最近日程加载失败。");
      }
    }

    autoRefreshToggle.addEventListener("change", () => {
      updateAutoRefreshState();
      startAutoRefresh();
    });

    saveAdminTokenBtn.addEventListener("click", async () => {
      const token = adminTokenInput.value.trim();
      window.localStorage.setItem("demo_admin_token", token);
      adminTokenStatus.textContent = token ? "admin token ????????..." : "admin token ???";
      autoRefreshToggle.checked = true;
      updateAutoRefreshState();
      startAutoRefresh();
      await loadDashboard();
    });

    initializeAdminTokenInput();
    updateAutoRefreshState();
    loadDashboard();
    startAutoRefresh();
  </script>
</body>
</html>
    """
    return html_page(html)

# 示和调试 agent 对话能力  agent 聊天交互演示页
@router.get("/demo/chat", response_class=HTMLResponse)
async def chat_demo_page() -> HTMLResponse:
    html = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Schedule Reminder Agent Chat Demo</title>
  <style>
    :root {
      --bg: #f4efe7;
      --panel: rgba(255, 255, 255, 0.78);
      --panel-strong: #ffffff;
      --text: #18212f;
      --muted: #5a6472;
      --line: rgba(24, 33, 47, 0.12);
      --accent: #163a5f;
      --accent-2: #c07d2d;
      --success: #0f766e;
      --warn: #9a3412;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(192, 125, 45, 0.18), transparent 30%),
        radial-gradient(circle at top right, rgba(22, 58, 95, 0.15), transparent 28%),
        linear-gradient(135deg, #f7f2ea 0%, #f3ede2 100%);
    }
    .shell {
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px;
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 24px;
      min-height: 100vh;
    }
    .brand, .panel {
      background: var(--panel);
      backdrop-filter: blur(16px);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 24px 60px rgba(24, 33, 47, 0.12);
    }
    .brand {
      padding: 28px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .kicker {
      text-transform: uppercase;
      letter-spacing: 0.22em;
      color: var(--accent-2);
      font-size: 12px;
      font-weight: 700;
    }
    h1 {
      margin: 12px 0 14px;
      font-size: 42px;
      line-height: 1.02;
    }
    .lede {
      color: var(--muted);
      line-height: 1.7;
      font-size: 16px;
      margin: 0 0 18px;
    }
    .tag {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(22, 58, 95, 0.09);
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 10px;
    }
    .meta {
      display: grid;
      gap: 14px;
      margin-top: 18px;
    }
    .meta-card {
      background: rgba(255, 255, 255, 0.58);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
    }
    .meta-card label {
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .meta-card code {
      font-size: 13px;
      word-break: break-all;
    }
    .panel {
      padding: 22px;
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      gap: 16px;
    }
    .toolbar {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
    }
    .toolbar .group {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    input, textarea, button {
      font: inherit;
    }
    .field, .text {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.82);
      color: var(--text);
      padding: 14px 16px;
      outline: none;
      transition: border-color .2s ease, box-shadow .2s ease;
    }
    .field:focus, .text:focus {
      border-color: rgba(22, 58, 95, 0.38);
      box-shadow: 0 0 0 4px rgba(22, 58, 95, 0.08);
    }
    .text {
      min-height: 160px;
      resize: vertical;
      line-height: 1.7;
    }
    .btn {
      border: 0;
      border-radius: 14px;
      padding: 12px 16px;
      cursor: pointer;
      font-weight: 700;
      transition: transform .15s ease, opacity .15s ease, box-shadow .15s ease;
    }
    .btn:hover { transform: translateY(-1px); }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent), #264d74);
      color: white;
      box-shadow: 0 10px 24px rgba(22, 58, 95, 0.25);
    }
    .btn-secondary {
      background: rgba(192, 125, 45, 0.12);
      color: #7c4a11;
    }
    .status {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      color: var(--muted);
      font-size: 14px;
    }
    .pill {
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(22, 58, 95, 0.08);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }
    .stream {
      display: grid;
      gap: 14px;
      align-content: start;
      overflow: auto;
      padding-right: 2px;
    }
    .bubble {
      padding: 16px 18px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.82);
      line-height: 1.7;
      white-space: pre-wrap;
    }
    .bubble.user {
      background: rgba(22, 58, 95, 0.08);
      margin-left: 40px;
    }
    .bubble.agent {
      margin-right: 40px;
    }
    .bubble.confirm {
      border-left: 4px solid var(--accent-2);
    }
    .bubble.reply {
      border-left: 4px solid var(--success);
    }
    .bubble.clarify {
      border-left: 4px solid var(--warn);
    }
    .footer {
      display: grid;
      gap: 12px;
    }
    .helper {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    @media (max-width: 980px) {
      .shell { grid-template-columns: 1fr; }
      h1 { font-size: 34px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="brand">
      <div>
        <div class="kicker">Schedule Reminder Agent</div>
        <h1>Chat<br/>Demo</h1>
        <p class="lede">
          这个页面用于直接体验 <code>/api/v1/agent/chat</code> 的完整会话流。
          你可以在这里测试提问、确认、执行和回复。
        </p>
        <div class="tag">企业级分层 · 会话确认 · 工具编排</div>
      </div>
      <div class="meta">
        <div class="meta-card">
          <label>Session ID</label>
          <code id="sessionLabel"></code>
        </div>
        <div class="meta-card">
          <label>当前状态</label>
          <code id="stateLabel">idle</code>
        </div>
        <div class="meta-card">
          <label>提示</label>
          <code>输入“明天下午5点提醒我开会”试试</code>
        </div>
      </div>
    </aside>

    <main class="panel">
      <div class="toolbar">
        <div class="group">
          <button class="btn btn-secondary" id="newSessionBtn">新会话</button>
          <button class="btn btn-secondary" id="reuseBtn">保留会话</button>
        </div>
        <div class="status">
          <span class="pill">LLM Fallback Ready</span>
          <span id="confirmHint">等待输入</span>
        </div>
      </div>

      <textarea id="input" class="text" placeholder="例如：明天下午5点提醒我开会"></textarea>

      <div class="toolbar">
        <div class="group">
          <button class="btn btn-primary" id="sendBtn">发送</button>
          <button class="btn btn-secondary" id="confirmBtn" disabled>确认执行</button>
        </div>
      </div>

      <div class="stream" id="stream"></div>

      <div class="footer">
        <div class="helper">
          这个 Demo 会把确认上下文保存在后端。第一次发送后如果返回 <code>confirm</code>，
          你可以点“确认执行”完成下一步。
        </div>
      </div>
    </main>
  </div>

  <script>
    const sessionKey = "schedule-reminder-session-id";
    const stream = document.getElementById("stream");
    const input = document.getElementById("input");
    const stateLabel = document.getElementById("stateLabel");
    const confirmHint = document.getElementById("confirmHint");
    const sessionLabel = document.getElementById("sessionLabel");
    const confirmBtn = document.getElementById("confirmBtn");
    const sendBtn = document.getElementById("sendBtn");
    const newSessionBtn = document.getElementById("newSessionBtn");
    const reuseBtn = document.getElementById("reuseBtn");

    let lastInput = "";
    let currentState = "idle";
    let sessionId = localStorage.getItem(sessionKey) || createSessionId();

    function createSessionId() {
      const id = `session-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
      localStorage.setItem(sessionKey, id);
      return id;
    }

    function setSessionId(nextId) {
      sessionId = nextId;
      localStorage.setItem(sessionKey, nextId);
      sessionLabel.textContent = nextId;
    }

    function renderMessage(text, role = "agent", state = "") {
      const node = document.createElement("div");
      node.className = `bubble ${role} ${state}`.trim();
      node.textContent = text;
      stream.appendChild(node);
      node.scrollIntoView({ behavior: "smooth", block: "end" });
    }

    function updateState(nextState, hint) {
      currentState = nextState;
      stateLabel.textContent = nextState;
      confirmHint.textContent = hint || "等待输入";
      confirmBtn.disabled = nextState !== "confirm";
    }

    async function sendPayload(confirmed) {
      const userInput = confirmed ? "确认" : input.value.trim();
      if (!userInput) return;

      lastInput = confirmed ? lastInput : userInput;
      renderMessage(confirmed ? "确认执行" : userInput, "user");

      const response = await fetch("/api/v1/agent/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          user_input: userInput,
          confirmed,
          context: {}
        })
      });

      const body = await response.json();
      const data = body.data;
      updateState(data.agent_state, data.user_message || "");
      renderMessage(
        [
          `状态：${data.agent_state}`,
          `意图：${data.intent}`,
          data.user_message ? `回复：${data.user_message}` : "",
          data.tool_name ? `工具：${data.tool_name}` : "",
          data.execution_result ? `结果：${JSON.stringify(data.execution_result, null, 2)}` : ""
        ].filter(Boolean).join("\\n"),
        "agent",
        data.agent_state
      );

      if (data.agent_state !== "confirm") {
        confirmBtn.disabled = true;
      }
    }

    sendBtn.addEventListener("click", async () => {
      try {
        await sendPayload(false);
      } catch (error) {
        renderMessage(`请求失败：${error.message}`, "agent");
      }
    });

    confirmBtn.addEventListener("click", async () => {
      try {
        await sendPayload(true);
      } catch (error) {
        renderMessage(`确认失败：${error.message}`, "agent");
      }
    });

    newSessionBtn.addEventListener("click", () => {
      setSessionId(createSessionId());
      updateState("idle", "已创建新会话");
      renderMessage("已切换到新会话。", "agent", "reply");
      confirmBtn.disabled = true;
    });

    reuseBtn.addEventListener("click", () => {
      setSessionId(localStorage.getItem(sessionKey) || createSessionId());
      updateState("idle", "正在使用当前会话");
      renderMessage("已保留当前会话。", "agent", "reply");
    });

    setSessionId(sessionId);
    renderMessage("欢迎来到日程提醒智能体 Chat Demo。", "agent", "reply");
  </script>
</body>
</html>
    """
    return html_page(html)

# 后台总览演示页
@router.get("/demo/dashboard", response_class=HTMLResponse)
async def dashboard_demo_page() -> HTMLResponse:
    html = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Schedule Reminder Dashboard</title>
  <style>
    :root {
      --bg: #f2ede6;
      --panel: rgba(255, 255, 255, 0.9);
      --text: #18212f;
      --muted: #5d6876;
      --line: rgba(24, 33, 47, 0.12);
      --accent: #163a5f;
      --accent-2: #b7791f;
      --ok: #0f766e;
      --warn: #9a3412;
      --code: #15202d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(183, 121, 31, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(22, 58, 95, 0.14), transparent 30%),
        linear-gradient(135deg, #f6f2ea 0%, #ebe2d4 100%);
    }
    .shell {
      max-width: 1360px;
      margin: 0 auto;
      padding: 28px;
      display: grid;
      gap: 22px;
    }
    .hero, .section {
      border: 1px solid var(--line);
      border-radius: 28px;
      background: var(--panel);
      box-shadow: 0 24px 60px rgba(24, 33, 47, 0.12);
      backdrop-filter: blur(16px);
    }
    .hero {
      padding: 28px;
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 22px;
      align-items: center;
    }
    .kicker {
      text-transform: uppercase;
      letter-spacing: 0.2em;
      font-size: 12px;
      font-weight: 800;
      color: var(--accent-2);
    }
    h1 {
      margin: 12px 0 14px;
      font-size: 44px;
      line-height: 1.02;
    }
    .lede {
      margin: 0;
      color: var(--muted);
      line-height: 1.8;
      font-size: 16px;
      max-width: 720px;
    }
    .hero-side {
      border: 1px solid var(--line);
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.82);
      padding: 18px 20px;
      display: grid;
      gap: 12px;
    }
    .hero-side label {
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .toolbar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    .btn {
      border: none;
      border-radius: 14px;
      padding: 11px 16px;
      font-weight: 800;
      cursor: pointer;
    }
    .btn:disabled {
      cursor: wait;
      opacity: 0.72;
    }
    .status-note {
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.72);
      padding: 10px 12px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    .status-note.error {
      color: #9a3412;
      border-color: rgba(154, 52, 18, 0.2);
      background: rgba(154, 52, 18, 0.06);
    }
    .status-note.success {
      color: #0f766e;
      border-color: rgba(15, 118, 110, 0.2);
      background: rgba(15, 118, 110, 0.06);
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent), #285983);
      color: #fff;
      box-shadow: 0 10px 24px rgba(22, 58, 95, 0.22);
    }
    .btn-secondary {
      background: rgba(22, 58, 95, 0.08);
      color: var(--accent);
    }
    .toggle {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(22, 58, 95, 0.08);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }
    .toggle input { accent-color: var(--ok); }
    .section {
      padding: 24px;
      display: grid;
      gap: 18px;
    }
    .section-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: end;
      flex-wrap: wrap;
    }
    .section-head h2 {
      margin: 0;
      font-size: 24px;
    }
    .section-head p {
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
      font-size: 14px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.86);
      padding: 18px;
      display: grid;
      gap: 10px;
    }
    .metric label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .metric strong {
      font-size: 30px;
      line-height: 1;
    }
    .metric .meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.7;
    }
    .layout {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }
    .stack {
      display: grid;
      gap: 18px;
    }
    .panel {
      border: 1px solid var(--line);
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.88);
      overflow: hidden;
    }
    .panel-head {
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(22, 58, 95, 0.04);
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }
    .panel-head strong {
      font-size: 17px;
    }
    .panel-body {
      padding: 16px 18px 18px;
      display: grid;
      gap: 12px;
    }
    .line {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.82);
      padding: 14px;
      display: grid;
      gap: 6px;
    }
    .line strong {
      font-size: 14px;
    }
    .line span {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.65;
    }
    pre {
      margin: 0;
      padding: 16px;
      border-radius: 18px;
      background: var(--code);
      color: #edf3fa;
      overflow: auto;
      font-size: 12px;
      line-height: 1.65;
    }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 18px;
      padding: 24px;
      text-align: center;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.62);
    }
    @media (max-width: 1120px) {
      .hero { grid-template-columns: 1fr; }
      .grid { grid-template-columns: 1fr 1fr; }
      .layout { grid-template-columns: 1fr; }
    }
    @media (max-width: 720px) {
      .shell { padding: 18px; }
      h1 { font-size: 34px; }
      .grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div>
        <div class="kicker">Operations Console</div>
        <h1>Unified Dashboard<br/>Overview</h1>
        <p class="lede">
          这个页面直接读取 <code>/api/v1/admin/dashboard/overview</code>，
          把 schedule、sessions、history、LLM audit、reminders 的核心统计集中到一个后台总览里。
        </p>
      </div>
      <div class="hero-side">
        <div>
          <label>Recommended Use</label>
          <div>先看全局概览，再决定进入哪个细分页面排查。</div>
        </div>
        <div class="toolbar">
          <button type="button" class="btn btn-primary" id="reloadBtn">Reload Dashboard</button>
          <a class="btn btn-secondary" href="/docs" target="_blank" rel="noreferrer">OpenAPI Docs</a>
        </div>
        <div class="toolbar">
          <input id="adminTokenInput" class="field" placeholder="Optional admin token" />
          <button type="button" class="btn btn-secondary" id="saveAdminTokenBtn">Save Token</button>
        </div>
        <div class="status-note" id="adminStatusNote">No admin token saved yet.</div>
        <div class="status-note" id="refreshStatusNote">Ready to load dashboard.</div>
        <div class="status-note" id="runtimeStatusNote">Runtime view is waiting for first load.</div>
        <label class="toggle">
          <input id="autoRefreshToggle" type="checkbox" checked />
          Auto refresh every 15s
        </label>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Key Metrics</h2>
          <p>这些卡片是后台首页最常见的“第一眼状态”数据。</p>
        </div>
      </div>
      <div class="grid" id="metricGrid">
        <div class="empty">Loading dashboard metrics...</div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <div>
          <h2>Module Breakdown</h2>
          <p>用于查看每个模块内部的统计分布，而不是只看一个总数。</p>
        </div>
      </div>
      <div class="layout">
        <section class="panel">
          <div class="panel-head">
            <strong>Readable Summary</strong>
            <a href="/api/v1/demo/chat" target="_blank" rel="noreferrer">Chat Demo</a>
          </div>
          <div class="panel-body" id="summaryList">
            <div class="empty">Loading module summary...</div>
          </div>
        </section>
        <div class="stack">
          <section class="panel">
            <div class="panel-head">
              <strong>Runtime Highlights</strong>
              <a href="/api/v1/demo/reminder-logs" target="_blank" rel="noreferrer">Reminder Logs</a>
            </div>
            <div class="panel-body" id="runtimeList">
              <div class="empty">Loading runtime highlights...</div>
            </div>
          </section>
          <section class="panel">
            <div class="panel-head">
              <strong>Raw Payload</strong>
              <a href="/docs" target="_blank" rel="noreferrer">OpenAPI Docs</a>
            </div>
            <div class="panel-body">
              <pre id="rawPayload">Loading...</pre>
            </div>
          </div>
        </div>
      </div>
    </section>
  </div>

  <script>
    const reloadBtn = document.getElementById("reloadBtn");
    const autoRefreshToggle = document.getElementById("autoRefreshToggle");
    const adminTokenInput = document.getElementById("adminTokenInput");
    const saveAdminTokenBtn = document.getElementById("saveAdminTokenBtn");
    const adminStatusNote = document.getElementById("adminStatusNote");
    const refreshStatusNote = document.getElementById("refreshStatusNote");
    const runtimeStatusNote = document.getElementById("runtimeStatusNote");
    const metricGrid = document.getElementById("metricGrid");
    const summaryList = document.getElementById("summaryList");
    const runtimeList = document.getElementById("runtimeList");
    const rawPayload = document.getElementById("rawPayload");
    let timer = null;

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function formatJson(value) {
      return JSON.stringify(value, null, 2);
    }

    function getAdminToken() {
      return window.localStorage.getItem("demo_admin_token") || "";
    }

    function buildAdminHeaders() {
      const token = getAdminToken();
      return token ? { "X-Admin-Token": token } : {};
    }

    function initializeAdminTokenInput() {
      adminTokenInput.value = getAdminToken();
      updateAdminStatus(getAdminToken() ? "Admin token loaded from browser storage." : "No admin token saved yet.", getAdminToken() ? "success" : "");
    }

    function updateAdminStatus(message, type = "") {
      adminStatusNote.textContent = message;
      adminStatusNote.className = type ? `status-note ${type}` : "status-note";
    }

    function updateRefreshStatus(message, type = "") {
      refreshStatusNote.textContent = message;
      refreshStatusNote.className = type ? `status-note ${type}` : "status-note";
    }

    function setReloadState(isLoading) {
      reloadBtn.disabled = isLoading;
      reloadBtn.textContent = isLoading ? "Reloading..." : "Reload Dashboard";
    }

    function currentTimeText() {
      return new Date().toLocaleTimeString("zh-CN", { hour12: false });
    }

    function renderRuntimeHighlights(overview, readiness, runtimeSummary) {
      const redis = readiness?.redis || { enabled: false, connected: false, queue_backlog: 0 };
      const workerQueue = overview.worker_queue;
      const schedulerItems = runtimeSummary?.items || [];
      const topJobs = schedulerItems.slice(0, 3);
      const lines = [
        `Redis: enabled=${redis.enabled} | connected=${redis.connected} | backlog=${redis.queue_backlog}`,
        `Worker queue: queued=${workerQueue.queued_count} | processing=${workerQueue.processing_count} | stale=${workerQueue.stale_processing_count} | failed=${workerQueue.failed_count}`,
      ];

      topJobs.forEach(item => {
        lines.push(
          `Job ${item.job_id}: runs=${item.total_runs} | success=${item.success_count} | failed=${item.failed_count} | last_status=${item.last_status || "-"} | processed_total=${item.total_processed_count}`
        );
      });

      runtimeList.innerHTML = lines.map(line => `
        <div class="line">
          <strong>Runtime Snapshot</strong>
          <span>${escapeHtml(line)}</span>
        </div>
      `).join("");
      runtimeStatusNote.textContent = `Runtime view updated at ${currentTimeText()}.`;
      runtimeStatusNote.className = "status-note success";
    }

    function renderMetrics(data) {
      const cards = [
        {
          label: "Schedules",
          value: data.schedule.total,
          meta: `active=${data.schedule.active_count} | done=${data.schedule.done_count} | overdue=${data.schedule.overdue_count}`,
        },
        {
          label: "Sessions",
          value: data.sessions.total,
          meta: `pending=${data.sessions.pending_confirmation_count} | expired=${data.sessions.expired_count}`,
        },
        {
          label: "History",
          value: data.history.total,
          meta: `confirm=${data.history.confirm_count} | clarify=${data.history.clarify_count} | reply=${data.history.reply_count}`,
        },
        {
          label: "LLM Audit",
          value: data.llm_audit.total,
          meta: `success=${data.llm_audit.success_count} | failed=${data.llm_audit.failed_count} | repair=${data.llm_audit.repair_count}`,
        },
        {
          label: "Reminders",
          value: data.reminders.total,
          meta: `sent=${data.reminders.sent_count} | failed=${data.reminders.failed_count} | pending=${data.reminders.pending_count}`,
        },
        {
          label: "Worker Queue",
          value: data.worker_queue.total,
          meta: `queued=${data.worker_queue.queued_count} | processing=${data.worker_queue.processing_count} | stale=${data.worker_queue.stale_processing_count}`,
        },
        {
          label: "Due Today",
          value: data.schedule.due_today_count,
          meta: `upcoming=${data.schedule.upcoming_count} | cancelled=${data.schedule.cancelled_count}`,
        },
      ];

      metricGrid.innerHTML = cards.map(card => `
        <article class="metric">
          <label>${escapeHtml(card.label)}</label>
          <strong>${escapeHtml(card.value)}</strong>
          <div class="meta">${escapeHtml(card.meta)}</div>
        </article>
      `).join("");
    }

    function renderSummary(data) {
      const lines = [
        `Schedule: total=${data.schedule.total}, active=${data.schedule.active_count}, cancelled=${data.schedule.cancelled_count}, done=${data.schedule.done_count}`,
        `Schedule due state: due_today=${data.schedule.due_today_count}, overdue=${data.schedule.overdue_count}, upcoming=${data.schedule.upcoming_count}`,
        `Sessions: total=${data.sessions.total}, pending_confirmation=${data.sessions.pending_confirmation_count}, expired=${data.sessions.expired_count}`,
        `History: total=${data.history.total}, confirm=${data.history.confirm_count}, clarify=${data.history.clarify_count}, reply=${data.history.reply_count}, llm_source=${data.history.llm_source_count}`,
        `LLM audit: total=${data.llm_audit.total}, success=${data.llm_audit.success_count}, failed=${data.llm_audit.failed_count}, repair=${data.llm_audit.repair_count}`,
        `Reminders: total=${data.reminders.total}, sent=${data.reminders.sent_count}, failed=${data.reminders.failed_count}, pending=${data.reminders.pending_count}`,
        `Worker queue: total=${data.worker_queue.total}, queued=${data.worker_queue.queued_count}, processing=${data.worker_queue.processing_count}, stale=${data.worker_queue.stale_processing_count}, redis_enabled=${data.worker_queue.redis_enabled}, redis_backlog=${data.worker_queue.redis_queue_backlog}`,
        `Scheduler jobs: total=${data.scheduler.total}, running=${data.scheduler.running_count}, success=${data.scheduler.success_count}, failed=${data.scheduler.failed_count}`,
      ];

      summaryList.innerHTML = lines.map(line => `
        <div class="line">
          <strong>Module Snapshot</strong>
          <span>${escapeHtml(line)}</span>
        </div>
      `).join("");
    }

    async function loadDashboard() {
      setReloadState(true);
      updateRefreshStatus("Refreshing dashboard...", "");
      try {
        const [overviewResponse, readinessResponse, runtimeResponse] = await Promise.all([
          fetch("/api/v1/admin/dashboard/overview", { headers: buildAdminHeaders() }),
          fetch("/api/v1/health/ready"),
          fetch("/api/v1/admin/scheduler/runtime-summary", { headers: buildAdminHeaders() }),
        ]);
        const body = await overviewResponse.json();
        if (overviewResponse.status === 401 || body?.error_code === "UNAUTHORIZED") {
          stopAutoRefresh();
          autoRefreshToggle.checked = false;
          updateAdminStatus("Admin access denied. Please enter the correct token and click Save Token.", "error");
          updateRefreshStatus("Refresh stopped because admin access failed.", "error");
          runtimeStatusNote.textContent = "Runtime view is unavailable until admin access succeeds.";
          runtimeStatusNote.className = "status-note error";
          metricGrid.innerHTML = '<div class="empty">Admin access denied. Please enter a valid admin token and click Save Token.</div>';
          summaryList.innerHTML = '<div class="empty">Dashboard summary is unavailable until admin access succeeds.</div>';
          runtimeList.innerHTML = '<div class="empty">Runtime highlights are unavailable until admin access succeeds.</div>';
          rawPayload.textContent = "401 Unauthorized: please save a valid admin token.";
          return;
        }
        const readinessBody = await readinessResponse.json();
        const runtimeBody = await runtimeResponse.json();
        if (!overviewResponse.ok || !body?.data) {
          throw new Error(`Dashboard request failed: ${overviewResponse.status}`);
        }
        updateAdminStatus("Admin request succeeded.", "success");
        updateRefreshStatus(`Last refreshed at ${currentTimeText()}.`, "success");
        renderMetrics(body.data);
        renderSummary(body.data);
        renderRuntimeHighlights(body.data, readinessBody?.data, runtimeBody?.data);
        rawPayload.textContent = formatJson({
          overview: body.data,
          readiness: readinessBody?.data || null,
          scheduler_runtime_summary: runtimeBody?.data || null,
        });
      } catch (error) {
        updateRefreshStatus(`Refresh failed: ${error.message}`, "error");
        runtimeStatusNote.textContent = "Runtime view failed to load.";
        runtimeStatusNote.className = "status-note error";
        metricGrid.innerHTML = '<div class="empty">Dashboard load failed.</div>';
        summaryList.innerHTML = '<div class="empty">Summary load failed.</div>';
        runtimeList.innerHTML = '<div class="empty">Runtime highlights failed to load.</div>';
        rawPayload.textContent = String(error);
      } finally {
        setReloadState(false);
      }
    }

    function startAutoRefresh() {
      stopAutoRefresh();
      if (!autoRefreshToggle.checked) return;
      timer = window.setInterval(loadDashboard, 15000);
    }

    function stopAutoRefresh() {
      if (timer != null) {
        window.clearInterval(timer);
        timer = null;
      }
    }

    reloadBtn.addEventListener("click", loadDashboard);
    autoRefreshToggle.addEventListener("change", startAutoRefresh);
    saveAdminTokenBtn.addEventListener("click", async () => {
      const token = adminTokenInput.value.trim();
      window.localStorage.setItem("demo_admin_token", token);
      if (!token) {
        updateAdminStatus("Empty token was saved. Admin requests will continue to return 401 until you enter the real token.", "error");
      } else {
        updateAdminStatus("Admin token saved. Trying dashboard request...", "success");
      }
      autoRefreshToggle.checked = true;
      startAutoRefresh();
      await loadDashboard();
    });

    initializeAdminTokenInput();
    loadDashboard();
    startAutoRefresh();
  </script>
</body>
</html>
    """
    return html_page(html)

# LLM 审计日志可视化页面
@router.get("/demo/llm-audit", response_class=HTMLResponse)
async def llm_audit_demo_page() -> HTMLResponse:
    html = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Audit Console</title>
  <style>
    :root {
      --bg: #f6f1e8;
      --panel: rgba(255, 255, 255, 0.82);
      --panel-strong: #ffffff;
      --text: #17212b;
      --muted: #687385;
      --line: rgba(23, 33, 43, 0.12);
      --accent: #0f3d57;
      --accent-2: #c9822b;
      --ok: #0f766e;
      --bad: #b42318;
      --chip: rgba(15, 61, 87, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(201, 130, 43, 0.15), transparent 26%),
        radial-gradient(circle at top right, rgba(15, 61, 87, 0.12), transparent 28%),
        linear-gradient(135deg, #f9f4eb 0%, #f3ede2 100%);
    }
    .shell {
      max-width: 1380px;
      margin: 0 auto;
      padding: 28px;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 24px;
    }
    .brand, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 24px 60px rgba(23, 33, 43, 0.12);
      backdrop-filter: blur(16px);
    }
    .brand {
      padding: 26px;
      display: flex;
      flex-direction: column;
      gap: 20px;
      height: fit-content;
      position: sticky;
      top: 24px;
    }
    .kicker {
      text-transform: uppercase;
      letter-spacing: 0.22em;
      color: var(--accent-2);
      font-size: 12px;
      font-weight: 700;
    }
    h1 {
      margin: 10px 0 12px;
      font-size: 36px;
      line-height: 1.08;
    }
    .lede {
      margin: 0;
      color: var(--muted);
      line-height: 1.75;
      font-size: 15px;
    }
    .tag {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(15, 61, 87, 0.09);
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
    }
    .fact {
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.62);
      border: 1px solid var(--line);
    }
    .fact label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .fact code {
      font-size: 13px;
      word-break: break-word;
    }
    .panel {
      padding: 22px;
      display: grid;
      gap: 18px;
      align-content: start;
    }
    .toolbar {
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
    }
    .field, select, button {
      font: inherit;
    }
    .field, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.88);
      padding: 12px 14px;
      color: var(--text);
      outline: none;
    }
    .field:focus, select:focus {
      border-color: rgba(15, 61, 87, 0.34);
      box-shadow: 0 0 0 4px rgba(15, 61, 87, 0.08);
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
    }
    .btn {
      border: 0;
      border-radius: 14px;
      padding: 12px 16px;
      cursor: pointer;
      font-weight: 700;
      transition: transform .15s ease, box-shadow .15s ease, opacity .15s ease;
    }
    .btn:hover { transform: translateY(-1px); }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent), #215779);
      color: white;
      box-shadow: 0 12px 28px rgba(15, 61, 87, 0.25);
    }
    .btn-secondary {
      background: rgba(201, 130, 43, 0.12);
      color: #8b5718;
    }
    .summary {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      color: var(--muted);
      font-size: 14px;
    }
    .pill {
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--chip);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }
    .grid {
      display: grid;
      gap: 14px;
    }
    .card {
      border-radius: 22px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.9);
      overflow: hidden;
    }
    .card-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(15, 61, 87, 0.04);
    }
    .card-title {
      display: grid;
      gap: 6px;
    }
    .card-title strong {
      font-size: 16px;
    }
    .session-link {
      color: var(--accent);
      text-decoration: none;
      border-bottom: 1px dashed rgba(15, 61, 87, 0.35);
      width: fit-content;
    }
    .session-link:hover {
      color: #0b5f86;
      border-bottom-color: rgba(11, 95, 134, 0.55);
    }
    .card-meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    .badge {
      padding: 7px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
      height: fit-content;
      white-space: nowrap;
    }
    .badge.ok {
      background: rgba(15, 118, 110, 0.12);
      color: var(--ok);
    }
    .badge.bad {
      background: rgba(180, 35, 24, 0.12);
      color: var(--bad);
    }
    .card-body {
      display: grid;
      gap: 14px;
      padding: 16px 18px 18px;
    }
    .row {
      display: grid;
      gap: 6px;
    }
    .row label {
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    pre {
      margin: 0;
      padding: 14px 16px;
      border-radius: 16px;
      background: #111927;
      color: #e5eef8;
      overflow: auto;
      font-size: 12px;
      line-height: 1.65;
    }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 22px;
      padding: 36px 24px;
      text-align: center;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.6);
    }
    @media (max-width: 1080px) {
      .shell { grid-template-columns: 1fr; }
      .brand { position: static; }
      .toolbar { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 720px) {
      .toolbar { grid-template-columns: 1fr; }
      h1 { font-size: 30px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="brand">
      <div>
        <div class="kicker">LLM Audit Console</div>
        <h1>AI 审计日志<br/>看板</h1>
        <p class="lede">
          这个页面直接读取 <code>/api/v1/admin/agent/llm-audit</code> 和
          <code>/api/v1/admin/agent/llm-audit/{session_id}</code>，
          用来查看模型调用、修复链路、失败原因和结构化结果。
        </p>
      </div>
      <div class="tag">AI 可观察性 · 会话回放 · 运维筛选</div>
      <div class="fact">
        <label>调试建议</label>
        <code>先发一次 /agent/chat，再用 session_id 过滤</code>
      </div>
      <div class="fact">
        <label>推荐组合</label>
        <code>success=false + parser_stage=repair</code>
      </div>
      <div class="fact">
        <label>当前接口</label>
        <code>/api/v1/admin/agent/llm-audit</code>
      </div>
      <div class="fact">
        <label>Admin Token</label>
        <input id="adminTokenInput" class="field" placeholder="可选：admin token" />
        <button class="btn btn-secondary" id="saveAdminTokenBtn" style="margin-top:10px;">保存 Token</button>
      </div>
    </aside>

    <main class="panel">
      <div class="toolbar">
        <input id="sessionId" class="field" placeholder="按 session_id 筛选，例如 llm-demo-2" />
        <select id="stage">
          <option value="">全部阶段</option>
          <option value="parse">parse</option>
          <option value="repair">repair</option>
        </select>
        <select id="success">
          <option value="">全部结果</option>
          <option value="true">success=true</option>
          <option value="false">success=false</option>
        </select>
        <input id="limit" class="field" type="number" min="1" max="200" value="20" placeholder="limit" />
        <input id="offset" class="field" type="number" min="0" value="0" placeholder="offset" />
      </div>

      <div class="actions">
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
          <button class="btn btn-primary" id="loadBtn">查询日志</button>
          <button class="btn btn-secondary" id="resetBtn">重置筛选</button>
          <a class="btn btn-secondary" id="exportBtn" href="/api/v1/admin/agent/llm-audit/export" target="_blank" rel="noreferrer">导出 CSV</a>
        </div>
        <div class="summary">
          <span class="pill" id="summaryTotal">总数 0</span>
          <span class="pill" id="summaryFilter">当前筛选：全部</span>
        </div>
      </div>

      <div class="summary">
        <span class="pill" id="detailTitle">当前视图：最近审计记录</span>
      </div>

      <div class="grid" id="result"></div>
    </main>
  </div>

  <script>
    const sessionIdInput = document.getElementById("sessionId");
    const stageInput = document.getElementById("stage");
    const successInput = document.getElementById("success");
    const limitInput = document.getElementById("limit");
    const offsetInput = document.getElementById("offset");
    const loadBtn = document.getElementById("loadBtn");
    const resetBtn = document.getElementById("resetBtn");
    const result = document.getElementById("result");
    const summaryTotal = document.getElementById("summaryTotal");
    const summaryFilter = document.getElementById("summaryFilter");
    const exportBtn = document.getElementById("exportBtn");
    const detailTitle = document.getElementById("detailTitle");
    const adminTokenInput = document.getElementById("adminTokenInput");
    const saveAdminTokenBtn = document.getElementById("saveAdminTokenBtn");

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function formatJson(value) {
      if (value == null) return "null";
      return JSON.stringify(value, null, 2);
    }

    function getAdminToken() {
      return window.localStorage.getItem("demo_admin_token") || "";
    }

    function buildAdminHeaders() {
      const token = getAdminToken();
      return token ? { "X-Admin-Token": token } : {};
    }

    function initializeAdminTokenInput() {
      adminTokenInput.value = getAdminToken();
      adminTokenStatus.textContent = getAdminToken() ? "??????? admin token" : "??? admin token";
    }

    function currentFilterText() {
      const parts = [];
      if (sessionIdInput.value.trim()) parts.push(`session_id=${sessionIdInput.value.trim()}`);
      if (stageInput.value) parts.push(`parser_stage=${stageInput.value}`);
      if (successInput.value) parts.push(`success=${successInput.value}`);
      if (!parts.length) return "全部";
      return parts.join(" | ");
    }

    function updateDetailTitle() {
      if (sessionIdInput.value.trim()) {
        detailTitle.textContent = `当前视图：会话 ${sessionIdInput.value.trim()} 的 AI 审计记录`;
        return;
      }
      detailTitle.textContent = "当前视图：最近审计记录";
    }

    function renderEmpty() {
      result.innerHTML = '<div class="empty">当前条件下没有找到 AI 审计日志。</div>';
    }

    function updateExportLink() {
      const params = new URLSearchParams();
      if (sessionIdInput.value.trim()) params.set("session_id", sessionIdInput.value.trim());
      if (stageInput.value) params.set("parser_stage", stageInput.value);
      if (successInput.value) params.set("success", successInput.value);
      exportBtn.href = `/api/v1/admin/agent/llm-audit/export?${params.toString()}`;
    }

    function renderCards(items) {
      if (!items.length) {
        renderEmpty();
        return;
      }

      result.innerHTML = items.map(item => {
        const badgeClass = item.success ? "ok" : "bad";
        const badgeText = item.success ? "SUCCESS" : "FAILED";
        return `
          <section class="card">
            <div class="card-head">
              <div class="card-title">
                <strong>${escapeHtml(item.session_id)}</strong>
                <a class="session-link" href="#" data-session-id="${escapeHtml(item.session_id)}">按此 session_id 查看明细</a>
                <div class="card-meta">
                  id=${item.id} · stage=${escapeHtml(item.parser_stage)} · provider=${escapeHtml(item.provider || "-")} · model=${escapeHtml(item.model_name || "-")}
                </div>
                <div class="card-meta">
                  created_at=${escapeHtml(item.created_at)} · updated_at=${escapeHtml(item.updated_at)}
                </div>
              </div>
              <span class="badge ${badgeClass}">${badgeText}</span>
            </div>
            <div class="card-body">
              <div class="row">
                <label>User Input</label>
                <pre>${escapeHtml(item.user_input)}</pre>
              </div>
              <div class="row">
                <label>Request Payload</label>
                <pre>${escapeHtml(formatJson(item.request_payload))}</pre>
              </div>
              <div class="row">
                <label>Parsed Response</label>
                <pre>${escapeHtml(formatJson(item.parsed_response))}</pre>
              </div>
              <div class="row">
                <label>Raw Response Text</label>
                <pre>${escapeHtml(item.raw_response_text || "")}</pre>
              </div>
              <div class="row">
                <label>Error Message</label>
                <pre>${escapeHtml(item.error_message || "")}</pre>
              </div>
            </div>
          </section>
        `;
      }).join("");

      result.querySelectorAll("[data-session-id]").forEach(node => {
        node.addEventListener("click", async (event) => {
          event.preventDefault();
          sessionIdInput.value = node.getAttribute("data-session-id") || "";
          offsetInput.value = "0";
          await loadAuditLogs();
        });
      });
    }

    async function loadAuditLogs() {
      const params = new URLSearchParams();
      if (sessionIdInput.value.trim()) params.set("session_id", sessionIdInput.value.trim());
      if (stageInput.value) params.set("parser_stage", stageInput.value);
      if (successInput.value) params.set("success", successInput.value);
      params.set("limit", limitInput.value || "20");
      params.set("offset", offsetInput.value || "0");

      const response = await fetch(`/api/v1/admin/agent/llm-audit?${params.toString()}`, { headers: buildAdminHeaders() });
      const body = await response.json();
      const data = body.data;
      summaryTotal.textContent = `总数 ${data.total}`;
      summaryFilter.textContent = `当前筛选：${currentFilterText()}`;
      updateDetailTitle();
      updateExportLink();
      renderCards(data.items);
    }

    loadBtn.addEventListener("click", async () => {
      try {
        await loadAuditLogs();
      } catch (error) {
        result.innerHTML = `<div class="empty">加载失败：${escapeHtml(error.message)}</div>`;
      }
    });

    resetBtn.addEventListener("click", async () => {
      sessionIdInput.value = "";
      stageInput.value = "";
      successInput.value = "";
      limitInput.value = "20";
      offsetInput.value = "0";
      await loadAuditLogs();
    });

    saveAdminTokenBtn.addEventListener("click", async () => {
      const token = adminTokenInput.value.trim();
      window.localStorage.setItem("demo_admin_token", token);
      adminTokenStatus.textContent = token ? "admin token ????????..." : "admin token ???";
      updateExportLink();
      await loadAuditLogs();
    });

    exportBtn.addEventListener("click", async (event) => {
      event.preventDefault();
      try {
        await downloadWithAdminToken(exportBtn.href, "admin_llm_audit.csv");
      } catch (error) {
        result.innerHTML = `<div class="empty">导出失败：${escapeHtml(error.message)}</div>`;
      }
    });

    initializeAdminTokenInput();
    updateExportLink();
    loadAuditLogs();
  </script>
</body>
</html>
    """
    return html_page(html)

# agent 会话历史查看页
@router.get("/demo/agent-history", response_class=HTMLResponse)
async def agent_history_demo_page() -> HTMLResponse:
    html = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agent Session History Console</title>
  <style>
    :root {
      --bg: #f3efe8;
      --panel: rgba(255, 255, 255, 0.84);
      --text: #16212b;
      --muted: #677281;
      --line: rgba(22, 33, 43, 0.12);
      --accent: #14425c;
      --accent-2: #b97528;
      --ok: #0f766e;
      --warn: #9a3412;
      --chip: rgba(20, 66, 92, 0.08);
      --code: #111927;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(185, 117, 40, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(20, 66, 92, 0.14), transparent 30%),
        linear-gradient(135deg, #f7f3ec 0%, #efe8dc 100%);
    }
    .shell {
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 24px;
    }
    .brand, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 24px 60px rgba(22, 33, 43, 0.12);
      backdrop-filter: blur(16px);
    }
    .brand {
      padding: 26px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      position: sticky;
      top: 24px;
      height: fit-content;
    }
    .kicker {
      text-transform: uppercase;
      letter-spacing: 0.22em;
      color: var(--accent-2);
      font-size: 12px;
      font-weight: 700;
    }
    h1 {
      margin: 10px 0 12px;
      font-size: 36px;
      line-height: 1.08;
    }
    .lede {
      margin: 0;
      color: var(--muted);
      line-height: 1.75;
      font-size: 15px;
    }
    .tag {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(20, 66, 92, 0.08);
      color: var(--accent);
      font-weight: 700;
      font-size: 13px;
    }
    .fact {
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.66);
      border: 1px solid var(--line);
    }
    .fact label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .fact code {
      font-size: 13px;
      word-break: break-word;
    }
    .panel {
      padding: 22px;
      display: grid;
      gap: 18px;
      align-content: start;
    }
    .toolbar {
      display: grid;
      grid-template-columns: 1.4fr 1fr 1fr 1fr 100px 100px;
      gap: 12px;
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.88);
      padding: 14px 16px;
    }
    .stat label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .stat strong {
      display: block;
      font-size: 26px;
      line-height: 1;
    }
    .summary {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      color: var(--muted);
      font-size: 14px;
    }
    .pill {
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--chip);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }
    .field, select, button {
      font: inherit;
    }
    .field, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.9);
      color: var(--text);
      padding: 12px 14px;
      outline: none;
    }
    .field:focus, select:focus {
      border-color: rgba(20, 66, 92, 0.34);
      box-shadow: 0 0 0 4px rgba(20, 66, 92, 0.08);
    }
    .btn {
      border: 0;
      border-radius: 14px;
      padding: 12px 16px;
      cursor: pointer;
      font-weight: 700;
      transition: transform .15s ease, box-shadow .15s ease;
    }
    .btn:hover { transform: translateY(-1px); }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent), #235c7e);
      color: white;
      box-shadow: 0 12px 28px rgba(20, 66, 92, 0.24);
    }
    .btn-secondary {
      background: rgba(185, 117, 40, 0.12);
      color: #7e4f18;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(360px, 520px) 1fr;
      gap: 18px;
      align-items: start;
    }
    .stack {
      display: grid;
      gap: 14px;
    }
    .card, .timeline {
      border-radius: 22px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      overflow: hidden;
    }
    .card-head, .timeline-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(20, 66, 92, 0.04);
    }
    .card-title, .timeline-title {
      display: grid;
      gap: 6px;
    }
    .card-title strong, .timeline-title strong {
      font-size: 16px;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    .session-link {
      color: var(--accent);
      text-decoration: none;
      border-bottom: 1px dashed rgba(20, 66, 92, 0.4);
      width: fit-content;
    }
    .session-link:hover {
      color: #0b618a;
      border-bottom-color: rgba(11, 97, 138, 0.55);
    }
    .badge {
      padding: 7px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
      height: fit-content;
      white-space: nowrap;
    }
    .badge.confirm { background: rgba(185, 117, 40, 0.14); color: #8b5718; }
    .badge.reply { background: rgba(15, 118, 110, 0.12); color: var(--ok); }
    .badge.clarify { background: rgba(154, 52, 18, 0.12); color: var(--warn); }
    .badge.execute { background: rgba(20, 66, 92, 0.12); color: var(--accent); }
    .card-body, .timeline-body {
      display: grid;
      gap: 14px;
      padding: 16px 18px 18px;
    }
    .row {
      display: grid;
      gap: 6px;
    }
    .row label {
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .timeline-step {
      position: relative;
      padding-left: 22px;
      display: grid;
      gap: 8px;
    }
    .timeline-step::before {
      content: "";
      position: absolute;
      left: 4px;
      top: 8px;
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--accent);
      box-shadow: 0 0 0 4px rgba(20, 66, 92, 0.08);
    }
    .timeline-step::after {
      content: "";
      position: absolute;
      left: 8px;
      top: 24px;
      bottom: -18px;
      width: 2px;
      background: rgba(20, 66, 92, 0.14);
    }
    .timeline-step:last-child::after {
      display: none;
    }
    .quick-links {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .quick-link {
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(20, 66, 92, 0.08);
      color: var(--accent);
      text-decoration: none;
      font-size: 12px;
      font-weight: 700;
    }
    .quick-link:hover {
      background: rgba(20, 66, 92, 0.14);
    }
    .detail-panel {
      border-radius: 22px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      overflow: hidden;
    }
    pre {
      margin: 0;
      padding: 14px 16px;
      border-radius: 16px;
      background: var(--code);
      color: #e5eef8;
      overflow: auto;
      font-size: 12px;
      line-height: 1.65;
    }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 22px;
      padding: 36px 24px;
      text-align: center;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.62);
    }
    @media (max-width: 1180px) {
      .shell { grid-template-columns: 1fr; }
      .brand { position: static; }
      .toolbar { grid-template-columns: 1fr 1fr; }
      .stats { grid-template-columns: 1fr 1fr; }
      .layout { grid-template-columns: 1fr; }
    }
    @media (max-width: 720px) {
      .toolbar { grid-template-columns: 1fr; }
      .stats { grid-template-columns: 1fr; }
      h1 { font-size: 30px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="brand">
      <div>
        <div class="kicker">Agent Timeline Console</div>
        <h1>Agent Session<br/>History</h1>
        <p class="lede">
          这个页面直接读取 <code>/api/v1/admin/agent/sessions/history</code> 和
          <code>/api/v1/admin/agent/sessions/{session_id}/history</code>，
          用来查看每一次自然语言请求如何从解析、确认到执行收敛成最终结果。
        </p>
      </div>
      <div class="tag">会话轨迹 · 后台调试 · Agent 可观察性</div>
      <div class="fact">
        <label>推荐动作</label>
        <code>先发两次 /agent/chat，再按 session_id 查看完整轨迹</code>
      </div>
      <div class="fact">
        <label>适合观察</label>
        <code>parser_source、agent_state、execution_result 的变化</code>
      </div>
      <div class="fact">
        <label>当前接口</label>
        <code>/api/v1/admin/agent/sessions/history</code>
      </div>
      <div class="fact">
        <label>Admin Token</label>
        <input id="adminTokenInput" class="field" placeholder="可选：admin token" />
        <button class="btn btn-secondary" id="saveAdminTokenBtn" style="margin-top:10px;">保存 Token</button>
      </div>
    </aside>

    <main class="panel">
      <div class="toolbar">
        <input id="sessionId" class="field" placeholder="按 session_id 筛选，例如 history-demo-1" />
        <select id="parserSource">
          <option value="">全部 parser_source</option>
          <option value="rule">rule</option>
          <option value="llm">llm</option>
          <option value="conversation">conversation</option>
        </select>
        <select id="agentState">
          <option value="">全部 agent_state</option>
          <option value="confirm">confirm</option>
          <option value="reply">reply</option>
          <option value="clarify">clarify</option>
          <option value="execute">execute</option>
        </select>
        <select id="intent">
          <option value="">全部 intent</option>
          <option value="create">create</option>
          <option value="update">update</option>
          <option value="delete">delete</option>
          <option value="query">query</option>
          <option value="unknown">unknown</option>
        </select>
        <input id="limit" class="field" type="number" min="1" max="100" value="20" placeholder="limit" />
        <input id="offset" class="field" type="number" min="0" value="0" placeholder="offset" />
      </div>

      <div class="actions">
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
          <button class="btn btn-primary" id="loadBtn">查询历史</button>
          <button class="btn btn-secondary" id="resetBtn">重置筛选</button>
          <a class="btn btn-secondary" id="exportBtn" href="/api/v1/admin/agent/sessions/history/export" target="_blank" rel="noreferrer">导出 CSV</a>
        </div>
        <div class="summary">
          <span class="pill" id="summaryTotal">总数 0</span>
          <span class="pill" id="summaryFilter">当前筛选：全部</span>
          <span class="pill" id="detailTitle">右侧轨迹：未选中 session</span>
        </div>
      </div>

      <div class="stats">
        <div class="stat">
          <label>当前列表总数</label>
          <strong id="statTotal">0</strong>
        </div>
        <div class="stat">
          <label>Confirm 条数</label>
          <strong id="statConfirm">0</strong>
        </div>
        <div class="stat">
          <label>Reply 条数</label>
          <strong id="statReply">0</strong>
        </div>
        <div class="stat">
          <label>LLM 来源条数</label>
          <strong id="statLlm">0</strong>
        </div>
      </div>

      <div class="layout">
        <section class="stack" id="listResult"></section>
        <div class="stack">
          <section class="timeline">
            <div class="timeline-head">
              <div class="timeline-title">
                <strong>Session Timeline</strong>
                <div class="meta" id="timelineMeta">点击左侧 session_id 查看完整会话轨迹</div>
              </div>
            </div>
            <div class="timeline-body" id="timelineResult">
              <div class="empty">还没有加载任何单个 session 的完整历史。</div>
            </div>
          </section>

          <section class="detail-panel">
            <div class="timeline-head">
              <div class="timeline-title">
                <strong>Related Schedule</strong>
                <div class="meta" id="scheduleMeta">当 execution_result 里包含日程 id 时，这里会自动展示详情。</div>
              </div>
            </div>
            <div class="timeline-body" id="scheduleResult">
              <div class="empty">还没有可展示的关联日程详情。</div>
            </div>
          </section>
        </div>
      </div>
    </main>
  </div>

  <script>
    const sessionIdInput = document.getElementById("sessionId");
    const parserSourceInput = document.getElementById("parserSource");
    const agentStateInput = document.getElementById("agentState");
    const intentInput = document.getElementById("intent");
    const limitInput = document.getElementById("limit");
    const offsetInput = document.getElementById("offset");
    const loadBtn = document.getElementById("loadBtn");
    const resetBtn = document.getElementById("resetBtn");
    const listResult = document.getElementById("listResult");
    const timelineResult = document.getElementById("timelineResult");
    const scheduleResult = document.getElementById("scheduleResult");
    const scheduleMeta = document.getElementById("scheduleMeta");
    const summaryTotal = document.getElementById("summaryTotal");
    const summaryFilter = document.getElementById("summaryFilter");
    const detailTitle = document.getElementById("detailTitle");
    const timelineMeta = document.getElementById("timelineMeta");
    const statTotal = document.getElementById("statTotal");
    const statConfirm = document.getElementById("statConfirm");
    const statReply = document.getElementById("statReply");
    const statLlm = document.getElementById("statLlm");
    const exportBtn = document.getElementById("exportBtn");
    const adminTokenInput = document.getElementById("adminTokenInput");
    const saveAdminTokenBtn = document.getElementById("saveAdminTokenBtn");
    const adminTokenStatus = document.getElementById("adminTokenStatus");

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function formatJson(value) {
      if (value == null) return "null";
      return JSON.stringify(value, null, 2);
    }

    function getAdminToken() {
      return window.localStorage.getItem("demo_admin_token") || "";
    }

    function buildAdminHeaders() {
      const token = getAdminToken();
      return token ? { "X-Admin-Token": token } : {};
    }

    function initializeAdminTokenInput() {
      adminTokenInput.value = getAdminToken();
      adminTokenStatus.textContent = getAdminToken() ? "??????? admin token" : "??? admin token";
    }

    function currentFilterText() {
      const parts = [];
      if (sessionIdInput.value.trim()) parts.push(`session_id=${sessionIdInput.value.trim()}`);
      if (parserSourceInput.value) parts.push(`parser_source=${parserSourceInput.value}`);
      if (agentStateInput.value) parts.push(`agent_state=${agentStateInput.value}`);
      if (intentInput.value) parts.push(`intent=${intentInput.value}`);
      return parts.length ? parts.join(" | ") : "全部";
    }

    function stateBadgeClass(value) {
      if (value === "confirm") return "confirm";
      if (value === "reply") return "reply";
      if (value === "clarify") return "clarify";
      return "execute";
    }

    function updateExportLink() {
      const params = new URLSearchParams();
      if (sessionIdInput.value.trim()) params.set("session_id", sessionIdInput.value.trim());
      if (parserSourceInput.value) params.set("parser_source", parserSourceInput.value);
      if (agentStateInput.value) params.set("agent_state", agentStateInput.value);
      if (intentInput.value) params.set("intent", intentInput.value);
      exportBtn.href = `/api/v1/admin/agent/sessions/history/export?${params.toString()}`;
    }

    function renderListEmpty() {
      listResult.innerHTML = '<div class="empty">当前条件下没有找到 Agent 会话历史。</div>';
    }

    function renderTimelineEmpty() {
      timelineMeta.textContent = "点击左侧 session_id 查看完整会话轨迹";
      timelineResult.innerHTML = '<div class="empty">还没有加载任何单个 session 的完整历史。</div>';
    }

    function renderScheduleEmpty() {
      scheduleMeta.textContent = "当 execution_result 里包含日程 id 时，这里会自动展示详情。";
      scheduleResult.innerHTML = '<div class="empty">还没有可展示的关联日程详情。</div>';
    }

    function updateStats(items, total) {
      statTotal.textContent = String(total);
      statConfirm.textContent = String(items.filter(item => item.agent_state === "confirm").length);
      statReply.textContent = String(items.filter(item => item.agent_state === "reply").length);
      statLlm.textContent = String(items.filter(item => item.parser_source === "llm").length);
    }

    function renderList(items) {
      if (!items.length) {
        renderListEmpty();
        return;
      }

      listResult.innerHTML = items.map(item => `
        <section class="card">
          <div class="card-head">
            <div class="card-title">
              <strong>${escapeHtml(item.session_id)}</strong>
              <a class="session-link" href="#" data-session-id="${escapeHtml(item.session_id)}">查看该 session 完整轨迹</a>
              <div class="meta">
                id=${item.id} · parser=${escapeHtml(item.parser_source || "-")} · intent=${escapeHtml(item.intent)} · state=${escapeHtml(item.agent_state)}
              </div>
              <div class="meta">
                created_at=${escapeHtml(item.created_at)} · confirmed=${escapeHtml(item.confirmed)}
              </div>
              <div class="quick-links">
                <a class="quick-link" href="#" data-session-id="${escapeHtml(item.session_id)}">时间线</a>
                <a class="quick-link" href="/api/v1/demo/llm-audit" target="_blank" rel="noreferrer">LLM 审计页</a>
              </div>
            </div>
            <span class="badge ${stateBadgeClass(item.agent_state)}">${escapeHtml(item.agent_state.toUpperCase())}</span>
          </div>
          <div class="card-body">
            <div class="row">
              <label>User Input</label>
              <pre>${escapeHtml(item.user_input)}</pre>
            </div>
            <div class="row">
              <label>User Message</label>
              <pre>${escapeHtml(item.user_message || "")}</pre>
            </div>
            <div class="row">
              <label>Suggested Inputs</label>
              <pre>${escapeHtml(formatJson(item.suggested_inputs || []))}</pre>
            </div>
            <div class="row">
              <label>Tool Arguments</label>
              <pre>${escapeHtml(formatJson(item.tool_arguments))}</pre>
            </div>
            <div class="row">
              <label>Execution Result</label>
              <pre>${escapeHtml(formatJson(item.execution_result))}</pre>
            </div>
          </div>
        </section>
      `).join("");

      listResult.querySelectorAll("[data-session-id]").forEach(node => {
        node.addEventListener("click", async (event) => {
          event.preventDefault();
          const sessionId = node.getAttribute("data-session-id") || "";
          await loadTimeline(sessionId);
        });
      });
    }

    function renderTimeline(sessionId, items) {
      timelineMeta.textContent = `当前 session：${sessionId}，共 ${items.length} 条历史`;
      detailTitle.textContent = `右侧轨迹：${sessionId}`;

      if (!items.length) {
        timelineResult.innerHTML = '<div class="empty">这个 session 当前没有历史记录。</div>';
        return;
      }

      timelineResult.innerHTML = items.map(item => `
        <section class="timeline-step">
          <div style="display:flex;justify-content:space-between;gap:12px;align-items:flex-start;">
            <div class="meta">
              <strong>${escapeHtml(item.agent_state)}</strong> · parser=${escapeHtml(item.parser_source || "-")} · intent=${escapeHtml(item.intent)}
              <br/>
              created_at=${escapeHtml(item.created_at)} · confirmed=${escapeHtml(item.confirmed)}
            </div>
            <span class="badge ${stateBadgeClass(item.agent_state)}">${escapeHtml(item.agent_state.toUpperCase())}</span>
          </div>
          <div class="row">
            <label>User Input</label>
            <pre>${escapeHtml(item.user_input)}</pre>
          </div>
          <div class="row">
            <label>Context</label>
            <pre>${escapeHtml(formatJson(item.context))}</pre>
          </div>
          <div class="row">
            <label>Tool Arguments</label>
            <pre>${escapeHtml(formatJson(item.tool_arguments))}</pre>
          </div>
          <div class="row">
            <label>User Message</label>
            <pre>${escapeHtml(item.user_message || "")}</pre>
          </div>
          <div class="row">
            <label>Suggested Inputs</label>
            <pre>${escapeHtml(formatJson(item.suggested_inputs || []))}</pre>
          </div>
          <div class="row">
            <label>Execution Result</label>
            <pre>${escapeHtml(formatJson(item.execution_result))}</pre>
          </div>
        </section>
      `).join("");

      const relatedSchedule = [...items].reverse().find(item =>
        item.execution_result && typeof item.execution_result === "object" && item.execution_result.id
      );
      if (relatedSchedule) {
        loadScheduleDetail(relatedSchedule.execution_result.id);
      } else {
        renderScheduleEmpty();
      }
    }

    async function loadScheduleDetail(scheduleId) {
      const response = await fetch(`/api/v1/schedule/${scheduleId}`);
      const body = await response.json();
      if (!body.data) {
        renderScheduleEmpty();
        return;
      }
      scheduleMeta.textContent = `当前关联日程 id=${scheduleId}`;
      scheduleResult.innerHTML = `
        <div class="row">
          <label>Schedule Payload</label>
          <pre>${escapeHtml(formatJson(body.data))}</pre>
        </div>
        <div class="quick-links">
          <a class="quick-link" href="/docs" target="_blank" rel="noreferrer">OpenAPI Docs</a>
          <a class="quick-link" href="/api/v1/demo/chat" target="_blank" rel="noreferrer">Agent Chat Demo</a>
        </div>
      `;
    }

    async function loadTimeline(sessionId) {
      const response = await fetch(`/api/v1/admin/agent/sessions/${encodeURIComponent(sessionId)}/history`, { headers: buildAdminHeaders() });
      const body = await response.json();
      renderTimeline(sessionId, body.data || []);
    }

    async function loadHistoryList() {
      const params = new URLSearchParams();
      if (sessionIdInput.value.trim()) params.set("session_id", sessionIdInput.value.trim());
      if (parserSourceInput.value) params.set("parser_source", parserSourceInput.value);
      if (agentStateInput.value) params.set("agent_state", agentStateInput.value);
      if (intentInput.value) params.set("intent", intentInput.value);
      params.set("limit", limitInput.value || "20");
      params.set("offset", offsetInput.value || "0");

      const response = await fetch(`/api/v1/admin/agent/sessions/history?${params.toString()}`, { headers: buildAdminHeaders() });
      const body = await response.json();
      const data = body.data;
      summaryTotal.textContent = `总数 ${data.total}`;
      summaryFilter.textContent = `当前筛选：${currentFilterText()}`;
      updateStats(data.items, data.total);
      updateExportLink();
      if (!sessionIdInput.value.trim()) {
        detailTitle.textContent = "右侧轨迹：未选中 session";
      }
      renderList(data.items);

      if (sessionIdInput.value.trim()) {
        await loadTimeline(sessionIdInput.value.trim());
      } else {
        renderTimelineEmpty();
        renderScheduleEmpty();
      }
    }

    loadBtn.addEventListener("click", async () => {
      try {
        await loadHistoryList();
      } catch (error) {
        listResult.innerHTML = `<div class="empty">加载失败：${escapeHtml(error.message)}</div>`;
      }
    });

    resetBtn.addEventListener("click", async () => {
      sessionIdInput.value = "";
      parserSourceInput.value = "";
      agentStateInput.value = "";
      intentInput.value = "";
      limitInput.value = "20";
      offsetInput.value = "0";
      renderTimelineEmpty();
      renderScheduleEmpty();
      await loadHistoryList();
    });

    saveAdminTokenBtn.addEventListener("click", async () => {
      const token = adminTokenInput.value.trim();
      window.localStorage.setItem("demo_admin_token", token);
      adminTokenStatus.textContent = token ? "admin token ????????..." : "admin token ???";
      updateExportLink();
      await loadHistoryList();
    });

    exportBtn.addEventListener("click", async (event) => {
      event.preventDefault();
      try {
        await downloadWithAdminToken(exportBtn.href, "admin_agent_history.csv");
      } catch (error) {
        listResult.innerHTML = `<div class="empty">导出失败：${escapeHtml(error.message)}</div>`;
      }
    });

    initializeAdminTokenInput();
    updateExportLink();
    renderScheduleEmpty();
    loadHistoryList();
  </script>
</body>
</html>
    """
    return html_page(html)

# 提醒链路日志与任务查看页
@router.get("/demo/reminder-logs", response_class=HTMLResponse)
async def reminder_logs_demo_page() -> HTMLResponse:
    html = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Reminder Logs Console</title>
  <style>
    :root {
      --bg: #f4efe7;
      --panel: rgba(255, 255, 255, 0.84);
      --text: #16212b;
      --muted: #667180;
      --line: rgba(22, 33, 43, 0.12);
      --accent: #12455e;
      --accent-2: #bf7b2f;
      --ok: #0f766e;
      --warn: #9a3412;
      --bad: #b42318;
      --code: #111927;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(191, 123, 47, 0.16), transparent 28%),
        radial-gradient(circle at top right, rgba(18, 69, 94, 0.14), transparent 30%),
        linear-gradient(135deg, #f8f3ea 0%, #efe7db 100%);
    }
    .shell {
      max-width: 1380px;
      margin: 0 auto;
      padding: 28px;
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 24px;
    }
    .brand, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: 0 24px 60px rgba(22, 33, 43, 0.12);
      backdrop-filter: blur(16px);
    }
    .brand {
      padding: 26px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      position: sticky;
      top: 24px;
      height: fit-content;
    }
    .kicker {
      text-transform: uppercase;
      letter-spacing: 0.22em;
      color: var(--accent-2);
      font-size: 12px;
      font-weight: 700;
    }
    h1 {
      margin: 10px 0 12px;
      font-size: 36px;
      line-height: 1.08;
    }
    .lede {
      margin: 0;
      color: var(--muted);
      line-height: 1.75;
      font-size: 15px;
    }
    .tag {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      width: fit-content;
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(18, 69, 94, 0.08);
      color: var(--accent);
      font-weight: 700;
      font-size: 13px;
    }
    .fact {
      padding: 14px 16px;
      border-radius: 18px;
      background: rgba(255, 255, 255, 0.66);
      border: 1px solid var(--line);
    }
    .fact label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .fact code {
      font-size: 13px;
      word-break: break-word;
    }
    .panel {
      padding: 22px;
      display: grid;
      gap: 18px;
      align-content: start;
    }
    .toolbar {
      display: grid;
      grid-template-columns: 1.4fr 1fr 120px 120px;
      gap: 12px;
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
    }
    .summary {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      color: var(--muted);
      font-size: 14px;
    }
    .pill {
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(18, 69, 94, 0.08);
      color: var(--accent);
      font-size: 12px;
      font-weight: 700;
    }
    .field, select, button {
      font: inherit;
    }
    .field, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255, 255, 255, 0.9);
      color: var(--text);
      padding: 12px 14px;
      outline: none;
    }
    .field:focus, select:focus {
      border-color: rgba(18, 69, 94, 0.34);
      box-shadow: 0 0 0 4px rgba(18, 69, 94, 0.08);
    }
    .btn {
      border: 0;
      border-radius: 14px;
      padding: 12px 16px;
      cursor: pointer;
      font-weight: 700;
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent), #245d7d);
      color: white;
      box-shadow: 0 12px 28px rgba(18, 69, 94, 0.24);
    }
    .btn-secondary {
      background: rgba(191, 123, 47, 0.12);
      color: #7f4f18;
    }
    .grid {
      display: grid;
      gap: 14px;
    }
    .card {
      border-radius: 22px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.92);
      overflow: hidden;
    }
    .card-head {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(18, 69, 94, 0.04);
    }
    .card-title {
      display: grid;
      gap: 6px;
    }
    .card-title strong {
      font-size: 16px;
    }
    .card-meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    .badge {
      padding: 7px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 800;
      height: fit-content;
      white-space: nowrap;
    }
    .badge.pending { background: rgba(191, 123, 47, 0.12); color: #8b5718; }
    .badge.sent { background: rgba(15, 118, 110, 0.12); color: var(--ok); }
    .badge.failed { background: rgba(180, 35, 24, 0.12); color: var(--bad); }
    .card-body {
      display: grid;
      gap: 14px;
      padding: 16px 18px 18px;
    }
    .row {
      display: grid;
      gap: 6px;
    }
    .row label {
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    pre {
      margin: 0;
      padding: 14px 16px;
      border-radius: 16px;
      background: var(--code);
      color: #e5eef8;
      overflow: auto;
      font-size: 12px;
      line-height: 1.65;
    }
    .empty {
      border: 1px dashed var(--line);
      border-radius: 22px;
      padding: 36px 24px;
      text-align: center;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.62);
    }
    @media (max-width: 1080px) {
      .shell { grid-template-columns: 1fr; }
      .brand { position: static; }
      .toolbar { grid-template-columns: 1fr 1fr; }
    }
    @media (max-width: 720px) {
      .toolbar { grid-template-columns: 1fr; }
      h1 { font-size: 30px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside class="brand">
      <div>
        <div class="kicker">Reminder Logs Console</div>
        <h1>Reminder Logs<br/>Board</h1>
        <p class="lede">
          这个页面直接读取 <code>/api/v1/admin/schedule/reminder-logs</code>，
          用来查看提醒发送日志、失败原因、计划触发时间和实际发送结果。
        </p>
      </div>
      <div class="tag">提醒执行轨迹 · 发送状态 · 运维排查</div>
      <div class="fact">
        <label>Recommended Filter</label>
        <code>先按 status=failed 看异常，再按 schedule_id 精确排查</code>
      </div>
      <div class="fact">
        <label>Current API</label>
        <code>/api/v1/admin/schedule/reminder-logs</code>
      </div>
      <div class="fact">
        <label>Admin Token</label>
        <input id="adminTokenInput" class="field" placeholder="Optional admin token" />
        <button class="btn btn-secondary" id="saveAdminTokenBtn" style="margin-top:10px;">Save Token</button>
        <div class="card-meta" id="adminTokenStatus" style="margin-top:10px;">No saved admin token</div>
      </div>
      <div class="fact">
        <label>Related Pages</label>
        <code>/api/v1/demo/dashboard</code>
      </div>
    </aside>

    <main class="panel">
      <div class="toolbar">
        <input id="scheduleId" class="field" placeholder="Filter by schedule_id" />
        <select id="status">
          <option value="">All status</option>
          <option value="pending">pending</option>
          <option value="sent">sent</option>
          <option value="failed">failed</option>
        </select>
        <input id="limit" class="field" type="number" min="1" max="100" value="20" placeholder="limit" />
        <input id="offset" class="field" type="number" min="0" value="0" placeholder="offset" />
      </div>

      <div class="actions">
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
          <button class="btn btn-primary" id="loadBtn">Load Logs</button>
          <button class="btn btn-secondary" id="resetBtn">Reset Filters</button>
          <a class="btn btn-secondary" id="exportBtn" href="/api/v1/admin/schedule/reminder-logs/export" target="_blank" rel="noreferrer">Export CSV</a>
        </div>
        <div class="summary">
          <span class="pill" id="summaryTotal">total 0</span>
          <span class="pill" id="summaryFilter">filters: all</span>
        </div>
      </div>

      <div class="grid" id="result"></div>
    </main>
  </div>

  <script>
    const scheduleIdInput = document.getElementById("scheduleId");
    const statusInput = document.getElementById("status");
    const limitInput = document.getElementById("limit");
    const offsetInput = document.getElementById("offset");
    const loadBtn = document.getElementById("loadBtn");
    const resetBtn = document.getElementById("resetBtn");
    const summaryTotal = document.getElementById("summaryTotal");
    const summaryFilter = document.getElementById("summaryFilter");
    const result = document.getElementById("result");
    const exportBtn = document.getElementById("exportBtn");
    const adminTokenInput = document.getElementById("adminTokenInput");
    const saveAdminTokenBtn = document.getElementById("saveAdminTokenBtn");

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function currentFilterText() {
      const parts = [];
      if (scheduleIdInput.value.trim()) parts.push(`schedule_id=${scheduleIdInput.value.trim()}`);
      if (statusInput.value) parts.push(`status=${statusInput.value}`);
      return parts.length ? parts.join(" | ") : "all";
    }

    function getAdminToken() {
      return window.localStorage.getItem("demo_admin_token") || "";
    }

    function buildAdminHeaders() {
      const token = getAdminToken();
      return token ? { "X-Admin-Token": token } : {};
    }

    function initializeAdminTokenInput() {
      adminTokenInput.value = getAdminToken();
      adminTokenStatus.textContent = getAdminToken() ? "Loaded saved admin token" : "No saved admin token";
    }

    function renderEmpty() {
      result.innerHTML = '<div class="empty">No reminder logs found under current filters.</div>';
    }

    function updateExportLink() {
      const params = new URLSearchParams();
      if (scheduleIdInput.value.trim()) params.set("schedule_id", scheduleIdInput.value.trim());
      if (statusInput.value) params.set("status", statusInput.value);
      exportBtn.href = `/api/v1/admin/schedule/reminder-logs/export?${params.toString()}`;
    }

    function renderCards(items) {
      if (!items.length) {
        renderEmpty();
        return;
      }

      result.innerHTML = items.map(item => `
        <section class="card">
          <div class="card-head">
            <div class="card-title">
              <strong>Reminder Log #${escapeHtml(item.id)}</strong>
              <div class="card-meta">
                schedule_id=${escapeHtml(item.schedule_id)} | planned=${escapeHtml(item.planned_trigger_at)}
              </div>
              <div class="card-meta">
                created_at=${escapeHtml(item.created_at)} | updated_at=${escapeHtml(item.updated_at)}
              </div>
            </div>
            <span class="badge ${escapeHtml(item.status)}">${escapeHtml(String(item.status).toUpperCase())}</span>
          </div>
          <div class="card-body">
            <div class="row">
              <label>Planned Trigger At</label>
              <pre>${escapeHtml(item.planned_trigger_at)}</pre>
            </div>
            <div class="row">
              <label>Reminded At</label>
              <pre>${escapeHtml(item.reminded_at || "null")}</pre>
            </div>
            <div class="row">
              <label>Error Message</label>
              <pre>${escapeHtml(item.error_message || "")}</pre>
            </div>
          </div>
        </section>
      `).join("");
    }

    async function loadLogs() {
      const params = new URLSearchParams();
      if (scheduleIdInput.value.trim()) params.set("schedule_id", scheduleIdInput.value.trim());
      if (statusInput.value) params.set("status", statusInput.value);
      params.set("limit", limitInput.value || "20");
      params.set("offset", offsetInput.value || "0");

      const response = await fetch(`/api/v1/admin/schedule/reminder-logs?${params.toString()}`, { headers: buildAdminHeaders() });
      const body = await response.json();
      summaryTotal.textContent = `total ${body.data.total}`;
      summaryFilter.textContent = `filters: ${currentFilterText()}`;
      updateExportLink();
      renderCards(body.data.items);
    }

    loadBtn.addEventListener("click", async () => {
      try {
        await loadLogs();
      } catch (error) {
        result.innerHTML = `<div class="empty">Load failed: ${escapeHtml(error.message)}</div>`;
      }
    });

    resetBtn.addEventListener("click", async () => {
      scheduleIdInput.value = "";
      statusInput.value = "";
      limitInput.value = "20";
      offsetInput.value = "0";
      await loadLogs();
    });

    saveAdminTokenBtn.addEventListener("click", async () => {
      const token = adminTokenInput.value.trim();
      window.localStorage.setItem("demo_admin_token", token);
      adminTokenStatus.textContent = token ? "Admin token saved. Refreshing logs..." : "Admin token cleared";
      updateExportLink();
      await loadLogs();
    });

    exportBtn.addEventListener("click", async (event) => {
      event.preventDefault();
      try {
        await downloadWithAdminToken(exportBtn.href, "admin_reminder_logs.csv");
      } catch (error) {
        result.innerHTML = `<div class="empty">Export failed: ${escapeHtml(error.message)}</div>`;
      }
    });

    initializeAdminTokenInput();
    updateExportLink();
    loadLogs();
  </script>
</body>
</html>
    """
    return html_page(html)
