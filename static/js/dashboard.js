async function fetchJson(url, options = {}) {
  const requestOptions = { ...options };
  if (!(requestOptions.body instanceof FormData)) {
    requestOptions.headers = { "Content-Type": "application/json", ...(requestOptions.headers || {}) };
  }

  const response = await fetch(url, requestOptions);
  const data = await response.json();
  if (!response.ok || data.success === false) {
    throw new Error(data.error || "请求失败");
  }
  return data;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatScore(value) {
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(4) : String(value);
}

function displayPredictionText(value) {
  if (value === "Attack") return "攻击";
  if (value === "Normal") return "正常";
  return value || "-";
}

function displayRiskLevel(value) {
  if (value === "High") return "高";
  if (value === "Medium") return "中";
  if (value === "Low") return "低";
  if (value === "Unknown") return "未知";
  return value || "未知";
}

function localizeAnalysisText(text) {
  const value = String(text || "").trim();
  if (!value) return "-";

  if (value.startsWith("The model classified this traffic as an attack because several key features appear abnormal")) {
    return value
      .replace(
        "The model classified this traffic as an attack because several key features appear abnormal, including ",
        "模型将该流量判定为攻击，原因是多个关键特征表现异常，包括："
      )
      .replace(/\.$/, "。");
  }

  if (value === "This traffic may lead to unauthorized access, abnormal resource consumption, or service disruption.") {
    return "该流量可能导致未授权访问、资源异常消耗或服务中断。";
  }

  if (value === "Inspect the source IP, session behavior, and target host state, then apply firewall or intrusion-prevention controls as needed.") {
    return "建议检查源 IP、会话行为和目标主机状态，并按需启用防火墙或入侵防护策略。";
  }

  if (value === "The model classified this traffic as normal and did not detect strong attack indicators.") {
    return "模型将该流量判定为正常，未发现明显攻击特征。";
  }

  if (value === "The short-term risk appears low, but continued monitoring is still recommended.") {
    return "当前短期风险较低，但仍建议持续观察相关流量。";
  }

  if (value === "Keep the relevant logs and continue observing similar sessions for abnormal changes.") {
    return "建议保留相关日志，并继续观察同类会话是否出现异常变化。";
  }

  if (value === "Manual review is still required.") {
    return "仍建议人工复核本次分析结果。";
  }

  if (value === "Cross-check the raw traffic and system logs before taking action.") {
    return "在采取处置动作前，请先交叉核对原始流量与系统日志。";
  }

  return value;
}

function getLlmSettings() {
  return {
    llm_model: document.getElementById("llmModel").value.trim(),
    llm_api_key: document.getElementById("llmApiKey").value.trim(),
    llm_base_url: document.getElementById("llmBaseUrl").value.trim(),
  };
}

function getFilterParams() {
  const params = new URLSearchParams();
  const modelName = document.getElementById("historyModel").value;
  const riskLevel = document.getElementById("historyRisk").value;
  const predictionText = document.getElementById("historyPrediction").value;
  const createdFrom = document.getElementById("historyFrom").value;
  const createdTo = document.getElementById("historyTo").value;
  const limit = document.getElementById("historyLimit").value || "20";
  const offset = document.getElementById("historyOffset").value || "0";

  params.set("limit", limit);
  params.set("offset", offset);
  if (modelName) params.set("model_name", modelName);
  if (riskLevel) params.set("risk_level", riskLevel);
  if (predictionText) params.set("prediction_text", predictionText);
  if (createdFrom) params.set("created_from", createdFrom);
  if (createdTo) params.set("created_to", createdTo);
  return params;
}

function badgeClassByPrediction(predictionText) {
  return predictionText === "Attack" ? "badge-attack" : "badge-normal";
}

function badgeClassByRisk(riskLevel) {
  const value = String(riskLevel || "").toLowerCase();
  if (value === "high" || value === "高") return "badge-high";
  if (value === "medium" || value === "中") return "badge-medium";
  if (value === "low" || value === "低") return "badge-low";
  return "badge-low";
}

function renderStats(summary) {
  document.getElementById("statTotal").textContent = summary.total ?? 0;
  document.getElementById("statAttack").textContent = summary.prediction_counts?.Attack ?? 0;
  document.getElementById("statHighRisk").textContent = summary.risk_counts?.High ?? 0;
  document.getElementById("statDefaultModel").textContent = summary.model_counts?.decision_tree ?? 0;
}

function renderChart(containerId, dataMap) {
  const container = document.getElementById(containerId);
  const entries = Object.entries(dataMap || {});
  if (!entries.length) {
    container.innerHTML = "<p class='empty-state'>暂无可展示数据。</p>";
    return;
  }

  const maxValue = Math.max(...entries.map(([, value]) => Number(value) || 0), 1);
  container.innerHTML = entries
    .map(([label, value]) => {
      const width = `${Math.max(((Number(value) || 0) / maxValue) * 100, 5)}%`;
      return `
        <div class="chart-row">
          <div class="chart-label">
            <span>${escapeHtml(label)}</span>
            <span>${escapeHtml(value)}</span>
          </div>
          <div class="chart-track">
            <div class="chart-fill" style="width:${width};"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderEmptyCard(targetId, text) {
  document.getElementById(targetId).innerHTML = `<div class="empty-note">${escapeHtml(text)}</div>`;
}

function renderMetrics(metrics) {
  return `
    <div class="summary-grid">
      ${metrics
        .map(
          (item) => `
            <div class="summary-metric">
              <span>${escapeHtml(item.label)}</span>
              <strong>${escapeHtml(item.value)}</strong>
            </div>
          `
        )
        .join("")}
    </div>
  `;
}

function renderDataBlock(title, content, asPre = false) {
  return `
    <div class="data-block">
      <span>${escapeHtml(title)}</span>
      ${asPre ? `<pre>${escapeHtml(pretty(content))}</pre>` : `<p>${escapeHtml(content || "-")}</p>`}
    </div>
  `;
}

function renderDetectionCard(targetId, result, extraMetrics = []) {
  const metrics = [
    { label: "检测分数", value: formatScore(result.prediction_score) },
    { label: "检测模型", value: result.model_name ?? "-" },
    { label: "记录编号", value: result.record_id ?? "-" },
    ...extraMetrics,
  ];

  document.getElementById(targetId).innerHTML = `
    <div class="status-strip">
      <div>
        <div class="summary-title">检测结论</div>
        <div class="summary-value">${escapeHtml(displayPredictionText(result.prediction_text || "-"))}</div>
      </div>
      <div class="result-badge ${badgeClassByPrediction(result.prediction_text)}">${escapeHtml(displayPredictionText(result.prediction_text || "-"))}</div>
    </div>
    ${renderMetrics(metrics)}
    ${renderDataBlock("关键特征", result.key_features || {}, true)}
  `;
}

function renderAnalysisCard(targetId, result) {
  const metrics = [
    { label: "风险等级", value: result.risk_level || "Unknown" },
    { label: "模型来源", value: document.getElementById("llmModel").value.trim() || "默认配置" },
    { label: "分析状态", value: result.explanation ? "已生成" : "待生成" },
  ];

  document.getElementById(targetId).innerHTML = `
    <div class="status-strip">
      <div>
        <div class="summary-title">辅助分析风险等级</div>
        <div class="summary-value">${escapeHtml(displayRiskLevel(result.risk_level || "Unknown"))}</div>
      </div>
      <div class="result-badge ${badgeClassByRisk(result.risk_level)}">${escapeHtml(displayRiskLevel(result.risk_level || "Unknown"))}</div>
    </div>
    ${renderMetrics(metrics)}
    <div class="analysis-list">
      ${renderDataBlock("分析解释", localizeAnalysisText(result.explanation || "-"))}
      ${renderDataBlock("影响说明", localizeAnalysisText(result.impact || "-"))}
      ${renderDataBlock("处置建议", localizeAnalysisText(result.suggestion || "-"))}
    </div>
  `;
}

function renderBatchResults(data) {
  const firstResult = data.results?.[0];
  if (!firstResult) {
    renderEmptyCard("csvDetectionCard", "未获取到批量检测结果。");
    renderEmptyCard("csvAnalysisCard", "未获取到大模型辅助分析结果。");
    return;
  }

  renderDetectionCard("csvDetectionCard", firstResult, [
    { label: "检测条数", value: data.count ?? 0 },
    { label: "攻击数量", value: data.summary?.prediction_counts?.Attack ?? 0 },
    { label: "高风险数量", value: data.summary?.risk_counts?.High ?? 0 },
  ]);
  renderAnalysisCard("csvAnalysisCard", firstResult);
}

function renderSingleResult(result) {
  renderDetectionCard("singleDetectionCard", result);
  renderAnalysisCard("singleAnalysisCard", result);
}

function renderHistoryMeta(history) {
  const start = history.total === 0 ? 0 : history.offset + 1;
  const end = history.offset + history.count;
  document.getElementById("historyMeta").textContent =
    `当前显示第 ${start} - ${end} 条，共 ${history.total} 条记录。`;
}

function renderHistorySummary(summary) {
  const metrics = [
    { label: "总记录数", value: summary.total ?? 0 },
    { label: "攻击记录", value: summary.prediction_counts?.Attack ?? 0 },
    { label: "正常记录", value: summary.prediction_counts?.Normal ?? 0 },
    { label: "高风险记录", value: summary.risk_counts?.High ?? 0 },
  ];

  document.getElementById("historySummary").innerHTML = metrics
    .map(
      (item) => `
        <div class="summary-metric">
          <span>${escapeHtml(item.label)}</span>
          <strong>${escapeHtml(item.value)}</strong>
        </div>
      `
    )
    .join("");
}

function renderHistoryTable(results) {
  const container = document.getElementById("historyTable");
  if (!results.length) {
    container.innerHTML = "<p class='empty-state'>当前筛选条件下没有历史记录。</p>";
    return;
  }

  const rows = results
    .map(
      (item) => `
        <tr class="history-row" data-record-id="${escapeHtml(item.record_id)}">
          <td>${escapeHtml(item.record_id)}</td>
          <td>${escapeHtml(item.model_name)}</td>
          <td><span class="result-badge ${badgeClassByPrediction(item.prediction_text)}">${escapeHtml(displayPredictionText(item.prediction_text))}</span></td>
          <td><span class="result-badge ${badgeClassByRisk(item.risk_level)}">${escapeHtml(displayRiskLevel(item.risk_level || "-"))}</span></td>
          <td>${escapeHtml(formatScore(item.prediction_score))}</td>
          <td>${escapeHtml(item.created_at)}</td>
        </tr>
      `
    )
    .join("");

  container.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>记录 ID</th>
          <th>模型</th>
          <th>检测结果</th>
          <th>风险等级</th>
          <th>分数</th>
          <th>检测时间</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;

  container.querySelectorAll(".history-row").forEach((row) => {
    row.addEventListener("click", () => loadRecordDetail(row.dataset.recordId));
  });
}

function renderHistoryDetail(result) {
  document.getElementById("historyDetectionDetail").innerHTML = `
    ${renderMetrics([
      { label: "记录编号", value: result.record_id ?? "-" },
      { label: "检测模型", value: result.model_name ?? "-" },
      { label: "检测结论", value: displayPredictionText(result.prediction_text ?? "-") },
      { label: "检测分数", value: formatScore(result.prediction_score) },
      { label: "检测时间", value: result.created_at ?? "-" },
      { label: "风险等级", value: displayRiskLevel(result.risk_level ?? "-") },
    ])}
    ${renderDataBlock("关键特征", result.key_features || {}, true)}
    ${renderDataBlock("原始特征", result.raw_features || {}, true)}
  `;

  document.getElementById("historyAnalysisDetail").innerHTML = `
    <div class="status-strip">
      <div>
        <div class="summary-title">辅助分析风险等级</div>
        <div class="summary-value">${escapeHtml(displayRiskLevel(result.risk_level || "Unknown"))}</div>
      </div>
      <div class="result-badge ${badgeClassByRisk(result.risk_level)}">${escapeHtml(displayRiskLevel(result.risk_level || "Unknown"))}</div>
    </div>
    <div class="analysis-list">
      ${renderDataBlock("分析解释", localizeAnalysisText(result.explanation || "-"))}
      ${renderDataBlock("影响说明", localizeAnalysisText(result.impact || "-"))}
      ${renderDataBlock("处置建议", localizeAnalysisText(result.suggestion || "-"))}
    </div>
  `;
}

async function loadRecordDetail(recordId) {
  document.getElementById("historyDetectionDetail").innerHTML = "<div class='empty-note'>正在加载检测详情...</div>";
  document.getElementById("historyAnalysisDetail").innerHTML = "<div class='empty-note'>正在加载大模型分析详情...</div>";

  try {
    const data = await fetchJson(`/api/history/${encodeURIComponent(recordId)}`);
    renderHistoryDetail(data.result);
  } catch (error) {
    document.getElementById("historyDetectionDetail").innerHTML = `<div class='empty-note'>${escapeHtml(error.message)}</div>`;
    document.getElementById("historyAnalysisDetail").innerHTML = `<div class='empty-note'>${escapeHtml(error.message)}</div>`;
  }
}

async function runCsvDetection() {
  const inputPath = document.getElementById("csvPath").value.trim();
  const fileInput = document.getElementById("csvFile");
  const selectedFile = fileInput.files[0];
  const modelName = document.getElementById("csvModel").value;
  const limit = Number(document.getElementById("csvLimit").value || 10);

  renderEmptyCard("csvDetectionCard", "正在执行批量检测...");
  renderEmptyCard("csvAnalysisCard", "正在生成大模型辅助分析...");

  try {
    let data;
    const llmSettings = getLlmSettings();

    if (selectedFile) {
      const formData = new FormData();
      formData.append("file", selectedFile);
      formData.append("model_name", modelName);
      formData.append("limit", String(limit));
      Object.entries(llmSettings).forEach(([key, value]) => {
        if (value) formData.append(key, value);
      });
      data = await fetchJson("/api/detect/csv", { method: "POST", body: formData });
    } else {
      data = await fetchJson("/api/detect/csv", {
        method: "POST",
        body: JSON.stringify({
          input_path: inputPath,
          model_name: modelName,
          limit,
          ...llmSettings,
        }),
      });
    }

    renderBatchResults(data);
    await loadHistory();
  } catch (error) {
    renderEmptyCard("csvDetectionCard", error.message);
    renderEmptyCard("csvAnalysisCard", error.message);
  }
}

async function runSingleDetection() {
  const sampleText = document.getElementById("singleSample").value;

  renderEmptyCard("singleDetectionCard", "正在执行单条检测...");
  renderEmptyCard("singleAnalysisCard", "正在生成大模型辅助分析...");

  try {
    const sample = JSON.parse(sampleText);
    const llmSettings = getLlmSettings();
    const data = await fetchJson("/api/detect/single", {
      method: "POST",
      body: JSON.stringify({ sample, ...llmSettings }),
    });

    renderSingleResult(data.result);
    await loadHistory();
  } catch (error) {
    renderEmptyCard("singleDetectionCard", error.message);
    renderEmptyCard("singleAnalysisCard", error.message);
  }
}

async function loadHistory() {
  const params = getFilterParams();
  document.getElementById("historySummary").innerHTML = "<div class='empty-note'>正在加载历史记录摘要...</div>";

  try {
    const summaryParams = new URLSearchParams(params);
    summaryParams.delete("limit");
    summaryParams.delete("offset");

    const [history, summary] = await Promise.all([
      fetchJson(`/api/history?${params.toString()}`),
      fetchJson(`/api/history/summary?${summaryParams.toString()}`),
    ]);

    renderHistorySummary(summary);
    renderHistoryMeta(history);
    renderHistoryTable(history.results);
    renderStats(summary);
    renderChart("predictionChart", summary.prediction_counts);
    renderChart("riskChart", summary.risk_counts);
    renderChart("modelChart", summary.model_counts);
  } catch (error) {
    document.getElementById("historySummary").innerHTML = `<div class='empty-note'>${escapeHtml(error.message)}</div>`;
    document.getElementById("historyMeta").textContent = "历史记录查询失败。";
    document.getElementById("historyTable").innerHTML = "";
    document.getElementById("predictionChart").innerHTML = "";
    document.getElementById("riskChart").innerHTML = "";
    document.getElementById("modelChart").innerHTML = "";
  }
}

function resetHistoryFilters() {
  document.getElementById("historyModel").value = "";
  document.getElementById("historyRisk").value = "";
  document.getElementById("historyPrediction").value = "";
  document.getElementById("historyFrom").value = "";
  document.getElementById("historyTo").value = "";
  document.getElementById("historyLimit").value = "20";
  document.getElementById("historyOffset").value = "0";
  loadHistory();
}

function syncSelectedFileToPath() {
  const fileInput = document.getElementById("csvFile");
  const pathInput = document.getElementById("csvPath");
  if (fileInput.files.length) {
    pathInput.value = fileInput.files[0].name;
  }
}

function exportHistory() {
  const params = getFilterParams();
  params.delete("limit");
  params.delete("offset");
  window.location.href = `/api/history/export?${params.toString()}`;
}

document.getElementById("runCsvBtn").addEventListener("click", runCsvDetection);
document.getElementById("runSingleBtn").addEventListener("click", runSingleDetection);
document.getElementById("loadHistoryBtn").addEventListener("click", loadHistory);
document.getElementById("resetHistoryBtn").addEventListener("click", resetHistoryFilters);
document.getElementById("exportHistoryBtn").addEventListener("click", exportHistory);
document.getElementById("csvFile").addEventListener("change", syncSelectedFileToPath);

loadHistory();
