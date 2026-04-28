async function fetchJson(url, options = {}) {
  const requestOptions = { ...options };
  if (!(requestOptions.body instanceof FormData)) {
    requestOptions.headers = { "Content-Type": "application/json", ...(requestOptions.headers || {}) };
  }
  const response = await fetch(url, requestOptions);
  const data = await response.json();
  if (!response.ok || data.success === false) {
    throw new Error(data.error || "\u8bf7\u6c42\u5931\u8d25");
  }
  return data;
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

function renderMessage(id, text) {
  document.getElementById(id).textContent = text;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
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
    container.innerHTML = "<p class='empty-state'>\u6682\u65e0\u53ef\u5c55\u793a\u6570\u636e\u3002</p>";
    return;
  }
  const maxValue = Math.max(...entries.map(([, value]) => value), 1);
  container.innerHTML = entries
    .map(([label, value]) => {
      const width = `${Math.max((value / maxValue) * 100, 4)}%`;
      return `
        <div class="chart-row">
          <div class="chart-label">
            <span>${escapeHtml(label)}</span>
            <span>${value}</span>
          </div>
          <div class="chart-bar-track">
            <div class="chart-bar-fill" style="width: ${width};"></div>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderHistoryMeta(history) {
  const start = history.total === 0 ? 0 : history.offset + 1;
  const end = history.offset + history.count;
  document.getElementById("historyMeta").textContent =
    `\u5f53\u524d\u663e\u793a\u7b2c ${start} - ${end} \u6761\uff0c\u5171 ${history.total} \u6761\u8bb0\u5f55\u3002`;
}

function renderHistoryTable(results) {
  const container = document.getElementById("historyTable");
  if (!results.length) {
    container.innerHTML = "<p class='empty-state'>\u5f53\u524d\u7b5b\u9009\u6761\u4ef6\u4e0b\u6ca1\u6709\u5386\u53f2\u8bb0\u5f55\u3002</p>";
    return;
  }

  const rows = results
    .map(
      (item) => `
    <tr class="history-row" data-record-id="${escapeHtml(item.record_id)}">
      <td>${item.record_id}</td>
      <td>${item.model_name}</td>
      <td>${item.prediction_text}</td>
      <td>${item.risk_level || ""}</td>
      <td>${item.prediction_score ?? ""}</td>
      <td>${item.created_at}</td>
    </tr>
  `
    )
    .join("");

  container.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>\u8bb0\u5f55 ID</th>
          <th>\u6a21\u578b</th>
          <th>\u68c0\u6d4b\u7ed3\u679c</th>
          <th>\u98ce\u9669\u7b49\u7ea7</th>
          <th>\u5206\u6570</th>
          <th>\u68c0\u6d4b\u65f6\u95f4</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
  container.querySelectorAll(".history-row").forEach((row) => {
    row.addEventListener("click", () => loadRecordDetail(row.dataset.recordId));
  });
}

async function loadRecordDetail(recordId) {
  renderMessage("recordDetail", "\u6b63\u5728\u52a0\u8f7d\u8bb0\u5f55\u8be6\u60c5...");
  try {
    const data = await fetchJson(`/api/history/${encodeURIComponent(recordId)}`);
    renderMessage("recordDetail", pretty(data.result));
  } catch (error) {
    renderMessage("recordDetail", error.message);
  }
}

async function runCsvDetection() {
  const inputPath = document.getElementById("csvPath").value.trim();
  const fileInput = document.getElementById("csvFile");
  const selectedFile = fileInput.files[0];
  const modelName = document.getElementById("csvModel").value;
  const limit = Number(document.getElementById("csvLimit").value || 10);
  renderMessage("csvSummary", "\u6b63\u5728\u6267\u884c\u6279\u91cf\u68c0\u6d4b...");

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
      data = await fetchJson("/api/detect/csv", {
        method: "POST",
        body: formData,
      });
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
    renderMessage(
      "csvSummary",
      pretty({
        count: data.count,
        model_name: data.model_name,
        summary: data.summary,
        first_result: data.results[0],
      })
    );
    await loadHistory();
  } catch (error) {
    renderMessage("csvSummary", error.message);
  }
}

async function runSingleDetection() {
  const sampleText = document.getElementById("singleSample").value;
  renderMessage("singleResult", "\u6b63\u5728\u6267\u884c\u5355\u6761\u68c0\u6d4b...");

  try {
    const sample = JSON.parse(sampleText);
    const llmSettings = getLlmSettings();
    const data = await fetchJson("/api/detect/single", {
      method: "POST",
      body: JSON.stringify({ sample, ...llmSettings }),
    });
    renderMessage("singleResult", pretty(data.result));
    await loadHistory();
  } catch (error) {
    renderMessage("singleResult", error.message);
  }
}

async function loadHistory() {
  const params = getFilterParams();
  renderMessage("historySummary", "\u6b63\u5728\u52a0\u8f7d\u5386\u53f2\u8bb0\u5f55...");

  try {
    const summaryParams = new URLSearchParams(params);
    summaryParams.delete("limit");
    summaryParams.delete("offset");

    const [history, summary] = await Promise.all([
      fetchJson(`/api/history?${params.toString()}`),
      fetchJson(`/api/history/summary?${summaryParams.toString()}`),
    ]);

    renderMessage("historySummary", pretty(summary));
    renderHistoryMeta(history);
    renderHistoryTable(history.results);
    renderStats(summary);
    renderChart("predictionChart", summary.prediction_counts);
    renderChart("riskChart", summary.risk_counts);
    renderChart("modelChart", summary.model_counts);
  } catch (error) {
    renderMessage("historySummary", error.message);
    document.getElementById("historyMeta").textContent = "\u5386\u53f2\u8bb0\u5f55\u67e5\u8be2\u5931\u8d25\u3002";
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
