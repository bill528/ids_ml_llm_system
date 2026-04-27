async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

function renderHistoryTable(results) {
  const container = document.getElementById("historyTable");
  if (!results.length) {
    container.innerHTML = "<p>暂无记录</p>";
    return;
  }

  const rows = results.map((item) => `
    <tr>
      <td>${item.record_id}</td>
      <td>${item.model_name}</td>
      <td>${item.prediction_text}</td>
      <td>${item.risk_level || ""}</td>
      <td>${item.prediction_score ?? ""}</td>
      <td>${item.created_at}</td>
    </tr>
  `).join("");

  container.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>记录ID</th>
          <th>模型</th>
          <th>检测结果</th>
          <th>风险等级</th>
          <th>分数</th>
          <th>时间</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

async function runCsvDetection() {
  const input_path = document.getElementById("csvPath").value.trim();
  const model_name = document.getElementById("csvModel").value;
  const limit = Number(document.getElementById("csvLimit").value || 10);
  const summaryBox = document.getElementById("csvSummary");
  summaryBox.textContent = "检测中...";
  try {
    const data = await fetchJson("/api/detect/csv", {
      method: "POST",
      body: JSON.stringify({ input_path, model_name, limit }),
    });
    summaryBox.textContent = pretty({
      count: data.count,
      model_name: data.model_name,
      summary: data.summary,
      first_result: data.results[0],
    });
  } catch (error) {
    summaryBox.textContent = error.message;
  }
}

async function runSingleDetection() {
  const sampleText = document.getElementById("singleSample").value;
  const resultBox = document.getElementById("singleResult");
  resultBox.textContent = "检测中...";
  try {
    const sample = JSON.parse(sampleText);
    const data = await fetchJson("/api/detect/single", {
      method: "POST",
      body: JSON.stringify({ sample }),
    });
    resultBox.textContent = pretty(data);
  } catch (error) {
    resultBox.textContent = error.message;
  }
}

async function loadHistory() {
  const model_name = document.getElementById("historyModel").value;
  const risk_level = document.getElementById("historyRisk").value;
  const prediction_text = document.getElementById("historyPrediction").value;
  const params = new URLSearchParams({ limit: "20" });
  if (model_name) params.set("model_name", model_name);
  if (risk_level) params.set("risk_level", risk_level);
  if (prediction_text) params.set("prediction_text", prediction_text);

  const historySummary = document.getElementById("historySummary");
  historySummary.textContent = "查询中...";
  try {
    const [history, summary] = await Promise.all([
      fetchJson(`/api/history?${params.toString()}`),
      fetchJson(`/api/history/summary?${params.toString().replace("limit=20&", "")}`),
    ]);
    historySummary.textContent = pretty(summary);
    renderHistoryTable(history.results);
  } catch (error) {
    historySummary.textContent = error.message;
  }
}

document.getElementById("runCsvBtn").addEventListener("click", runCsvDetection);
document.getElementById("runSingleBtn").addEventListener("click", runSingleDetection);
document.getElementById("loadHistoryBtn").addEventListener("click", loadHistory);

loadHistory();
