async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok || data.success === false) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

function renderMessage(id, text) {
  document.getElementById(id).textContent = text;
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

function renderHistoryMeta(history) {
  const start = history.total === 0 ? 0 : history.offset + 1;
  const end = history.offset + history.count;
  document.getElementById("historyMeta").textContent =
    `Showing records ${start} - ${end} of ${history.total}.`;
}

function renderHistoryTable(results) {
  const container = document.getElementById("historyTable");
  if (!results.length) {
    container.innerHTML = "<p class='empty-state'>No history records match the current filters.</p>";
    return;
  }

  const rows = results
    .map(
      (item) => `
    <tr>
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
          <th>Record ID</th>
          <th>Model</th>
          <th>Prediction</th>
          <th>Risk Level</th>
          <th>Score</th>
          <th>Created At</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

async function runCsvDetection() {
  const inputPath = document.getElementById("csvPath").value.trim();
  const modelName = document.getElementById("csvModel").value;
  const limit = Number(document.getElementById("csvLimit").value || 10);
  renderMessage("csvSummary", "Running batch detection...");

  try {
    const data = await fetchJson("/api/detect/csv", {
      method: "POST",
      body: JSON.stringify({ input_path: inputPath, model_name: modelName, limit }),
    });
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
  renderMessage("singleResult", "Running single detection...");

  try {
    const sample = JSON.parse(sampleText);
    const data = await fetchJson("/api/detect/single", {
      method: "POST",
      body: JSON.stringify({ sample }),
    });
    renderMessage("singleResult", pretty(data.result));
    await loadHistory();
  } catch (error) {
    renderMessage("singleResult", error.message);
  }
}

async function loadHistory() {
  const params = getFilterParams();
  renderMessage("historySummary", "Loading history...");

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
  } catch (error) {
    renderMessage("historySummary", error.message);
    document.getElementById("historyMeta").textContent = "History query failed.";
    document.getElementById("historyTable").innerHTML = "";
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

document.getElementById("runCsvBtn").addEventListener("click", runCsvDetection);
document.getElementById("runSingleBtn").addEventListener("click", runSingleDetection);
document.getElementById("loadHistoryBtn").addEventListener("click", loadHistory);
document.getElementById("resetHistoryBtn").addEventListener("click", resetHistoryFilters);

loadHistory();
