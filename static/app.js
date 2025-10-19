const $  = (s) => document.querySelector(s);

let currentSource   = "forecasts"; // "forecasts" | "actuals"
let onlyFlagEquals1 = false;       // forecasts filter

function showFormForActuals(show) {
  $("#actualsFormWrap").style.display = show ? "" : "none";
}
function showForecastFilters(show) {
  $("#forecastFilters").style.display = show ? "" : "none";
}
function setToggleButtons() {
  $("#flagAll").classList.toggle("primary", !onlyFlagEquals1);
  $("#flagOnly1").classList.toggle("primary",  onlyFlagEquals1);
}
function setTitle() {
  $("#which").textContent = currentSource === "forecasts" ? "forecasts_future" : "actuals_raw";
  $("#showForecasts").classList.toggle("primary", currentSource === "forecasts");
  $("#showActuals").classList.toggle("primary", currentSource === "actuals");
  showFormForActuals(currentSource === "actuals");
  showForecastFilters(currentSource === "forecasts");
  setToggleButtons();
}

async function loadData(limit = 500) {
  const params = new URLSearchParams({
    source: currentSource,
    limit: String(limit)
  });
  if (currentSource === "forecasts") {
    params.set("only1", onlyFlagEquals1 ? "1" : "0");
  }

  const res = await fetch(`/api/data?${params.toString()}`);
  const json = await res.json();

  // Clamp input to backend-applied limit
  if (typeof json.total_rows === "number" && typeof json.applied_limit === "number") {
    const requested = Number($("#limitInput").value || 500);
    if (requested > json.total_rows || requested !== json.applied_limit) {
      $("#limitInput").value = String(json.applied_limit);
    }
  }

  renderTable(Array.isArray(json.rows) ? json.rows : []);
}

function renderTable(rows) {
  const thead = $("#theadRow");
  const body  = $("#dataBody");
  const empty = $("#emptyMsg");

  thead.innerHTML = "";
  body.innerHTML  = "";

  if (!rows.length) {
    empty.style.display = "";
    return;
  }
  empty.style.display = "none";

  const cols = Object.keys(rows[0]);
  cols.forEach(k => {
    const th = document.createElement("th");
    th.textContent = k;
    thead.appendChild(th);
  });

  rows.forEach(r => {
    const tr = document.createElement("tr");
    cols.forEach(k => {
      const td = document.createElement("td");
      const v = r[k];
      td.textContent = (v === null || v === undefined) ? "" : v;
      tr.appendChild(td);
    });
    body.appendChild(tr);
  });
}

// ---- Actuals form (unchanged) ----
function formToJSON(form) {
  const data = Object.fromEntries(new FormData(form).entries());
  const numeric = ["store_id","product_id","TOTAL_OUT_THE_DOOR","TOTAL_AVAILABLE_TO_SELL"];
  numeric.forEach(k => {
    if (data[k] === "" || data[k] === null || data[k] === undefined) {
      data[k] = null;
    } else {
      const n = Number(data[k]);
      data[k] = Number.isFinite(n) ? n : null;
    }
  });
  ["product", "category"].forEach(k => { if (data[k] === "") data[k] = null; });
  return data;
}
async function upsertActuals(payload) {
  const res = await fetch("/api/actuals", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  return res.json();
}

// ---- Predict next X days (unchanged) ----
async function runForecast() {
  const btn = $("#predictBtn");
  const msg = $("#predictMsg");
  const days = Number($("#predictDays").value || 7);
  btn.disabled = true;
  msg.textContent = "Running pipeline…";

  try {
    const res  = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ days })
    });
    const json = await res.json();
    if (!res.ok || !json.ok) {
      msg.textContent = `Pipeline failed: ${(json && (json.error || json.errors)) || res.status}`;
    } else {
      msg.textContent = `Pipeline ok (horizon=${json.days}). Refreshing forecasts…`;
      currentSource = "forecasts";
      setTitle();
      loadData(Number($("#limitInput").value || 500));
    }
  } catch (err) {
    msg.textContent = `Error: ${err.message || err}`;
  } finally {
    btn.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setTitle();
  loadData(500);

  $("#showForecasts").addEventListener("click", () => {
    currentSource = "forecasts";
    setTitle();
    loadData(Number($("#limitInput").value || 500));
  });
  $("#showActuals").addEventListener("click", () => {
    currentSource = "actuals";
    setTitle();
    loadData(Number($("#limitInput").value || 500));
  });
  $("#refreshBtn").addEventListener("click", () => {
    loadData(Number($("#limitInput").value || 500));
  });

  $("#actualsForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const msg = $("#formMsg");
    msg.textContent = "Saving...";
    try {
      const out = await upsertActuals(formToJSON(e.target));
      if (out.ok) {
        msg.textContent = `Success: ${out.action}`;
        if (currentSource === "actuals") {
          loadData(Number($("#limitInput").value || 500));
        }
        e.target.reset();
      } else {
        msg.textContent = `Error: ${out.error || "Unknown error"}`;
      }
    } catch (err) {
      msg.textContent = `Error: ${err.message || err}`;
    }
  });

  // Forecast filter toggles
  $("#flagAll").addEventListener("click", () => {
    onlyFlagEquals1 = false;
    setToggleButtons();
    if (currentSource === "forecasts") {
      loadData(Number($("#limitInput").value || 500));
    }
  });
  $("#flagOnly1").addEventListener("click", () => {
    onlyFlagEquals1 = true;
    setToggleButtons();
    if (currentSource === "forecasts") {
      loadData(Number($("#limitInput").value || 500));
    }
  });

  $("#predictBtn").addEventListener("click", runForecast);
});