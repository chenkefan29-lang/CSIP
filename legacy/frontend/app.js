let trendChart = null;

const sceneSelect = document.getElementById("scene");
const aerationInput = document.getElementById("aeration");
const recycleInput = document.getElementById("recycleRatio");
const aerationValue = document.getElementById("aerationValue");
const recycleValue = document.getElementById("recycleValue");
const simulateButton = document.getElementById("simulateButton");

function updateSliderLabels() {
  aerationValue.textContent = Number(aerationInput.value).toFixed(1);
  recycleValue.textContent = Number(recycleInput.value).toFixed(1);
}

function renderAlerts(alerts) {
  const alertsBox = document.getElementById("alerts");

  if (alerts.length === 0) {
    alertsBox.innerHTML = '<div class="no-alert">当前没有异常告警。</div>';
    return;
  }

  alertsBox.innerHTML = alerts
    .map((alert) => `<div class="alert-item">${alert}</div>`)
    .join("");
}

function renderChart(series) {
  const canvas = document.getElementById("trendChart");
  const labels = series.map((item) => `${item.hour}:00`);

  if (trendChart) {
    trendChart.destroy();
  }

  trendChart = new Chart(canvas, {
    type: "line",
    data: {
      labels: labels,
      datasets: [
        {
          label: "出水 COD (mg/L)",
          data: series.map((item) => item.cod),
          borderColor: "#087f8c",
          backgroundColor: "rgba(8, 127, 140, 0.12)",
          tension: 0.3,
        },
        {
          label: "出水氨氮 (mg/L)",
          data: series.map((item) => item.ammonia),
          borderColor: "#e17d22",
          backgroundColor: "rgba(225, 125, 34, 0.12)",
          tension: 0.3,
        },
        {
          label: "DO (mg/L)",
          data: series.map((item) => item.do),
          borderColor: "#2463a6",
          backgroundColor: "rgba(36, 99, 166, 0.12)",
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: "index",
        intersect: false,
      },
      scales: {
        y: {
          beginAtZero: true,
        },
      },
    },
  });
}

async function loadSimulation() {
  const scene = sceneSelect.value;
  const aeration = aerationInput.value;
  const recycleRatio = recycleInput.value;

  simulateButton.disabled = true;
  simulateButton.textContent = "正在计算...";

  try {
    const query = new URLSearchParams({
      scene: scene,
      aeration: aeration,
      recycle_ratio: recycleRatio,
    });

    const response = await fetch(`/api/simulate?${query.toString()}`);

    if (!response.ok) {
      throw new Error("后端接口返回错误");
    }

    const result = await response.json();
    const current = result.current;

    document.getElementById("codValue").textContent = current.cod;
    document.getElementById("ammoniaValue").textContent = current.ammonia;
    document.getElementById("doValue").textContent = current.do;
    document.getElementById("mlssValue").textContent = current.mlss;

    document.getElementById("advice").textContent = result.advice;
    document.getElementById("statusBadge").textContent =
      result.alerts.length === 0 ? "运行正常" : `存在 ${result.alerts.length} 条告警`;

    renderAlerts(result.alerts);
    renderChart(result.series);
  } catch (error) {
    document.getElementById("advice").textContent =
      "无法获取模拟数据，请确认后端服务仍在运行。";
    document.getElementById("alerts").innerHTML =
      `<div class="alert-item">${error.message}</div>`;
    document.getElementById("statusBadge").textContent = "连接失败";
  } finally {
    simulateButton.disabled = false;
    simulateButton.textContent = "更新模拟结果";
  }
}

aerationInput.addEventListener("input", updateSliderLabels);
recycleInput.addEventListener("input", updateSliderLabels);
simulateButton.addEventListener("click", loadSimulation);

updateSliderLabels();
loadSimulation();