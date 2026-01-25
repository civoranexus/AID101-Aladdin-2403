const API_BASE = "http://127.0.0.1:8000";

const CROPS_100 = [
  "wheat","rice","maize","sugarcane","cotton","potato","onion","tomato","mustard",
  "groundnut","soybean","sunflower","chickpea","pigeonpea","lentil","sorghum",
  "pearlmillet","barley","jute","tea","coffee","banana","mango","orange","apple",
  "grapes","papaya","pomegranate","guava","watermelon","muskmelon","pineapple",
  "coconut","cashew","arecanut","rubber","cocoa","brinjal","cauliflower","cabbage",
  "carrot","radish","peas","beans","okra","spinach","capsicum","cucumber","pumpkin",
  "bottle_gourd","bitter_gourd","ridge_gourd","garlic","ginger","turmeric","chilli",
  "coriander","cumin","fennel","fenugreek","cardamom","clove","black_pepper",
  "cinnamon","bay_leaf","safflower","sesame","castor","linseed","niger","moong",
  "urad","masoor","rajma","cowpea","mothbean","horsegram","kidneybean","tobacco",
  "bajra","jowar","ragi","foxtail_millet","little_millet","kodo_millet","barnyard_millet",
  "proso_millet","amaranth","sugarbeet","sweet_potato","tapioca","jackfruit","litchi",
  "custard_apple","fig","date_palm","jamun","kiwi","strawberry","dragon_fruit"
];

// ✅ Toast
const toast = (msg) => {
  const el = document.getElementById("toast");
  el.textContent = msg;
  el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 2400);
};

// ✅ API status dot
const setApiStatus = (ok) => {
  const dot = document.getElementById("apiDot");
  const text = document.getElementById("apiStatus");

  if (ok) {
    dot.style.background = "#22C55E";
    dot.style.boxShadow = "0 0 0 6px rgba(34,197,94,0.15)";
    text.textContent = "Backend Online (FastAPI running)";
  } else {
    dot.style.background = "#EF4444";
    dot.style.boxShadow = "0 0 0 6px rgba(239,68,68,0.15)";
    text.textContent = "Backend Offline (start uvicorn)";
  }
};

async function pingBackend() {
  try {
    const res = await fetch(`${API_BASE}/docs`, { method: "GET" });
    setApiStatus(res.ok);
  } catch (err) {
    setApiStatus(false);
  }
}

function pretty(obj) {
  return JSON.stringify(obj, null, 2);
}

/* =========================
   ✅ Crop Dropdown + Search
========================= */
function fillCropDropdown(selectEl, crops, defaultCrop = "wheat") {
  selectEl.innerHTML = "";
  crops.forEach(crop => {
    const opt = document.createElement("option");
    opt.value = crop;
    opt.textContent = crop.replaceAll("_", " ").toUpperCase();
    selectEl.appendChild(opt);
  });
  selectEl.value = defaultCrop;
}

function filterCrops(query) {
  const q = query.trim().toLowerCase();
  if (!q) return CROPS_100;
  return CROPS_100.filter(c => c.includes(q));
}

function setupCropSearch(searchInputId, selectId, defaultCrop) {
function formatCropLabel(crop) {
  return crop.replaceAll("_", " ").toUpperCase();
}

function fillCropDropdown(selectEl, crops, defaultCrop = "wheat") {
  selectEl.innerHTML = "";
  crops.forEach(crop => {
    const opt = document.createElement("option");
    opt.value = crop;
    opt.textContent = formatCropLabel(crop);
    selectEl.appendChild(opt);
  });
  selectEl.value = defaultCrop;
}

function getSuggestions(query, limit = 8) {
  const q = query.trim().toLowerCase();
  if (!q) return [];

  return CROPS_100
    .filter(c => c.includes(q))
    .slice(0, limit);
}

function renderSuggestions(container, suggestions, activeIndex) {
  container.innerHTML = "";

  suggestions.forEach((crop, i) => {
    const item = document.createElement("div");
    item.className = "suggestion-item" + (i === activeIndex ? " active" : "");

    item.innerHTML = `
      <span class="suggestion-name">${formatCropLabel(crop)}</span>
      <span class="suggestion-tag">Crop</span>
    `;

    item.dataset.value = crop;
    container.appendChild(item);
  });
}

function setupCropAutoSuggest(searchInputId, suggestionsId, selectId, defaultCrop) {
  const input = document.getElementById(searchInputId);
  const suggestionsBox = document.getElementById(suggestionsId);
  const select = document.getElementById(selectId);

  fillCropDropdown(select, CROPS_100, CROPS_100[0]);
  select.value = "";
  // input.value = defaultCrop;

  let activeIndex = -1;
  let current = [];

  const hide = () => {
    suggestionsBox.classList.add("hidden");
    suggestionsBox.innerHTML = "";
    activeIndex = -1;
    current = [];
  };

  const show = () => {
    suggestionsBox.classList.remove("hidden");
  };

  const applyCrop = (crop) => {
    if (!crop) return;
    input.value = crop;
    select.value = crop;
    hide();
  };

  input.addEventListener("input", () => {
    const q = input.value;
    current = getSuggestions(q, 10);
    activeIndex = -1;

    if (!current.length) {
      hide();
      return;
    }

    renderSuggestions(suggestionsBox, current, activeIndex);
    show();
  });

  input.addEventListener("focus", () => {
    const q = input.value;
    current = getSuggestions(q, 10);
    if (!current.length) return;
    renderSuggestions(suggestionsBox, current, activeIndex);
    show();
  });

  input.addEventListener("keydown", (e) => {
    if (!current.length && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
      current = getSuggestions(input.value, 10);
      if (!current.length) return;
      renderSuggestions(suggestionsBox, current, activeIndex);
      show();
    }

    if (e.key === "ArrowDown") {
      e.preventDefault();
      activeIndex = Math.min(activeIndex + 1, current.length - 1);
      renderSuggestions(suggestionsBox, current, activeIndex);
      return;
    }

    if (e.key === "ArrowUp") {
      e.preventDefault();
      activeIndex = Math.max(activeIndex - 1, 0);
      renderSuggestions(suggestionsBox, current, activeIndex);
      return;
    }

    if (e.key === "Enter") {
      if (activeIndex >= 0 && current[activeIndex]) {
        e.preventDefault();
        applyCrop(current[activeIndex]);
      }
      return;
    }

    if (e.key === "Escape") {
      hide();
      return;
    }
  });

  suggestionsBox.addEventListener("click", (e) => {
    const item = e.target.closest(".suggestion-item");
    if (!item) return;
    applyCrop(item.dataset.value);
  });

  // if user manually changes dropdown, update input
  select.addEventListener("change", () => {
    input.value = select.value;
    hide();
  });

  // click outside closes suggestions
  document.addEventListener("click", (e) => {
    if (!suggestionsBox.contains(e.target) && e.target !== input) {
      hide();
    }
  });
}
}

/* =========================
   ✅ Weather Widget
========================= */
async function loadWeather(city) {
  const wCityLabel = document.getElementById("weatherCityLabel");
  const wTemp = document.getElementById("wTemp");
  const wHum = document.getElementById("wHum");
  const wCond = document.getElementById("wCond");

  wCityLabel.textContent = `City: ${city || "--"}`;
  wTemp.textContent = "--";
  wHum.textContent = "--";
  wCond.textContent = "--";

  if (!city) return;

  try {
    const res = await fetch(`${API_BASE}/weather?city=${encodeURIComponent(city)}`);
    const data = await res.json();

    if (!res.ok || data.error) {
      toast("Weather fetch failed ❌");
      wCond.textContent = "Weather API error";
      return;
    }

    wTemp.textContent = data.temperature ?? "--";
    wHum.textContent = data.humidity ?? "--";
    wCond.textContent = data.condition ?? "--";
  } catch (err) {
    toast("Backend offline ❌ (cannot load weather)");
  }
}

/* =========================
   ✅ Charts
========================= */
let yieldChart, irrChart;
const yieldHistory = [];
const irrHistory = [];

function initCharts() {
  const yc = document.getElementById("yieldChart");
  const ic = document.getElementById("irrChart");

  yieldChart = new Chart(yc, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "Predicted Yield (tons/acre)",
        data: []
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: true } }
    }
  });

  irrChart = new Chart(ic, {
    type: "line",
    data: {
      labels: [],
      datasets: [{
        label: "Recommended Water (mm)",
        data: []
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: true } }
    }
  });
}

function pushYield(value) {
  const t = new Date().toLocaleTimeString();
  yieldHistory.push({ t, value });
  if (yieldHistory.length > 8) yieldHistory.shift();

  yieldChart.data.labels = yieldHistory.map(x => x.t);
  yieldChart.data.datasets[0].data = yieldHistory.map(x => x.value);
  yieldChart.update();
}

function pushIrr(value) {
  const t = new Date().toLocaleTimeString();
  irrHistory.push({ t, value });
  if (irrHistory.length > 8) irrHistory.shift();

  irrChart.data.labels = irrHistory.map(x => x.t);
  irrChart.data.datasets[0].data = irrHistory.map(x => x.value);
  irrChart.update();
}

/* =========================
   ✅ Yield Prediction
========================= */
document.getElementById("yieldForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const crop = document.getElementById("yieldCrop").value;
  const city = document.getElementById("yieldCity").value.trim();
  const rainfall = document.getElementById("yieldRainfall").value;
  const fertilizer = document.getElementById("yieldFertilizer").value;

  // Update weather widget based on yield city
  await loadWeather(city);

  const url =
    `${API_BASE}/predict-yield?crop=${encodeURIComponent(crop)}` +
    `&rainfall=${encodeURIComponent(rainfall)}` +
    `&fertilizer=${encodeURIComponent(fertilizer)}` +
    `&city=${encodeURIComponent(city)}`;

  try {
    const res = await fetch(url, { method: "POST" });
    const data = await res.json();

    document.getElementById("yieldOutput").textContent = pretty(data);

    if (!res.ok) {
      toast("Yield prediction failed ❌");
      return;
    }

    toast("Yield predicted successfully ✅");
    if (data.predicted_yield !== undefined) pushYield(data.predicted_yield);
  } catch (err) {
    toast("Backend not reachable ❌");
  }
});

/* =========================
   ✅ Irrigation Prediction
========================= */
document.getElementById("irrigationForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const crop = document.getElementById("irrCrop").value;
  const city = document.getElementById("irrCity").value.trim();
  const soil = document.getElementById("soilMoisture").value;

  // Update weather widget based on irrigation city
  await loadWeather(city);

  const url =
    `${API_BASE}/predict-irrigation?soil_moisture=${encodeURIComponent(soil)}` +
    `&crop=${encodeURIComponent(crop)}` +
    `&city=${encodeURIComponent(city)}`;

  try {
    const res = await fetch(url, { method: "POST" });
    const data = await res.json();

    document.getElementById("irrOutput").textContent = pretty(data);

    if (!res.ok) {
      toast("Irrigation prediction failed ❌");
      return;
    }

    toast("Irrigation recommendation ready ✅");
    if (data.recommended_water_mm !== undefined) pushIrr(data.recommended_water_mm);
  } catch (err) {
    toast("Backend not reachable ❌");
  }
});

/* =========================
   ✅ Refresh
========================= */
document.getElementById("btnRefresh").addEventListener("click", async () => {
  await pingBackend();
  const city = document.getElementById("yieldCity").value.trim() || "Delhi";
  await loadWeather(city);
  toast("Dashboard refreshed ✅");
});

document.getElementById("btnWeatherRefresh").addEventListener("click", async () => {
  const city =
    document.getElementById("yieldCity").value.trim() ||
    document.getElementById("irrCity").value.trim();

  if (!city) {
    toast("Please enter a city first ❌");
    return;
  }

  await loadWeather(city);
  toast("Weather updated ✅");
});

function formatCropLabel(crop) {
  return crop.replaceAll("_", " ").toUpperCase();
}

function fillCropDropdown(selectEl, crops, defaultCrop = "wheat") {
  selectEl.innerHTML = "";
  crops.forEach(crop => {
    const opt = document.createElement("option");
    opt.value = crop;
    opt.textContent = formatCropLabel(crop);
    selectEl.appendChild(opt);
  });
  selectEl.value = defaultCrop;
}

function getSuggestions(query, limit = 8) {
  const q = query.trim().toLowerCase();
  if (!q) return [];
  return CROPS_100.filter(c => c.includes(q)).slice(0, limit);
}

function renderSuggestions(container, suggestions, activeIndex) {
  container.innerHTML = "";

  suggestions.forEach((crop, i) => {
    const item = document.createElement("div");
    item.className = "suggestion-item" + (i === activeIndex ? " active" : "");

    item.innerHTML = `
      <span class="suggestion-name">${formatCropLabel(crop)}</span>
      <span class="suggestion-tag">Crop</span>
    `;

    item.dataset.value = crop;
    container.appendChild(item);
  });
}

function setupCropAutoSuggest(searchInputId, suggestionsId, selectId, defaultCrop) {
  const input = document.getElementById(searchInputId);
  const suggestionsBox = document.getElementById(suggestionsId);
  const select = document.getElementById(selectId);

  if (!input || !suggestionsBox || !select) {
    console.error("AutoSuggest missing element:", { searchInputId, suggestionsId, selectId });
    return;
  }

  fillCropDropdown(select, CROPS_100, defaultCrop);
  input.value = defaultCrop;

  let activeIndex = -1;
  let current = [];

  const hide = () => {
    suggestionsBox.classList.add("hidden");
    suggestionsBox.innerHTML = "";
    activeIndex = -1;
    current = [];
  };

  const show = () => {
    suggestionsBox.classList.remove("hidden");
  };

  const applyCrop = (crop) => {
    if (!crop) return;
    input.value = crop;
    select.value = crop;
    hide();
  };

  input.addEventListener("input", () => {
    current = getSuggestions(input.value, 10);
    activeIndex = -1;

    if (!current.length) {
      hide();
      return;
    }

    renderSuggestions(suggestionsBox, current, activeIndex);
    show();
  });

  input.addEventListener("focus", () => {
    current = getSuggestions(input.value, 10);
    if (!current.length) return;
    renderSuggestions(suggestionsBox, current, activeIndex);
    show();
  });

  input.addEventListener("keydown", (e) => {
    if (!current.length && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
      current = getSuggestions(input.value, 10);
      if (!current.length) return;
      renderSuggestions(suggestionsBox, current, activeIndex);
      show();
    }

    if (e.key === "ArrowDown") {
      e.preventDefault();
      activeIndex = Math.min(activeIndex + 1, current.length - 1);
      renderSuggestions(suggestionsBox, current, activeIndex);
      return;
    }

    if (e.key === "ArrowUp") {
      e.preventDefault();
      activeIndex = Math.max(activeIndex - 1, 0);
      renderSuggestions(suggestionsBox, current, activeIndex);
      return;
    }

    if (e.key === "Enter") {
      if (activeIndex >= 0 && current[activeIndex]) {
        e.preventDefault();
        applyCrop(current[activeIndex]);
      }
      return;
    }

    if (e.key === "Escape") {
      hide();
      return;
    }
  });

  suggestionsBox.addEventListener("click", (e) => {
    const item = e.target.closest(".suggestion-item");
    if (!item) return;
    applyCrop(item.dataset.value);
  });

  select.addEventListener("change", () => {
    input.value = select.value;
    hide();
  });

  document.addEventListener("click", (e) => {
    if (!suggestionsBox.contains(e.target) && e.target !== input) {
      hide();
    }
  });
}

/* =========================
   ✅ Init
========================= */

initCharts();
pingBackend();
loadWeather("");

setupCropAutoSuggest("yieldCropSearch", "yieldSuggestions", "yieldCrop", "wheat");
setupCropAutoSuggest("irrCropSearch", "irrSuggestions", "irrCrop", "rice");
