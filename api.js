/**
 * api.js — FloodSight Python Backend Bridge (Safe Version)
 * =========================================================
 * This file ONLY adds the Python AI predictions on top of
 * the existing simulation. It never touches the map, cursor,
 * clock, canvas, or any core simulation code.
 *
 * Place this AFTER app.js in index.html:
 *   <script src="app.js"></script>
 *   <script src="api.js"></script>
 */

const FloodAPI = (function () {
  const BASE = "http://localhost:8000";

  async function apiFetch(path, options) {
    options = options || {};
    var res = await fetch(BASE + path, Object.assign(
      { headers: { "Content-Type": "application/json" } },
      options
    ));
    if (!res.ok) throw new Error("API error " + res.status);
    return res.json();
  }

  return {
    predict: function (cityKey) {
      return apiFetch("/api/predict", {
        method: "POST",
        body: JSON.stringify({ city: cityKey }),
      });
    },
    forecast: function (cityKey, steps) {
      return apiFetch("/api/forecast", {
        method: "POST",
        body: JSON.stringify({ city: cityKey, steps: steps || 24 }),
      });
    },
    weather: function (cityKey) {
      return apiFetch("/api/weather/" + cityKey);
    },
    health: async function () {
      try {
        var d = await apiFetch("/");
        return d.status === "ok";
      } catch (e) {
        return false;
      }
    }
  };
})();


// ── Safely integrate with FloodSight UI ──────────────────────────────────────
// Uses window.onload so app.js has fully finished initialising first.

window.addEventListener("load", function () {

  // Small delay to make sure app.js has set everything up
  setTimeout(async function () {
    try {

      // ── 1. Check backend is alive ───────────────────────────────────────
      var alive = await FloodAPI.health();

      if (!alive) {
        showBanner("⚠️ Python backend offline — using simulation mode", "#ffb347");
        console.warn("[FloodAPI] Backend not reachable at " + "http://localhost:8000");
        return;   // ← stop here, don't touch anything else
      }

      showBanner("🐍 Python AI backend connected ✓", "#7de8b0");
      console.log("[FloodAPI] Backend connected ✓");

      // ── 2. Hook city pill buttons to load live AI data ──────────────────
      var pills = document.querySelectorAll(".lpill");
      pills.forEach(function (pill) {
        pill.addEventListener("click", function () {
          var cityKey = (pill.dataset.city || pill.textContent.trim()).toLowerCase();
          loadLiveData(cityKey);
        });
      });

      // ── 3. Load data for the default active city ────────────────────────
      var activePill = document.querySelector(".lpill.active");
      if (activePill) {
        var defaultCity = (activePill.dataset.city || activePill.textContent.trim()).toLowerCase();
        loadLiveData(defaultCity);
      }

    } catch (e) {
      console.error("[FloodAPI] Startup error:", e);
      // Silently fail — the original simulation keeps working
    }
  }, 1500);   // wait 1.5s after page load

});


// ── Load live prediction for a city ──────────────────────────────────────────

async function loadLiveData(cityKey) {
  try {
    console.log("[FloodAPI] Fetching data for:", cityKey);

    var data = await FloodAPI.predict(cityKey);
    var level = data.predicted_level_m;
    var risk  = data.risk_category;
    var conf  = data.confidence_pct;

    console.log("[FloodAPI] Prediction:", level + "m  Risk:", risk);

    // ── Only update the AI/mascot bubble — don't touch simulation ────────
    updateMascotBubble(level, risk, conf);

    // ── Update feature importance text if AI insight panel exists ────────
    if (data.feature_importance) {
      updateAIInsight(data.feature_importance);
    }

    // ── Show weather info if returned ─────────────────────────────────────
    if (data.weather) {
      console.log("[FloodAPI] Live weather:", data.weather.description,
                  data.weather.temperature_c + "°C",
                  "Rain:", data.weather.rain_1h_mm + "mm");
    }

  } catch (e) {
    console.warn("[FloodAPI] Could not load live data:", e.message);
    // Silently fail — simulation still runs
  }
}


// ── UI helpers ────────────────────────────────────────────────────────────────

function updateMascotBubble(level, risk, conf) {
  try {
    var bubble = document.querySelector(".mascot-bubble");
    if (!bubble) return;

    var emoji = { Safe: "😊", Watch: "👀", Warning: "⚠️", Danger: "😨", Critical: "🚨" };
    var e = emoji[risk] || "🤔";

    bubble.innerHTML =
      e + " <strong>AI Prediction: " + level.toFixed(2) + "m</strong><br>" +
      "Risk: <strong>" + risk + "</strong> &nbsp;|&nbsp; Confidence: " + conf + "%<br>" +
      "<span style='font-size:.58rem;color:var(--txt2)'>🐍 Live PyTorch LSTM prediction</span>";
  } catch (e) {
    // ignore
  }
}

function updateAIInsight(importance) {
  try {
    var insight = document.querySelector(".ai-insight");
    if (!insight) return;

    var top = Object.entries(importance)
      .sort(function (a, b) { return b[1] - a[1]; })
      .slice(0, 3)
      .map(function (kv) {
        return "<strong>" + kv[0].replace(/_/g, " ") + "</strong> (" + kv[1] + "%)";
      })
      .join(", ");

    insight.innerHTML = "🧠 Top LSTM drivers: " + top;
  } catch (e) {
    // ignore
  }
}

function showBanner(msg, color) {
  try {
    var old = document.getElementById("floodapi-banner");
    if (old) old.remove();

    var banner = document.createElement("div");
    banner.id = "floodapi-banner";
    banner.style.cssText = [
      "position:fixed", "bottom:72px", "left:50%",
      "transform:translateX(-50%)", "z-index:9990",
      "padding:6px 20px", "border-radius:100px",
      "background:rgba(255,255,255,.95)",
      "border:2px solid " + color,
      "font-family:'Fredoka One',cursive",
      "font-size:.72rem", "color:#4a3550",
      "box-shadow:0 4px 18px rgba(0,0,0,.1)",
      "white-space:nowrap", "pointer-events:none"
    ].join(";");
    banner.textContent = msg;
    document.body.appendChild(banner);
    setTimeout(function () {
      if (banner.parentNode) banner.remove();
    }, 4000);
  } catch (e) {
    // ignore
  }
}