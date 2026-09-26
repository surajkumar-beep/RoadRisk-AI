/* ==========================================================================
   RoadRisk AI — progressive UI enhancements (no framework, no dependencies).

   Everything here is additive: the pages are fully server-rendered and remain
   usable if JavaScript is disabled. Animations respect
   prefers-reduced-motion and never block interaction.
   ========================================================================== */
(function () {
  "use strict";

  var reduceMotion = false;
  try {
    reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  } catch (err) {
    reduceMotion = false;
  }

  /* ------------------------------------------------------------ helpers --- */
  function parseNumeric(text) {
    // "12,316" -> 12316 ; "15.4%" -> 15.4 ; "Rainy" -> null
    if (!text) return null;
    var cleaned = String(text).trim().replace(/,/g, "").replace(/%$/, "");
    if (!/^-?\d+(\.\d+)?$/.test(cleaned)) return null;
    return { value: parseFloat(cleaned), decimals: (cleaned.split(".")[1] || "").length };
  }

  function formatNumber(value, decimals, withCommas, suffix) {
    var fixed = value.toFixed(decimals);
    if (withCommas) {
      var parts = fixed.split(".");
      parts[0] = parts[0].replace(/\B(?=(\d{3})+(?!\d))/g, ",");
      fixed = parts.join(".");
    }
    return fixed + suffix;
  }

  /* ------------------------------------------------------- KPI count-up --- */
  function countUp(el) {
    var target = parseNumeric(el.textContent);
    if (!target || target.value === 0) return;
    var original = el.textContent.trim();
    var hasCommas = original.indexOf(",") !== -1;
    var hasPercent = /%$/.test(original);
    var duration = Math.min(900, 320 + Math.abs(target.value) * 0.6);
    var start = null;

    function step(ts) {
      if (start === null) start = ts;
      var progress = Math.min((ts - start) / duration, 1);
      // easeOutCubic
      var eased = 1 - Math.pow(1 - progress, 3);
      var current = target.value * eased;
      el.textContent = formatNumber(
        current, target.decimals, hasCommas || current >= 1000, hasPercent ? "%" : ""
      );
      if (progress < 1) {
        window.requestAnimationFrame(step);
      } else {
        el.textContent = original; // exact server-rendered value
      }
    }
    window.requestAnimationFrame(step);
  }

  /* ----------------------------------------------------- reveal on view --- */
  function initReveal() {
    var nodes = Array.prototype.slice.call(document.querySelectorAll(".reveal"));
    if (!nodes.length) return;

    if (reduceMotion || !("IntersectionObserver" in window)) {
      nodes.forEach(function (el) { el.classList.add("in"); });
      return;
    }

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry, i) {
        if (!entry.isIntersecting) return;
        var el = entry.target;
        // small stagger for items entering together
        window.setTimeout(function () { el.classList.add("in"); }, Math.min(i * 35, 210));
        observer.unobserve(el);
      });
    }, { rootMargin: "0px 0px -40px 0px", threshold: 0.04 });

    nodes.forEach(function (el) { observer.observe(el); });
  }

  /* ---------------------------------------------------- charts & toasts --- */
  function initProbBars() {
    document.querySelectorAll(".prob-fill").forEach(function (el) {
      var target = parseFloat(el.getAttribute("data-width") || "0");
      el.style.width = "0%";
      if (reduceMotion) {
        el.style.width = target + "%";
      } else {
        window.setTimeout(function () { el.style.width = target + "%"; }, 180);
      }
    });
  }

  function initRings() {
    document.querySelectorAll(".conf-ring").forEach(function (ring) {
      var pct = parseFloat(ring.getAttribute("data-pct") || "0");
      ring.style.setProperty("--pct", pct);
    });
  }

  function initToasts() {
    document.querySelectorAll(".flash").forEach(function (flash) {
      window.setTimeout(function () {
        flash.style.transition = "opacity .35s ease";
        flash.style.opacity = "0";
        window.setTimeout(function () {
          if (flash.parentNode) flash.parentNode.removeChild(flash);
        }, 380);
      }, 5200);
    });
  }

  function resizeCharts() {
    if (!window.Plotly) return;
    document.querySelectorAll(".js-plotly .plotly-graph-div").forEach(function (gd) {
      try { window.Plotly.Plots.resize(gd); } catch (err) { /* non-fatal */ }
    });
  }

  function initResizeTriggers() {
    // Resize Plotly after layout changes (filter reset links, window resize).
    document.querySelectorAll("[data-js-resize]").forEach(function (el) {
      el.addEventListener("click", function () { window.setTimeout(resizeCharts, 340); });
    });
    var t = null;
    window.addEventListener("resize", function () {
      window.clearTimeout(t);
      t = window.setTimeout(resizeCharts, 180);
    });
  }

  /* ------------------------------------------------------------- startup --- */
  document.addEventListener("DOMContentLoaded", function () {
    initReveal();
    initRings();
    initProbBars();
    initToasts();
    initResizeTriggers();

    if (!reduceMotion) {
      document.querySelectorAll("[data-countup]").forEach(function (el) { countUp(el); });
    }

    // Charts are embedded as static divs; nudge Plotly once fonts/fonts settle.
    window.setTimeout(resizeCharts, 260);
  });
})();