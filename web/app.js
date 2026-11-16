(function () {
  "use strict";

  var BUILD = "__BUILD__";
  var GPS_RE = /\.(fit|gpx|tcx)(\.gz)?$/i;

  var $ = function (id) { return document.getElementById(id); };
  var els = {
    drop: $("drop"),
    fileInput: $("file-input"),
    folderInput: $("folder-input"),
    selection: $("selection"),
    options: $("options"),
    generate: $("generate"),
    demo: $("demo"),
    cancel: $("cancel"),
    progress: $("progress"),
    barFill: $("bar-fill"),
    progressText: $("progress-text"),
    engine: $("engine"),
    error: $("error"),
    result: $("result"),
    img: $("result-img"),
    viewer: $("viewer"),
    download: $("download"),
  };

  var state = {
    worker: null,
    ready: false,
    busy: false,
    files: [],
    resultUrl: null,
    previewUrl: null,
    startedAt: 0,
  };

  // ---- worker ------------------------------------------------------------

  function startWorker() {
    state.ready = false;
    state.worker = new Worker("worker.js?v=" + BUILD);
    state.worker.onmessage = onWorkerMessage;
    state.worker.onerror = function (e) {
      showError("The processing engine crashed" + (e.message ? ": " + e.message : "."));
      finish();
    };
    els.engine.classList.remove("ready");
    els.engine.textContent = "Starting up…";
  }

  function onWorkerMessage(e) {
    var msg = e.data;
    if (msg.type === "status") {
      if (!state.ready) els.engine.textContent = msg.text;
      else els.progressText.textContent = msg.text;
    } else if (msg.type === "ready") {
      state.ready = true;
      els.engine.textContent = "Ready. Everything runs on this device.";
      els.engine.classList.add("ready");
      updateButtons();
    } else if (msg.type === "progress") {
      showProgress(msg.stage, msg.done, msg.total);
    } else if (msg.type === "result") {
      showResult(msg.png, msg.preview, msg.meta);
      finish();
    } else if (msg.type === "error") {
      if (!state.ready) els.engine.textContent = "The engine couldn't start. Check your connection and reload the page.";
      showError(msg.message);
      if (msg.detail) console.error(msg.detail);
      finish();
    }
  }

  // ---- picking files -------------------------------------------------------

  function wanted(name) {
    var lower = name.toLowerCase();
    return lower === "activities.csv" || lower.endsWith(".zip") || GPS_RE.test(lower);
  }

  function setFiles(list) {
    var seen = Object.create(null);
    var files = [];
    list.forEach(function (f) {
      if (!wanted(f.name) || seen[f.name]) return;
      seen[f.name] = true;
      files.push({ name: f.name, file: f });
    });
    var zips = files.filter(function (f) { return /\.zip$/i.test(f.name); });
    var gps = files.filter(function (f) { return GPS_RE.test(f.name); });
    var csv = files.some(function (f) { return f.name.toLowerCase() === "activities.csv"; });

    if (zips.length > 1 || (zips.length && gps.length)) {
      // Can't sensibly mix several exports; keep just the first zip.
      files = [zips[0]];
      zips = [zips[0]];
      gps = [];
      csv = false;
    }
    state.files = files;

    var text;
    if (!files.length) {
      text = list.length ? "None of those look like a Strava export or GPS files." : "";
    } else if (zips.length) {
      text = "Export zip: " + zips[0].name + " (" + formatBytes(zips[0].file.size) + ")";
    } else {
      text = gps.length + " GPS file" + (gps.length === 1 ? "" : "s") +
        (csv ? " + activities.csv" : " (no activities.csv, so types can't be filtered)");
    }
    els.selection.textContent = text;
    els.selection.classList.toggle("ok", files.length > 0);
    updateButtons();
  }

  // Walk a dropped folder. readEntries only returns ~100 entries per call.
  function readEntry(entry, out) {
    return new Promise(function (resolve) {
      if (entry.isFile) {
        entry.file(function (f) { out.push(f); resolve(); }, function () { resolve(); });
      } else if (entry.isDirectory) {
        var reader = entry.createReader();
        var all = [];
        (function batch() {
          reader.readEntries(function (entries) {
            if (!entries.length) {
              Promise.all(all.map(function (en) { return readEntry(en, out); })).then(resolve);
            } else {
              all = all.concat(entries);
              batch();
            }
          }, function () { resolve(); });
        })();
      } else {
        resolve();
      }
    });
  }

  function onDrop(e) {
    e.preventDefault();
    els.drop.classList.remove("over");
    var items = e.dataTransfer.items;
    var entries = [];
    if (items && items.length && items[0].webkitGetAsEntry) {
      for (var i = 0; i < items.length; i++) {
        var en = items[i].webkitGetAsEntry();
        if (en) entries.push(en);
      }
    }
    if (!entries.length) {
      setFiles(Array.from(e.dataTransfer.files));
      return;
    }
    var out = [];
    els.selection.textContent = "Reading folder…";
    Promise.all(entries.map(function (en) { return readEntry(en, out); })).then(function () {
      setFiles(out);
    });
  }

  ["dragenter", "dragover"].forEach(function (t) {
    els.drop.addEventListener(t, function (e) { e.preventDefault(); els.drop.classList.add("over"); });
  });
  els.drop.addEventListener("dragleave", function () { els.drop.classList.remove("over"); });
  els.drop.addEventListener("drop", onDrop);
  // Dropping outside the zone shouldn't navigate away from the page.
  window.addEventListener("dragover", function (e) { e.preventDefault(); });
  window.addEventListener("drop", function (e) { e.preventDefault(); });

  $("pick-files").addEventListener("click", function (e) { e.stopPropagation(); els.fileInput.click(); });
  $("pick-folder").addEventListener("click", function (e) { e.stopPropagation(); els.folderInput.click(); });
  els.drop.addEventListener("click", function () { els.fileInput.click(); });
  els.drop.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); els.fileInput.click(); }
  });
  els.fileInput.addEventListener("change", function () { setFiles(Array.from(els.fileInput.files)); els.fileInput.value = ""; });
  els.folderInput.addEventListener("change", function () { setFiles(Array.from(els.folderInput.files)); els.folderInput.value = ""; });

  // ---- options -------------------------------------------------------------

  function readOptions() {
    var f = els.options.elements;
    var area = f.area.value;
    return {
      size: parseInt(f.size.value, 10),
      colormap: f.colormap.value,
      scale: f.scale.value,
      line_width: parseFloat(f.line_width.value),
      glow: parseFloat(f.glow.value),
      include_types: f.include_types.value,
      exclude_commutes: f.exclude_commutes.checked,
      region: area === "all" ? "all" : "auto",
      area: area === "all" ? 1 : parseInt(area, 10),
    };
  }

  els.options.elements.line_width.addEventListener("input", function (e) {
    $("lw-out").textContent = parseFloat(e.target.value).toFixed(1);
  });
  els.options.elements.glow.addEventListener("input", function (e) {
    $("glow-out").textContent = parseFloat(e.target.value).toFixed(2);
  });
  els.options.addEventListener("submit", function (e) { e.preventDefault(); });

  // ---- running -------------------------------------------------------------

  function updateButtons() {
    els.generate.disabled = state.busy || !state.files.length;
    els.demo.disabled = state.busy;
    els.cancel.hidden = !state.busy;
    els.generate.textContent = state.busy ? "Working…" : "Generate heatmap";
  }

  function begin(message) {
    hideError();
    state.busy = true;
    state.startedAt = performance.now();
    document.body.dataset.state = "working";
    els.progress.hidden = false;
    els.barFill.style.width = "0%";
    els.progressText.textContent = state.ready ? "Starting…" : "Waiting for the engine to finish loading…";
    updateButtons();
    state.worker.postMessage(message);
  }

  function finish() {
    state.busy = false;
    els.progress.hidden = true;
    updateButtons();
  }

  els.generate.addEventListener("click", function () {
    if (!state.files.length) return;
    begin({ type: "run", files: state.files, options: readOptions() });
  });

  els.demo.addEventListener("click", function () {
    begin({ type: "demo", options: readOptions() });
  });

  els.cancel.addEventListener("click", function () {
    state.worker.terminate();
    finish();
    document.body.dataset.state = "idle";
    startWorker();
  });

  var STAGES = { Reading: [0, 70], Drawing: [70, 96], Colouring: [96, 100] };

  function showProgress(stage, done, total) {
    var span = STAGES[stage] || [0, 100];
    var frac = total ? done / total : 0;
    var pct = span[0] + (span[1] - span[0]) * frac;
    els.barFill.style.width = pct.toFixed(1) + "%";
    var label = {
      Reading: "Reading activities",
      Drawing: "Drawing routes",
      Colouring: "Colouring in",
    }[stage] || stage;
    els.progressText.textContent = stage === "Colouring"
      ? label + "…"
      : label + ": " + done.toLocaleString() + " of " + total.toLocaleString();
  }

  // ---- results ---------------------------------------------------------------

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function fmt(n) { return Math.round(n).toLocaleString(); }

  function formatBytes(n) {
    if (n > 1e9) return (n / 1e9).toFixed(1) + " GB";
    if (n > 1e6) return (n / 1e6).toFixed(0) + " MB";
    return Math.max(1, Math.round(n / 1e3)) + " KB";
  }

  function showResult(png, preview, meta) {
    if (state.resultUrl) URL.revokeObjectURL(state.resultUrl);
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.resultUrl = URL.createObjectURL(new Blob([png], { type: "image/png" }));
    state.previewUrl = URL.createObjectURL(new Blob([preview], { type: "image/png" }));
    els.download.href = state.resultUrl;
    els.result.hidden = false;
    els.viewer.style.setProperty("--ar", meta.width + " / " + meta.height);
    viewer.load(state.previewUrl, state.resultUrl, meta.width, meta.height, meta.preview_width);

    var s = meta.stats || {};
    var first = s.first_activity ? s.first_activity.slice(0, 4) : null;
    var last = s.last_activity ? s.last_activity.slice(0, 4) : null;
    var secs = (performance.now() - state.startedAt) / 1000;
    var cards = [
      [fmt(s.activities_drawn || 0), "activities on the map"],
      [fmt(s.distance_km || 0) + " km", "covered by those activities"],
      [fmt(s.max_overlap || 0) + "×", "most-travelled stretch"],
      [first ? (first === last ? first : first + "–" + last) : "–", "recorded"],
      [s.area_km ? s.area_km[0].toFixed(0) + " × " + s.area_km[1].toFixed(0) + " km" : "–", "map area"],
      [secs < 60 ? secs.toFixed(0) + " s" : (secs / 60).toFixed(1) + " min", "to make"],
    ];
    var statsBox = $("stats");
    statsBox.textContent = "";
    cards.forEach(function (c) {
      var card = el("div", "stat");
      card.appendChild(el("span", "value", c[0]));
      card.appendChild(el("span", "label", c[1]));
      statsBox.appendChild(card);
    });

    var types = s.types || {};
    var names = Object.keys(types);
    var max = Math.max.apply(null, names.map(function (k) { return types[k]; }).concat([1]));
    var typesBox = $("types");
    typesBox.textContent = "";
    if (!names.length) typesBox.appendChild(el("p", "muted", "No type information."));
    names.slice(0, 10).forEach(function (name) {
      var row = el("div", "type-row");
      row.appendChild(el("span", null, name));
      var bar = el("div", "type-bar");
      var fill = el("span");
      fill.style.width = (100 * types[name] / max).toFixed(1) + "%";
      bar.appendChild(fill);
      row.appendChild(bar);
      row.appendChild(el("span", "type-count", fmt(types[name])));
      typesBox.appendChild(row);
    });

    var skippedBox = $("skipped");
    skippedBox.textContent = "";
    var skipped = meta.skipped || {};
    var reasons = Object.keys(skipped);
    if (!reasons.length) skippedBox.appendChild(el("li", null, "Nothing. Every activity made it in."));
    reasons.forEach(function (r) { skippedBox.appendChild(el("li", null, skipped[r] + " × " + r)); });
    if (s.areas_found > 1) {
      skippedBox.appendChild(el("li", null, "Activities in " + (s.areas_found - 1) +
        " other area" + (s.areas_found > 2 ? "s" : "") + " (pick them under Area)"));
    }

    var warn = $("warnings");
    var list = $("warnings-list");
    list.textContent = "";
    warn.hidden = !meta.warning_count;
    $("warnings-summary").textContent = meta.warning_count + " file warning" + (meta.warning_count === 1 ? "" : "s");
    (meta.warnings || []).forEach(function (w) { list.appendChild(el("li", null, w)); });
    if (meta.warning_count > (meta.warnings || []).length) {
      list.appendChild(el("li", null, "…and " + (meta.warning_count - meta.warnings.length) + " more"));
    }

    document.body.dataset.state = "done";
    els.result.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function showError(text) {
    els.error.textContent = text;
    els.error.hidden = false;
    document.body.dataset.state = "error";
  }

  function hideError() { els.error.hidden = true; }

  // ---- zoomable viewer -------------------------------------------------------

  var viewer = (function () {
    var box = els.viewer, img = els.img;
    var natW = 1, natH = 1, scale = 1, minScale = 1, x = 0, y = 0;
    var previewSrc = "", fullSrc = "", previewW = 1, showingFull = false;
    var pointers = new Map();
    var pinch = 0;

    function apply() {
      // The preview is sharp until the image is drawn bigger than it is;
      // past that, swap in the full-resolution version.
      var wantFull = natW * scale > previewW * 1.05;
      if (wantFull !== showingFull && fullSrc) {
        showingFull = wantFull;
        img.src = wantFull ? fullSrc : previewSrc;
      }
      img.style.width = natW + "px";
      img.style.height = natH + "px";
      img.style.transform = "translate(" + x + "px," + y + "px) scale(" + scale + ")";
    }

    function clamp() {
      var w = natW * scale, h = natH * scale, bw = box.clientWidth, bh = box.clientHeight;
      x = w <= bw ? (bw - w) / 2 : Math.min(0, Math.max(bw - w, x));
      y = h <= bh ? (bh - h) / 2 : Math.min(0, Math.max(bh - h, y));
    }

    function fit() {
      if (!box.clientWidth) return;
      minScale = Math.min(box.clientWidth / natW, box.clientHeight / natH);
      scale = minScale;
      clamp();
      apply();
    }

    function zoomAt(f, cx, cy) {
      var next = Math.max(minScale, Math.min(4, scale * f));
      x = cx - (cx - x) * (next / scale);
      y = cy - (cy - y) * (next / scale);
      scale = next;
      clamp();
      apply();
    }

    function local(e) {
      var r = box.getBoundingClientRect();
      return [e.clientX - r.left, e.clientY - r.top];
    }

    box.addEventListener("wheel", function (e) {
      e.preventDefault();
      var p = local(e);
      zoomAt(Math.exp(-e.deltaY * 0.0015), p[0], p[1]);
    }, { passive: false });

    box.addEventListener("pointerdown", function (e) {
      box.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, local(e));
      box.classList.add("dragging");
    });

    box.addEventListener("pointermove", function (e) {
      if (!pointers.has(e.pointerId)) return;
      var prev = pointers.get(e.pointerId), cur = local(e);
      pointers.set(e.pointerId, cur);
      if (pointers.size === 1) {
        x += cur[0] - prev[0];
        y += cur[1] - prev[1];
        clamp();
        apply();
      } else if (pointers.size === 2) {
        var p = Array.from(pointers.values());
        var d = Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]);
        if (pinch) zoomAt(d / pinch, (p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2);
        pinch = d;
      }
    });

    function end(e) {
      pointers.delete(e.pointerId);
      if (pointers.size < 2) pinch = 0;
      if (!pointers.size) box.classList.remove("dragging");
    }
    box.addEventListener("pointerup", end);
    box.addEventListener("pointercancel", end);
    box.addEventListener("dblclick", fit);
    box.addEventListener("keydown", function (e) {
      var cx = box.clientWidth / 2, cy = box.clientHeight / 2;
      if (e.key === "+" || e.key === "=") zoomAt(1.25, cx, cy);
      else if (e.key === "-") zoomAt(0.8, cx, cy);
      else if (e.key === "0") fit();
    });
    window.addEventListener("resize", function () { if (!els.result.hidden) fit(); });

    return {
      load: function (preview, full, w, h, pw) {
        natW = w;
        natH = h;
        previewSrc = preview;
        fullSrc = full;
        previewW = pw || w;
        showingFull = false;
        img.src = preview;
        fit();
      },
    };
  })();

  // ---- start -------------------------------------------------------------------

  if (!window.Worker || !window.WebAssembly) {
    els.engine.textContent = "Sorry, this browser can't run MappaTrakka (it needs Web Workers and WebAssembly).";
    els.demo.disabled = true;
    return;
  }
  startWorker();
  updateButtons();
  document.body.dataset.state = "idle";
  // ?demo in the URL runs the demo straight away (handy for sharing a link).
  if (/[?&]demo\b/.test(location.search)) els.demo.click();
})();
