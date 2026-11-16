// Runs the mappatrakka Python package under Pyodide, off the main thread.
//
// Messages in:  {type: "run", files: [{name, file}], options}
//               {type: "demo", options}
// Messages out: {type: "status", text}
//               {type: "ready"}
//               {type: "progress", stage, done, total}
//               {type: "result", png, preview (ArrayBuffers), meta}
//               {type: "error", message}

/* global importScripts, loadPyodide */
"use strict";

var PYODIDE_VERSION = "0.28.3";
var PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v" + PYODIDE_VERSION + "/full/";
var BUILD = "__BUILD__"; // replaced by scripts/build_web.py
var UPLOADS = "/uploads";

importScripts(PYODIDE_URL + "pyodide.js");

function status(text) {
  self.postMessage({ type: "status", text: text });
}

var ready = (async function () {
  status("Loading Python…");
  var pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

  status("Loading numpy and Pillow…");
  await pyodide.loadPackage(["numpy", "pillow", "micropip"]);

  status("Loading FIT decoder…");
  var micropip = pyodide.pyimport("micropip");
  await micropip.install("fitdecode==0.11.0");

  status("Loading MappaTrakka…");
  var resp = await fetch("py/mappatrakka.zip?v=" + BUILD);
  if (!resp.ok) throw new Error("couldn't download the Python package (" + resp.status + ")");
  pyodide.unpackArchive(await resp.arrayBuffer(), "zip", { extractDir: "/home/pyodide" });
  pyodide.runPython("import mappatrakka.web");

  self.postMessage({ type: "ready" });
  return pyodide;
})();

ready.catch(function (err) {
  self.postMessage({ type: "error", message: "Couldn't start: " + (err && err.message ? err.message : err) });
});

function mountUploads(pyodide, files) {
  var FS = pyodide.FS;
  try {
    FS.unmount(UPLOADS);
  } catch (e) { /* not mounted yet */ }
  try {
    FS.mkdir(UPLOADS);
  } catch (e) { /* already exists */ }
  // WORKERFS reads straight from the File objects when Python asks for
  // bytes, so a multi-gigabyte export zip is never copied into memory.
  FS.mount(FS.filesystems.WORKERFS, {
    blobs: files.map(function (f) { return { name: f.name, data: f.file }; }),
  }, UPLOADS);
}

function progress(stage, done, total) {
  self.postMessage({ type: "progress", stage: stage, done: done, total: total });
}

// Pull a bytes value out of the result dict as an ArrayBuffer we can transfer.
function takeBytes(result, key) {
  var proxy = result.get(key);
  var buf = proxy.getBuffer("u8");
  var copy = buf.data.slice().buffer;
  buf.release();
  proxy.destroy();
  result.set(key, null);
  return copy;
}

async function generate(msg) {
  var pyodide = await ready;
  var web = pyodide.pyimport("mappatrakka.web");
  var options = pyodide.toPy(msg.options || {});
  var result;
  try {
    if (msg.type === "demo") {
      status("Building a demo export…");
      result = web.generate_demo(options, progress);
    } else {
      mountUploads(pyodide, msg.files);
      result = web.generate(UPLOADS, options, progress);
    }
    var png = takeBytes(result, "png");
    var preview = takeBytes(result, "preview");
    var meta = JSON.parse(pyodide.pyimport("json").dumps(result));
    self.postMessage({ type: "result", png: png, preview: preview, meta: meta }, [png, preview]);
  } finally {
    options.destroy();
    if (result) result.destroy();
    web.destroy();
  }
}

self.onmessage = function (e) {
  generate(e.data).catch(function (err) {
    var text = String(err && err.message ? err.message : err);
    // Python tracebacks are long; the last line is the useful bit.
    var lines = text.trim().split("\n");
    self.postMessage({ type: "error", message: lines[lines.length - 1], detail: text });
  });
};
