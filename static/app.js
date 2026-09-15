/* selflabel web UI -- vanilla JS, no framework, no build step. */
"use strict";

const $ = (s, el) => (el || document).querySelector(s);
const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));

const S = {
  // annotate tab
  images: [],            // [{filename, items:[ann], mode}]
  cur: null,             // {filename, img, w, h, items, mode}
  mode: "detect",
  drawMode: false,
  sel: -1,               // selected item index in cur.items
  selKpt: -1,            // selected keypoint index of the selected item
  drag: null,
  // review tab
  ds: null, classes: [], cls: null,
  files: [], loaded: 0, total: 0,
  selected: new Set(), lastIdx: -1,
  suggestions: {}, jobTimer: null, dsGen: 0, clsGen: 0,
};

/* ---------------------------------------------------------------- utils */

function toast(msg, ms = 2600) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.add("hidden"), ms);
}

async function api(url, opts) {
  let r;
  try {
    r = await fetch(url, opts);
  } catch (e) {
    throw new Error("网络错误: " + e.message);
  }
  const ct = r.headers.get("content-type") || "";
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    if (ct.includes("json")) { try { msg = (await r.json()).error || msg; } catch (_) {} }
    throw new Error(msg);
  }
  if (ct.includes("json")) return r.json();
  return r; // caller reads blob/text
}

function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

function parseLines(text) {
  // "name=description" per line -> [{name, description}]
  return (text || "").split("\n").map(l => l.trim()).filter(Boolean).map(l => {
    const i = l.indexOf("=");
    return i < 0 ? { name: l, description: "" }
                 : { name: l.slice(0, i).trim(), description: l.slice(i + 1).trim() };
  }).filter(x => x.name);
}

function clsColor(i) {
  return `hsl(${(i * 67 + 8) % 360} 85% 60%)`;
}

/* ----------------------------------------------------------- tabs/health */

function switchTab(name) {
  $("#page-annotate").classList.toggle("hidden", name !== "annotate");
  $("#page-review").classList.toggle("hidden", name !== "review");
  $("#tab-annotate").classList.toggle("active", name === "annotate");
  $("#tab-review").classList.toggle("active", name === "review");
  if (name === "review") initReview();
}

async function pingHealth() {
  const el = $("#health");
  try {
    const h = await api("/health");
    el.className = "health " + (h.ok ? "ok" : "bad");
    el.textContent = h.ok ? `模型在线 ${h.server}` : "模型未响应";
  } catch (e) {
    el.className = "health bad";
    el.textContent = "服务未启动";
  }
}

/* ================================================================ 标注页 */

function annState(filename) {
  let e = S.images.find(x => x.filename === filename);
  if (!e) { e = { filename, items: [], mode: S.mode }; S.images.push(e); }
  return e;
}

async function uploadFiles(files) {
  for (const f of files) {
    if (!f.type.startsWith("image/")) continue;
    const fd = new FormData();
    fd.append("file", f, f.name);
    try {
      const r = await api("/upload", { method: "POST", body: fd });
      annState(r.filename);
      addListItem(r.filename);
    } catch (e) { toast(`上传失败 ${f.name}: ${e.message}`); }
  }
  if (!S.cur && S.images.length) openImage(S.images[0].filename);
}

function addListItem(filename) {
  const div = document.createElement("div");
  div.className = "ann-item";
  div.dataset.fn = filename;
  const img = document.createElement("img");
  img.src = `/static/${encodeURIComponent(filename)}`;
  const span = document.createElement("span");
  span.textContent = filename;
  const st = document.createElement("span");
  st.className = "status";
  div.append(img, span, st);
  div.onclick = () => openImage(filename);
  $("#ann-list").appendChild(div);
  refreshListItem(filename);
}

function refreshListItem(filename) {
  const e = annState(filename);
  const div = $(`.ann-item[data-fn="${CSS.escape(filename)}"]`);
  if (!div) return;
  div.querySelector(".status").textContent = e.items.length ? `✓${e.items.length}` : "";
  div.classList.toggle("active", S.cur && S.cur.filename === filename);
}

async function openImage(filename) {
  const img = new Image();
  img.src = `/static/${encodeURIComponent(filename)}`;
  await img.decode();
  const e = annState(filename);
  e.mode = S.mode;
  S.cur = { filename, img, w: img.naturalWidth, h: img.naturalHeight, items: e.items, mode: e.mode };
  S.sel = -1; S.selKpt = -1;
  $("#cv-empty").classList.add("hidden");
  $("#mode").value = S.mode;
  fitCanvas();
  draw();
  refreshAllListItems();
}

function refreshAllListItems() { S.images.forEach(x => refreshListItem(x.filename)); }

function fitCanvas() {
  if (!S.cur) return;
  const wrap = $("#canvas-wrap");
  // downscale big frames as before, but allow small slices to be magnified (max 6x)
  const scale = Math.min(6, (wrap.clientWidth - 16) / S.cur.w, (wrap.clientHeight - 16) / S.cur.h);
  const cv = $("#cv");
  cv.width = Math.max(1, Math.round(S.cur.w * scale));
  cv.height = Math.max(1, Math.round(S.cur.h * scale));
  cv.style.width = cv.width + "px";
  cv.style.height = cv.height + "px";
}

function draw() {
  const cv = $("#cv"), ctx = cv.getContext("2d");
  if (!S.cur) return;
  ctx.imageSmoothingEnabled = S.cur.w * 4 < cv.width || S.cur.h * 4 < cv.height ? false : true;
  ctx.drawImage(S.cur.img, 0, 0, cv.width, cv.height);
  const k = (n) => n * cv.width, kY = (n) => n * cv.height;
  const skeleton = parseSkeleton();

  S.cur.items.forEach((a, i) => {
    const col = clsColor(a.class_id);
    const x1 = k(a.x1), y1 = kY(a.y1), x2 = k(a.x2), y2 = kY(a.y2);
    ctx.lineWidth = i === S.sel ? 3 : 2;
    ctx.strokeStyle = col;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const lbl = `${a.class_id}:${a.label}${a.score ? " " + a.score.toFixed(2) : ""}`;
    ctx.font = "12px sans-serif";
    const tw = ctx.measureText(lbl).width;
    ctx.fillStyle = col;
    ctx.fillRect(x1, Math.max(0, y1 - 15), tw + 6, 14);
    ctx.fillStyle = "#111";
    ctx.fillText(lbl, x1 + 3, Math.max(11, y1 - 4));

    if (S.mode === "pose" && a.keypoints) {
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "rgba(255,255,255,.65)";
      skeleton.forEach(([m, n]) => {
        const p1 = a.keypoints[m], p2 = a.keypoints[n];
        if (p1 && p2 && p1[2] > 0 && p2[2] > 0) {
          ctx.beginPath(); ctx.moveTo(k(p1[0]), kY(p1[1])); ctx.lineTo(k(p2[0]), kY(p2[1])); ctx.stroke();
        }
      });
      a.keypoints.forEach((p, j) => {
        const sel = i === S.sel && j === S.selKpt;
        ctx.beginPath();
        ctx.arc(k(p[0]), kY(p[1]), sel ? 7 : 5, 0, Math.PI * 2);
        ctx.fillStyle = p[2] > 0 ? "#ffd166" : "rgba(229,72,77,.9)";
        ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = sel ? "#fff" : "#222"; ctx.stroke();
      });
    }
    if (i === S.sel) {  // corner handles
      ctx.fillStyle = "#fff";
      [[x1, y1], [x2, y1], [x2, y2], [x1, y2]].forEach(([hx, hy]) =>
        ctx.fillRect(hx - 4, hy - 4, 8, 8));
    }
  });
  $("#cv-info").textContent = S.cur ? `${S.cur.filename} · ${S.cur.w}×${S.cur.h} · ${S.cur.items.length} 个实例` : "";
}

function parseSkeleton() {
  try { return JSON.parse($("#skeleton").value || "[]"); } catch (_) { return []; }
}

/* --- canvas mouse --- */

function evPos(ev) {
  const r = $("#cv").getBoundingClientRect();
  return { x: (ev.clientX - r.left) / r.width, y: (ev.clientY - r.top) / r.height };
}

function hitTest(p) {
  const items = S.cur ? S.cur.items : [];
  if (S.mode === "pose") {  // keypoints first
    for (let i = items.length - 1; i >= 0; i--) {
      const a = items[i];
      for (let j = 0; j < (a.keypoints || []).length; j++) {
        const kp = a.keypoints[j];
        if (Math.hypot((kp[0] - p.x) * $("#cv").width, (kp[1] - p.y) * $("#cv").height) < 9)
          return { type: "kpt", item: i, kpt: j };
      }
    }
  }
  for (let i = items.length - 1; i >= 0; i--) {
    const a = items[i];
    const cs = [[a.x1, a.y1, "x1", "y1"], [a.x2, a.y1, "x2", "y1"],
                [a.x2, a.y2, "x2", "y2"], [a.x1, a.y2, "x1", "y2"]];
    for (const [cx, cy, kx, ky] of cs) {
      if (Math.hypot((cx - p.x) * $("#cv").width, (cy - p.y) * $("#cv").height) < 9)
        return { type: "resize", item: i, kx, ky };
    }
  }
  for (let i = items.length - 1; i >= 0; i--) {
    const a = items[i];
    if (p.x >= a.x1 && p.x <= a.x2 && p.y >= a.y1 && p.y <= a.y2)
      return { type: "move", item: i };
  }
  return null;
}

function onMouseDown(ev) {
  if (!S.cur) return;
  const p = evPos(ev);
  if (S.drawMode) {
    S.drag = { type: "new", x0: p.x, y0: p.y, x1: p.x, y1: p.y };
  } else {
    const hit = hitTest(p);
    if (hit) {
      S.sel = hit.item;
      S.selKpt = hit.kpt !== undefined ? hit.kpt : -1;
      if (hit.type === "kpt") S.drag = { type: "kpt", item: hit.item, kpt: hit.kpt };
      else if (hit.type === "resize") S.drag = { type: "resize", item: hit.item, kx: hit.kx, ky: hit.ky };
      else S.drag = { type: "move", item: hit.item, px: p.x, py: p.y, orig: { ...hitCoords(hit.item) } };
    } else { S.sel = -1; S.selKpt = -1; }
    draw();
    syncSelClass();
  }
}

function hitCoords(i) {
  const a = S.cur.items[i];
  return { x1: a.x1, y1: a.y1, x2: a.x2, y2: a.y2 };
}

function onMouseMove(ev) {
  if (!S.cur || !S.drag) return;
  const p = evPos(ev);
  const d = S.drag, a = S.cur.items[d.item];
  const clamp = (v) => Math.max(0, Math.min(1, v));
  if (d.type === "new") {
    d.x1 = p.x; d.y1 = p.y;
    draw();
    const ctx = $("#cv").getContext("2d");
    ctx.strokeStyle = "#4c7ef3"; ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(Math.min(d.x0, d.x1) * $("#cv").width, Math.min(d.y0, d.y1) * $("#cv").height,
      Math.abs(d.x1 - d.x0) * $("#cv").width, Math.abs(d.y1 - d.y0) * $("#cv").height);
    ctx.setLineDash([]);
    return;
  }
  if (d.type === "move") {
    const dx = p.x - d.px, dy = p.y - d.py;
    a.x1 = clamp(d.orig.x1 + dx); a.y1 = clamp(d.orig.y1 + dy);
    a.x2 = clamp(d.orig.x2 + dx); a.y2 = clamp(d.orig.y2 + dy);
  } else if (d.type === "resize") {
    a[d.kx] = clamp(p.x); a[d.ky] = clamp(p.y);
    if (a.x2 < a.x1) [a.x1, a.x2] = [a.x2, a.x1];
    if (a.y2 < a.y1) [a.y1, a.y2] = [a.y2, a.y1];
  } else if (d.type === "kpt") {
    const kp = a.keypoints[d.kpt];
    kp[0] = clamp(p.x); kp[1] = clamp(p.y);
    if (kp[2] === 0) kp[2] = 2;
  }
  draw();
}

function onMouseUp() {
  const d = S.drag;
  S.drag = null;
  if (!S.cur || !d) return;
  if (d.type === "new") {
    const x1 = Math.min(d.x0, d.x1), x2 = Math.max(d.x0, d.x1);
    const y1 = Math.min(d.y0, d.y1), y2 = Math.max(d.y0, d.y1);
    if ((x2 - x1) * $("#cv").width > 8 && (y2 - y1) * $("#cv").height > 8) {
      const ci = parseInt($("#sel-class").value || "0", 10);
      const item = {
        class_id: ci, label: classDefs()[ci] ? classDefs()[ci].name : "object",
        x1, y1, x2, y2, keypoints: null, score: 1,
      };
      if (S.mode === "pose") {
        const n = keypointDefs().length || 4;
        item.keypoints = [[x1, y1, 2], [x2, y1, 2], [x2, y2, 2], [x1, y2, 2],
          ...Array(Math.max(0, n - 4)).fill([x1, y1, 0])].slice(0, n).map(k => [...k]);
      }
      S.cur.items.push(item);
      S.sel = S.cur.items.length - 1;
      S.cur.mode = S.mode;
      annState(S.cur.filename).items = S.cur.items;
      refreshListItem(S.cur.filename);
    }
  } else {
    annState(S.cur.filename).items = S.cur.items;
    refreshListItem(S.cur.filename);
  }
  draw();
}

/* --- annotate action --- */

function classDefs() { return parseLines($("#classes").value); }
function keypointDefs() { return parseLines($("#keypoints").value); }

async function annotateCurrent() {
  if (!S.cur) { toast("先上传/打开一张图片"); return; }
  const classes = classDefs();
  if (!classes.length) { toast("请先填写至少一个类别"); return; }
  const mode = $("#mode").value;
  const btn = $("#btn-annotate");
  btn.disabled = true; btn.textContent = "⏳ 模型推理中…";
  try {
    const r = await api("/annotate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: S.cur.filename, mode, classes,
        keypoints: keypointDefs(),
        kpt: parseInt($("#kpt-n")?.value || "4", 10),
        prompt: $("#prompt").value,
        conf: parseFloat($("#conf").value) || 0,
        iou: parseFloat($("#iou").value) || 0.5,
      }),
    });
    S.mode = mode; S.cur.mode = mode;
    S.cur.items = r.items;
    S.sel = -1; S.selKpt = -1;
    annState(S.cur.filename).items = r.items;
    $("#kpt-block").classList.toggle("hidden", mode !== "pose");
    draw(); refreshListItem(S.cur.filename);
    toast(`完成：${r.n} 个实例（${(r.t || 0) || ""}）`);
  } catch (e) { toast("标注失败: " + e.message, 4000); }
  finally { btn.disabled = false; btn.textContent = "▶ 自动标注本图"; }
}

function syncSelClass() {
  const sel = $("#sel-class");
  if (!S.cur || S.sel < 0 || !S.cur.items[S.sel]) return;
  sel.value = String(S.cur.items[S.sel].class_id);
}

/* --- exports --- */

async function exportCurrentTxt() {
  if (!S.cur || !S.cur.items.length) { toast("当前图没有标注"); return; }
  const r = await api("/export", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ format: "labels", mode: S.mode, items: S.cur.items,
                           filename: S.cur.filename }),
  });
  download(await r.blob(), `${S.cur.filename.replace(/\.[^.]+$/, "")}_${S.mode}.txt`);
}

async function exportClassesTxt() {
  const r = await api("/export", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ format: "classes", classes: classDefs() }),
  });
  download(await r.blob(), "classes.txt");
}

async function exportZip() {
  const images = S.images.filter(x => x.items.length)
    .map(x => ({ filename: x.filename, items: x.items }));
  if (!images.length) { toast("还没有任何已标注的图片"); return; }
  const btn = $("#btn-export-zip");
  btn.disabled = true;
  try {
    const r = await api("/export", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ format: "zip", fmt: $("#zip-fmt").value, images,
                             classes: classDefs(), keypoints: keypointDefs(),
                             skeleton: parseSkeleton() }),
    });
    download(await r.blob(), `selflabel_${$("#zip-fmt").value}.zip`);
    toast("导出完成");
  } catch (e) { toast("导出失败: " + e.message, 4000); }
  finally { btn.disabled = false; }
}

/* ================================================================ 审查页 */

let reviewInited = false;
async function initReview() {
  if (reviewInited) return;
  reviewInited = true;
  await refreshDatasets();
  const obs = new IntersectionObserver((ents) => {
    ents.forEach(e => {
      if (e.isIntersecting) { e.target.src = e.target.dataset.src; obs.unobserve(e.target); }
    });
  }, { root: $("#grid"), rootMargin: "300px" });
  S.gridObs = obs;
}

async function refreshDatasets() {
  try {
    const r = await api("/api/datasets");
    const sel = $("#ds");
    sel.innerHTML = "";
    (r.datasets || []).forEach(d => {
      const o = document.createElement("option");
      o.value = d.name; o.textContent = `${d.name} (${(d.classes || []).length} 类)`;
      sel.appendChild(o);
    });
    if (r.default_extra) $("#job-extra").value = r.default_extra;
    if (!r.datasets.length) { toast("review root 下没有发现「文件夹=类别」数据集"); return; }
    S.ds = sel.value;
    await onDatasetChange();
  } catch (e) { toast("加载数据集失败: " + e.message); }
}

async function onDatasetChange() {
  const dsgen = ++S.dsGen;
  S.ds = $("#ds").value;
  S.selected.clear(); S.lastIdx = -1;
  $("#grid").innerHTML = ""; $("#grid-more").classList.add("hidden");
  const r = await api("/api/datasets");
  if (dsgen !== S.dsGen) return;                 // a newer selection superseded us
  const d = (r.datasets || []).find(x => x.name === S.ds);
  S.classes = d ? d.classes : [];
  const clsList = $("#rev-classes");
  clsList.innerHTML = "";
  S.classes.forEach((c, i) => {
    const div = document.createElement("div");
    div.className = "rev-cls";
    div.innerHTML = `<span>${c.name}</span><span class="n">${c.count}</span>`;
    div.onclick = () => loadClass(c.name);
    clsList.appendChild(div);
  });
  $("#move-to").innerHTML = S.classes.map(c => `<option>${c.name}</option>`).join("");
  $("#job-cls").innerHTML = '<option value="">全部</option>' +
    S.classes.map(c => `<option>${c.name}</option>`).join("");
  if (S.classes.length) loadClass(S.classes[0].name);   // no await: grid loads in parallel
  const rc = await api(`/api/recheck?ds=${encodeURIComponent(S.ds)}`);
  if (dsgen !== S.dsGen) return;
  S.suggestions = rc.found ? rc.suggestions : {};
  if (S.files.length) renderGrid(true);          // repaint cells so badges show up
  updateCount();
  const n = Object.keys(S.suggestions).length;
  toast(rc.found ? `已加载 VLM 复核建议 ${n} 条` : "该数据集暂无 VLM 复核结果", 1800);
}

async function loadClass(cls) {
  const cgen = ++S.clsGen;
  const dsgen = S.dsGen;
  S.cls = cls;
  S.selected.clear(); S.lastIdx = -1;
  $$("#rev-classes .rev-cls").forEach(el =>
    el.classList.toggle("active", el.firstChild.textContent === cls));
  const r = await api(`/api/images?ds=${encodeURIComponent(S.ds)}&cls=${encodeURIComponent(cls)}&start=0&count=500`);
  if (cgen !== S.clsGen || dsgen !== S.dsGen) return;   // superseded
  S.files = r.files; S.total = r.total; S.loaded = r.files.length;
  renderGrid(true);
  updateCount();
}

function updateCount() {
  $("#rev-count").textContent =
    `${S.ds} / ${S.cls}: 已加载 ${S.loaded}/${S.total} · 选中 ${S.selected.size}`;
}

function cellEl(name, idx) {
  const div = document.createElement("div");
  div.className = "cell";
  div.dataset.name = name;
  div.dataset.idx = idx;
  const img = document.createElement("img");
  img.loading = "lazy";
  img.dataset.src = `/img?ds=${encodeURIComponent(S.ds)}&cls=${encodeURIComponent(S.cls)}&name=${encodeURIComponent(name)}`;
  img.style.filter = `brightness(${$("#brightness").value})`;
  div.appendChild(img);
  const s = S.suggestions[S.cls + "/" + name];
  if (s && s.label) {
    const b = document.createElement("div");
    b.className = "badge" + (s.agree ? " ok-badge" : "");
    div.classList.toggle("ok-badge", !!s.agree);
    b.textContent = `VLM:${s.label} ${s.conf.toFixed(2)}`;
    div.appendChild(b);
    if (!s.agree) div.classList.add("diff");
  }
  div.onmousedown = (ev) => {
    if (ev.shiftKey && S.lastIdx >= 0) {
      const [a, b2] = [Math.min(S.lastIdx, idx), Math.max(S.lastIdx, idx)];
      for (let i = a; i <= b2; i++) { S.selected.add(i); }
    } else {
      S.brushAdd = !S.selected.has(idx);
      toggleSel(idx);
      S.brushing = true;
    }
    S.lastIdx = idx;
    paintSelection();
  };
  div.onmouseenter = () => {
    if (S.brushing) { toggleSel(idx, S.brushAdd); paintSelection(); }
  };
  return div;
}

function toggleSel(idx, force) {
  const want = force !== undefined ? force : !S.selected.has(idx);
  if (want) S.selected.add(idx); else S.selected.delete(idx);
}

function paintSelection() {
  $$("#grid .cell").forEach(c => {
    const i = parseInt(c.dataset.idx, 10);
    c.classList.toggle("sel", S.selected.has(i));
    c.style.display = ($("#only-diff").checked && !c.classList.contains("diff")) ? "none" : "";
  });
  updateCount();
}

function renderGrid(reset) {
  const g = $("#grid");
  if (reset) { g.innerHTML = ""; S.loaded = 0; }
  const start = S.loaded;
  const slice = S.files.slice(start);
  slice.forEach((name, k) => {
    const c = cellEl(name, start + k);
    g.appendChild(c);
    S.gridObs.observe(c.querySelector("img"));
  });
  S.loaded = S.files.length;
  $("#grid-more").classList.toggle("hidden", S.loaded >= S.total);
  paintSelection();
}

async function loadMore() {
  const r = await api(`/api/images?ds=${encodeURIComponent(S.ds)}&cls=${encodeURIComponent(S.cls)}&start=${S.loaded}&count=500`);
  S.files = S.files.concat(r.files);
  S.total = r.total;
  renderGrid(false);
}

async function selectedByClass() {
  const byCls = {};
  $$("#grid .cell.sel").forEach(c => {
    const cls = S.cls;
    (byCls[cls] = byCls[cls] || []).push(c.dataset.name);
  });
  return byCls;
}

async function moveSelected(to) {
  const names = $$("#grid .cell.sel").map(c => c.dataset.name);
  if (!names.length) { toast("先选中样本（单击/Shift 连选/拖动刷选）"); return; }
  if (!to) { toast("选择目标类别"); return; }
  try {
    const r = await api("/api/move", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ds: S.ds, cls: S.cls, names, to }),
    });
    toast(`已移动 ${r.moved} 张 → ${to}`);
    await afterMutation();
  } catch (e) { toast("移动失败: " + e.message, 3500); }
}

async function adoptSuggestions() {
  const cells = $$("#grid .cell.sel");
  if (!cells.length) { toast("先选中样本"); return; }
  const byTarget = {};
  let skipped = 0;
  cells.forEach(c => {
    const s = S.suggestions[S.cls + "/" + c.dataset.name];
    if (s && s.label && !s.agree && s.label !== S.cls) {
      (byTarget[s.label] = byTarget[s.label] || []).push(c.dataset.name);
    } else skipped++;
  });
  if (!Object.keys(byTarget).length) { toast("选中的样本没有可采纳的分歧建议"); return; }
  try {
    let n = 0;
    for (const [to, names] of Object.entries(byTarget)) {
      const r = await api("/api/move", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ds: S.ds, cls: S.cls, names, to }),
      });
      n += r.moved;
    }
    toast(`已按 VLM 建议移动 ${n} 张${skipped ? `（跳过 ${skipped}）` : ""}`);
    await afterMutation();
  } catch (e) { toast("采纳失败: " + e.message, 3500); }
}

let deleteArmed = false, deleteTimer = null;
async function deleteSelected() {
  const names = $$("#grid .cell.sel").map(c => c.dataset.name);
  if (!names.length) { toast("先选中样本"); return; }
  if (!deleteArmed) {
    deleteArmed = true;
    $("#btn-del-rev").textContent = `⚠ 再点一次确认删除 ${names.length} 张`;
    $("#btn-del-rev").classList.add("danger");
    clearTimeout(deleteTimer);
    deleteTimer = setTimeout(disarmDelete, 4000);
    return;
  }
  disarmDelete();
  try {
    const r = await api("/api/delete", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ds: S.ds, cls: S.cls, names, confirm: true }),
    });
    toast(`已删除 ${r.deleted} 张`);
    await afterMutation();
  } catch (e) { toast("删除失败: " + e.message, 3500); }
}

function disarmDelete() {
  deleteArmed = false;
  $("#btn-del-rev").textContent = "🗑 删除选中";
  $("#btn-del-rev").classList.remove("danger");
}

async function newClass() {
  const name = prompt("新类别名：");
  if (!name) return;
  try {
    await api("/api/newclass", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ds: S.ds, name }),
    });
    toast(`已创建类别 ${name}`);
    await refreshDatasets();
  } catch (e) { toast("创建失败: " + e.message); }
}

async function afterMutation() {
  // re-list current class (files moved away are gone), refresh counts + suggestions
  const rc = await api(`/api/recheck?ds=${encodeURIComponent(S.ds)}`);
  S.suggestions = rc.found ? rc.suggestions : {};
  await loadClass(S.cls);
}

/* --- background recheck job --- */

async function startJob() {
  const body = {
    ds: S.ds,
    cls: $("#job-cls").value,
    sample: parseInt($("#job-sample").value || "0", 10),
    upscale: parseInt($("#job-upscale").value || "6", 10),
    extra: $("#job-extra").value,
  };
  try {
    const r = await api("/api/jobs", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    toast(`后台复核已启动 #${r.id}`);
    $("#job-progress").classList.remove("hidden");
    pollJob(r.id);
  } catch (e) { toast("启动失败: " + e.message, 3500); }
}

function pollJob(id) {
  clearInterval(S.jobTimer);
  const t0 = Date.now();
  S.jobTimer = setInterval(async () => {
    let j;
    try { j = await api(`/api/jobs/${id}`); } catch (_) { return; }
    const pct = j.total ? Math.round(j.done / j.total * 100) : 0;
    $("#job-bar").style.width = pct + "%";
    const elapsed = (Date.now() - t0) / 1000;
    const rate = j.done > 0 && elapsed > 0 ? j.done / elapsed : 0;
    const eta = rate > 0 && j.total ? Math.round((j.total - j.done) / rate) : 0;
    $("#job-text").textContent =
      `${j.status} ${j.done}/${j.total} (${pct}%) · 分歧 ${j.disagree} · 错误 ${j.errors}` +
      (rate ? ` · ${rate.toFixed(1)} 张/s · 剩余约 ${eta}s` : "");
    if (["done", "cancelled", "error"].includes(j.status)) {
      clearInterval(S.jobTimer);
      $("#btn-job").disabled = false;
      toast(j.status === "done" ? `复核完成：${j.done} 张，分歧 ${j.disagree}`
            : j.status === "cancelled" ? "复核已取消" : `复核出错: ${j.error}`, 4000);
      await afterMutation();
      setTimeout(() => $("#job-progress").classList.add("hidden"), 1500);
    }
  }, 2000);
}

/* ----------------------------------------------------------------- wire */

function initAnnotate() {
  $("#btn-upload").onclick = () => $("#file-input").click();
  $("#file-input").onchange = (e) => { uploadFiles([...e.target.files]); e.target.value = ""; };
  const wrap = $("#canvas-wrap");
  wrap.addEventListener("dragover", e => e.preventDefault());
  wrap.addEventListener("drop", e => {
    e.preventDefault();
    uploadFiles([...e.dataTransfer.files]);
  });
  const cv = $("#cv");
  cv.addEventListener("mousedown", onMouseDown);
  window.addEventListener("mousemove", onMouseMove);
  window.addEventListener("mouseup", onMouseUp);
  window.addEventListener("resize", () => { fitCanvas(); draw(); });
  window.addEventListener("keydown", (ev) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
    if (!S.cur) return;
    if ((ev.key === "Delete" || ev.key === "Backspace") && S.sel >= 0) {
      S.cur.items.splice(S.sel, 1);
      annState(S.cur.filename).items = S.cur.items;
      S.sel = -1; S.selKpt = -1;
      draw(); refreshListItem(S.cur.filename);
    } else if (ev.key === "v" && S.sel >= 0 && S.selKpt >= 0) {
      const kp = S.cur.items[S.sel].keypoints[S.selKpt];
      kp[2] = ((kp[2] || 0) + 1) % 3;
      draw();
    } else if (/^[0-9]$/.test(ev.key) && S.sel >= 0) {
      const ci = parseInt(ev.key, 10);
      if (classDefs()[ci]) {
        S.cur.items[S.sel].class_id = ci;
        S.cur.items[S.sel].label = classDefs()[ci].name;
        draw(); syncSelClass(); refreshListItem(S.cur.filename);
      }
    } else if (ev.key === "Escape") { S.drawMode = false; $("#btn-draw").classList.remove("on"); }
  });
  $("#btn-draw").onclick = () => {
    S.drawMode = !S.drawMode;
    $("#btn-draw").classList.toggle("on", S.drawMode);
  };
  $("#btn-del").onclick = () => {
    if (S.sel >= 0) {
      S.cur.items.splice(S.sel, 1);
      annState(S.cur.filename).items = S.cur.items;
      S.sel = -1; draw(); refreshListItem(S.cur.filename);
    }
  };
  $("#btn-annotate").onclick = annotateCurrent;
  $("#mode").onchange = () => {
    S.mode = $("#mode").value;
    if (S.cur) { S.cur.mode = S.mode; annState(S.cur.filename).mode = S.mode; }
    $("#kpt-block").classList.toggle("hidden", S.mode !== "pose");
    draw();
  };
  $("#sel-class").onchange = () => {
    if (S.sel >= 0 && S.cur.items[S.sel]) {
      const ci = parseInt($("#sel-class").value, 10);
      S.cur.items[S.sel].class_id = ci;
      S.cur.items[S.sel].label = (classDefs()[ci] || {}).name || "object";
      draw(); refreshListItem(S.cur.filename);
    }
  };
  $("#btn-export-txt").onclick = exportCurrentTxt;
  $("#btn-export-classes").onclick = exportClassesTxt;
  $("#btn-export-zip").onclick = exportZip;
  $("#classes").addEventListener("input", () => {
    $("#sel-class").innerHTML = classDefs().map((c, i) => `<option value="${i}">${i}:${c.name}</option>`).join("");
  });
  $("#classes").dispatchEvent(new Event("input"));
  $("#mode").dispatchEvent(new Event("change"));
}

function initReviewWire() {
  $("#tab-annotate").onclick = () => switchTab("annotate");
  $("#tab-review").onclick = () => switchTab("review");
  $("#ds").onchange = onDatasetChange;
  $("#btn-ds-refresh").onclick = refreshDatasets;
  $("#btn-more").onclick = loadMore;
  $("#btn-move").onclick = () => moveSelected($("#move-to").value);
  $("#btn-adopt").onclick = adoptSuggestions;
  $("#btn-del-rev").onclick = deleteSelected;
  $("#btn-newclass").onclick = newClass;
  $("#btn-job").onclick = startJob;
  $("#btn-job-cancel").onclick = async () => {
    const jobs = await api("/api/jobs");
    const running = (jobs.jobs || []).find(j => j.status === "running" || j.status === "pending");
    if (running) await api(`/api/jobs/${running.id}/cancel`, { method: "POST" });
    toast("已请求取消");
  };
  $("#brightness").oninput = () => {
    const v = $("#brightness").value;
    $$("#grid .cell img").forEach(im => im.style.filter = `brightness(${v})`);
  };
  $("#only-diff").onchange = paintSelection;
  window.addEventListener("mouseup", () => { S.brushing = false; });
}

initAnnotate();
initReviewWire();
pingHealth();
setInterval(pingHealth, 30000);
