import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const statusPath = path.join(root, "docs/reports/breast_transfer_research_status.json");
const outputPath = path.join(root, "docs/reports/breast_transfer_research_progress.html");
const s = JSON.parse(await fs.readFile(statusPath, "utf8"));

const esc = (value) => String(value ?? "").replace(/[&<>\"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
}[c]));
const localLink = (rel) => encodeURI(rel.replaceAll("\\", "/"));
const fmt = (n) => new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 5 }).format(n);
const dataCards = s.datasets.map((d) => `
  <article class="dataset">
    <div class="dataset-head"><h3>${esc(d.id)}</h3><span class="pill">${esc(d.status)}</span></div>
    <dl><dt>对象</dt><dd>${esc(d.objects)}</dd><dt>表达数据</dt><dd>${esc(d.matrix)}</dd><dt>处理条件</dt><dd>${esc(d.conditions)}</dd></dl>
    <div class="links">${d.evidence.map((p) => p.startsWith("http") ? `<a href="${esc(p)}">官方 GEO 页面</a>` : `<a href="${localLink(p)}">${esc(p)}</a>`).join("")}</div>
  </article>`).join("");
const workRows = s.work_packages.map((w) => `<tr><th scope="row">${esc(w.id)} ${esc(w.title)}</th><td>${esc(w.status)}</td><td>${w.result == null ? "尚无新结果" : esc(w.result)}</td></tr>`).join("");
const histRows = s.science.reusable_historical_results.map((r) => `<tr><td>${esc(r.contrast)}</td><td>${esc(r.top_pathway)}</td><td>${fmt(r.fdr)}</td></tr>`).join("");
const evidence = s.evidence_links.map((e) => {
  const href = e.path.startsWith("http") ? e.path : localLink(e.path);
  return `<li><a href="${esc(href)}">${esc(e.label)}</a> <code>${esc(e.path)}</code></li>`;
}).join("");
const limits = s.science.limits.map((x) => `<li>${esc(x)}</li>`).join("");
const next = s.next_actions.map((x, i) => `<li><span class="n">${i + 1}</span>${esc(x)}</li>`).join("");
const runtimeCards = `
  <div class="runtime-card"><b>执行环境</b><span>${esc(s.runtime.environment)}</span><small>${esc(s.runtime.environment_check)}</small></div>
  <div class="runtime-card"><b>训练/评估进程</b><span>${esc(s.runtime.active_drugscreenlab_processes)}</span><small>${esc(s.runtime.gpu_status_summary)}</small></div>
  <div class="runtime-card"><b>新增GPU预算</b><span>${fmt(s.experiment.settled_gpu_hours)} / ${fmt(s.experiment.authorized_gpu_hours)} 小时</span><small>单次运行上限 ${fmt(s.experiment.single_run_cap_gpu_hours)} 小时；${s.experiment.training_started ? "本研究已启动GPU训练。" : "本研究尚未启动GPU训练。"}</small></div>
  <div class="runtime-card"><b>存储与设备</b><span>${fmt(s.runtime.filesystem_free_gib)} GiB 可用</span><small>${esc(s.runtime.filesystem_note)} ${esc(s.runtime.unattributed_gpu)}</small></div>`;

const html = `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${esc(s.title)}｜研究进展</title>
<style>
:root{color-scheme:light;--ink:#15263b;--muted:#5b6b7b;--line:#dce4eb;--blue:#155e75;--teal:#0f766e;--amber:#9a5b00;--paper:#f5f8fa;--card:#fff}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 "Microsoft YaHei", "Noto Sans CJK SC", system-ui, sans-serif}main{max-width:1120px;margin:auto;padding:28px 22px 52px}header{background:linear-gradient(125deg,#10354a,#155e75 62%,#2b7780);color:white;border-radius:20px;padding:28px 32px;box-shadow:0 12px 32px #173a4c22}h1{font-size:30px;line-height:1.25;margin:0 0 10px}header p{max-width:850px;color:#e2f0f3;margin:0}.topline{display:flex;gap:10px;align-items:center;justify-content:space-between;margin-bottom:18px;color:#d2e7eb;font-size:13px}.stamp{background:#ffffff1b;border:1px solid #ffffff55;padding:5px 10px;border-radius:999px}.stale{display:none;background:#fff1d6;color:#754700;border:1px solid #e8bb61;border-radius:12px;padding:10px 14px;margin:14px 0}.stale.show{display:block}.facts{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0 24px}.fact{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px}.fact b{display:block;font-size:12px;text-transform:uppercase;color:var(--muted);letter-spacing:.03em}.fact span{display:block;font-size:17px;font-weight:700;margin-top:4px}.fact small{color:var(--muted)}section{margin:26px 0}h2{font-size:21px;margin:0 0 11px}h3{font-size:17px;margin:0}.sub{color:var(--muted);margin:-4px 0 12px}.notice{border-left:4px solid #d39827;background:#fff9e9;padding:13px 16px;border-radius:0 12px 12px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.dataset{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;min-width:0}.dataset-head{display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:space-between}.pill{font-size:12px;border-radius:999px;background:#e7f1f3;color:#124f61;padding:4px 9px}.dataset dl{display:grid;grid-template-columns:70px 1fr;gap:5px 8px;margin:14px 0}.dataset dt{color:var(--muted)}.dataset dd{margin:0}.links{display:flex;flex-direction:column;gap:4px;overflow-wrap:anywhere;font-size:12px}a{color:#075985;text-decoration:none}a:hover{text-decoration:underline}table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}th,td{text-align:left;vertical-align:top;padding:10px 12px;border-bottom:1px solid var(--line)}thead th{background:#eaf0f4;color:#42566a;font-size:13px}tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}tbody th{width:25%}.runtime{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.runtime-card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;display:flex;flex-direction:column;gap:5px}.runtime-card b{font-size:12px;color:var(--muted)}.runtime-card span{font-weight:700}.runtime-card small{color:var(--muted)}ol,ul{padding-left:0;list-style:none;margin:0}.action-list{display:grid;gap:9px}.action-list li{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:11px 14px;display:flex;gap:10px;align-items:flex-start}.n{flex:none;background:#dceef0;color:#155e75;width:25px;height:25px;text-align:center;border-radius:50%;font-weight:700}.limits{display:grid;gap:7px}.limits li{padding-left:18px;position:relative}.limits li:before{content:"•";color:#b7791f;position:absolute;left:2px}code{font-size:12px;color:#435568;overflow-wrap:anywhere}.decision{background:#eaf5f4;border:1px solid #c7e2de;border-radius:14px;padding:16px}.decision b{display:block;margin-bottom:5px}.footer{border-top:1px solid var(--line);margin-top:30px;padding-top:14px;color:var(--muted);font-size:13px}@media(max-width:850px){.facts,.runtime{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}}@media(max-width:520px){main{padding:14px}.facts,.runtime{grid-template-columns:1fr}header{padding:22px}h1{font-size:24px}}
</style></head><body><main>
<header><div class="topline"><span>DrugScreenLab · ${esc(s.experiment.id)} · 分阶段研究进展</span><span class="stamp">更新于 <time>${esc(s.updated_at.replace("T", " ").replace("+08:00", " 北京时间"))}</time></span></div><h1>${esc(s.title)}</h1><p>${esc(s.goal)}</p></header>
<div id="stale" class="stale" role="status">此页状态已超过10分钟未更新。请检查当前接续任务和源状态文件；未刷新前不要把状态文字当作实时进展。</div>
<div class="facts"><div class="fact"><b>当前结论</b><span>${esc(s.science.headline)}</span><small>${esc(s.science.result_scope)}</small></div><div class="fact"><b>运行状态</b><span>${esc(s.experiment.status)}</span><small>${esc(s.runtime.active_drugscreenlab_processes)}</small></div><div class="fact"><b>新增资源账</b><span>${fmt(s.experiment.settled_gpu_hours)} / ${fmt(s.experiment.authorized_gpu_hours)} GPU小时</span><small>与未知历史账分开记录</small></div><div class="fact"><b>研究窗口</b><span>${esc(s.window.started_at.slice(0, 16).replace("T", " "))}</span><small>${s.window.deadline_at ? `截止 ${esc(s.window.deadline_at.slice(0, 16).replace("T", " "))}（北京时间）` : "无固定日历截止；完成研究目标或用户终止时收口"}</small></div></div>
<section><h2>现在在哪里看进度</h2><p class="notice">本页是持续维护的唯一HTML入口：<code>docs/reports/breast_transfer_research_progress.html</code>。恢复任务卡与状态源为 <code>docs/reports/breast_transfer_research_status.json</code>；用 <code>node scripts/reporting/generate_breast_transfer_progress.mjs</code> 重建页面。自动接续状态：${esc(s.window.heartbeat_status)}；关联标识 <code>${esc(s.window.heartbeat_id)}</code>；原计划间隔 ${esc(s.window.heartbeat_interval_minutes)} 分钟，线程 <code>${esc(s.window.heartbeat_target_thread_id)}</code>。</p></section>
<section><h2>公开数据与对象范围</h2><p class="sub">候选系列页面数字只作为接入目标；真正对象数、独立患者数、重复性质以样本键核查为准。</p><div class="grid">${dataCards}</div></section>
<section><h2>研究步骤状态</h2><table><thead><tr><th>步骤</th><th>当前状态</th><th>可报告结果</th></tr></thead><tbody>${workRows}</tbody></table></section>
<section><h2>可复用的历史证据</h2><p class="sub">以下结果来自 EXP-010 四个时间匹配的描述性通路分析，仅用于复用与后续解释；它们不是本线程新结果，也不衡量药效。</p><table><thead><tr><th>比较</th><th>排名最高的通路</th><th>历史 FDR</th></tr></thead><tbody>${histRows}</tbody></table><p class="sub">FDR为历史通路检验的多重校正结果；不表示效应大小。DOX=多柔比星，BTZ=硼替佐米。</p></section>
<section><h2>边界与未知</h2><ul class="limits">${limits}</ul></section>
<section><h2>执行环境、设备与预算</h2><div class="runtime">${runtimeCards}</div></section>
<section><h2>监督结论与当前取舍</h2><div class="decision"><b>${esc(s.supervision.actual_call)}</b><span>${esc(s.supervision.result)}<br>${esc(s.supervision.decision)}</span></div></section>
<section><h2>下一步</h2><ol class="action-list">${next}</ol></section>
<section><h2>证据入口</h2><ul>${evidence}</ul></section>
<div class="footer">状态更新时间以本地时间为准。只有写入实际矩阵、身份键、运行回执或复算结果后才补入数值；缺失项保持“未知/待核实”，不填零。</div>
</main><script>const updated=new Date(${JSON.stringify(s.updated_at)});const age=(Date.now()-updated.getTime())/60000;if(age>10)document.getElementById("stale").classList.add("show");</script></body></html>`;

const temporary = `${outputPath}.tmp`;
await fs.writeFile(temporary, html, "utf8");
await fs.rename(temporary, outputPath);
console.log(`updated ${path.relative(root, outputPath)} from ${path.relative(root, statusPath)}`);
