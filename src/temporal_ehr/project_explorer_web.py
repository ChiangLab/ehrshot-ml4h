"""Dependency-free local web UI for the governed project explorer."""

# ruff: noqa: E501

from __future__ import annotations

import json
from datetime import date, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import duckdb

from temporal_ehr.data.contracts import ContractError
from temporal_ehr.project_explorer import ALLOWED_SPLITS, TASKS, embedding_preview

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Temporal EHR Project Explorer</title>
<style>
:root{--blue:#2774ae;--dark:#1f2933;--line:#d6dbe1;--muted:#627080;--bg:#f5f7f9;--good:#287a45;--warn:#9a6700}
*{box-sizing:border-box}body{margin:0;font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--dark);background:var(--bg)}
header{background:white;border-bottom:1px solid var(--line);padding:18px 26px}h1{margin:0;font-size:23px;font-weight:650}header p{margin:4px 0 0;color:var(--muted)}
.privacy{background:#fff7d6;border-bottom:1px solid #e2ca70;padding:9px 26px;font-size:13px}.layout{display:grid;grid-template-columns:220px 1fr;min-height:calc(100vh - 112px)}
nav{background:white;border-right:1px solid var(--line);padding:18px 12px}nav button{width:100%;text-align:left;border:0;background:transparent;border-radius:5px;padding:10px 12px;margin:2px 0;font-size:14px;cursor:pointer}nav button.active,nav button:hover{background:#eaf2f8;color:#145b8c}
main{padding:22px;max-width:1500px}.panel{display:none}.panel.active{display:block}.card{background:white;border:1px solid var(--line);border-radius:6px;padding:16px;margin:0 0 16px;box-shadow:0 1px 2px #00000008}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:16px}.metric{font-size:26px;font-weight:650}.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
h2{font-size:20px;margin:0 0 14px}h3{font-size:16px;margin:0 0 10px}.controls{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:12px}select,input,button.action{border:1px solid #abb5bf;border-radius:4px;background:white;padding:7px 9px;font:inherit}button.action{cursor:pointer;background:var(--blue);color:white;border-color:var(--blue)}
.table-wrap{overflow:auto;max-height:520px;border:1px solid var(--line)}table{border-collapse:collapse;width:100%;background:white;font-size:13px}th,td{border-bottom:1px solid #e5e8eb;padding:7px 9px;text-align:left;white-space:nowrap}th{position:sticky;top:0;background:#eef2f5;z-index:1}tr.clickable:hover{background:#eef6fb;cursor:pointer}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f7f8fa;border:1px solid var(--line);padding:12px;max-height:480px;overflow:auto;font:12px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}.muted{color:var(--muted)}.good{color:var(--good);font-weight:600}.warn{color:var(--warn);font-weight:600}.error{color:#a61b1b;white-space:pre-wrap}
.bar-row{display:grid;grid-template-columns:170px 1fr 64px;align-items:center;gap:8px;margin:6px 0}.bar-bg{height:14px;background:#e5e9ed}.bar{height:100%;background:var(--blue)}.pill{display:inline-block;border:1px solid var(--line);border-radius:12px;padding:2px 8px;margin:2px;font-size:12px}.split{display:grid;grid-template-columns:1fr 1fr;gap:14px}@media(max-width:900px){.layout{grid-template-columns:1fr}nav{border-right:0;border-bottom:1px solid var(--line);display:flex;overflow:auto}nav button{width:auto;white-space:nowrap}.split{grid-template-columns:1fr}}
</style></head><body>
<header><h1>Temporal EHR Project Explorer</h1><p>Governed train/validation review of EHRSHOT, B1-B4, and M1-M3</p></header>
<div class="privacy"><strong>Private DGX view.</strong> Do not screenshot or export row-level data. Held-out content is not indexed. The repository defines no M4.</div>
<div class="layout"><nav id="nav">
<button data-panel="overview" class="active">Overview</button><button data-panel="cohort">Cohort & events</button><button data-panel="m1">M1 inputs</button><button data-panel="m2">M2 summaries</button><button data-panel="embeddings">Embeddings</button><button data-panel="runs">Run status</button>
</nav><main>
<section id="overview" class="panel active"><h2>Project overview</h2><div id="overview-cards" class="cards"></div><div class="split"><div class="card"><h3>Interfaces</h3><div id="methods"></div></div><div><div class="card"><h3>Original EHRSHOT event fields</h3><div id="source-columns"></div></div><div class="card"><h3>Development cohort</h3><div id="cohort-chart"></div></div></div></div><div class="card"><h3>Validation AUROC</h3><div class="controls"><select id="metric-method"></select><select id="metric-condition"></select></div><div id="metric-chart"></div></div></section>
<section id="cohort" class="panel"><h2>Cohort and strictly pre-prediction events</h2><div class="card"><div class="controls"><select id="cohort-task"></select><select id="cohort-split"><option>train</option><option>validation</option></select><button class="action" id="load-cohort">Load cohort</button></div><div id="cohort-table"></div></div><div class="card"><h3>Selected instance event history</h3><p class="muted">Click a cohort row. This view retains the original seven EHRSHOT fields, then adds days-before-T and the task-specific 30/90-day window index. Events at or after T are absent.</p><div id="event-table" class="muted">No instance selected.</div></div></section>
<section id="m1" class="panel"><h2>M1 serialized inputs</h2><div class="card"><div class="controls"><select id="m1-task"></select><select id="m1-split"><option>train</option><option>validation</option></select><select id="m1-condition"></select><button class="action" id="load-m1">Load inputs</button></div><div id="m1-table"></div></div><div class="card"><h3>Full selected serialization</h3><pre id="m1-detail">Click a row above.</pre></div></section>
<section id="m2" class="panel"><h2>M2 structured summaries</h2><div class="card"><div class="controls"><select id="m2-task"></select><select id="m2-condition"></select><button class="action" id="load-m2">Load summaries</button></div><div id="m2-table"></div></div><div class="split"><div class="card"><h3>Ordered source</h3><pre id="m2-source">Click a row above.</pre></div><div class="card"><h3>Validated structured summary</h3><pre id="m2-summary">Click a row above.</pre></div></div></section>
<section id="embeddings" class="panel"><h2>Embedding artifacts and row previews</h2><div class="card"><p class="muted">All full arrays remain in immutable governed files. The catalog indexes every vector row; selecting a row reads only that vector.</p><div id="artifact-table"></div></div><div class="card"><div class="controls"><select id="artifact-select"></select><input id="vector-row" type="number" min="0" value="0"><button class="action" id="load-vector">Preview vector</button></div><div id="vector-summary"></div><pre id="vector-values">Select an artifact and row.</pre></div></section>
<section id="runs" class="panel"><h2>Run status</h2><div class="card"><div id="run-table"></div></div></section>
</main></div>
<script>
const state={overview:null,artifacts:[]};
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path){const r=await fetch(path,{cache:'no-store'});const j=await r.json();if(!r.ok)throw new Error(j.error||r.statusText);return j}
function table(rows,click){if(!rows.length)return '<p class="muted">No rows available yet.</p>';const cols=Object.keys(rows[0]);return `<div class="table-wrap"><table><thead><tr>${cols.map(c=>`<th>${esc(c)}</th>`).join('')}</tr></thead><tbody>${rows.map((r,i)=>`<tr ${click?`class="clickable" data-index="${i}"`:''}>${cols.map(c=>`<td title="${esc(r[c])}">${esc(String(r[c]??'').slice(0,160))}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`}
function options(id,values){document.getElementById(id).innerHTML=values.map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('')}
function bars(target,rows,labelKey,valueKey,max=1){document.getElementById(target).innerHTML=rows.map(r=>`<div class="bar-row"><span>${esc(r[labelKey])}</span><div class="bar-bg"><div class="bar" style="width:${Math.max(0,Math.min(100,100*Number(r[valueKey])/max))}%"></div></div><span>${Number(r[valueKey]).toFixed(3)}</span></div>`).join('')||'<p class="muted">No data.</p>'}
document.getElementById('nav').onclick=e=>{if(!e.target.dataset.panel)return;document.querySelectorAll('nav button,.panel').forEach(x=>x.classList.remove('active'));e.target.classList.add('active');document.getElementById(e.target.dataset.panel).classList.add('active')}
async function loadOverview(){const x=await api('/api/overview');state.overview=x;document.getElementById('overview-cards').innerHTML=[['Cohort instances',x.cohort_total],['M1 input rows',x.m1_inputs],['Embedding artifacts',x.embedding_artifacts],['Indexed vectors',x.embedding_rows],['M2 summaries',x.m2_summaries]].map(([l,v])=>`<div class="card"><div class="label">${l}</div><div class="metric">${Number(v).toLocaleString()}</div></div>`).join('');document.getElementById('methods').innerHTML=x.methods.map(m=>`<div><span class="pill">${esc(m.interface)}</span><strong>${esc(m.descriptive_name)}</strong> - <span class="${m.status.includes('complete')?'good':'muted'}">${esc(m.status)}</span><br><span class="muted">${esc(m.input_representation)} -> ${esc(m.classifier)}</span></div><hr>`).join('');document.getElementById('source-columns').innerHTML=x.source_columns.map(r=>`<span class="pill">${esc(r.column_name)}: ${esc(r.logical_type)}</span>`).join('');bars('cohort-chart',x.cohort.map(r=>({...r,label:`${r.task} / ${r.split}`,value:r.instances})),'label','value',Math.max(...x.cohort.map(r=>r.instances)));const methods=[...new Set(x.metrics.map(r=>r.method))];options('metric-method',methods);refreshMetricConditions()}
function refreshMetricConditions(){const method=document.getElementById('metric-method').value;const values=[...new Set(state.overview.metrics.filter(r=>r.method===method).map(r=>r.condition))];options('metric-condition',values);drawMetrics()}
function drawMetrics(){const m=document.getElementById('metric-method').value,c=document.getElementById('metric-condition').value;const rows=state.overview.metrics.filter(r=>r.method===m&&r.condition===c&&r.metric==='auroc').map(r=>({...r,label:r.task+' / '+r.scope}));bars('metric-chart',rows,'label','value',1)}
document.getElementById('metric-method').onchange=refreshMetricConditions;document.getElementById('metric-condition').onchange=drawMetrics;
const taskOpts=()=>TASKS;const TASKS=['lab_anemia','guo_readmission','new_hypertension','chexpert'];['cohort-task','m1-task','m2-task'].forEach(id=>options(id,TASKS));
async function loadCohort(){const t=document.getElementById('cohort-task').value,s=document.getElementById('cohort-split').value,x=await api(`/api/cohort?task=${encodeURIComponent(t)}&split=${encodeURIComponent(s)}`);document.getElementById('cohort-table').innerHTML=table(x.rows,true);document.querySelectorAll('#cohort-table tr[data-index]').forEach(tr=>tr.onclick=()=>loadEvents(x.rows[Number(tr.dataset.index)].instance_key))}
async function loadEvents(key){const x=await api(`/api/events?instance_key=${encodeURIComponent(key)}`);document.getElementById('event-table').innerHTML=table(x.rows)}document.getElementById('load-cohort').onclick=()=>guard(loadCohort);
async function initConditions(){const x=await api('/api/filters');options('m1-condition',x.m1_conditions);options('m2-condition',x.m2_conditions.length?x.m2_conditions:['C1_correct_compact_time'])}
async function loadM1(){const q=new URLSearchParams({task:document.getElementById('m1-task').value,split:document.getElementById('m1-split').value,condition:document.getElementById('m1-condition').value});const x=await api('/api/m1?'+q);document.getElementById('m1-table').innerHTML=table(x.rows,true);document.querySelectorAll('#m1-table tr[data-index]').forEach(tr=>tr.onclick=()=>loadM1Detail(x.rows[Number(tr.dataset.index)].representation_key))}
async function loadM1Detail(key){const x=await api('/api/m1_detail?representation_key='+encodeURIComponent(key));document.getElementById('m1-detail').textContent=x.serialized_text}document.getElementById('load-m1').onclick=()=>guard(loadM1);
async function loadM2(){const q=new URLSearchParams({task:document.getElementById('m2-task').value,condition:document.getElementById('m2-condition').value});const x=await api('/api/m2?'+q);document.getElementById('m2-table').innerHTML=table(x.rows,true);document.querySelectorAll('#m2-table tr[data-index]').forEach(tr=>tr.onclick=()=>loadM2Detail(x.rows[Number(tr.dataset.index)].generation_key))}
async function loadM2Detail(key){const x=await api('/api/m2_detail?generation_key='+encodeURIComponent(key));document.getElementById('m2-source').textContent=x.source_text;document.getElementById('m2-summary').textContent=x.summary_text}document.getElementById('load-m2').onclick=()=>guard(loadM2);
async function loadArtifacts(){const x=await api('/api/embeddings');state.artifacts=x.rows;document.getElementById('artifact-table').innerHTML=table(x.rows.map(r=>({method:r.method,task:r.task,condition:r.condition,rows:r.rows,dimensions:r.dimensions,integrity:r.integrity_status,norm_min:r.norm_min,norm_max:r.norm_max})));document.getElementById('artifact-select').innerHTML=x.rows.map(r=>`<option value="${esc(r.artifact_id)}">${esc(r.method+' | '+r.task+' | '+r.condition+' | '+r.rows+'x'+r.dimensions)}</option>`).join('')}
async function loadVector(){const q=new URLSearchParams({artifact_id:document.getElementById('artifact-select').value,row:document.getElementById('vector-row').value});const x=await api('/api/embedding_preview?'+q);document.getElementById('vector-summary').innerHTML=`<p><strong>${esc(x.method)} / ${esc(x.task)} / ${esc(x.condition)}</strong></p><p>row ${x.row_index.toLocaleString()} · dimension ${x.dimensions.toLocaleString()} · norm ${x.norm.toFixed(8)} · split ${esc(x.split)} · sequence position ${esc(x.sequence_position)}</p>`;document.getElementById('vector-values').textContent=JSON.stringify(x.first_coordinates,null,2)}document.getElementById('load-vector').onclick=()=>guard(loadVector);
async function loadRuns(){const x=await api('/api/runs');document.getElementById('run-table').innerHTML=table(x.rows)}
async function guard(fn){try{await fn()}catch(e){alert(e.message)}}
Promise.all([loadOverview(),initConditions(),loadArtifacts(),loadRuns()]).catch(e=>document.querySelector('main').innerHTML=`<p class="error">${esc(e.message)}</p>`);
</script></body></html>"""


def _json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _rows(connection: duckdb.DuckDBPyConnection, sql: str, parameters: list[Any] | None = None) -> list[dict[str, Any]]:
    cursor = connection.execute(sql, parameters or [])
    columns = [item[0] for item in cursor.description]
    return [{column: _json_value(value) for column, value in zip(columns, row, strict=True)} for row in cursor.fetchall()]


class ExplorerApplication:
    """Bounded query surface; deliberately exposes no arbitrary SQL or export route."""

    def __init__(self, database_path: Path):
        self.database_path = database_path.resolve()
        if not self.database_path.is_file():
            raise ContractError("Project explorer database does not exist.")

    def connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.database_path), read_only=True)

    def query(self, path: str, parameters: dict[str, list[str]]) -> dict[str, Any]:
        with self.connect() as connection:
            if path == "/api/overview":
                return {
                    "cohort_total": connection.execute("SELECT count(*) FROM clinical.cohort_instances").fetchone()[0],
                    "m1_inputs": connection.execute("SELECT count(*) FROM representations.m1_inputs").fetchone()[0],
                    "embedding_artifacts": connection.execute("SELECT count(*) FROM catalog.embedding_artifacts").fetchone()[0],
                    "embedding_rows": connection.execute("SELECT count(*) FROM representations.embedding_rows").fetchone()[0],
                    "m2_summaries": connection.execute("SELECT count(*) FROM representations.m2_summaries").fetchone()[0],
                    "methods": _rows(connection, "SELECT * FROM catalog.method_interfaces ORDER BY interface"),
                    "source_columns": _rows(connection, "SELECT * FROM catalog.source_columns"),
                    "cohort": _rows(connection, "SELECT * FROM clinical.cohort_summary"),
                    "metrics": _rows(connection, "SELECT * FROM results.validation_metrics ORDER BY method, condition, task, metric"),
                }
            if path == "/api/filters":
                return {
                    "m1_conditions": [row[0] for row in connection.execute(
                        "SELECT DISTINCT representation FROM representations.m1_inputs ORDER BY representation"
                    ).fetchall()],
                    "m2_conditions": [row[0] for row in connection.execute(
                        "SELECT DISTINCT condition FROM representations.m2_summaries ORDER BY condition"
                    ).fetchall()],
                }
            if path == "/api/cohort":
                task = self._task(parameters)
                split = self._split(parameters)
                return {"rows": _rows(connection, """
                    SELECT instance_key, subject_id, prediction_time, task, family, split, target,
                           eligible_prediction_time_count, selected_prediction_time_index
                    FROM clinical.cohort_instances WHERE task = ? AND split = ?
                    ORDER BY instance_key LIMIT 100
                """, [task, split])}
            if path == "/api/events":
                key = self._required(parameters, "instance_key", 64)
                exists = connection.execute(
                    "SELECT count(*) FROM clinical.cohort_instances WHERE instance_key = ?", [key]
                ).fetchone()[0]
                if exists != 1:
                    raise ContractError("Unknown development instance key.")
                return {"rows": _rows(connection, """
                    SELECT event_time, days_before_prediction, prediction_relative_window, code,
                           numeric_value, text_value, unit, omop_table
                    FROM clinical.deduplicated_events_pre_prediction WHERE instance_key = ?
                    ORDER BY event_time DESC, code, numeric_value NULLS FIRST, text_value NULLS FIRST
                    LIMIT 2000
                """, [key])}
            if path == "/api/m1":
                task = self._task(parameters)
                split = self._split(parameters)
                condition = self._required(parameters, "condition", 120)
                return {"rows": _rows(connection, """
                    SELECT representation_key, instance_key, target, token_count, truncated,
                           substr(serialized_text, 1, 240) AS preview
                    FROM representations.m1_inputs
                    WHERE task = ? AND split = ? AND representation = ?
                    ORDER BY instance_key LIMIT 100
                """, [task, split, condition])}
            if path == "/api/m1_detail":
                key = self._required(parameters, "representation_key", 128)
                row = connection.execute(
                    "SELECT serialized_text FROM representations.m1_inputs WHERE representation_key = ?", [key]
                ).fetchone()
                if row is None:
                    raise ContractError("Unknown representation key.")
                return {"serialized_text": row[0]}
            if path == "/api/m2":
                task = self._task(parameters)
                condition = self._required(parameters, "condition", 120)
                return {"rows": _rows(connection, """
                    SELECT generation_key, evidence_status, length(source_text) AS source_characters,
                           length(summary_text) AS summary_characters,
                           substr(summary_text, 1, 240) AS summary_preview
                    FROM representations.m2_summaries WHERE task = ? AND condition = ?
                    ORDER BY generation_key LIMIT 100
                """, [task, condition])}
            if path == "/api/m2_detail":
                key = self._required(parameters, "generation_key", 128)
                row = connection.execute(
                    "SELECT source_text, summary_text, validation_json FROM representations.m2_summaries WHERE generation_key = ?",
                    [key],
                ).fetchone()
                if row is None:
                    raise ContractError("Unknown M2 generation key.")
                return {"source_text": row[0], "summary_text": row[1], "validation": json.loads(row[2])}
            if path == "/api/embeddings":
                return {"rows": _rows(connection, """
                    SELECT artifact_id, method, task, condition, rows, metadata_rows, dimensions,
                           dtype, all_finite, norm_min, norm_max, unit_normalized, integrity_status
                    FROM catalog.embedding_artifacts ORDER BY method, task, condition
                """)}
            if path == "/api/embedding_preview":
                artifact = self._required(parameters, "artifact_id", 128)
                try:
                    row = int(self._required(parameters, "row", 20))
                except ValueError as exc:
                    raise ContractError("Embedding row must be an integer.") from exc
                return embedding_preview(connection, artifact, row)
            if path == "/api/runs":
                return {"rows": _rows(connection, """
                    SELECT job_id, state, run_kind, substr(git_commit, 1, 12) AS git_commit,
                           started_at, ended_at, exit_status
                    FROM catalog.run_status ORDER BY try_cast(regexp_extract(job_id, '^[0-9]+') AS BIGINT) DESC,
                                                     job_id DESC LIMIT 250
                """)}
        raise ContractError("Unknown explorer route.")

    @staticmethod
    def _required(parameters: dict[str, list[str]], key: str, maximum: int) -> str:
        value = parameters.get(key, [""])[0]
        if not value or len(value) > maximum:
            raise ContractError(f"Missing or invalid {key}.")
        return value

    def _task(self, parameters: dict[str, list[str]]) -> str:
        task = self._required(parameters, "task", 40)
        if task not in TASKS:
            raise ContractError("Unknown locked task.")
        return task

    def _split(self, parameters: dict[str, list[str]]) -> str:
        split = self._required(parameters, "split", 20)
        if split not in ALLOWED_SPLITS:
            raise ContractError("Only train/validation splits are available.")
        return split


def handler_factory(application: ExplorerApplication) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(HTTPStatus.OK, HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if not parsed.path.startswith("/api/"):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                return
            try:
                result = application.query(parsed.path, parse_qs(parsed.query, keep_blank_values=True))
                self._send_json(HTTPStatus.OK, result)
            except (ContractError, duckdb.Error, OSError, ValueError) as exc:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            self._send(status, json.dumps(payload, default=_json_value).encode("utf-8"), "application/json")

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'; connect-src 'self'")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            print(f"project-explorer: {format % args}", flush=True)

    return Handler


def serve(database_path: Path, *, host: str, port: int) -> None:
    if host not in {"127.0.0.1", "localhost"}:
        raise ContractError("The governed explorer must bind only to localhost.")
    application = ExplorerApplication(database_path)
    server = ThreadingHTTPServer((host, port), handler_factory(application))
    print(json.dumps({"event": "project_explorer_ready", "host": host, "port": port,
                      "database": str(database_path.resolve())}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
