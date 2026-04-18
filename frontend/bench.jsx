// Benchmarks page — bar chart + results table
function BenchmarkBar({ rows, width }) {
  const h = 220;
  const pad = { l: 44, r: 20, t: 36, b: 36 };
  const iw = Math.max(width - pad.l - pad.r, 40);
  const ih = h - pad.t - pad.b;

  const COLORS = { baseline: "#7AB0FF", smboost: "#00FF9D", upper_bound: "#F5B95A" };
  const LABELS = { baseline: "baseline", smboost: "smboost", upper_bound: "upper bound" };

  const barW = Math.min(80, Math.floor((iw - 80) / 3));
  const gap = Math.floor((iw - 3 * barW) / 4);
  const yTicks = [0, 25, 50, 75, 100];

  return (
    <svg width={width} height={h} style={{ display: "block" }}>
      {yTicks.map(t => {
        const y = pad.t + ih - (t / 100) * ih;
        return (
          <g key={t}>
            <line x1={pad.l} x2={pad.l + iw} y1={y} y2={y} stroke="rgba(255,255,255,0.05)" />
            <text x={pad.l - 8} y={y + 3.5} fontSize="10"
              fontFamily="SF Mono, monospace" fill="#68686E" textAnchor="end">{t}%</text>
          </g>
        );
      })}
      {rows.map((r, i) => {
        const x = pad.l + gap + i * (barW + gap);
        const barH = Math.max(2, r.passAt1 * ih);
        const y = pad.t + ih - barH;
        const color = COLORS[r.mode] || "#888";
        const gid = "bg" + r.mode;
        return (
          <g key={r.mode}>
            <defs>
              <linearGradient id={gid} x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity="0.9" />
                <stop offset="100%" stopColor={color} stopOpacity="0.55" />
              </linearGradient>
            </defs>
            <rect x={x} y={y} width={barW} height={barH}
              fill={`url(#${gid})`} rx="3" />
            <text x={x + barW / 2} y={y - 7} fontSize="11.5"
              fontFamily="SF Mono, monospace" fill={color}
              textAnchor="middle" fontWeight="600">
              {(r.passAt1 * 100).toFixed(1)}%
            </text>
            <text x={x + barW / 2} y={pad.t + ih + 18} fontSize="10"
              fontFamily="SF Mono, monospace" fill="#68686E" textAnchor="middle">
              {LABELS[r.mode] || r.mode}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function BenchmarkPage({ data }) {
  const wrap = React.useRef(null);
  const [chartW, setChartW] = React.useState(700);

  React.useEffect(() => {
    if (!wrap.current) return;
    const ro = new ResizeObserver(() => setChartW(wrap.current.clientWidth));
    ro.observe(wrap.current);
    return () => ro.disconnect();
  }, []);

  if (!data) {
    return (
      <div style={{ padding: "80px 40px", textAlign: "center" }}>
        <div className="mono" style={{ color: "var(--fg-3)", fontSize: "11px", textTransform: "uppercase", letterSpacing: "0.12em" }}>No results</div>
        <div style={{ fontSize: "18px", color: "var(--fg-0)", marginTop: "10px" }}>Run the benchmark first</div>
        <div className="mono" style={{ color: "var(--fg-3)", fontSize: "11.5px", marginTop: "14px", lineHeight: 1.8 }}>
          python -m benchmarks.run_humaneval --mode baseline --n-tasks 50<br />
          python -m benchmarks.run_humaneval --mode smboost --n-tasks 50<br />
          python -m benchmarks.run_humaneval --mode upper_bound --n-tasks 50<br />
          python benchmarks/export_results.py
        </div>
      </div>
    );
  }

  const COLORS = { baseline: "#7AB0FF", smboost: "#00FF9D", upper_bound: "#F5B95A" };

  return (
    <div>
      <div className="ph">
        <div>
          <h1 className="ph-title">Benchmarks</h1>
          <div className="ph-sub">
            <span>Generated {data.generatedAt}</span>
            <span className="sep">·</span>
            <span>{data.nTasks} tasks</span>
            <span className="sep">·</span>
            <span className="mono">{data.model}</span>
          </div>
        </div>
      </div>

      <div className="content-1">
        <div className="glass">
          <div className="panel-h">
            <div className="panel-t">
              HumanEval pass@1
              <span className="hint">{data.nTasks} tasks · {data.model}</span>
            </div>
            <div className="panel-act" style={{ display: "flex", gap: "12px" }}>
              {data.rows.map(r => (
                <span key={r.mode} className="sw2">
                  <span className="bar" style={{ background: COLORS[r.mode] || "#888" }} />
                  {r.mode.replace("_", " ")}
                </span>
              ))}
            </div>
          </div>
          <div ref={wrap}>
            <BenchmarkBar rows={data.rows} width={chartW} />
          </div>
        </div>

        <div className="glass" style={{ marginTop: "16px" }}>
          <div className="panel-h">
            <div className="panel-t">Results</div>
          </div>
          <div className="ag-head" style={{ gridTemplateColumns: "1fr 1fr 1fr 1fr" }}>
            <div>Mode</div>
            <div>pass@1</div>
            <div>Avg Retries</div>
            <div>Avg Latency</div>
          </div>
          {data.rows.map(r => (
            <div key={r.mode} className="ag-row"
              style={{
                gridTemplateColumns: "1fr 1fr 1fr 1fr",
                borderLeft: r.mode === "smboost" ? "2px solid var(--accent)" : "2px solid transparent",
              }}>
              <div className="mono" style={{ color: COLORS[r.mode] || "var(--fg-1)" }}>
                {r.mode.replace("_", " ")}
              </div>
              <div className="mono" style={{ color: "var(--fg-0)", fontWeight: 600 }}>
                {(r.passAt1 * 100).toFixed(1)}%
              </div>
              <div className="mono dim">
                {r.avgRetries !== null ? r.avgRetries.toFixed(1) : "—"}
              </div>
              <div className="mono dim">
                {r.avgLatency.toFixed(2)}s
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

Object.assign(window, { BenchmarkPage });
