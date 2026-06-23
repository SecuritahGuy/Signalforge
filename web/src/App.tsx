import {
  Activity,
  ArrowUpRight,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  CircleDollarSign,
  ClipboardList,
  Database,
  FlaskConical,
  FolderKanban,
  LineChart,
  ListFilter,
  Play,
  Radar,
  RefreshCcw,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  TerminalSquare,
  TrendingUp,
  WalletCards
} from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { checkHealth, fetchScripts, triggerScript, streamRun, type ScriptCategory, type ScriptArg, type ScriptMeta, type RunState } from "./api";

type HistoryRow = {
  runId: string;
  time: string;
  mode: string;
  latestPriceDate: string;
  equity: number;
  cash: number;
  auditStatus: string;
  openPositions: number;
};

type DashboardData = {
  generatedAt: string;
  paper: {
    tracking: Record<string, unknown>;
    account: Record<string, unknown>;
    monitor: Record<string, unknown>;
    audit: Record<string, unknown>;
    actionability: Record<string, unknown>;
    history: HistoryRow[];
    allHistory: HistoryRow[];
    positions: Record<string, unknown>[];
    watchlist: Record<string, unknown>[];
    orderLedger: Record<string, unknown>[];
    tradingLedger: Record<string, unknown>[];
  };
  discovery: {
    summary: Record<string, unknown>;
    candidates: Record<string, unknown>[];
    monitoringState: Record<string, unknown>[];
    exclusions: Record<string, unknown>[];
    promotionCandidates: Record<string, unknown>[];
    lanes: Record<string, Record<string, unknown>[]>;
    availableUniverse: Record<string, unknown>;
    currentWatchlistExcluded: Record<string, unknown>;
  };
  promotion: {
    summary: Record<string, unknown>;
    plan: Record<string, unknown>[];
  };
  actionability: {
    candidates: Record<string, unknown>[];
  };
  stocks: Record<string, StockDetail>;
  model: Record<string, unknown>;
  research: Record<string, unknown>;
  backtests: Record<string, unknown>;
  universe: {
    tracked: Record<string, unknown>[];
    sp500: Record<string, unknown>[];
    summary: Record<string, unknown>;
  };
  runs: {
    history: HistoryRow[];
    latestRun: Record<string, unknown>;
  };
  reports: Record<string, string>;
};

type StockDetail = {
  profile: Record<string, unknown>;
  metrics: Record<string, unknown>;
  prices: Record<string, unknown>[];
  paperPosition: Record<string, unknown> | null;
  watchlist: Record<string, unknown> | null;
  discovery: Record<string, unknown> | null;
  monitoring: Record<string, unknown> | null;
  actionability: Record<string, unknown> | null;
  promotion: Record<string, unknown> | null;
};

type ChartPoint = {
  date: string;
  close: number;
  open: number;
  high: number;
  low: number;
  volume: number;
  returnFromStart: number;
  dailyReturn: number;
  rangePct: number;
  volumeRatio: number;
};

type SortDirection = "asc" | "desc";

const fallbackData: DashboardData = {
  generatedAt: "",
  paper: {
    tracking: {},
    account: {},
    monitor: {},
    audit: {},
    actionability: {},
    history: [],
    allHistory: [],
    positions: [],
    watchlist: [],
    orderLedger: [],
    tradingLedger: []
  },
  discovery: {
    summary: {},
    candidates: [],
    monitoringState: [],
    exclusions: [],
    promotionCandidates: [],
    lanes: {},
    availableUniverse: {},
    currentWatchlistExcluded: {}
  },
  promotion: { summary: {}, plan: [] },
  actionability: { candidates: [] },
  stocks: {},
  model: {},
  research: {},
  backtests: {},
  universe: { tracked: [], sp500: [], summary: {} },
  runs: { history: [], latestRun: {} },
  reports: {}
};

const navItems = [
  ["Overview", Activity],
  ["Stock Detail", Search],
  ["Paper", LineChart],
  ["Positions", WalletCards],
  ["Discovery", Radar],
  ["Promotion", ClipboardList],
  ["Model", BarChart3],
  ["R&D", FlaskConical],
  ["Backtests", TrendingUp],
  ["Universe", Database],
  ["Runs", FolderKanban],
  ["Reports", TerminalSquare],
  ["Server", Play]
] as const;

type PageId = (typeof navItems)[number][0];

const pageCopy: Record<PageId, string> = {
  Overview: "Paper trading, discovery monitoring, and promotion review in one local console.",
  "Stock Detail": "One-symbol view of price, risk, model, discovery, and paper-trading context.",
  Paper: "Paper account movement, trading blockers, and current review state.",
  Positions: "Current holdings, watchlist rows, and generated order ledgers.",
  Discovery: "Monitored candidates from the R&D discovery feed before promotion.",
  Promotion: "Promotion readiness and gates before anything enters the traded universe.",
  Model: "Model visibility, prediction drift, score buckets, and pick explanations.",
  "R&D": "Experiment leaderboards, feature ablations, and portfolio-rule research.",
  Backtests: "Backtest curves, monthly returns, symbol contribution, and trade ledgers.",
  Universe: "Tracked symbols, broad universe coverage, exclusions, and discovery lanes.",
  Runs: "Daily run history and the latest run state.",
  Reports: "Local markdown report outputs generated by the daily review bundle.",
  Server: "Run scripts, view live output, and monitor the SignalForge API server."
};

function App() {
  const [data, setData] = useState<DashboardData>(fallbackData);
  const [selectedNav, setSelectedNav] = useState<PageId>("Overview");
  const [selectedCandidate, setSelectedCandidate] = useState<string | null>(null);
  const [selectedStock, setSelectedStock] = useState<string | null>(null);
  const [notice, setNotice] = useState("");
  const [serverOnline, setServerOnline] = useState(false);
  const [scriptCategories, setScriptCategories] = useState<ScriptCategory[]>([]);
  const [scriptRuns, setScriptRuns] = useState<Map<string, RunState>>(new Map());
  const operationsCommand = ".venv/bin/python scripts/run_daily_operations.py --loop --interval-minutes 30";

  const loadData = () => {
    fetch("/data/dashboard.json")
      .then((response) => response.json())
      .then((payload) => {
        setData(payload);
        setNotice("Data reloaded from local bundle");
      })
      .catch(() => {
        setData(fallbackData);
        setNotice("Could not load local dashboard data");
      });
  };

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    const poll = () => {
      checkHealth().then((ok) => {
        setServerOnline(ok);
        if (ok) fetchScripts().then(setScriptCategories);
      });
    };
    poll();
    const id = setInterval(poll, 10_000);
    return () => clearInterval(id);
  }, []);

  const runScript = useCallback(async (name: string, args?: Record<string, unknown>) => {
    const runId = await triggerScript(name, args);
    if (!runId) {
      setNotice(`Failed to start ${name}`);
      return;
    }
    const partial: RunState = {
      run_id: runId,
      script_name: name,
      status: "running",
      started_at: new Date().toISOString(),
      finished_at: null,
      exit_code: null,
      output: [],
    };
    setScriptRuns((prev) => new Map(prev).set(runId, partial));
    setSelectedNav("Server");

    const close = streamRun(runId, (line) => {
      setScriptRuns((prev) => {
        const cur = prev.get(runId);
        if (!cur) return prev;
        const next = { ...cur, output: [...cur.output, line] };
        return new Map(prev).set(runId, next);
      });
    }, (status) => {
      setScriptRuns((prev) => {
        const cur = prev.get(runId);
        if (!cur) return prev;
        const next = { ...cur, status: status as RunState["status"] };
        return new Map(prev).set(runId, next);
      });
    });

    return close;
  }, []);

  const copyCommand = () => {
    navigator.clipboard
      ?.writeText(operationsCommand)
      .then(() => setNotice("Review command copied"))
      .catch(() => setNotice("Copy unavailable in this browser"));
  };

  const selected = useMemo(() => {
    if (!selectedCandidate) return data.discovery.candidates[0];
    return (
      data.discovery.candidates.find((row) => String(row.symbol) === selectedCandidate) ??
      data.discovery.candidates[0]
    );
  }, [data.discovery.candidates, selectedCandidate]);

  const selectedStockSymbol =
    selectedStock ??
    selectedCandidate ??
    String(data.paper.positions[0]?.symbol ?? data.discovery.candidates[0]?.symbol ?? "");
  const selectedStockDetail = selectedStockSymbol ? data.stocks[selectedStockSymbol] : undefined;

  const actionCounts = objectNumberMap(data.paper.actionability.effective_action_counts);
  const latestSnapshot = value(data.paper.tracking.latest_snapshot, "No snapshot");
  const latestPriceDate = value(data.paper.tracking.latest_price_date, "No price date");

  const openDiscoveryCandidate = (symbol: string) => {
    setSelectedCandidate(symbol);
    setSelectedNav("Discovery");
  };

  const openStock = (symbol: string) => {
    setSelectedStock(symbol);
    setSelectedNav("Stock Detail");
  };

  const renderPage = () => {
    switch (selectedNav) {
      case "Stock Detail":
        return (
          <StockDetailPage
            symbol={selectedStockSymbol}
            detail={selectedStockDetail}
            candidates={Object.keys(data.stocks)}
            onSelect={openStock}
          />
        );
      case "Paper":
        return (
          <section className="main-grid page-view" data-page="Paper">
            <div className="panel span-2">
              <PanelHeader
                icon={<LineChart size={18} />}
                title="Paper Equity Timeline"
                action={`${data.paper.history.length} snapshots`}
              />
              <EquityChart rows={data.paper.history} />
            </div>
            <DailyReviewPanel
              latestAudit={value(data.paper.tracking.latest_audit_status, "unknown")}
              latestPriceDate={latestPriceDate}
              assessment={value(data.paper.tracking.assessment, "pending")}
              latestSnapshot={latestSnapshot}
              command={operationsCommand}
            />
            {serverOnline && (
              <div className="panel" style={{ display: "flex", gap: 8, alignItems: "center", padding: "12px 16px" }}>
                <span style={{ fontSize: 13, color: "var(--muted)" }}>Server actions:</span>
                <button className="primary-button" style={{ padding: "4px 12px", fontSize: 13, gap: 4 }} onClick={() => runScript("run_paper_monitor")}>
                  <Play size={14} /> Paper Monitor
                </button>
                <button className="primary-button" style={{ padding: "4px 12px", fontSize: 13, gap: 4 }} onClick={() => runScript("update_paper_ledger")}>
                  <Play size={14} /> Update Ledger
                </button>
                <button className="primary-button" style={{ padding: "4px 12px", fontSize: 13, gap: 4 }} onClick={() => runScript("run_paper_tracking_report")}>
                  <Play size={14} /> Tracking Report
                </button>
                <button className="primary-button" style={{ padding: "4px 12px", fontSize: 13, gap: 4 }} onClick={() => runScript("run_intraday_risk_monitor")}>
                  <Play size={14} /> Risk Monitor
                </button>
              </div>
            )}
            <div className="panel">
              <PanelHeader icon={<BarChart3 size={18} />} title="Actionability Blockers" action="current" />
              <BlockerChart counts={actionCounts} />
            </div>
            <div className="panel span-2">
              <PanelHeader icon={<ListFilter size={18} />} title="Actionability Candidates" action="review" />
              <ActionabilityTable rows={data.actionability.candidates.slice(0, 10)} onSymbolSelect={openStock} />
            </div>
          </section>
        );
      case "Positions":
        return (
          <section className="main-grid page-view" data-page="Positions">
            <div className="panel span-3">
              <PanelHeader icon={<WalletCards size={18} />} title="Open Positions" action={`${data.paper.positions.length} rows`} />
              <DataTable
                rows={data.paper.positions}
                columns={[
                  "symbol",
                  "sector",
                  "action",
                  "shares",
                  "entry_price",
                  "latest_price",
                  "unrealized_pnl",
                  "unrealized_return",
                  "days_open",
                  "current_score",
                  "entry_score",
                  "highest_close_since_entry",
                  "trailing_stop_activated",
                  "target_exit_date",
                  "exit_reason"
                ]}
                onSymbolSelect={openStock}
              />
            </div>
            <div className="panel span-2">
              <PanelHeader icon={<Search size={18} />} title="Portfolio Watchlist" action={`${data.paper.watchlist.length} rows`} />
              <DataTable rows={data.paper.watchlist.slice(0, 25)} columns={["symbol", "sector", "score", "adj_close", "relative_volume_20d", "volatility_20d", "sector_rank_volatility_20d"]} onSymbolSelect={openStock} />
            </div>
            <div className="panel">
              <PanelHeader icon={<BarChart3 size={18} />} title="Sector Exposure" action="mark value" />
              <BlockerChart counts={objectNumberMap(data.paper.monitor.sector_exposure)} />
            </div>
            <div className="panel span-3">
              <PanelHeader icon={<ClipboardList size={18} />} title="Order Ledger" action={`${data.paper.orderLedger.length} rows`} />
              <DataTable rows={data.paper.orderLedger.slice(0, 30)} columns={["status", "date", "symbol", "sector", "score", "shares", "estimated_required_cash", "skip_reason"]} onSymbolSelect={openStock} />
            </div>
          </section>
        );
      case "Discovery":
        return (
          <section className="main-grid page-view" data-page="Discovery">
            {serverOnline && (
              <div className="panel span-3" style={{ display: "flex", gap: 8, alignItems: "center", padding: "12px 16px" }}>
                <span style={{ fontSize: 13, color: "var(--muted)" }}>Server actions:</span>
                <button className="primary-button" style={{ padding: "4px 12px", fontSize: 13, gap: 4 }} onClick={() => runScript("run_symbol_discovery_rd")}>
                  <Play size={14} /> Run Discovery R&D
                </button>
                <button className="primary-button" style={{ padding: "4px 12px", fontSize: 13, gap: 4 }} onClick={() => runScript("run_stock_discovery")}>
                  <Play size={14} /> Run Stock Discovery
                </button>
                <button className="primary-button" style={{ padding: "4px 12px", fontSize: 13, gap: 4 }} onClick={() => runScript("promote_discovery_candidates")}>
                  <Play size={14} /> Promote Candidates
                </button>
              </div>
            )}
            <div className="panel span-2">
              <PanelHeader icon={<Radar size={18} />} title="Discovery Candidates" action="monitored" />
              <CandidateTable
                rows={data.discovery.candidates.slice(0, 30)}
                selected={selectedCandidate}
                onSelect={openDiscoveryCandidate}
                onSymbolSelect={openStock}
              />
            </div>
            <div className="panel">
              <PanelHeader icon={<SlidersHorizontal size={18} />} title="Candidate Detail" action="monitoring" />
              <CandidateDetail row={selected} />
            </div>
            <div className="panel span-3">
              <PanelHeader icon={<Search size={18} />} title="Discovery Gate Summary" action="current" />
              <SummaryGrid
                items={[
                  ["Eligible after filters", number(data.discovery.summary.eligible_after_filters)],
                  ["Monitored candidates", number(data.discovery.summary.monitored_candidate_count)],
                  ["Promotion candidates", number(data.discovery.summary.promotion_candidate_count)],
                  ["Ready to promote", number(data.promotion.summary.ready_to_promote_count)]
                ]}
              />
            </div>
            <div className="panel span-2">
              <PanelHeader icon={<ClipboardList size={18} />} title="Monitoring State" action={`${data.discovery.monitoringState.length} rows`} />
              <DataTable rows={data.discovery.monitoringState.slice(0, 30)} columns={["symbol", "sector", "appearances", "latest_discovery_score", "max_discovery_score", "promotion_status", "promotion_blockers"]} onSymbolSelect={openStock} />
            </div>
            <div className="panel">
              <PanelHeader icon={<ShieldCheck size={18} />} title="Exclusions" action={`${data.discovery.exclusions.length} rows`} />
              <DataTable rows={data.discovery.exclusions.slice(0, 12)} columns={["symbol", "sector", "exclusion_reasons"]} onSymbolSelect={openStock} />
            </div>
          </section>
        );
      case "Promotion":
        return (
          <section className="main-grid page-view" data-page="Promotion">
            <div className="panel span-3">
              <PanelHeader icon={<ClipboardList size={18} />} title="Promotion Plan" action="review only" />
              <PromotionTable rows={data.promotion.plan.slice(0, 30)} onSymbolSelect={openStock} />
            </div>
            <div className="panel span-2">
              <PanelHeader icon={<CheckCircle2 size={18} />} title="Promotion Readiness" action="gated" />
              <SummaryGrid
                items={[
                  ["Ready to promote", number(data.promotion.summary.ready_to_promote_count)],
                  ["Plan rows", data.promotion.plan.length],
                  ["Review candidates", number(data.discovery.summary.promotion_candidate_count)],
                  ["Discovery monitored", number(data.discovery.summary.monitored_candidate_count)]
                ]}
              />
            </div>
            <div className="panel">
              <PanelHeader icon={<ShieldCheck size={18} />} title="Promotion Rule" action="guardrail" />
              <p className="panel-copy">
                Candidates stay in monitoring until they pass promotion checks. This page is intentionally review-only.
              </p>
            </div>
          </section>
        );
      case "Model":
        return (
          <section className="main-grid page-view" data-page="Model">
            <div className="panel span-2">
              <PanelHeader icon={<BarChart3 size={18} />} title="Prediction Drift" action={`${rows(data.model.predictionDrift).length} months`} />
              <DataTable rows={rows(data.model.predictionDrift).slice(-24)} columns={["month", "rows", "symbols", "score_mean", "score_p10", "score_p50", "score_p90", "above_threshold"]} />
            </div>
            <div className="panel">
              <PanelHeader icon={<SlidersHorizontal size={18} />} title="Top Features" action="importance" />
              <p className="panel-copy compact">
                Aggregated across walk-forward splits. A split is one train/validation time window.
              </p>
              <DataTable rows={rows(data.model.featureImportanceSummary).slice(0, 10)} columns={["feature", "mean_importance", "mean_rank", "top3_rate", "splits"]} />
            </div>
            <div className="panel span-2">
              <PanelHeader icon={<Search size={18} />} title="Paper Pick Explanations" action={`${rows(data.model.paperPickExplanations).length} rows`} />
              <DataTable rows={rows(data.model.paperPickExplanations).slice(0, 25)} columns={["symbol", "sector", "score", "shares", "estimated_required_cash", "top_model_features"]} onSymbolSelect={openStock} />
            </div>
            <div className="panel">
              <PanelHeader icon={<LineChart size={18} />} title="Score Buckets" action={`${rows(data.model.scoreBuckets).length} rows`} />
              <DataTable rows={rows(data.model.scoreBuckets).slice(0, 20)} columns={["split_id", "bucket_label", "rows", "score_mean", "mean_realized_return", "win_rate", "ic_spearman"]} />
            </div>
          </section>
        );
      case "R&D":
        return (
          <section className="main-grid page-view" data-page="R&D">
            {serverOnline && (
              <div className="panel span-3" style={{ display: "flex", gap: 8, alignItems: "center", padding: "12px 16px" }}>
                <span style={{ fontSize: 13, color: "var(--muted)" }}>Server actions:</span>
                <button className="primary-button" style={{ padding: "4px 12px", fontSize: 13, gap: 4 }} onClick={() => runScript("run_experiments")}>
                  <Play size={14} /> Run Experiments
                </button>
                <button className="primary-button" style={{ padding: "4px 12px", fontSize: 13, gap: 4 }} onClick={() => runScript("run_rd_experiments")}>
                  <Play size={14} /> Run R&D Experiments
                </button>
                <button className="primary-button" style={{ padding: "4px 12px", fontSize: 13, gap: 4 }} onClick={() => runScript("train_baseline")}>
                  <Play size={14} /> Train Baseline
                </button>
                <button className="primary-button" style={{ padding: "4px 12px", fontSize: 13, gap: 4 }} onClick={() => runScript("run_visibility_report")}>
                  <Play size={14} /> Visibility Report
                </button>
              </div>
            )}
            <div className="panel span-3">
              <PanelHeader icon={<FlaskConical size={18} />} title="Risk Experiment Leaderboard" action={`${rows(data.research.riskExperimentLeaderboard).length} rows`} />
              <DataTable rows={rows(data.research.riskExperimentLeaderboard).slice(0, 25)} columns={["experiment", "feature_set", "model", "ic_mean", "directional_hit_rate", "risk_backtest_sharpe", "risk_backtest_max_drawdown"]} />
            </div>
            <div className="panel span-2">
              <PanelHeader icon={<BarChart3 size={18} />} title="Feature Ablation" action="research" />
              <DataTable rows={rows(data.research.featureAblation).slice(0, 20)} columns={["experiment", "feature_set", "model", "ic_mean", "risk_backtest_sharpe", "risk_backtest_max_drawdown"]} />
            </div>
            <div className="panel">
              <PanelHeader icon={<SlidersHorizontal size={18} />} title="Portfolio Rules" action="rules" />
              <DataTable rows={rows(data.research.portfolioRules).slice(0, 12)} columns={["rule", "total_return", "sharpe", "max_drawdown", "filled_trades", "win_rate"]} />
            </div>
          </section>
        );
      case "Backtests":
        return (
          <section className="main-grid page-view" data-page="Backtests">
            <div className="panel span-2">
              <PanelHeader icon={<LineChart size={18} />} title="Paper Style Equity" action={`${rows(data.backtests.paperStyleDailyEquity).length} rows`} />
              <DataTable rows={rows(data.backtests.paperStyleDailyEquity).slice(-30)} columns={["date", "cash", "equity", "open_positions", "realized_pnl", "unrealized_pnl"]} />
            </div>
            <div className="panel">
              <PanelHeader icon={<TrendingUp size={18} />} title="Backtest Summary" action="current" />
              <SummaryGrid
                items={[
                  ["Ending equity", formatMoney(number(record(data.backtests.paperStyleSummary).ending_equity))],
                  ["Total return", formatPercent(number(record(data.backtests.paperStyleSummary).total_return))],
                  ["Sharpe", number(record(data.backtests.paperStyleSummary).sharpe).toFixed(2)],
                  ["Max drawdown", formatPercent(number(record(data.backtests.paperStyleSummary).max_drawdown))]
                ]}
              />
            </div>
            <div className="panel span-2">
              <PanelHeader icon={<BarChart3 size={18} />} title="Symbol Contributions" action={`${rows(data.backtests.topExperimentSymbolContributions).length} rows`} />
              <DataTable rows={rows(data.backtests.topExperimentSymbolContributions).slice(0, 25)} columns={["symbol", "long_days", "short_days", "gross_contribution", "avg_daily_contribution", "avg_abs_weight"]} onSymbolSelect={openStock} />
            </div>
            <div className="panel">
              <PanelHeader icon={<ClipboardList size={18} />} title="Monthly Returns" action={`${rows(data.backtests.topExperimentMonthlyReturns).length} rows`} />
              <DataTable rows={rows(data.backtests.topExperimentMonthlyReturns).slice(-18)} columns={["month", "raw_return", "risk_return", "trading_days"]} />
            </div>
          </section>
        );
      case "Universe":
        return (
          <section className="main-grid page-view" data-page="Universe">
            <div className="panel span-3">
              <PanelHeader icon={<Database size={18} />} title="Universe Summary" action="local" />
              <SummaryGrid
                items={[
                  ["Tracked symbols", number(data.universe.summary.trackedCount)],
                  ["S&P 500 symbols", number(data.universe.summary.sp500Count)],
                  ["Discovery candidates", data.discovery.candidates.length],
                  ["Exclusions", data.discovery.exclusions.length]
                ]}
              />
            </div>
            <div className="panel span-2">
              <PanelHeader icon={<Search size={18} />} title="Tracked Universe" action={`${data.universe.tracked.length} rows`} />
              <DataTable rows={data.universe.tracked.slice(0, 50)} columns={["symbol", "name", "category", "sector", "industry"]} onSymbolSelect={openStock} />
            </div>
            <div className="panel">
              <PanelHeader icon={<Radar size={18} />} title="Momentum Lane" action="top" />
              <DataTable rows={rows(data.discovery.lanes.momentumBreakouts).slice(0, 12)} columns={["symbol", "sector", "discovery_score", "return_20d", "return_60d", "lane_reason"]} onSymbolSelect={openStock} />
            </div>
            <div className="panel span-3">
              <PanelHeader icon={<Database size={18} />} title="S&P 500 Universe" action={`${data.universe.sp500.length} rows`} />
              <DataTable rows={data.universe.sp500.slice(0, 75)} columns={["symbol", "name", "sector", "industry", "notes"]} onSymbolSelect={openStock} />
            </div>
          </section>
        );
      case "Runs":
        return (
          <section className="main-grid page-view" data-page="Runs">
            <div className="panel span-3">
              <PanelHeader icon={<FolderKanban size={18} />} title="Daily Run History" action={`${data.runs.history.length} rows`} />
              <DataTable rows={data.runs.history.slice().reverse().slice(0, 50)} columns={["runId", "time", "mode", "latestPriceDate", "equity", "cash", "auditStatus", "openPositions"]} />
            </div>
            <div className="panel span-3">
              <PanelHeader icon={<LineChart size={18} />} title="Recent Equity Timeline" action={`${data.paper.history.length} snapshots`} />
              <EquityChart rows={data.paper.history} />
            </div>
          </section>
        );
      case "Reports":
        return (
          <section className="main-grid page-view" data-page="Reports">
            <ReportPanel title="Daily Ops Review" text={data.reports.dailyOps} />
            <ReportPanel title="Paper Tracking" text={data.reports.paperTracking} />
            <ReportPanel title="Paper Monitor" text={data.reports.paperMonitor} />
            <ReportPanel title="Paper Realism Audit" text={data.reports.paperRealismAudit} />
            <ReportPanel title="Paper Style Backtest" text={data.reports.paperStyleBacktest} />
            <ReportPanel title="Model Visibility" text={data.reports.modelVisibility} />
            <ReportPanel title="Symbol Discovery" text={data.reports.symbolDiscovery} />
            <ReportPanel title="Promotion Plan" text={data.reports.promotionPlan} />
            <ReportPanel title="R&D Summary" text={data.reports.research} />
          </section>
        );
      case "Server":
        return (
          <section className="main-grid page-view" data-page="Server">
            <ScriptRunnerPanel
              categories={scriptCategories}
              runs={scriptRuns}
              online={serverOnline}
              onRun={runScript}
            />
          </section>
        );
      case "Overview":
      default:
        return (
          <section className="main-grid page-view" data-page="Overview">
            <div className="panel span-2">
              <PanelHeader
                icon={<LineChart size={18} />}
                title="Paper Equity Timeline"
                action={`${data.paper.history.length} snapshots`}
              />
              <EquityChart rows={data.paper.history} />
            </div>

            <DailyReviewPanel
              latestAudit={value(data.paper.tracking.latest_audit_status, "unknown")}
              latestPriceDate={latestPriceDate}
              assessment={value(data.paper.tracking.assessment, "pending")}
              latestSnapshot={latestSnapshot}
              command={operationsCommand}
            />

            <div className="panel span-2">
              <PanelHeader icon={<Radar size={18} />} title="Top Discovery Candidates" action="top third" />
              <CandidateTable
                rows={data.discovery.candidates.slice(0, 12)}
                selected={selectedCandidate}
                onSelect={openDiscoveryCandidate}
                onSymbolSelect={openStock}
              />
            </div>

            <div className="panel">
              <PanelHeader icon={<SlidersHorizontal size={18} />} title="Candidate Detail" action="monitoring" />
              <CandidateDetail row={selected} />
            </div>

            <div className="panel">
              <PanelHeader icon={<BarChart3 size={18} />} title="Actionability Blockers" action="current" />
              <BlockerChart counts={actionCounts} />
            </div>

            <div className="panel span-3">
              <PanelHeader icon={<ClipboardList size={18} />} title="Promotion Plan" action="review only" />
              <PromotionTable rows={data.promotion.plan.slice(0, 8)} onSymbolSelect={openStock} />
            </div>
          </section>
        );
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <TrendingUp size={18} />
          </div>
          <div>
            <strong>SignalForge</strong>
            <span>Local operations</span>
          </div>
        </div>
        <nav className="nav-list" aria-label="Primary">
          {navItems.map(([label, Icon]) => (
            <button
              className={selectedNav === label ? "nav-item active" : "nav-item"}
              key={label}
              onClick={() => setSelectedNav(label)}
              aria-current={selectedNav === label ? "page" : undefined}
            >
              <Icon size={17} />
              <span>{label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-status">
          <span className={`status-dot ${serverOnline ? "online" : "offline"}`} />
          <div>
            <strong>{serverOnline ? "Server online" : "Server offline"}</strong>
            <span>{serverOnline ? "api ready on :8080" : "start with npm run server"}</span>
          </div>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div>
            <h1>{selectedNav === "Overview" ? "SignalForge Operations" : selectedNav}</h1>
            <p>{pageCopy[selectedNav]}</p>
          </div>
          <div className="topbar-actions">
            <div className="refresh-meta">
              <span>Last sync</span>
              <strong>{formatDateTime(data.generatedAt)}</strong>
              {notice ? <em>{notice}</em> : null}
            </div>
            <button className="icon-button" title="Reload synced data" onClick={loadData}>
              <RefreshCcw size={17} />
            </button>
            <button className="primary-button" onClick={copyCommand}>
              <Play size={17} />
              Copy Command
            </button>
          </div>
        </header>

        <section className="kpi-grid" aria-label="Key metrics">
          <KpiCard
            label="Paper equity"
            value={formatMoney(number(data.paper.tracking.paper_equity))}
            detail={`${formatPercent(number(data.paper.tracking.paper_total_return))} total return`}
            tone="green"
            icon={<CircleDollarSign size={18} />}
          />
          <KpiCard
            label="Cash"
            value={formatMoney(number(data.paper.tracking.paper_cash))}
            detail={`${number(data.paper.tracking.open_positions)} open positions`}
            tone="slate"
            icon={<Database size={18} />}
          />
          <KpiCard
            label="Actionable orders"
            value={String(number(data.paper.actionability.actionable_new_order_count))}
            detail={
              data.paper.actionability.plan_is_latest_price_date === false
                ? `last plan ${value(data.paper.actionability.as_of_date, "unknown")}; buys after close`
                : `${number(data.paper.actionability.blocked_by_active_symbol_count)} active-symbol blocks`
            }
            tone="amber"
            icon={<ListFilter size={18} />}
          />
          <KpiCard
            label="Monitored symbols"
            value={String(number(data.discovery.summary.monitored_candidate_count))}
            detail={`${number(data.discovery.summary.eligible_after_filters)} eligible after filters`}
            tone="blue"
            icon={<Search size={18} />}
          />
          <KpiCard
            label="Ready to promote"
            value={String(number(data.promotion.summary.ready_to_promote_count))}
            detail={`${number(data.discovery.summary.promotion_candidate_count)} review candidates`}
            tone="green"
            icon={<CheckCircle2 size={18} />}
          />
        </section>

        {renderPage()}
      </main>
    </div>
  );
}

function KpiCard({
  label,
  value: displayValue,
  detail,
  tone,
  icon
}: {
  label: string;
  value: string;
  detail: string;
  tone: "green" | "blue" | "amber" | "slate";
  icon: ReactNode;
}) {
  return (
    <article className={`kpi-card ${tone}`}>
      <div className="kpi-icon">{icon}</div>
      <div>
        <span>{label}</span>
        <strong>{displayValue}</strong>
        <small>{detail}</small>
      </div>
    </article>
  );
}

function DailyReviewPanel({
  latestAudit,
  latestPriceDate,
  assessment,
  latestSnapshot,
  command
}: {
  latestAudit: string;
  latestPriceDate: string;
  assessment: string;
  latestSnapshot: string;
  command: string;
}) {
  return (
    <div className="panel">
      <PanelHeader icon={<ShieldCheck size={18} />} title="Daily Review" action="status" />
      <div className="review-stack">
        <ReviewLine label="Audit" value={latestAudit} tone="green" />
        <ReviewLine label="Latest price date" value={latestPriceDate} tone="slate" />
        <ReviewLine label="Assessment" value={assessment} tone="amber" />
        <ReviewLine label="Latest snapshot" value={latestSnapshot} tone="slate" />
      </div>
      <div className="command-box">
        <TerminalSquare size={16} />
        <code>{command}</code>
      </div>
    </div>
  );
}

function PanelHeader({ icon, title, action }: { icon: ReactNode; title: string; action: string }) {
  return (
    <div className="panel-header">
      <div>
        {icon}
        <h2>{title}</h2>
      </div>
      <span>{action}</span>
    </div>
  );
}

function SummaryGrid({ items }: { items: Array<[string, string | number]> }) {
  return (
    <div className="summary-grid">
      {items.map(([label, itemValue]) => (
        <div className="summary-item" key={label}>
          <span>{label}</span>
          <strong>{itemValue}</strong>
        </div>
      ))}
    </div>
  );
}

function StockDetailPage({
  symbol,
  detail,
  candidates,
  onSelect
}: {
  symbol: string;
  detail?: StockDetail;
  candidates: string[];
  onSelect: (symbol: string) => void;
}) {
  if (!symbol || !detail) {
    return (
      <section className="main-grid page-view" data-page="Stock Detail">
        <div className="panel span-3">
          <PanelHeader icon={<Search size={18} />} title="Stock Detail" action="select symbol" />
          <p className="empty">Select a symbol from any table to open its detail page.</p>
        </div>
      </section>
    );
  }

  const profile = detail.profile ?? {};
  const metrics = detail.metrics ?? {};
  const latestPrice = detail.prices[detail.prices.length - 1];
  const priorPrice = detail.prices[Math.max(0, detail.prices.length - 21)] ?? detail.prices[0];
  const priceChange = number(latestPrice?.adj_close) && number(priorPrice?.adj_close)
    ? number(latestPrice?.adj_close) / number(priorPrice?.adj_close) - 1
    : 0;
  const relatedRows = [
    detail.paperPosition ? { source: "paper position", ...detail.paperPosition } : null,
    detail.watchlist ? { source: "watchlist", ...detail.watchlist } : null,
    detail.discovery ? { source: "discovery", ...detail.discovery } : null,
    detail.monitoring ? { source: "monitoring", ...detail.monitoring } : null,
    detail.actionability ? { source: "actionability", ...detail.actionability } : null,
    detail.promotion ? { source: "promotion", ...detail.promotion } : null
  ].filter(Boolean) as Record<string, unknown>[];
  const symbols = candidates.slice(0, 24);
  const chartPoints = buildChartPoints(detail.prices);

  return (
    <section className="main-grid page-view" data-page="Stock Detail">
      <div className="panel stock-hero span-3">
        <div className="stock-title">
          <div>
            <span>{value(profile.sector, value(metrics.sector, "Unknown sector"))}</span>
            <h2>{symbol}</h2>
            <p>{value(profile.name, "No company profile synced.")}</p>
          </div>
          <div className="stock-pills">
            {detail.paperPosition ? <span className="pill strong">paper position</span> : null}
            {detail.discovery ? <span className="pill strong">discovery</span> : null}
            {detail.actionability ? <span className="pill strong">actionability</span> : null}
            {detail.promotion ? <span className="pill strong">promotion</span> : null}
          </div>
        </div>
        <SummaryGrid
          items={[
            ["Latest close", formatMoney(number(latestPrice?.adj_close))],
            ["20d price move", formatPercent(priceChange)],
            ["Research 20d return", formatPercent(number(metrics.return_20d))],
            ["Research 60d return", formatPercent(number(metrics.return_60d))]
          ]}
        />
      </div>

      <div className="panel span-3">
        <PanelHeader icon={<LineChart size={18} />} title="Price Trend Explorer" action={`${detail.prices.length} closes`} />
        <StockExplorer rows={chartPoints} metrics={metrics} />
      </div>
      <div className="panel span-3">
        <PanelHeader icon={<BarChart3 size={18} />} title="Daily Tape" action="sortable" />
        <DailyTapeTable rows={chartPoints} />
      </div>

      <div className="panel">
        <PanelHeader icon={<ShieldCheck size={18} />} title="Risk & Liquidity" action="latest" />
        <MetricList
          items={[
            ["Vol 20d", formatPercent(number(metrics.volatility_20d))],
            ["Vol 60d", formatPercent(number(metrics.volatility_60d))],
            ["Vol ratio", number(metrics.volatility_ratio_20d_60d).toFixed(2)],
            ["Relative volume", number(metrics.relative_volume_20d).toFixed(2)],
            ["Avg dollar volume", compactMoney(number(metrics.avg_dollar_volume_20d))]
          ]}
        />
      </div>
      <div className="panel">
        <PanelHeader icon={<Radar size={18} />} title="Relative Strength" action="latest" />
        <MetricList
          items={[
            ["Vs market 20d", formatPercent(number(metrics.stock_minus_market_return_20d))],
            ["Vs market 60d", formatPercent(number(metrics.stock_minus_market_return_60d))],
            ["Vs sector 20d", formatPercent(number(metrics.stock_minus_sector_return_20d))],
            ["Beta 60d", number(metrics.beta_60d).toFixed(2)],
            ["Market corr 60d", number(metrics.correlation_to_market_60d).toFixed(2)]
          ]}
        />
      </div>
      <div className="panel">
        <PanelHeader icon={<SlidersHorizontal size={18} />} title="Trend Position" action="latest" />
        <MetricList
          items={[
            ["Above SMA 20", formatPercent(number(metrics.price_above_sma_20))],
            ["Above SMA 50", formatPercent(number(metrics.price_above_sma_50))],
            ["20d high distance", formatPercent(number(metrics.distance_from_20d_high))],
            ["60d drawdown", formatPercent(number(metrics.drawdown_60d))],
            ["Sector vol rank", formatPercent(number(metrics.sector_rank_volatility_20d))]
          ]}
        />
      </div>

      <div className="panel span-2">
        <PanelHeader icon={<ClipboardList size={18} />} title="System Context" action={`${relatedRows.length} sources`} />
        <DataTable
          rows={relatedRows}
          columns={[
            "source",
            "symbol",
            "status",
            "action",
            "score",
            "current_score",
            "shares",
            "unrealized_pnl",
            "unrealized_return",
            "target_exit_date",
            "exit_reason",
            "trailing_stop_activated",
            "promotion_status",
            "promotion_blockers",
            "skip_reason"
          ]}
          onSymbolSelect={onSelect}
        />
      </div>
      <div className="panel">
        <PanelHeader icon={<Search size={18} />} title="Quick Symbol Jump" action={`${candidates.length} symbols`} />
        <div className="symbol-jump">
          {symbols.map((item) => (
            <button className={item === symbol ? "symbol-chip active" : "symbol-chip"} key={item} onClick={() => onSelect(item)}>
              {item}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

function MetricList({ items }: { items: Array<[string, string]> }) {
  return (
    <dl className="metric-list">
      {items.map(([label, itemValue]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{itemValue}</dd>
        </div>
      ))}
    </dl>
  );
}

function StockExplorer({ rows, metrics }: { rows: ChartPoint[]; metrics: Record<string, unknown> }) {
  const windows = [
    ["1M", 21],
    ["3M", 63],
    ["6M", 126],
    ["1Y", 252],
    ["All", rows.length]
  ] as const;
  const [windowDays, setWindowDays] = useState<number>(63);
  const selectedRows = rows.slice(-Math.min(windowDays, rows.length));
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const activeIndex = hoverIndex ?? Math.max(0, selectedRows.length - 1);
  const active = selectedRows[activeIndex];
  if (!selectedRows.length) return <p className="empty">No price history synced for this symbol.</p>;

  const closes = selectedRows.map((row) => row.close);
  const volumes = selectedRows.map((row) => row.volume);
  const pricePath = linePath(closes, 100, 70, 8);
  const first = selectedRows[0];
  const last = selectedRows[selectedRows.length - 1];
  const windowReturn = last.close / first.close - 1;
  const minClose = Math.min(...closes);
  const maxClose = Math.max(...closes);
  const avgVolume = volumes.reduce((sum, item) => sum + item, 0) / Math.max(1, volumes.length);
  const maxVolume = Math.max(...volumes, 1);
  const crosshairX = selectedRows.length <= 1 ? 0 : (activeIndex / (selectedRows.length - 1)) * 100;
  const crosshairY = chartY(active.close, closes, 70, 8);
  const color = windowReturn >= 0 ? "#148a4a" : "#b42318";
  const lessons = stockLessons({ rows: selectedRows, metrics, active, avgVolume });

  return (
    <div className="market-explorer">
      <div className="explorer-toolbar">
        <div className="segmented-control" aria-label="Price history window">
          {windows.map(([label, days]) => (
            <button
              key={label}
              className={Math.min(windowDays, rows.length) === Math.min(days, rows.length) ? "active" : ""}
              onClick={() => setWindowDays(days)}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="explorer-readout">
          <span>{active.date}</span>
          <strong>{formatMoney(active.close)}</strong>
          <em className={active.dailyReturn >= 0 ? "positive" : "negative"}>
            {formatPercent(active.dailyReturn)}
          </em>
        </div>
      </div>

      <div className="explorer-grid">
        <div className="interactive-chart">
          <svg
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            aria-label="Interactive stock price and volume chart"
            onMouseLeave={() => setHoverIndex(null)}
            onMouseMove={(event) => {
              const rect = event.currentTarget.getBoundingClientRect();
              const pct = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
              setHoverIndex(Math.round(pct * (selectedRows.length - 1)));
            }}
          >
            <defs>
              <linearGradient id="stockPriceFill" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity="0.18" />
                <stop offset="100%" stopColor={color} stopOpacity="0" />
              </linearGradient>
            </defs>
            {selectedRows.map((row, index) => {
              const x = selectedRows.length <= 1 ? 0 : (index / (selectedRows.length - 1)) * 100;
              const height = Math.max(2, (row.volume / maxVolume) * 22);
              return (
                <rect
                  key={`${row.date}-volume`}
                  x={x}
                  y={96 - height}
                  width={Math.max(0.18, 84 / selectedRows.length)}
                  height={height}
                  fill={row.dailyReturn >= 0 ? "#148a4a" : "#b42318"}
                  opacity="0.22"
                />
              );
            })}
            <path d={`${pricePath} L 100 96 L 0 96 Z`} fill="url(#stockPriceFill)" />
            <path d={pricePath} fill="none" stroke={color} strokeWidth="2.4" vectorEffect="non-scaling-stroke" />
            <line x1={crosshairX} x2={crosshairX} y1="4" y2="96" stroke="#344054" strokeOpacity="0.28" vectorEffect="non-scaling-stroke" />
            <circle cx={crosshairX} cy={crosshairY} r="1.5" fill={color} vectorEffect="non-scaling-stroke" />
          </svg>
        </div>

        <div className="explorer-side">
          <SummaryGrid
            items={[
              ["Window return", formatPercent(windowReturn)],
              ["Close range", `${formatMoney(minClose)} - ${formatMoney(maxClose)}`],
              ["Avg volume", compactMoney(avgVolume)],
              ["Active volume", `${active.volumeRatio.toFixed(1)}x avg`]
            ]}
          />
          <div className="lesson-stack">
            {lessons.map((item) => (
              <div className="lesson-line" key={item.label}>
                <span>{item.label}</span>
                <strong>{item.value}</strong>
                <p>{item.note}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function DailyTapeTable({ rows }: { rows: ChartPoint[] }) {
  const [sortKey, setSortKey] = useState<keyof ChartPoint>("date");
  const [direction, setDirection] = useState<SortDirection>("desc");
  const [visibleRows, setVisibleRows] = useState(30);
  const columns: Array<[keyof ChartPoint, string]> = [
    ["date", "date"],
    ["close", "close"],
    ["dailyReturn", "daily move"],
    ["returnFromStart", "window move"],
    ["rangePct", "range"],
    ["volume", "volume"],
    ["volumeRatio", "volume x"]
  ];
  const tableRows = useMemo(() => {
    return rows
      .slice(-visibleRows)
      .sort((a, b) => {
        const left = a[sortKey];
        const right = b[sortKey];
        const result = typeof left === "string" || typeof right === "string"
          ? String(left).localeCompare(String(right))
          : Number(left) - Number(right);
        return direction === "asc" ? result : -result;
      });
  }, [rows, sortKey, direction, visibleRows]);
  if (!rows.length) return <p className="empty">No daily rows synced for this symbol.</p>;
  return (
    <div className="daily-tape">
      <div className="tape-toolbar">
        <label>
          Rows
          <input
            type="number"
            min="5"
            max={rows.length}
            value={visibleRows}
            onChange={(event) => setVisibleRows(Math.max(5, Math.min(rows.length, Number(event.target.value) || 30)))}
          />
        </label>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {columns.map(([key, label]) => (
                <th key={key}>
                  <button
                    className="sort-button"
                    onClick={() => {
                      if (sortKey === key) {
                        setDirection(direction === "asc" ? "desc" : "asc");
                      } else {
                        setSortKey(key);
                        setDirection(key === "date" ? "desc" : "desc");
                      }
                    }}
                  >
                    {label}
                    <span>{sortKey === key ? (direction === "asc" ? "up" : "down") : ""}</span>
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tableRows.map((row) => (
              <tr key={row.date}>
                <td>{row.date}</td>
                <td>{formatMoney(row.close)}</td>
                <td className={row.dailyReturn >= 0 ? "positive" : "negative"}>{formatPercent(row.dailyReturn)}</td>
                <td className={row.returnFromStart >= 0 ? "positive" : "negative"}>{formatPercent(row.returnFromStart)}</td>
                <td>{formatPercent(row.rangePct)}</td>
                <td>{compactMoney(row.volume)}</td>
                <td>{row.volumeRatio.toFixed(1)}x</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function buildChartPoints(rows: Record<string, unknown>[]): ChartPoint[] {
  const parsed = rows
    .map((row) => ({
      date: value(row.date, ""),
      close: number(row.adj_close || row.close),
      open: number(row.open),
      high: number(row.high),
      low: number(row.low),
      volume: number(row.volume)
    }))
    .filter((row) => row.date && Number.isFinite(row.close) && row.close > 0);
  const firstClose = parsed[0]?.close || 1;
  const avgVolume = parsed.reduce((sum, row) => sum + row.volume, 0) / Math.max(1, parsed.length);
  return parsed.map((row, index) => {
    const priorClose = parsed[Math.max(0, index - 1)]?.close || row.close;
    const high = row.high || row.close;
    const low = row.low || row.close;
    return {
      ...row,
      high,
      low,
      returnFromStart: row.close / firstClose - 1,
      dailyReturn: index === 0 ? 0 : row.close / priorClose - 1,
      rangePct: high && low ? high / low - 1 : 0,
      volumeRatio: avgVolume ? row.volume / avgVolume : 0
    };
  });
}

function linePath(points: number[], width: number, maxY: number, minY: number): string {
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;
  return points
    .map((point, index) => {
      const x = points.length <= 1 ? 0 : (index / (points.length - 1)) * width;
      const y = maxY - ((point - min) / range) * (maxY - minY);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function chartY(point: number, points: number[], maxY: number, minY: number): number {
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;
  return maxY - ((point - min) / range) * (maxY - minY);
}

function stockLessons({
  rows,
  metrics,
  active,
  avgVolume
}: {
  rows: ChartPoint[];
  metrics: Record<string, unknown>;
  active: ChartPoint;
  avgVolume: number;
}): Array<{ label: string; value: string; note: string }> {
  const last = rows[rows.length - 1];
  const windowReturn = last && rows[0] ? last.close / rows[0].close - 1 : 0;
  const drawdown = number(metrics.drawdown_60d);
  const relStrength = number(metrics.stock_minus_sector_return_20d);
  const volumeTone = active.volume > avgVolume * 1.5 ? "heavy" : active.volume < avgVolume * 0.7 ? "quiet" : "normal";
  return [
    {
      label: "Trend",
      value: windowReturn >= 0 ? "up window" : "down window",
      note: "Window return shows whether the selected history is compounding or fading."
    },
    {
      label: "Participation",
      value: volumeTone,
      note: "Volume above average means the move had more market participation."
    },
    {
      label: "Relative strength",
      value: formatPercent(relStrength),
      note: "Positive values mean it has recently beaten its sector benchmark."
    },
    {
      label: "Drawdown",
      value: formatPercent(drawdown),
      note: "A deeper drawdown means price is further below its recent peak."
    }
  ];
}

function DataTable({
  rows,
  columns,
  onSymbolSelect
}: {
  rows: Record<string, unknown>[];
  columns: string[];
  onSymbolSelect?: (symbol: string) => void;
}) {
  if (!rows.length) return <p className="empty">No synced rows for this table.</p>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column.replace(/_/g, " ")}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowKey(row, rowIndex)}>
              {columns.map((column) => {
                const symbol = value(row[column], "");
                return (
                  <td key={column}>
                    {column === "symbol" && symbol && onSymbolSelect ? (
                      <button className="symbol-link" onClick={() => onSymbolSelect(symbol)}>
                        {symbol}
                      </button>
                    ) : (
                      formatCell(row[column])
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EquityChart({ rows }: { rows: HistoryRow[] }) {
  const points = rows.map((row) => row.equity).filter((item) => Number.isFinite(item));
  if (!points.length) {
    return (
      <div className="chart-wrap empty-chart">
        <p className="empty">No paper equity history synced.</p>
      </div>
    );
  }
  const min = Math.min(...points, 0);
  const max = Math.max(...points, 1);
  const range = max - min || 1;
  const path = points
    .map((point, index) => {
      const x = points.length <= 1 ? 0 : (index / (points.length - 1)) * 100;
      const y = 86 - ((point - min) / range) * 72;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
  const latest = rows[rows.length - 1];
  return (
    <div className="chart-wrap">
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-label="Paper equity line chart">
        <defs>
          <linearGradient id="equityFill" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#18a058" stopOpacity="0.18" />
            <stop offset="100%" stopColor="#18a058" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d={`${path} L 100 100 L 0 100 Z`} fill="url(#equityFill)" />
        <path d={path} fill="none" stroke="#148a4a" strokeWidth="2.4" vectorEffect="non-scaling-stroke" />
      </svg>
      <div className="chart-meta">
        <span>{latest ? formatDateTime(latest.time) : "No history"}</span>
        <strong>{latest ? formatMoney(latest.equity) : "$0.00"}</strong>
      </div>
    </div>
  );
}

function BlockerChart({ counts }: { counts: Record<string, number> }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const max = Math.max(...entries.map(([, count]) => count), 1);
  if (!entries.length) return <p className="empty">No blocker data.</p>;
  return (
    <div className="blocker-list">
      {entries.map(([label, count]) => (
        <div className="blocker-row" key={label}>
          <div>
            <span>{label.replace(/_/g, " ")}</span>
            <strong>{count}</strong>
          </div>
          <div className="bar-track">
            <div style={{ width: `${(count / max) * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function CandidateTable({
  rows,
  selected,
  onSelect,
  onSymbolSelect
}: {
  rows: Record<string, unknown>[];
  selected: string | null;
  onSelect: (symbol: string) => void;
  onSymbolSelect?: (symbol: string) => void;
}) {
  if (!rows.length) return <p className="empty">No discovery candidates synced.</p>;
  return (
    <div className="table-wrap candidate-table">
      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Name</th>
            <th>Sector</th>
            <th>Score</th>
            <th>20d</th>
            <th>Lanes</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const symbol = String(row.symbol);
            return (
              <tr
                className={selected === symbol ? "selected" : ""}
                key={symbol}
                onClick={() => onSelect(symbol)}
              >
                <td>
                  {onSymbolSelect ? (
                    <button
                      className="symbol-link"
                      onClick={(event) => {
                        event.stopPropagation();
                        onSymbolSelect(symbol);
                      }}
                    >
                      {symbol}
                    </button>
                  ) : (
                    <strong>{symbol}</strong>
                  )}
                </td>
                <td>{value(row.name, "")}</td>
                <td>{value(row.sector, "")}</td>
                <td>{number(row.discovery_score).toFixed(1)}</td>
                <td>{formatPercent(number(row.return_20d))}</td>
                <td>{number(row.lane_count)}</td>
                <td><span className="pill">{value(row.promotion_status, "monitoring")}</span></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function ActionabilityTable({
  rows,
  onSymbolSelect
}: {
  rows: Record<string, unknown>[];
  onSymbolSelect?: (symbol: string) => void;
}) {
  if (!rows.length) return <p className="empty">No actionability candidates synced.</p>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Action</th>
            <th>Score</th>
            <th>Shares</th>
            <th>Reason</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${value(row.symbol, "row")}-${index}`}>
              <td>
                {onSymbolSelect ? (
                  <button className="symbol-link" onClick={() => onSymbolSelect(value(row.symbol, ""))}>
                    {value(row.symbol, "")}
                  </button>
                ) : (
                  <strong>{value(row.symbol, "")}</strong>
                )}
              </td>
              <td><span className="pill">{value(row.effective_action, value(row.action, "review"))}</span></td>
              <td>{number(row.score).toFixed(2)}</td>
              <td>{number(row.target_shares || row.shares)}</td>
              <td>
                {value(
                  row.skip_reason,
                  value(
                    row.reason,
                    row.effective_action === "stale_plan_wait_for_after_close"
                      ? `plan ${value(row.plan_date, "")}; latest ${value(row.latest_price_date, "")}`
                      : ""
                  )
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CandidateDetail({ row }: { row?: Record<string, unknown> }) {
  if (!row) return <p className="empty">Select a candidate.</p>;
  return (
    <div className="detail-card">
      <div className="detail-title">
        <div>
          <span>{value(row.sector, "")}</span>
          <strong>{value(row.symbol, "")}</strong>
        </div>
        <ArrowUpRight size={18} />
      </div>
      <p>{value(row.why_flagged, "No explanation available.")}</p>
      <dl>
        <div><dt>Discovery score</dt><dd>{number(row.discovery_score).toFixed(2)}</dd></div>
        <div><dt>60d return</dt><dd>{formatPercent(number(row.return_60d))}</dd></div>
        <div><dt>Lane count</dt><dd>{number(row.lane_count)}</dd></div>
        <div><dt>Blockers</dt><dd>{value(row.promotion_blockers, "none")}</dd></div>
      </dl>
    </div>
  );
}

function PromotionTable({
  rows,
  onSymbolSelect
}: {
  rows: Record<string, unknown>[];
  onSymbolSelect?: (symbol: string) => void;
}) {
  if (!rows.length) return <p className="empty">No ready promotion plan rows. Monitoring gates are still active.</p>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Name</th>
            <th>Sector</th>
            <th>Score</th>
            <th>Appearances</th>
            <th>Age</th>
            <th>Plan status</th>
            <th>Blockers</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={String(row.symbol)}>
              <td>
                {onSymbolSelect ? (
                  <button className="symbol-link" onClick={() => onSymbolSelect(value(row.symbol, ""))}>
                    {value(row.symbol, "")}
                  </button>
                ) : (
                  <strong>{value(row.symbol, "")}</strong>
                )}
              </td>
              <td>{value(row.name, "")}</td>
              <td>{value(row.sector, "")}</td>
              <td>{number(row.discovery_score).toFixed(1)}</td>
              <td>{number(row.appearances)}</td>
              <td>{number(row.monitoring_age_days)}d</td>
              <td><span className="pill">{value(row.promotion_plan_status, "")}</span></td>
              <td>{value(row.promotion_plan_blockers, "")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ReportPanel({ title, text }: { title: string; text: string }) {
  const content = text.trim();
  return (
    <div className="panel report-panel span-2">
      <PanelHeader icon={<ChevronRight size={18} />} title={title} action={content ? "local" : "missing"} />
      {content ? <pre>{content.slice(0, 2800)}</pre> : <p className="empty">Report was not found in the synced local bundle.</p>}
    </div>
  );
}

function ReviewLine({ label, value: lineValue, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className="review-line">
      <span>{label}</span>
      <strong className={tone}>{lineValue}</strong>
    </div>
  );
}

function ArgsForm({ args, values, onChange }: {
  args: ScriptArg[];
  values: Record<string, string>;
  onChange: (key: string, val: string) => void;
}) {
  if (args.length === 0) return null;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 6, alignItems: "center" }}>
      {args.map((a) => {
        const key = a.name.replace(/^--/, "");
        if (a.type === "bool") return null;
        return (
          <label key={key} style={{ fontSize: 12, display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ color: "var(--muted)", whiteSpace: "nowrap" }}>{a.name.replace(/^--/, "")}:</span>
            {a.choices ? (
              <select
                value={values[key] ?? String(a.default ?? "")}
                onChange={(e) => onChange(key, e.target.value)}
                style={{ fontSize: 12, padding: "2px 6px", borderRadius: 4, border: "1px solid var(--line)" }}
              >
                {a.choices.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            ) : (
              <input
                type={a.type === "int" || a.type === "float" ? "number" : "text"}
                value={values[key] ?? String(a.default ?? "")}
                onChange={(e) => onChange(key, e.target.value)}
                placeholder={a.default != null ? String(a.default) : key}
                style={{ width: 120, fontSize: 12, padding: "2px 6px", borderRadius: 4, border: "1px solid var(--line)" }}
              />
            )}
          </label>
        );
      })}
    </div>
  );
}

function ScriptRunnerPanel({
  categories,
  runs,
  online,
  onRun,
}: {
  categories: ScriptCategory[];
  runs: Map<string, RunState>;
  online: boolean;
  onRun: (name: string, args?: Record<string, unknown>) => void;
}) {
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [expandedCat, setExpandedCat] = useState<string | null>(null);
  const [argValues, setArgValues] = useState<Record<string, Record<string, string>>>({});

  const runList = Array.from(runs.values());
  const active = selectedRun ? runs.get(selectedRun) ?? null : null;

  return (
    <>
      <div className="panel span-2">
        <PanelHeader
          icon={<Play size={18} />}
          title="Run Script"
          action={online ? "connected" : "disconnected"}
        />
        {!online ? (
          <p className="empty">
            Server is offline. Start it with <code>npm run server</code> from the project root.
          </p>
        ) : categories.length === 0 ? (
          <p className="empty">No scripts available.</p>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {categories.map((cat) => (
              <details key={cat.category} open={expandedCat === cat.category}
                onToggle={(e) => setExpandedCat((e.target as HTMLDetailsElement).open ? cat.category : null)}
              >
                <summary style={{ cursor: "pointer", fontWeight: 600, fontSize: 13, padding: "6px 0", color: "var(--ink)" }}>
                  {cat.label}
                  <span style={{ marginLeft: 8, fontWeight: 400, color: "var(--muted)" }}>({cat.scripts.length})</span>
                </summary>
                <table className="data-table" style={{ width: "100%", marginTop: 4 }}>
                  <thead>
                    <tr>
                      <th style={{ width: 180 }}>Script</th>
                      <th>Description</th>
                      <th style={{ width: 100 }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cat.scripts.map((s) => {
                      const scriptArgs = argValues[s.name] ?? {};
                      return (
                        <tr key={s.name}>
                          <td><code style={{ fontSize: 12 }}>{s.name}</code></td>
                          <td className="muted" style={{ fontSize: 12 }}>{s.description}</td>
                          <td>
                            <button
                              className="primary-button"
                              style={{ padding: "4px 10px", fontSize: 12, gap: 3 }}
                              onClick={() => {
                                const hasArgs = s.args.length > 0;
                                const merged: Record<string, unknown> = {};
                                if (hasArgs) {
                                  for (const a of s.args) {
                                    const key = a.name.replace(/^--/, "");
                                    const raw = scriptArgs[key] ?? String(a.default ?? "");
                                    if (a.type === "int") merged[key] = parseInt(raw) || 0;
                                    else if (a.type === "float") merged[key] = parseFloat(raw) || 0;
                                    else if (a.type === "bool") merged[key] = a.default === true;
                                    else merged[key] = raw;
                                  }
                                }
                                onRun(s.name, hasArgs ? merged : undefined);
                              }}
                            >
                              <Play size={13} />
                              Run
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </details>
            ))}
          </div>
        )}
      </div>

      {runList.length > 0 && (
        <div className="panel span-2">
          <PanelHeader
            icon={<TerminalSquare size={18} />}
            title="Recent Runs"
            action={`${runList.length} total`}
          />
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
            {runList.map((r) => (
              <button
                key={r.run_id}
                className={selectedRun === r.run_id ? "nav-item active" : "nav-item"}
                style={{ padding: "4px 10px", fontSize: 13 }}
                onClick={() => setSelectedRun(r.run_id)}
              >
                {r.script_name}
                <span style={{ marginLeft: 6, fontSize: 11, opacity: 0.7 }}>
                  {r.status === "running" ? "⏳" : r.status === "completed" ? "✅" : r.status === "failed" ? "❌" : "⏸"}
                </span>
              </button>
            ))}
          </div>

          {active && (
            <div>
              <div style={{ display: "flex", gap: 16, marginBottom: 8, fontSize: 13 }}>
                <span><strong>Status:</strong> {active.status}</span>
                {active.exit_code !== null && <span><strong>Exit code:</strong> {active.exit_code}</span>}
                {active.started_at && <span><strong>Started:</strong> {new Date(active.started_at).toLocaleTimeString()}</span>}
                {active.finished_at && <span><strong>Finished:</strong> {new Date(active.finished_at).toLocaleTimeString()}</span>}
              </div>
              <pre
                style={{
                  background: "#1c2430",
                  color: "#e0e7ef",
                  padding: "12px 16px",
                  borderRadius: 8,
                  fontSize: 12,
                  lineHeight: 1.5,
                  maxHeight: 400,
                  overflow: "auto",
                  fontFamily: "ui-monospace, SFMono-Regular, monospace",
                }}
              >
                {active.output.length > 0
                  ? active.output.join("\n")
                  : <span style={{ opacity: 0.5 }}>Waiting for output...</span>}
              </pre>
            </div>
          )}
        </div>
      )}
    </>
  );
}

function objectNumberMap(value: unknown): Record<string, number> {
  if (!value || typeof value !== "object") return {};
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>).map(([key, item]) => [key, number(item)])
  );
}

function rows(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? (value as Record<string, unknown>[]) : [];
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function rowKey(row: Record<string, unknown>, fallback: number): string {
  const symbol = value(row.symbol, "");
  const id = value(row.runId, value(row.run_id, ""));
  const date = value(row.date, value(row.month, ""));
  return `${symbol || id || date || "row"}-${fallback}`;
}

function number(value: unknown): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function value(value: unknown, fallback: string): string {
  return value === null || value === undefined || value === "" ? fallback : String(value);
}

function formatCell(value: unknown): string {
  if (typeof value === "number") {
    if (Math.abs(value) > 0 && Math.abs(value) < 1) return value.toFixed(4);
    if (Math.abs(value) >= 1000) return value.toLocaleString("en-US", { maximumFractionDigits: 2 });
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  return value === null || value === undefined || value === "" ? "-" : String(value);
}

function formatMoney(value: number): string {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(value);
}

function compactMoney(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    notation: "compact",
    maximumFractionDigits: 1
  }).format(value);
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

function formatDateTime(value: string): string {
  if (!value) return "Not synced";
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return value;
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit"
  }).format(date);
}

export default App;
