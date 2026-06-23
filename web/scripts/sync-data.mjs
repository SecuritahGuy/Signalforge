import fs from "node:fs";
import path from "node:path";
import readline from "node:readline";

const repoRoot = path.resolve(import.meta.dirname, "../..");
const publicDataDir = path.join(repoRoot, "web/public/data");

function readJson(relativePath, fallback = {}) {
  const fullPath = path.join(repoRoot, relativePath);
  if (!fs.existsSync(fullPath)) return fallback;
  const text = fs
    .readFileSync(fullPath, "utf8")
    .replace(/\bNaN\b/g, "null")
    .replace(/-Infinity\b/g, "null")
    .replace(/\bInfinity\b/g, "null");
  return JSON.parse(text);
}

function readText(relativePath, fallback = "") {
  const fullPath = path.join(repoRoot, relativePath);
  if (!fs.existsSync(fullPath)) return fallback;
  return fs.readFileSync(fullPath, "utf8");
}

function readCsv(relativePath) {
  const fullPath = path.join(repoRoot, relativePath);
  if (!fs.existsSync(fullPath)) return [];
  const text = fs.readFileSync(fullPath, "utf8").trim();
  if (!text) return [];
  const [headerLine, ...lines] = parseCsvRows(text);
  return lines.map((row) =>
    Object.fromEntries(headerLine.map((key, index) => [key, normalizeValue(row[index] ?? "")]))
  );
}

function parseCsvRows(text) {
  const rows = [];
  let row = [];
  let field = "";
  let inQuotes = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];
    if (char === '"' && inQuotes && next === '"') {
      field += '"';
      index += 1;
      continue;
    }
    if (char === '"') {
      inQuotes = !inQuotes;
      continue;
    }
    if (char === "," && !inQuotes) {
      row.push(field);
      field = "";
      continue;
    }
    if ((char === "\n" || char === "\r") && !inQuotes) {
      if (char === "\r" && next === "\n") index += 1;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
      continue;
    }
    field += char;
  }
  row.push(field);
  rows.push(row);
  return rows.filter((items) => items.some((item) => item !== ""));
}

function parseCsvLine(line) {
  return parseCsvRows(line)[0] ?? [];
}

function normalizeValue(value) {
  const trimmed = String(value).trim();
  if (trimmed === "") return "";
  const number = Number(trimmed);
  return Number.isFinite(number) && /^-?\d+(\.\d+)?(e[+-]?\d+)?$/i.test(trimmed)
    ? number
    : trimmed;
}

function latestHistoryRows(limit = 36) {
  const rows = readCsv("reports/daily_runs/history.csv");
  return rows
    .filter((row) => row.local_time && row.account_equity !== "")
    .slice(-limit)
    .map((row) => ({
      runId: row.run_id,
      time: row.local_time,
      mode: row.mode,
      latestPriceDate: row.latest_price_date,
      equity: Number(row.account_equity || 0),
      cash: Number(row.account_cash || 0),
      auditStatus: row.audit_status || "",
      openPositions: Number(row.account_open_positions || 0)
    }));
}

function historyRows() {
  return readCsv("reports/daily_runs/history.csv").map((row) => ({
    runId: row.run_id,
    time: row.local_time,
    mode: row.mode,
    latestPriceDate: row.latest_price_date,
    equity: Number(row.account_equity || 0),
    cash: Number(row.account_cash || 0),
    auditStatus: row.audit_status || "",
    openPositions: Number(row.account_open_positions || 0)
  }));
}

function topRows(rows, count, scoreField = "discovery_score") {
  return [...rows]
    .sort((a, b) => Number(b[scoreField] || 0) - Number(a[scoreField] || 0))
    .slice(0, count);
}

function aggregateFeatureImportance(rows) {
  const byFeature = new Map();
  for (const row of rows) {
    const feature = String(row.feature ?? "");
    if (!feature) continue;
    const current =
      byFeature.get(feature) ??
      {
        feature,
        importanceTotal: 0,
        rankTotal: 0,
        top3Count: 0,
        splitIds: new Set(),
        rows: 0
      };
    const importance = Number(row.importance || 0);
    const rank = Number(row.rank || 0);
    current.importanceTotal += importance;
    current.rankTotal += rank;
    current.top3Count += rank > 0 && rank <= 3 ? 1 : 0;
    if (row.split_id !== "") current.splitIds.add(row.split_id);
    current.rows += 1;
    byFeature.set(feature, current);
  }
  return [...byFeature.values()]
    .map((item) => ({
      feature: item.feature,
      mean_importance: item.importanceTotal / item.rows,
      mean_rank: item.rankTotal / item.rows,
      top3_rate: item.top3Count / item.rows,
      splits: item.splitIds.size
    }))
    .sort((a, b) => b.mean_importance - a.mean_importance);
}

function collectSymbols(...rowGroups) {
  const symbols = new Set();
  for (const rows of rowGroups) {
    for (const row of rows) {
      const symbol = String(row.symbol ?? "").trim();
      if (symbol) symbols.add(symbol);
    }
  }
  return symbols;
}

async function readLatestRowsForSymbols(relativePath, symbols) {
  const fullPath = path.join(repoRoot, relativePath);
  if (!fs.existsSync(fullPath) || symbols.size === 0) return {};
  const reader = readline.createInterface({
    input: fs.createReadStream(fullPath),
    crlfDelay: Infinity
  });
  let header = null;
  let symbolIndex = -1;
  const latest = {};
  for await (const line of reader) {
    if (!line) continue;
    if (!header) {
      header = parseCsvLine(line);
      symbolIndex = header.indexOf("symbol");
      continue;
    }
    if (symbolIndex < 0) return {};
    const values = parseCsvLine(line);
    const symbol = String(values[symbolIndex] ?? "").trim();
    if (!symbols.has(symbol)) continue;
    latest[symbol] = Object.fromEntries(
      header.map((key, index) => [key, normalizeValue(values[index] ?? "")])
    );
  }
  return latest;
}

async function readPriceHistoryForSymbols(relativePath, symbols, limitPerSymbol = 260) {
  const fullPath = path.join(repoRoot, relativePath);
  if (!fs.existsSync(fullPath) || symbols.size === 0) return {};
  const reader = readline.createInterface({
    input: fs.createReadStream(fullPath),
    crlfDelay: Infinity
  });
  let header = null;
  let symbolIndex = -1;
  const bySymbol = {};
  for await (const line of reader) {
    if (!line) continue;
    if (!header) {
      header = parseCsvLine(line);
      symbolIndex = header.indexOf("symbol");
      continue;
    }
    if (symbolIndex < 0) return {};
    const values = parseCsvLine(line);
    const symbol = String(values[symbolIndex] ?? "").trim();
    if (!symbols.has(symbol)) continue;
    const row = Object.fromEntries(
      header.map((key, index) => [key, normalizeValue(values[index] ?? "")])
    );
    const rows = bySymbol[symbol] ?? [];
    rows.push(row);
    if (rows.length > limitPerSymbol) rows.shift();
    bySymbol[symbol] = rows;
  }
  return bySymbol;
}

function unusedLegacyReadLatestRowsForSymbols(relativePath, symbols) {
  const fullPath = path.join(repoRoot, relativePath);
  if (!fs.existsSync(fullPath) || symbols.size === 0) return {};
  const lines = fs.readFileSync(fullPath, "utf8").trim().split(/\r?\n/);
  if (lines.length < 2) return {};
  const header = parseCsvLine(lines[0]);
  const symbolIndex = header.indexOf("symbol");
  if (symbolIndex < 0) return {};
  const latest = {};
  for (const line of lines.slice(1)) {
    const values = parseCsvLine(line);
    const symbol = String(values[symbolIndex] ?? "").trim();
    if (!symbols.has(symbol)) continue;
    latest[symbol] = Object.fromEntries(
      header.map((key, index) => [key, normalizeValue(values[index] ?? "")])
    );
  }
  return latest;
}

function unusedLegacyReadPriceHistoryForSymbols(relativePath, symbols, limitPerSymbol = 260) {
  const fullPath = path.join(repoRoot, relativePath);
  if (!fs.existsSync(fullPath) || symbols.size === 0) return {};
  const lines = fs.readFileSync(fullPath, "utf8").trim().split(/\r?\n/);
  if (lines.length < 2) return {};
  const header = parseCsvLine(lines[0]);
  const symbolIndex = header.indexOf("symbol");
  if (symbolIndex < 0) return {};
  const bySymbol = {};
  for (const line of lines.slice(1)) {
    const values = parseCsvLine(line);
    const symbol = String(values[symbolIndex] ?? "").trim();
    if (!symbols.has(symbol)) continue;
    const row = Object.fromEntries(
      header.map((key, index) => [key, normalizeValue(values[index] ?? "")])
    );
    const rows = bySymbol[symbol] ?? [];
    rows.push(row);
    if (rows.length > limitPerSymbol) rows.shift();
    bySymbol[symbol] = rows;
  }
  return bySymbol;
}

function mergeRecordMaps(primary, fallback) {
  return { ...fallback, ...primary };
}

function mergePriceMaps(primary, fallback) {
  const merged = { ...fallback };
  for (const [symbol, rows] of Object.entries(primary)) {
    const byDate = new Map((merged[symbol] ?? []).map((row) => [row.date, row]));
    for (const row of rows) byDate.set(row.date, row);
    merged[symbol] = [...byDate.values()]
      .sort((a, b) => String(a.date).localeCompare(String(b.date)))
      .slice(-260);
  }
  return merged;
}

function rowBySymbol(rows) {
  return Object.fromEntries(
    rows
      .filter((row) => row.symbol)
      .map((row) => [String(row.symbol), row])
  );
}

async function buildStockDetails({
  symbols,
  trackedUniverse,
  sp500Universe,
  positionRows,
  watchlistRows,
  discoveryCandidates,
  monitoringState,
  actionabilityCandidates,
  promotionPlan
}) {
  const trackedBySymbol = rowBySymbol(trackedUniverse);
  const sp500BySymbol = rowBySymbol(sp500Universe);
  const positionBySymbol = rowBySymbol(positionRows);
  const watchlistBySymbol = rowBySymbol(watchlistRows);
  const discoveryBySymbol = rowBySymbol(discoveryCandidates);
  const monitoringBySymbol = rowBySymbol(monitoringState);
  const actionabilityBySymbol = rowBySymbol(actionabilityCandidates);
  const promotionBySymbol = rowBySymbol(promotionPlan);
  const trackedResearch = await readLatestRowsForSymbols("data/processed/research_frame.csv", symbols);
  const sp500Research = await readLatestRowsForSymbols("data/processed/sp500_research_frame.csv", symbols);
  const metricsBySymbol = mergeRecordMaps(trackedResearch, sp500Research);
  const trackedPrices = await readPriceHistoryForSymbols("data/raw/yahoo_prices.csv", symbols);
  const sp500Prices = await readPriceHistoryForSymbols("data/raw/sp500_yahoo_prices.csv", symbols);
  const pricesBySymbol = mergePriceMaps(trackedPrices, sp500Prices);

  return Object.fromEntries(
    [...symbols].sort().map((symbol) => {
      const profile = trackedBySymbol[symbol] ?? sp500BySymbol[symbol] ?? {};
      const metrics = metricsBySymbol[symbol] ?? {};
      return [
        symbol,
        {
          profile,
          metrics,
          prices: pricesBySymbol[symbol] ?? [],
          paperPosition: positionBySymbol[symbol] ?? null,
          watchlist: watchlistBySymbol[symbol] ?? null,
          discovery: discoveryBySymbol[symbol] ?? null,
          monitoring: monitoringBySymbol[symbol] ?? null,
          actionability: actionabilityBySymbol[symbol] ?? null,
          promotion: promotionBySymbol[symbol] ?? null
        }
      ];
    })
  );
}

fs.mkdirSync(publicDataDir, { recursive: true });

const discoveryCandidates = readCsv("reports/symbol_discovery_rd/candidates.csv");
const promotionPlan = readCsv("reports/symbol_discovery_promotion_plan_candidates.csv");
const actionabilityCandidates = readCsv("reports/paper_actionability_candidates.csv");
const positionRows = readCsv("reports/paper_monitor_positions.csv");
const watchlistRows = readCsv("reports/paper_portfolio_watchlist.csv");
const orderLedgerRows = readCsv("reports/paper_portfolio_order_ledger.csv");
const tradingLedgerRows = readCsv("data/paper/paper_trading_ledger.csv");
const trackedUniverse = readCsv("data/reference/tracked_universe.csv");
const sp500Universe = readCsv("data/reference/sp500_universe.csv");
const featureImportanceBySplit = readCsv("reports/model_visibility_feature_importance_by_split.csv");
const monitoringState = topRows(readCsv("reports/symbol_discovery_rd/monitoring_state.csv"), 200, "latest_discovery_score");
const dashboardSymbols = collectSymbols(
  positionRows,
  watchlistRows,
  discoveryCandidates,
  monitoringState,
  actionabilityCandidates,
  promotionPlan,
  trackedUniverse,
  sp500Universe
);

const payload = {
  generatedAt: new Date().toISOString(),
  paper: {
    tracking: readJson("reports/paper_tracking_summary.json"),
    account: readJson("reports/paper_account_summary.json"),
    monitor: readJson("reports/paper_monitor_summary.json"),
    audit: readJson("reports/paper_realism_audit_summary.json"),
    actionability: readJson("reports/paper_actionability_summary.json"),
    history: latestHistoryRows(),
    allHistory: historyRows(),
    positions: positionRows,
    watchlist: topRows(watchlistRows, 100, "score"),
    orderLedger: orderLedgerRows.slice(0, 200),
    tradingLedger: tradingLedgerRows.slice(-200)
  },
  discovery: {
    summary: readJson("reports/symbol_discovery_rd/summary.json"),
    candidates: topRows(discoveryCandidates, 200),
    monitoringState,
    exclusions: readCsv("reports/symbol_discovery_rd/exclusions.csv").slice(0, 200),
    promotionCandidates: readCsv("reports/symbol_discovery_rd/promotion_candidates.csv"),
    lanes: {
      momentumBreakouts: topRows(readCsv("reports/discovery_sp500_ex_watchlist/lane_momentum_breakouts.csv"), 75),
      sectorLeaders: topRows(readCsv("reports/discovery_sp500_ex_watchlist/lane_sector_leaders.csv"), 75),
      valueRecoveries: topRows(readCsv("reports/discovery_sp500_ex_watchlist/lane_value_recoveries.csv"), 75),
      volumeAnomalies: topRows(readCsv("reports/discovery_sp500_ex_watchlist/lane_volume_anomalies.csv"), 75)
    },
    availableUniverse: {
      summary: readJson("reports/discovery_available_universe/summary.json"),
      scoredCandidates: topRows(readCsv("reports/discovery_available_universe/scored_candidates.csv"), 100)
    },
    currentWatchlistExcluded: {
      summary: readJson("reports/discovery_current_watchlist_excluded/summary.json"),
      scoredCandidates: topRows(readCsv("reports/discovery_current_watchlist_excluded/scored_candidates.csv"), 100)
    }
  },
  promotion: {
    summary: readJson("reports/symbol_discovery_promotion_plan_summary.json"),
    plan: topRows(promotionPlan, 50)
  },
  actionability: {
    candidates: topRows(actionabilityCandidates, 150, "score")
  },
  model: {
    comparison: readCsv("reports/model_comparison.csv"),
    scoreBuckets: readCsv("reports/model_visibility_score_buckets.csv"),
    predictionDrift: readCsv("reports/model_visibility_prediction_drift.csv"),
    paperPickExplanations: topRows(readCsv("reports/model_visibility_paper_pick_explanations.csv"), 100, "score"),
    featureImportance: topRows(featureImportanceBySplit, 100, "importance"),
    featureImportanceSummary: aggregateFeatureImportance(featureImportanceBySplit),
    sectorContribution: readCsv("reports/model_visibility_sector_contribution.csv"),
    randomForestFeatureImportance: topRows(readCsv("reports/random_forest_feature_importance.csv"), 100, "importance"),
    baselineMetadata: readJson("reports/baseline_model_metadata.json"),
    randomForestMetadata: readJson("reports/random_forest_model_metadata.json")
  },
  research: {
    featureAblation: topRows(readCsv("reports/rd_feature_ablation.csv"), 100, "risk_backtest_sharpe"),
    portfolioRules: topRows(readCsv("reports/rd_portfolio_rules.csv"), 100, "sharpe"),
    experimentLeaderboard: topRows(readCsv("reports/experiment_leaderboard.csv"), 100, "backtest_sharpe"),
    riskExperimentLeaderboard: topRows(readCsv("reports/risk_experiment_leaderboard.csv"), 100, "risk_backtest_sharpe"),
    scoreThresholdSensitivity: readCsv("reports/exec_score_threshold_sensitivity.csv"),
    eventRiskSensitivity: readCsv("reports/exec_event_risk_sensitivity.csv")
  },
  backtests: {
    topExperimentMonthlyReturns: readCsv("reports/top_experiment_monthly_returns.csv"),
    topExperimentSymbolContributions: topRows(readCsv("reports/top_experiment_symbol_contributions.csv"), 100, "gross_contribution"),
    topExperimentDailyReturns: readCsv("reports/top_experiment_daily_returns.csv").slice(-200),
    topExperimentTradeLedger: readCsv("reports/top_experiment_trade_ledger.csv").slice(-200),
    paperStyleDailyEquity: readCsv("reports/paper_style_backtest_daily_equity.csv").slice(-200),
    paperStyleLedger: readCsv("reports/paper_style_backtest_ledger.csv").slice(-200),
    paperStyleSummary: readJson("reports/paper_style_backtest_summary.json"),
    execTopMetadata: readJson("reports/exec_top_experiment_min_score_001_metadata.json")
  },
  universe: {
    tracked: trackedUniverse,
    sp500: sp500Universe,
    summary: {
      trackedCount: trackedUniverse.length,
      sp500Count: sp500Universe.length
    }
  },
  runs: {
    history: historyRows(),
    latestRun: historyRows().at(-1) ?? {}
  },
  stocks: await buildStockDetails({
    symbols: dashboardSymbols,
    trackedUniverse,
    sp500Universe,
    positionRows,
    watchlistRows,
    discoveryCandidates,
    monitoringState,
    actionabilityCandidates,
    promotionPlan
  }),
  reports: {
    dailyOps: readText("reports/daily_ops_review.md"),
    paperTracking: readText("reports/paper_tracking_report.md"),
    symbolDiscovery: readText("reports/symbol_discovery_rd/report.md"),
    promotionPlan: readText("reports/symbol_discovery_promotion_plan_report.md"),
    modelVisibility: readText("reports/model_visibility_summary.md"),
    paperMonitor: readText("reports/paper_monitor_report.md"),
    paperRealismAudit: readText("reports/paper_realism_audit_report.md"),
    paperStyleBacktest: readText("reports/paper_style_backtest_report.md"),
    research: readText("reports/rd_summary.md")
  }
};

fs.writeFileSync(
  path.join(publicDataDir, "dashboard.json"),
  `${JSON.stringify(payload, null, 2)}\n`
);

console.log(`wrote dashboard data to ${path.relative(repoRoot, publicDataDir)}/dashboard.json`);
