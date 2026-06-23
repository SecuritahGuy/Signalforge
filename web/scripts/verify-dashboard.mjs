import { chromium } from "playwright";

const url = process.env.DASHBOARD_URL ?? "http://127.0.0.1:5173/";
const viewports = [
  { name: "desktop", width: 1440, height: 1000 },
  { name: "mobile", width: 390, height: 900 }
];

const browser = await chromium.launch({ headless: true });
const results = [];

for (const viewport of viewports) {
  const page = await browser.newPage({
    viewport: { width: viewport.width, height: viewport.height },
    deviceScaleFactor: 1
  });
  const consoleErrors = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  await page.goto(url, { waitUntil: "networkidle" });
  const pageChecks = {};
  for (const pageName of [
    "Overview",
    "Stock Detail",
    "Paper",
    "Positions",
    "Discovery",
    "Promotion",
    "Model",
    "R&D",
    "Backtests",
    "Universe",
    "Runs",
    "Reports"
  ]) {
    await page.getByRole("button", { name: pageName }).click();
    await page.waitForSelector(`[data-page="${pageName}"]`);
    pageChecks[pageName] = await page.evaluate((expectedPage) => {
      const activeNav = document.querySelector(".nav-item.active")?.textContent?.trim() ?? "";
      const pageView = document.querySelector(".page-view")?.getAttribute("data-page") ?? "";
      const text = document.body.innerText;
      return {
        activeNav,
        pageView,
        hasExpectedPage: pageView === expectedPage,
        hasVisibleContent: text.length > 200
      };
    }, pageName);
  }
  await page.getByRole("button", { name: "Overview" }).click();
  await page.waitForSelector('[data-page="Overview"]');
  const firstSymbolLink = page.locator(".candidate-table .symbol-link").first();
  const stockSymbol = (await firstSymbolLink.innerText()).trim();
  await firstSymbolLink.click();
  await page.waitForSelector('[data-page="Stock Detail"]');
  const stockClick = await page.evaluate((symbol) => {
    const activeNav = document.querySelector(".nav-item.active")?.textContent?.trim() ?? "";
    const text = document.body.innerText;
    return {
      activeNav,
      clickedSymbol: symbol,
      routedToStock: activeNav === "Stock Detail",
      hasSymbol: text.includes(symbol),
      hasPriceTrend: text.includes("Price Trend"),
      hasRisk: text.includes("Risk & Liquidity"),
      hasSystemContext: text.includes("System Context")
    };
  }, stockSymbol);
  await page.getByRole("button", { name: "Overview" }).click();
  await page.waitForSelector('[data-page="Overview"]');
  const checks = await page.evaluate(() => {
    const text = document.body.innerText;
    const root = document.querySelector("#root");
    const shell = document.querySelector(".app-shell");
    const rect = shell?.getBoundingClientRect();
    return {
      title: document.title,
      hasSignalForge: text.includes("SignalForge Operations"),
      hasDiscovery: text.includes("Top Discovery Candidates"),
      hasPromotion: text.includes("Promotion Plan"),
      kpiCount: document.querySelectorAll(".kpi-card").length,
      panelCount: document.querySelectorAll(".panel").length,
      tableRows: document.querySelectorAll("tbody tr").length,
      rootChildren: root?.children.length ?? 0,
      shellWidth: Math.round(rect?.width ?? 0),
      shellHeight: Math.round(rect?.height ?? 0)
    };
  });
  const firstCandidateSymbol = await page.locator(".candidate-table tbody tr").first().locator("td").first().innerText();
  await page.locator(".candidate-table tbody tr").first().click();
  await page.waitForSelector('[data-page="Discovery"]');
  const candidateClick = await page.evaluate((symbol) => {
    const activeNav = document.querySelector(".nav-item.active")?.textContent?.trim() ?? "";
    const selectedSymbol = document.querySelector(".candidate-table tbody tr.selected td .symbol-link")?.textContent?.trim() ?? "";
    const detailSymbol = document.querySelector(".detail-title strong")?.textContent?.trim() ?? "";
    return {
      activeNav,
      selectedSymbol,
      detailSymbol,
      clickedSymbol: symbol,
      routedToDiscovery: activeNav === "Discovery",
      selectedMatches: selectedSymbol === symbol && detailSymbol === symbol
    };
  }, firstCandidateSymbol.trim());
  const screenshot = `/private/tmp/signalforge-dashboard-${viewport.name}.png`;
  await page.screenshot({ path: screenshot, fullPage: true });
  await page.close();
  results.push({ viewport: viewport.name, screenshot, consoleErrors, pageChecks, stockClick, candidateClick, checks });
}

await browser.close();

const failures = results.flatMap((result) => {
  const checks = result.checks;
  const failed = [];
  if (!checks.hasSignalForge) failed.push(`${result.viewport}: missing title text`);
  if (!checks.hasDiscovery) failed.push(`${result.viewport}: missing discovery section`);
  if (!checks.hasPromotion) failed.push(`${result.viewport}: missing promotion section`);
  if (checks.kpiCount < 5) failed.push(`${result.viewport}: expected at least 5 KPI cards`);
  if (checks.panelCount < 5) failed.push(`${result.viewport}: expected at least 5 panels`);
  if (checks.rootChildren < 1) failed.push(`${result.viewport}: React root did not render`);
  for (const [pageName, pageCheck] of Object.entries(result.pageChecks)) {
    if (!pageCheck.hasExpectedPage) failed.push(`${result.viewport}: ${pageName} page did not activate`);
    if (!pageCheck.hasVisibleContent) failed.push(`${result.viewport}: ${pageName} page has too little content`);
  }
  if (!result.candidateClick.routedToDiscovery) {
    failed.push(`${result.viewport}: candidate click did not route to Discovery`);
  }
  if (!result.candidateClick.selectedMatches) {
    failed.push(`${result.viewport}: candidate click did not preserve selected symbol`);
  }
  if (!result.stockClick.routedToStock) {
    failed.push(`${result.viewport}: symbol click did not route to Stock Detail`);
  }
  if (!result.stockClick.hasSymbol || !result.stockClick.hasPriceTrend || !result.stockClick.hasRisk || !result.stockClick.hasSystemContext) {
    failed.push(`${result.viewport}: stock detail page is missing expected content`);
  }
  if (result.consoleErrors.length) failed.push(`${result.viewport}: console errors present`);
  return failed;
});

console.log(JSON.stringify({ url, results, failures }, null, 2));

if (failures.length) {
  process.exitCode = 1;
}
