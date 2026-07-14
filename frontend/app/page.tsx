"use client";

import { useEffect, useMemo, useState } from "react";
import PlayerProfileSection from "./components/PlayerProfileSection";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type PlayerSummary = {
  player_id: number;
  display_name: string;
  position?: string | null;
  current_form_rating?: number | null;
  rolling_3_match_rating?: number | null;
  rolling_20_match_rating?: number | null;
  latest_valuation_date?: string | null;
  current_market_value_eur?: number | null;
  estimated_value_eur?: number | null;
  estimated_lower_eur?: number | null;
  estimated_upper_eur?: number | null;
  predicted_pct_change?: number | null;
  probability_value_increase?: number | null;
  valuation_model_version?: string | null;
  confidence?: number | null;
  direction?: string | null;
  refreshed_at?: string | null;
};

type ValuationPoint = {
  valuation_date: string;
  value_eur: number;
  source?: string;
};

type MatchImpact = {
  match_id: number;
  match_datetime: string;
  rating?: number | null;
  minutes?: number | null;
  explanation?: string;
  performance_impact_score?: number | null;
  impact_direction?: string;
  estimated_value_delta_eur?: number | null;
};

type PlayerDetail = PlayerSummary & {
  valuation_history: ValuationPoint[];
  match_impacts: MatchImpact[];
};

type View = "player" | "watchlist" | "catalog" | "health";

function money(value?: number | null) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return new Intl.NumberFormat("en", {
    style: "currency",
    currency: "EUR",
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(Number(value));
}

function parseDate(value: string) {
  return new Date(value.includes("T") ? value : `${value}T12:00:00Z`);
}

function shortDate(value?: string | null) {
  if (!value) return "—";
  return new Intl.DateTimeFormat("en", {
    day: "numeric",
    month: "short",
  }).format(parseDate(value));
}

function fullDate(value?: string | null) {
  if (!value) return "Date unavailable";
  return new Intl.DateTimeFormat("en", {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(parseDate(value));
}

function initials(name?: string) {
  return (name ?? "Player")
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0])
    .join("")
    .toUpperCase();
}

function signedPercent(value?: number | null) {
  if (value == null || !Number.isFinite(value)) return "—";
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}%`;
}

function directionCopy(direction?: string | null) {
  if (direction === "rising") return "Likely rising";
  if (direction === "falling") return "Likely falling";
  if (direction === "stable") return "Broadly stable";
  return "Not yet projected";
}

function trendClass(value?: number | null) {
  if (value == null) return "neutral";
  if (value > 0.1) return "positive";
  if (value < -0.1) return "negative";
  return "neutral";
}

function ArrowIcon({ direction = "right" }: { direction?: "up" | "down" | "right" }) {
  const path = direction === "up" ? "M5 15 15 5m0 0H8m7 0v7" : direction === "down" ? "M5 5l10 10m0 0V8m0 7H8" : "M4 10h12m0 0-5-5m5 5-5 5";
  return <svg viewBox="0 0 20 20" aria-hidden="true"><path d={path} /></svg>;
}

function MarketValueChart({
  history,
  projection,
  lower,
  upper,
}: {
  history: ValuationPoint[];
  projection?: number | null;
  lower?: number | null;
  upper?: number | null;
}) {
  if (!history.length) {
    return <div className="empty compact">No Transfermarkt valuation history is available.</div>;
  }

  const chronological = [...history]
    .sort((a, b) => parseDate(a.valuation_date).getTime() - parseDate(b.valuation_date).getTime())
    .slice(-10);
  const forecastAvailable = projection != null;
  const allValues = [
    ...chronological.map((point) => Number(point.value_eur)),
    ...(forecastAvailable ? [Number(projection)] : []),
    ...(lower != null ? [Number(lower)] : []),
    ...(upper != null ? [Number(upper)] : []),
  ];
  const minimumValue = Math.min(...allValues);
  const maximumValue = Math.max(...allValues);
  const padding = Math.max((maximumValue - minimumValue) * 0.18, maximumValue * 0.04, 1);
  const domainMinimum = Math.max(0, minimumValue - padding);
  const domainMaximum = maximumValue + padding;
  const valueRange = Math.max(domainMaximum - domainMinimum, 1);
  const width = 920;
  const height = 300;
  const left = 68;
  const right = 36;
  const top = 24;
  const bottom = 48;
  const historicalRight = forecastAvailable ? width - right - 108 : width - right;
  const xStep = chronological.length > 1
    ? (historicalRight - left) / (chronological.length - 1)
    : 0;
  const xForIndex = (index: number) => chronological.length > 1
    ? left + index * xStep
    : left + (historicalRight - left) / 2;
  const yFor = (value: number) => top + ((domainMaximum - value) / valueRange) * (height - top - bottom);
  const historicalPoints = chronological
    .map((point, index) => `${xForIndex(index)},${yFor(Number(point.value_eur))}`)
    .join(" ");
  const forecastX = width - right;
  const latest = chronological[chronological.length - 1];
  const latestX = xForIndex(chronological.length - 1);
  const latestY = yFor(Number(latest.value_eur));
  const projectionY = forecastAvailable ? yFor(Number(projection)) : null;
  const gridValues = [0, 0.5, 1].map((ratio) => domainMaximum - ratio * valueRange);

  return (
    <div className="chartWrap">
      <svg className="marketChart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Transfermarkt valuation history and model projection">
        <defs>
          <linearGradient id="valueArea" x1="0" y1="0" x2="0" y2="1">
            <stop stopColor="#3157d5" stopOpacity=".16" />
            <stop offset="1" stopColor="#3157d5" stopOpacity="0" />
          </linearGradient>
        </defs>
        {gridValues.map((value) => {
          const y = yFor(value);
          return (
            <g key={value}>
              <line x1={left} y1={y} x2={width - right} y2={y} className="chartGrid" />
              <text x={left - 12} y={y + 4} textAnchor="end" className="chartTick">{money(value)}</text>
            </g>
          );
        })}
        {chronological.length > 1 && (
          <path d={`M${historicalPoints} L${historicalRight},${height - bottom} L${left},${height - bottom} Z`} className="chartArea" />
        )}
        <polyline points={historicalPoints} className="actualLine" />
        {chronological.map((point, index) => (
          <g key={`${point.valuation_date}-${index}`}>
            <circle cx={xForIndex(index)} cy={yFor(Number(point.value_eur))} r={index === chronological.length - 1 ? 5.5 : 4} className="actualDot">
              <title>{`${fullDate(point.valuation_date)}: ${money(point.value_eur)}`}</title>
            </circle>
            {(index === 0 || index === chronological.length - 1) && (
              <text x={xForIndex(index)} y={height - 18} textAnchor={index === 0 ? "start" : "middle"} className="chartDate">
                {shortDate(point.valuation_date)}
              </text>
            )}
          </g>
        ))}
        {forecastAvailable && projectionY != null && (
          <g>
            <line x1={latestX} y1={latestY} x2={forecastX} y2={projectionY} className="forecastLine" />
            {lower != null && upper != null && (
              <g>
                <line x1={forecastX} y1={yFor(Number(upper))} x2={forecastX} y2={yFor(Number(lower))} className="intervalLine" />
                <line x1={forecastX - 9} y1={yFor(Number(upper))} x2={forecastX + 9} y2={yFor(Number(upper))} className="intervalCap" />
                <line x1={forecastX - 9} y1={yFor(Number(lower))} x2={forecastX + 9} y2={yFor(Number(lower))} className="intervalCap" />
              </g>
            )}
            <rect x={forecastX - 6} y={projectionY - 6} width="12" height="12" rx="2" transform={`rotate(45 ${forecastX} ${projectionY})`} className="forecastDot">
              <title>{`Model projection: ${money(projection)}`}</title>
            </rect>
            <text x={forecastX} y={height - 18} textAnchor="middle" className="chartDate forecastDate">Projection</text>
          </g>
        )}
      </svg>
    </div>
  );
}

function FormChart({ matches }: { matches: MatchImpact[] }) {
  const points = [...matches]
    .filter((match): match is MatchImpact & { rating: number } => match.rating != null)
    .sort((a, b) => parseDate(a.match_datetime).getTime() - parseDate(b.match_datetime).getTime())
    .slice(-10);
  if (!points.length) return <div className="empty compact">No rated appearances are available.</div>;

  const width = 920;
  const height = 260;
  const left = 54;
  const right = 28;
  const top = 22;
  const bottom = 48;
  const minimum = 4;
  const maximum = 10;
  const xStep = points.length > 1 ? (width - left - right) / (points.length - 1) : 0;
  const xFor = (index: number) => points.length > 1 ? left + index * xStep : width / 2;
  const yFor = (value: number) => top + ((maximum - value) / (maximum - minimum)) * (height - top - bottom);
  const linePoints = points.map((point, index) => `${xFor(index)},${yFor(point.rating)}`).join(" ");

  return (
    <div className="chartWrap">
      <svg className="formChart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Recent position-adjusted match ratings">
        {[6, 8, 10].map((value) => (
          <g key={value}>
            <line x1={left} y1={yFor(value)} x2={width - right} y2={yFor(value)} className={value === 6 ? "baselineLine" : "chartGrid"} />
            <text x={left - 12} y={yFor(value) + 4} textAnchor="end" className="chartTick">{value.toFixed(1)}</text>
          </g>
        ))}
        <polyline points={linePoints} className="formLine" />
        {points.map((point, index) => (
          <g key={`${point.match_id}-${index}`}>
            <circle cx={xFor(index)} cy={yFor(point.rating)} r="5" className={`formDot ${trendClass(point.rating - 6)}`}>
              <title>{`${fullDate(point.match_datetime)} · Rating ${point.rating.toFixed(2)} · ${point.explanation ?? "Position-adjusted performance"}`}</title>
            </circle>
            <text x={xFor(index)} y={height - 18} textAnchor="middle" className="chartDate">{shortDate(point.match_datetime)}</text>
          </g>
        ))}
        <text x={width - right} y={yFor(6) - 8} textAnchor="end" className="baselineLabel">6.0 baseline</text>
      </svg>
    </div>
  );
}

function JsonPanel({ title, value }: { title: string; value: unknown }) {
  return (
    <article className="wide technicalPanel">
      <p className="label">{title}</p>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </article>
  );
}

export default function Home() {
  const [players, setPlayers] = useState<PlayerSummary[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [player, setPlayer] = useState<PlayerDetail | null>(null);
  const [query, setQuery] = useState("");
  const [view, setView] = useState<View>("player");
  const [watchlist, setWatchlist] = useState<Set<number>>(new Set());
  const [catalog, setCatalog] = useState<unknown>(null);
  const [lineage, setLineage] = useState<unknown>(null);
  const [runs, setRuns] = useState<Record<string, unknown>[]>([]);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/api/players`)
      .then((response) => {
        if (!response.ok) throw new Error(`Player API returned ${response.status}`);
        return response.json();
      })
      .then((payload) => {
        const rows = (payload.data ?? []) as PlayerSummary[];
        setPlayers(rows);
        if (rows.length) setSelectedId(rows[0].player_id);
      })
      .catch((reason) => setError(String(reason)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (selectedId == null) return;
    setError(null);
    setDetailLoading(true);
    fetch(`${API_URL}/api/players/${selectedId}`)
      .then((response) => {
        if (!response.ok) throw new Error(`Player detail API returned ${response.status}`);
        return response.json();
      })
      .then((payload) => setPlayer(payload.data as PlayerDetail))
      .catch((reason) => setError(String(reason)))
      .finally(() => setDetailLoading(false));
  }, [selectedId]);

  useEffect(() => {
    if (view === "catalog" && catalog == null) {
      Promise.all([
        fetch(`${API_URL}/api/catalog`).then((response) => response.json()),
        fetch(`${API_URL}/api/lineage`).then((response) => response.json()),
      ])
        .then(([catalogPayload, lineagePayload]) => {
          setCatalog(catalogPayload.data);
          setLineage(lineagePayload.data);
        })
        .catch((reason) => setError(String(reason)));
    }
    if (view === "health") {
      fetch(`${API_URL}/api/pipeline-runs`)
        .then((response) => response.json())
        .then((payload) => setRuns(payload.data ?? []))
        .catch((reason) => setError(String(reason)));
    }
  }, [view, catalog]);

  const filteredPlayers = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return players.filter((candidate) =>
      candidate.display_name.toLowerCase().includes(normalized),
    );
  }, [players, query]);

  const recentMatches = (player?.match_impacts ?? []).slice(0, 10);
  const recentFive = recentMatches
    .filter((match): match is MatchImpact & { rating: number } => match.rating != null)
    .slice(0, 5);
  const history = player?.valuation_history ?? [];
  const latestHistory = [...history].sort(
    (a, b) => parseDate(a.valuation_date).getTime() - parseDate(b.valuation_date).getTime(),
  );
  const watchedPlayers = players.filter((candidate) => watchlist.has(candidate.player_id));
  const formTrend = player?.rolling_3_match_rating != null && player?.rolling_20_match_rating != null
    ? player.rolling_3_match_rating - player.rolling_20_match_rating
    : null;
  const projectedChange = player?.estimated_value_eur != null && player?.current_market_value_eur
    ? player.estimated_value_eur / player.current_market_value_eur - 1
    : player?.predicted_pct_change ?? null;
  const projectedDelta = player?.estimated_value_eur != null && player?.current_market_value_eur != null
    ? player.estimated_value_eur - player.current_market_value_eur
    : null;
  const projectionAvailable = player?.estimated_value_eur != null;
  const direction = player?.direction ?? "unscored";

  function toggleWatchlist() {
    if (!player) return;
    setWatchlist((current) => {
      const next = new Set(current);
      if (next.has(player.player_id)) next.delete(player.player_id);
      else next.add(player.player_id);
      return next;
    });
  }

  function submitPlayerSearch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalized = query.trim().toLowerCase();
    if (!normalized) return;
    const candidate = players.find(
      (item) => item.display_name.toLowerCase() === normalized,
    ) ?? filteredPlayers[0];
    if (candidate) {
      setSelectedId(candidate.player_id);
      setQuery(candidate.display_name);
      setView("player");
    }
  }

  const viewTitle = view === "player"
    ? "Player outlook"
    : view === "watchlist"
      ? "Squad watchlist"
      : view === "catalog"
        ? "Data catalog & lineage"
        : "Pipeline health";

  return (
    <main>
      <aside>
        <div className="brand"><span>MV</span><div>Market Value Pulse<small>Football intelligence</small></div></div>
        <nav aria-label="Primary navigation">
          <button className={view === "player" ? "active" : ""} onClick={() => setView("player")}>Player outlook</button>
          <button className={view === "watchlist" ? "active" : ""} onClick={() => setView("watchlist")}>Watchlist <small>{watchlist.size}</small></button>
          <button className={view === "catalog" ? "active" : ""} onClick={() => setView("catalog")}>Data & lineage</button>
          <button className={view === "health" ? "active" : ""} onClick={() => setView("health")}>Pipeline health</button>
        </nav>
        <div className="source"><i className={error ? "offline" : ""} /> {error ? "API unavailable" : "Data service connected"}<br /><b>{players.length} mapped players</b></div>
      </aside>

      <section className="content">
        <header>
          <div><p className="eyebrow">MARKET VALUE INTELLIGENCE</p><h1>{viewTitle}</h1></div>
          {view === "player" && (
            <form className="search" onSubmit={submitPlayerSearch}>
              <label htmlFor="player-search">Find a player</label>
              <div className="searchControls">
                <input id="player-search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search by name" />
                <select aria-label="Select player" value={selectedId ?? ""} onChange={(event) => setSelectedId(Number(event.target.value))}>
                  {filteredPlayers.map((candidate) => <option key={candidate.player_id} value={candidate.player_id}>{candidate.display_name}</option>)}
                </select>
                <button type="submit" disabled={filteredPlayers.length === 0}>View</button>
              </div>
            </form>
          )}
        </header>

        {error && <div className="error"><b>We could not load the latest data.</b><span>{error}</span></div>}
        {(loading || detailLoading) && view === "player" && <div className="loading"><i /><span>Loading player outlook…</span></div>}
        {view === "player" && !loading && !detailLoading && !player && !error && <div className="empty">No serving data yet. Run the serving build and database load.</div>}

        {view === "player" && player && !detailLoading && (
          <>
            <section className="playerHeader">
              <div className="avatar">{initials(player.display_name)}</div>
              <div className="identity">
                <div className="identityLine"><h2>{player.display_name}</h2><span>{player.position ?? "Position unavailable"}</span></div>
                <p>WhoScored ID {player.player_id} <i /> Latest match {fullDate(recentMatches[0]?.match_datetime)}</p>
              </div>
              <button className="watch" onClick={toggleWatchlist}>{watchlist.has(player.player_id) ? "✓ On watchlist" : "+ Add to watchlist"}</button>
            </section>

            <section className="storyChapter">
              <div className="chapterHeader">
                <span>01</span>
                <div>
                  <p className="label">CURRENT OUTLOOK</p>
                  <h2>Value and recent form</h2>
                  <p>The observed market baseline, the model forecast and the performances currently driving the outlook.</p>
                </div>
              </div>

              <div className="outlookGrid">
                <article className="outlookCard valuationOutlook">
                  <div className="outlookCardHead">
                    <div>
                      <p className="label">MARKET VALUE</p>
                      <h3>Published value and model projection</h3>
                    </div>
                    {projectionAvailable && <span className={`statusPill ${direction}`}>{directionCopy(direction)}</span>}
                  </div>

                  <div className="valuationHero">
                    <div className="valuationMetric">
                      <span>Latest Transfermarkt value</span>
                      <strong>{money(player.current_market_value_eur)}</strong>
                      <small>Published {fullDate(player.latest_valuation_date)}</small>
                    </div>

                    <div className="valuationTransition">
                      <div className="valueArrow"><ArrowIcon direction={projectedChange == null ? "right" : projectedChange >= 0 ? "up" : "down"} /></div>
                      <b className={trendClass(projectedChange)}>{signedPercent(projectedChange)}</b>
                    </div>

                    <div className="valuationMetric projectedMetric">
                      <span>Model projection</span>
                      <strong>{money(player.estimated_value_eur)}</strong>
                      <small className={trendClass(projectedChange)}>
                        {projectedDelta == null ? "Awaiting projected delta" : `${projectedDelta >= 0 ? "+" : ""}${money(projectedDelta)} from the published baseline`}
                      </small>
                    </div>
                  </div>

                  {projectionAvailable ? (
                    <>
                      <div className="rangeStrip">
                        <span>90% projected range</span>
                        <b>{money(player.estimated_lower_eur)} — {money(player.estimated_upper_eur)}</b>
                      </div>
                      <div className="probabilityMeter">
                        <div>
                          <span>Probability of an increase</span>
                          <b>{player.probability_value_increase == null ? "—" : `${Math.round(player.probability_value_increase * 100)}%`}</b>
                        </div>
                        <span><i style={{ width: `${Math.max(0, Math.min(100, Math.round((player.probability_value_increase ?? 0) * 100)))}%` }} /></span>
                      </div>
                    </>
                  ) : (
                    <div className="noProjection"><b>No current projection</b><span>A projection appears after eligible EPL match data is scored with an active valuation model.</span></div>
                  )}

                  <div className="embeddedChart">
                    <div className="embeddedChartHead">
                      <span>Valuation trajectory</span>
                      <div className="legend"><span><i className="actualLegend" /> Published</span><span><i className="forecastLegend" /> Projection</span></div>
                    </div>
                    <MarketValueChart history={latestHistory} projection={player.estimated_value_eur} lower={player.estimated_lower_eur} upper={player.estimated_upper_eur} />
                  </div>

                  <p className="outlookFootnote">The projection is not a new Transfermarkt valuation. It estimates current value from eligible domestic EPL performances after the latest published observation.</p>
                </article>

                <article className="outlookCard formOutlook">
                  <div className="outlookCardHead">
                    <div>
                      <p className="label">PLAYER FORM</p>
                      <h3>Recent domestic performance</h3>
                    </div>
                    <span className={`statusPill ${trendClass(formTrend)}`}>{formTrend == null ? "Trend unavailable" : formTrend > 0.1 ? "Improving" : formTrend < -0.1 ? "Declining" : "Steady"}</span>
                  </div>

                  <div className="formHero">
                    <div className="formScore">
                      <strong>{player.current_form_rating?.toFixed(2) ?? "—"}</strong>
                      <span>/ 10</span>
                      <small>90-day minutes-weighted form</small>
                    </div>

                    <div className="formQuickStats">
                      <div><span>Last 3</span><b>{player.rolling_3_match_rating?.toFixed(2) ?? "—"}</b></div>
                      <div><span>Last 20</span><b>{player.rolling_20_match_rating?.toFixed(2) ?? "—"}</b></div>
                      <div><span>Trend</span><b className={trendClass(formTrend)}>{formTrend == null ? "—" : `${formTrend >= 0 ? "+" : ""}${formTrend.toFixed(2)}`}</b></div>
                    </div>
                  </div>

                  <div className="embeddedChart formEmbeddedChart">
                    <div className="embeddedChartHead">
                      <span>Last 10 rated appearances</span>
                      <small>6.0 is the neutral baseline</small>
                    </div>
                    <FormChart matches={recentMatches} />
                  </div>

                  <div className="recentRatingRow" aria-label="Last five match ratings">
                    {recentFive.length ? [...recentFive].reverse().map((match) => (
                      <div key={match.match_id} className={trendClass(match.rating - 6)} title={`${fullDate(match.match_datetime)} · ${match.explanation ?? "Position-adjusted performance"}`}>
                        <span>{shortDate(match.match_datetime)}</span>
                        <b>{match.rating.toFixed(1)}</b>
                      </div>
                    )) : <span className="muted">No recent rated matches</span>}
                  </div>
                </article>
              </div>
            </section>

            <section className="storyChapter seasonStory">
              <div className="chapterHeader">
                <span>02</span>
                <div>
                  <p className="label">SEASON PERFORMANCE</p>
                  <h2>What type of player is he this season?</h2>
                  <p>Role-relative output across the full season, separated from short-term form and market-value forecasting.</p>
                </div>
              </div>

              <PlayerProfileSection
                playerId={player.player_id}
                onSelectPlayer={(playerId) => {
                  setSelectedId(playerId);
                  setView("player");
                  window.scrollTo({ top: 0, behavior: "smooth" });
                }}
              />
            </section>

            <article className="sectionCard wide matchSection">
              <div className="articleHead"><div><p className="label">RECENT MATCH DRIVERS</p><h3>What shaped the player&apos;s form signal</h3><p>Component explanations show why each performance rated above or below the 6.0 baseline.</p></div><span className="method">Euro deltas appear only when replay-scored</span></div>
              <div className="matchTable">
                <div className="matchTableHead"><span>Match</span><span>Rating</span><span>Main drivers</span><span>Form impact</span></div>
                {recentMatches.slice(0, 8).map((match) => {
                  const impact = match.performance_impact_score;
                  return (
                    <div className="matchRow" key={match.match_id}>
                      <div><b>Match {match.match_id}</b><small>{fullDate(match.match_datetime)}{match.minutes != null ? ` · ${Math.round(match.minutes)} min` : ""}</small></div>
                      <div className="ratingCell"><b>{match.rating?.toFixed(2) ?? "—"}</b><span>out of 10</span></div>
                      <div className="why">{match.explanation ?? "Position-adjusted performance"}</div>
                      <div className={`impactCell ${trendClass(impact)}`}><b>{impact == null ? "—" : `${impact >= 0 ? "+" : ""}${impact.toFixed(2)}`}</b><span>{match.estimated_value_delta_eur != null ? `Replay Δ ${money(match.estimated_value_delta_eur)}` : match.impact_direction ?? "form signal"}</span></div>
                    </div>
                  );
                })}
              </div>
            </article>

            <section className="explanationGrid">
              <article className="sectionCard explanationCard">
                <p className="label">HOW TO READ THE FORECAST</p>
                <h3>One projection, with uncertainty</h3>
                <p>The latest Transfermarkt valuation is the observed baseline. The projected value applies the active model to domestic match performance recorded since that date.</p>
                <ul><li>The midpoint is the model&apos;s current estimate.</li><li>The 90% range communicates forecast uncertainty.</li><li>The direction probability shows how strongly the model leans toward an increase.</li></ul>
              </article>
              <article className="sectionCard limitationsCard">
                <p className="label">KNOWN LIMITATIONS</p>
                <h3>What the model does not include</h3>
                <ul><li>International and national-team matches</li><li>Injuries not represented in match data</li><li>Contract duration, salary and release clauses</li><li>Transfer demand and club negotiating position</li><li>Proprietary scouting and off-field information</li></ul>
                <p className="scopeNote">Current performance scope: domestic EPL matches only.</p>
              </article>
            </section>

            <details className="modelDetails">
              <summary>Model and data details</summary>
              <div><span>Valuation model</span><b>{player.valuation_model_version ?? "No active model"}</b><span>Forecast refreshed</span><b>{fullDate(player.refreshed_at)}</b><span>Valuation source</span><b>Transfermarkt</b><span>Performance source</span><b>WhoScored · EPL</b></div>
            </details>
          </>
        )}

        {view === "watchlist" && <div className="cards">{watchedPlayers.length ? watchedPlayers.map((candidate) => <button className="playerCard" key={candidate.player_id} onClick={() => { setSelectedId(candidate.player_id); setView("player"); }}><span>{candidate.position ?? "Position unavailable"}</span><b>{candidate.display_name}</b><div><strong>{money(candidate.current_market_value_eur)}</strong><small>Published value</small></div><div><strong>{candidate.current_form_rating?.toFixed(2) ?? "—"}</strong><small>Current form</small></div></button>) : <div className="empty">Add players from Player Outlook to build a local shortlist.</div>}</div>}

        {view === "catalog" && <div className="technicalGrid"><JsonPanel title="DATA CATALOG" value={catalog ?? "Loading…"} /><JsonPanel title="LINEAGE" value={lineage ?? "Loading…"} /></div>}

        {view === "health" && <article className="sectionCard wide"><p className="label">RECENT PIPELINE RUNS</p><div className="runs">{runs.length ? runs.map((run, index) => <div className="run" key={String(run.run_id ?? index)}><b>{String(run.pipeline ?? "pipeline")}</b><span className={String(run.status ?? "unknown")}>{String(run.status ?? "unknown")}</span><small>{String(run.started_at ?? "")}</small></div>) : <div className="empty">No database pipeline runs have been recorded yet.</div>}</div></article>}
      </section>
    </main>
  );
}
