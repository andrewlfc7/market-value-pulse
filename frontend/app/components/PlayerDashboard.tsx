"use client";

import PlayerProfileSection from "./PlayerProfileSection";
import styles from "./PlayerDashboard.module.css";
import {
  directionCopy,
  fullDate,
  money,
  parseDate,
  ratingClass,
  shortDate,
  signedPercent,
  trendClass,
} from "../lib/player-format";
import type {
  MatchImpact,
  PlayerDetail,
  ValuationPoint,
} from "../lib/player-types";

type Props = {
  player: PlayerDetail;
  watched: boolean;
  onToggleWatchlist: () => void;
  onSelectPlayer: (playerId: number) => void;
};

function toneClass(tone: string, base: string) {
  return `${base} ${styles[tone] ?? ""}`;
}

function ArrowIcon({
  direction = "right",
}: {
  direction?: "up" | "down" | "right";
}) {
  const path =
    direction === "up"
      ? "M5 15 15 5m0 0H8m7 0v7"
      : direction === "down"
        ? "M5 5l10 10m0 0V8m0 7H8"
        : "M4 10h12m0 0-5-5m5 5-5 5";

  return (
    <svg viewBox="0 0 20 20" aria-hidden="true">
      <path d={path} />
    </svg>
  );
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
    return (
      <div className={styles.emptyChart}>
        No Transfermarkt valuation history is available.
      </div>
    );
  }

  const chronological = [...history]
    .sort(
      (left, right) =>
        parseDate(left.valuation_date).getTime() -
        parseDate(right.valuation_date).getTime(),
    )
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
  const padding = Math.max(
    (maximumValue - minimumValue) * 0.18,
    maximumValue * 0.04,
    1,
  );
  const domainMinimum = Math.max(0, minimumValue - padding);
  const domainMaximum = maximumValue + padding;
  const valueRange = Math.max(domainMaximum - domainMinimum, 1);
  const width = 920;
  const height = 290;
  const left = 70;
  const right = 38;
  const top = 25;
  const bottom = 48;
  const historicalRight = forecastAvailable
    ? width - right - 110
    : width - right;
  const xStep =
    chronological.length > 1
      ? (historicalRight - left) / (chronological.length - 1)
      : 0;
  const xForIndex = (index: number) =>
    chronological.length > 1
      ? left + index * xStep
      : left + (historicalRight - left) / 2;
  const yFor = (value: number) =>
    top +
    ((domainMaximum - value) / valueRange) *
      (height - top - bottom);
  const historicalPoints = chronological
    .map(
      (point, index) =>
        `${xForIndex(index)},${yFor(Number(point.value_eur))}`,
    )
    .join(" ");
  const forecastX = width - right;
  const latest = chronological[chronological.length - 1];
  const latestX = xForIndex(chronological.length - 1);
  const latestY = yFor(Number(latest.value_eur));
  const projectionY = forecastAvailable
    ? yFor(Number(projection))
    : null;
  const gridValues = [0, 0.5, 1].map(
    (ratio) => domainMaximum - ratio * valueRange,
  );

  return (
    <div className={styles.chartWrap}>
      <svg
        className={styles.marketChart}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Transfermarkt valuation history and model projection"
      >
        <defs>
          <linearGradient
            id="darkValueArea"
            x1="0"
            y1="0"
            x2="0"
            y2="1"
          >
            <stop stopColor="#35c69b" stopOpacity=".24" />
            <stop offset="1" stopColor="#35c69b" stopOpacity="0" />
          </linearGradient>
        </defs>

        {gridValues.map((value) => {
          const y = yFor(value);
          return (
            <g key={value}>
              <line
                x1={left}
                y1={y}
                x2={width - right}
                y2={y}
                className={styles.chartGrid}
              />
              <text
                x={left - 12}
                y={y + 4}
                textAnchor="end"
                className={styles.chartTick}
              >
                {money(value)}
              </text>
            </g>
          );
        })}

        {chronological.length > 1 && (
          <path
            d={`M${historicalPoints} L${historicalRight},${height - bottom} L${left},${height - bottom} Z`}
            className={styles.chartArea}
          />
        )}

        <polyline
          points={historicalPoints}
          className={styles.actualLine}
        />

        {chronological.map((point, index) => (
          <g key={`${point.valuation_date}-${index}`}>
            <circle
              cx={xForIndex(index)}
              cy={yFor(Number(point.value_eur))}
              r={index === chronological.length - 1 ? 5.5 : 4}
              className={styles.actualDot}
            >
              <title>{`${fullDate(point.valuation_date)}: ${money(point.value_eur)}`}</title>
            </circle>
            {(index === 0 || index === chronological.length - 1) && (
              <text
                x={xForIndex(index)}
                y={height - 17}
                textAnchor={index === 0 ? "start" : "middle"}
                className={styles.chartDate}
              >
                {shortDate(point.valuation_date)}
              </text>
            )}
          </g>
        ))}

        {forecastAvailable && projectionY != null && (
          <g>
            <line
              x1={latestX}
              y1={latestY}
              x2={forecastX}
              y2={projectionY}
              className={styles.forecastLine}
            />
            {lower != null && upper != null && (
              <g>
                <line
                  x1={forecastX}
                  y1={yFor(Number(upper))}
                  x2={forecastX}
                  y2={yFor(Number(lower))}
                  className={styles.intervalLine}
                />
                <line
                  x1={forecastX - 9}
                  y1={yFor(Number(upper))}
                  x2={forecastX + 9}
                  y2={yFor(Number(upper))}
                  className={styles.intervalCap}
                />
                <line
                  x1={forecastX - 9}
                  y1={yFor(Number(lower))}
                  x2={forecastX + 9}
                  y2={yFor(Number(lower))}
                  className={styles.intervalCap}
                />
              </g>
            )}
            <rect
              x={forecastX - 6}
              y={projectionY - 6}
              width="12"
              height="12"
              rx="2"
              transform={`rotate(45 ${forecastX} ${projectionY})`}
              className={styles.forecastDot}
            >
              <title>{`Model projection: ${money(projection)}`}</title>
            </rect>
            <text
              x={forecastX}
              y={height - 17}
              textAnchor="middle"
              className={`${styles.chartDate} ${styles.forecastDate}`}
            >
              Projection
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}

function FormChart({ matches }: { matches: MatchImpact[] }) {
  const points = [...matches]
    .filter(
      (match): match is MatchImpact & { rating: number } =>
        match.rating != null,
    )
    .sort(
      (left, right) =>
        parseDate(left.match_datetime).getTime() -
        parseDate(right.match_datetime).getTime(),
    )
    .slice(-10);

  if (!points.length) {
    return (
      <div className={styles.emptyChart}>
        No rated appearances are available.
      </div>
    );
  }

  const width = 920;
  const height = 300;
  const left = 58;
  const right = 30;
  const top = 24;
  const bottom = 52;
  const minimum = 4;
  const maximum = 10;
  const xStep =
    points.length > 1
      ? (width - left - right) / (points.length - 1)
      : 0;
  const xFor = (index: number) =>
    points.length > 1 ? left + index * xStep : width / 2;
  const yFor = (value: number) =>
    top +
    ((maximum - value) / (maximum - minimum)) *
      (height - top - bottom);
  const linePoints = points
    .map((point, index) => `${xFor(index)},${yFor(point.rating)}`)
    .join(" ");

  return (
    <div className={styles.chartWrap}>
      <svg
        className={styles.formChart}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Recent position-adjusted match ratings"
      >
        {[4, 6, 8, 10].map((value) => (
          <g key={value}>
            <line
              x1={left}
              y1={yFor(value)}
              x2={width - right}
              y2={yFor(value)}
              className={
                value === 6
                  ? styles.baselineLine
                  : styles.chartGrid
              }
            />
            <text
              x={left - 12}
              y={yFor(value) + 4}
              textAnchor="end"
              className={styles.chartTick}
            >
              {value.toFixed(1)}
            </text>
          </g>
        ))}

        <polyline points={linePoints} className={styles.formLine} />

        {points.map((point, index) => {
          const tone = ratingClass(point.rating);
          return (
            <g key={`${point.match_id}-${index}`}>
              <circle
                cx={xFor(index)}
                cy={yFor(point.rating)}
                r={index === points.length - 1 ? 6 : 5}
                className={toneClass(tone, styles.formDot)}
              >
                <title>
                  {`${fullDate(point.match_datetime)} · Rating ${point.rating.toFixed(2)} · ${point.minutes == null ? "Minutes unavailable" : `${Math.round(point.minutes)} min`} · ${point.explanation ?? "Position-adjusted performance"}`}
                </title>
              </circle>
              <text
                x={xFor(index)}
                y={height - 18}
                textAnchor="middle"
                className={styles.chartDate}
              >
                {shortDate(point.match_datetime)}
              </text>
            </g>
          );
        })}

        <text
          x={width - right}
          y={yFor(6) - 9}
          textAnchor="end"
          className={styles.baselineLabel}
        >
          6.0 neutral baseline
        </text>
      </svg>
    </div>
  );
}

function ChapterHeader({
  number,
  eyebrow,
  title,
  description,
}: {
  number: string;
  eyebrow: string;
  title: string;
  description: string;
}) {
  return (
    <div className={styles.chapterHeader}>
      <span>{number}</span>
      <div>
        <p className="label">{eyebrow}</p>
        <h2>{title}</h2>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function PlayerDashboard({
  player,
  watched,
  onToggleWatchlist,
  onSelectPlayer,
}: Props) {
  const recentMatches = (player.match_impacts ?? []).slice(0, 10);
  const recentFive = recentMatches
    .filter(
      (match): match is MatchImpact & { rating: number } =>
        match.rating != null,
    )
    .slice(0, 5)
    .reverse();
  const history = [...(player.valuation_history ?? [])].sort(
    (left, right) =>
      parseDate(left.valuation_date).getTime() -
      parseDate(right.valuation_date).getTime(),
  );
  const formTrend =
    player.rolling_3_match_rating != null &&
    player.rolling_20_match_rating != null
      ? player.rolling_3_match_rating -
        player.rolling_20_match_rating
      : null;
  const projectedChange =
    player.estimated_value_eur != null &&
    player.current_market_value_eur
      ? player.estimated_value_eur /
          player.current_market_value_eur -
        1
      : player.predicted_pct_change ?? null;
  const projectedDelta =
    player.estimated_value_eur != null &&
    player.current_market_value_eur != null
      ? player.estimated_value_eur -
        player.current_market_value_eur
      : null;
  const projectionAvailable = player.estimated_value_eur != null;
  const direction = player.direction ?? "unscored";
  const probabilityPercent = Math.max(
    0,
    Math.min(
      100,
      Math.round((player.probability_value_increase ?? 0) * 100),
    ),
  );
  const rangeMinimum = player.estimated_lower_eur;
  const rangeMaximum = player.estimated_upper_eur;
  const rangePosition =
    player.estimated_value_eur != null &&
    rangeMinimum != null &&
    rangeMaximum != null &&
    rangeMaximum > rangeMinimum
      ? Math.max(
          0,
          Math.min(
            100,
            ((player.estimated_value_eur - rangeMinimum) /
              (rangeMaximum - rangeMinimum)) *
              100,
          ),
        )
      : 50;
  const formScorePosition = Math.max(
    0,
    Math.min(100, ((player.current_form_rating ?? 0) / 10) * 100),
  );

  return (
    <>
      <section className={styles.playerHeader}>
        <div className={styles.avatar}>
          {player.display_name
            .split(" ")
            .filter(Boolean)
            .slice(0, 2)
            .map((part) => part[0])
            .join("")
            .toUpperCase()}
        </div>
        <div className={styles.identity}>
          <div className={styles.identityLine}>
            <h2>{player.display_name}</h2>
            <span>{player.position ?? "Position unavailable"}</span>
          </div>
          <p>
            WhoScored ID {player.player_id}
            <i />
            Latest match {fullDate(recentMatches[0]?.match_datetime)}
          </p>
        </div>
        <button
          type="button"
          className={styles.watchButton}
          onClick={onToggleWatchlist}
        >
          {watched ? "✓ On watchlist" : "+ Add to watchlist"}
        </button>
      </section>

      <section className={styles.chapter}>
        <ChapterHeader
          number="01"
          eyebrow="CURRENT OUTLOOK"
          title="Value and recent form"
          description="The market baseline, the model estimate and the current performance signal in one decision view."
        />

        <div className={styles.outlookGrid}>
          <article className={`${styles.card} ${styles.valuationCard}`}>
            <div className={styles.cardHead}>
              <div>
                <p className="label">MARKET VALUE</p>
                <h3>Published value and model projection</h3>
              </div>
              {projectionAvailable && (
                <span
                  className={toneClass(
                    direction === "rising"
                      ? "positive"
                      : direction === "falling"
                        ? "negative"
                        : "neutral",
                    styles.statusPill,
                  )}
                >
                  {directionCopy(direction)}
                </span>
              )}
            </div>

            <div className={styles.valuationHero}>
              <div className={styles.valueMetric}>
                <span>Latest Transfermarkt value</span>
                <strong>{money(player.current_market_value_eur)}</strong>
                <small>
                  Published {fullDate(player.latest_valuation_date)}
                </small>
              </div>

              <div
                className={toneClass(
                  trendClass(projectedChange),
                  styles.valueTransition,
                )}
              >
                <div>
                  <ArrowIcon
                    direction={
                      projectedChange == null
                        ? "right"
                        : projectedChange >= 0
                          ? "up"
                          : "down"
                    }
                  />
                </div>
                <b>{signedPercent(projectedChange)}</b>
              </div>

              <div className={`${styles.valueMetric} ${styles.projectedValue}`}>
                <span>Model estimate</span>
                <strong>{money(player.estimated_value_eur)}</strong>
                <small
                  className={styles[trendClass(projectedChange)]}
                >
                  {projectedDelta == null
                    ? "Awaiting projected delta"
                    : `${projectedDelta >= 0 ? "+" : ""}${money(projectedDelta)} from baseline`}
                </small>
              </div>
            </div>

            {projectionAvailable ? (
              <div className={styles.forecastBlock}>
                <div className={styles.rangeLabels}>
                  <span>{money(rangeMinimum)}</span>
                  <b>{money(player.estimated_value_eur)}</b>
                  <span>{money(rangeMaximum)}</span>
                </div>
                <div className={styles.rangeTrack}>
                  <i
                    className={styles.rangeEstimate}
                    style={{ left: `${rangePosition}%` }}
                  />
                </div>
                <div className={styles.rangeCaptions}>
                  <span>Lower</span>
                  <span>90% projected range</span>
                  <span>Upper</span>
                </div>

                <div className={styles.probabilityRow}>
                  <div>
                    <span>Probability value increased</span>
                    <b>{probabilityPercent}%</b>
                  </div>
                  <span className={styles.probabilityTrack}>
                    <i style={{ width: `${probabilityPercent}%` }} />
                  </span>
                </div>
              </div>
            ) : (
              <div className={styles.noProjection}>
                <b>No current projection</b>
                <span>
                  A projection appears after eligible EPL match data is
                  scored with an active model.
                </span>
              </div>
            )}

            <div className={styles.embeddedChart}>
              <div className={styles.embeddedChartHead}>
                <span>Valuation trajectory</span>
                <div>
                  <span><i className={styles.actualLegend} /> Published</span>
                  <span><i className={styles.forecastLegend} /> Projection</span>
                </div>
              </div>
              <MarketValueChart
                history={history}
                projection={player.estimated_value_eur}
                lower={player.estimated_lower_eur}
                upper={player.estimated_upper_eur}
              />
            </div>
          </article>

          <article className={`${styles.card} ${styles.formCard}`}>
            <div className={styles.cardHead}>
              <div>
                <p className="label">PLAYER FORM</p>
                <h3>Current domestic performance</h3>
              </div>
              <span
                className={toneClass(
                  trendClass(formTrend),
                  styles.statusPill,
                )}
              >
                {formTrend == null
                  ? "Trend unavailable"
                  : formTrend > 0.1
                    ? "Improving"
                    : formTrend < -0.1
                      ? "Declining"
                      : "Steady"}
              </span>
            </div>

            <div className={styles.formHero}>
              <div className={styles.formScore}>
                <strong>
                  {player.current_form_rating?.toFixed(2) ?? "—"}
                </strong>
                <span>/ 10</span>
                <small>90-day minutes-weighted form</small>
              </div>

              <div className={styles.formScale}>
                <div className={styles.formScaleTrack}>
                  <i style={{ left: `${formScorePosition}%` }} />
                </div>
                <div>
                  <span>Poor</span>
                  <span>Neutral</span>
                  <span>Strong</span>
                </div>
              </div>
            </div>

            <div className={styles.formStats}>
              <div>
                <span>Last 3</span>
                <b>
                  {player.rolling_3_match_rating?.toFixed(2) ?? "—"}
                </b>
              </div>
              <div>
                <span>Last 20</span>
                <b>
                  {player.rolling_20_match_rating?.toFixed(2) ?? "—"}
                </b>
              </div>
              <div>
                <span>Difference</span>
                <b className={styles[trendClass(formTrend)]}>
                  {formTrend == null
                    ? "—"
                    : `${formTrend >= 0 ? "+" : ""}${formTrend.toFixed(2)}`}
                </b>
              </div>
            </div>

            <div className={styles.formInterpretation}>
              <span>Current signal</span>
              <strong>
                {player.current_form_rating == null
                  ? "Insufficient recent data"
                  : player.current_form_rating >= 7
                    ? "Strong current form"
                    : player.current_form_rating >= 6
                      ? "Around the neutral baseline"
                      : "Below the neutral baseline"}
              </strong>
              <p>
                Recent form is shown separately from the full-season
                profile and the market-value estimate.
              </p>
            </div>
          </article>
        </div>
      </section>

      <section className={styles.chapter}>
        <ChapterHeader
          number="02"
          eyebrow="RECENT PERFORMANCES"
          title="How recent matches shaped the signal"
          description="A full-width view of recent ratings, followed by the five latest rated appearances and optional match-level drivers."
        />

        <article className={`${styles.card} ${styles.recentCard}`}>
          <div className={styles.recentChartHead}>
            <div>
              <p className="label">FORM TREND</p>
              <h3>Last 10 rated appearances</h3>
            </div>
            <span>6.0 is the neutral baseline</span>
          </div>

          <FormChart matches={recentMatches} />

          <div className={styles.appearanceGrid}>
            {recentFive.length ? (
              recentFive.map((match) => {
                const tone = ratingClass(match.rating);
                return (
                  <article
                    key={match.match_id}
                    className={toneClass(tone, styles.appearanceCard)}
                  >
                    <div>
                      <span>{shortDate(match.match_datetime)}</span>
                      <b>{match.rating.toFixed(1)}</b>
                    </div>
                    <small>
                      {match.minutes == null
                        ? "Minutes unavailable"
                        : `${Math.round(match.minutes)} min`}
                    </small>
                    <p>
                      {match.explanation ??
                        "Position-adjusted performance"}
                    </p>
                  </article>
                );
              })
            ) : (
              <div className={styles.emptyChart}>
                No recent rated appearances are available.
              </div>
            )}
          </div>

          <details className={styles.matchDrivers}>
            <summary>
              <span>View recent match-level drivers</span>
              <small>Eight latest rated appearances</small>
            </summary>
            <div className={styles.matchTable}>
              <div className={styles.matchTableHead}>
                <span>Match</span>
                <span>Rating</span>
                <span>Main drivers</span>
                <span>Form impact</span>
              </div>
              {recentMatches.slice(0, 8).map((match) => {
                const impact = match.performance_impact_score;
                return (
                  <div className={styles.matchRow} key={match.match_id}>
                    <div>
                      <b>Match {match.match_id}</b>
                      <small>
                        {fullDate(match.match_datetime)}
                        {match.minutes != null
                          ? ` · ${Math.round(match.minutes)} min`
                          : ""}
                      </small>
                    </div>
                    <div className={styles.ratingCell}>
                      <b>{match.rating?.toFixed(2) ?? "—"}</b>
                      <span>out of 10</span>
                    </div>
                    <div className={styles.driverCopy}>
                      {match.explanation ??
                        "Position-adjusted performance"}
                    </div>
                    <div
                      className={toneClass(
                        trendClass(impact),
                        styles.impactCell,
                      )}
                    >
                      <b>
                        {impact == null
                          ? "—"
                          : `${impact >= 0 ? "+" : ""}${impact.toFixed(2)}`}
                      </b>
                      <span>
                        {match.estimated_value_delta_eur != null
                          ? `Replay Δ ${money(match.estimated_value_delta_eur)}`
                          : match.impact_direction ?? "form signal"}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </details>
        </article>
      </section>

      <PlayerProfileSection
        playerId={player.player_id}
        onSelectPlayer={onSelectPlayer}
      />

      <section className={styles.supportingSection}>
        <div className={styles.supportingHeader}>
          <p className="label">SUPPORTING INFORMATION</p>
          <h2>Methodology, limitations and data coverage</h2>
        </div>

        <div className={styles.supportingDetails}>
          <details>
            <summary>How to read the forecast</summary>
            <div>
              <p>
                The latest Transfermarkt valuation is the observed
                baseline. The model applies domestic match performance
                recorded after that date.
              </p>
              <ul>
                <li>The midpoint is the model&apos;s current estimate.</li>
                <li>The 90% range communicates forecast uncertainty.</li>
                <li>
                  The direction probability shows how strongly the model
                  leans toward an increase.
                </li>
              </ul>
            </div>
          </details>

          <details>
            <summary>Known limitations</summary>
            <div>
              <ul>
                <li>International and national-team matches</li>
                <li>Injuries not represented in match data</li>
                <li>Contract duration, salary and release clauses</li>
                <li>Transfer demand and club negotiating position</li>
                <li>Proprietary scouting and off-field information</li>
              </ul>
            </div>
          </details>

          <details>
            <summary>Model and data details</summary>
            <div className={styles.modelGrid}>
              <span>Valuation model</span>
              <b>{player.valuation_model_version ?? "No active model"}</b>
              <span>Forecast refreshed</span>
              <b>{fullDate(player.refreshed_at)}</b>
              <span>Valuation source</span>
              <b>Transfermarkt</b>
              <span>Performance source</span>
              <b>WhoScored · EPL</b>
            </div>
          </details>
        </div>
      </section>
    </>
  );
}
