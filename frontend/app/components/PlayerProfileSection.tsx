"use client";

import { useEffect, useMemo, useState } from "react";
import styles from "./PlayerProfileSection.module.css";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type ProfileMetric = {
  key: string;
  label: string;
  phase: string;
  value?: number | null;
  percentile?: number | null;
  display_order?: number;
};

type ProfileData = {
  player_id: number;
  player_name?: string | null;
  season?: string | null;
  primary_role?: string | null;
  secondary_role?: string | null;
  primary_role_share?: number | null;
  minutes?: number | null;
  appearances?: number | null;
  sample_status?: string | null;
  benchmark?: {
    competition?: string | null;
    role?: string | null;
    minimum_minutes?: number | null;
  };
  metrics: ProfileMetric[];
};

type SimilarPlayer = {
  similar_player_id?: number;
  whoscored_player_id?: number;
  player_id?: number;
  similar_player_name?: string;
  player_name?: string;
  primary_role?: string | null;
  secondary_role?: string | null;
  minutes?: number | null;
  metrics_used?: number | null;
  similarity?: number | null;
  profile_similarity?: number | null;
  rank?: number | null;
};

type Props = {
  playerId: number;
  onSelectPlayer?: (playerId: number) => void;
};

const PHASE_COLORS: Record<string, string> = {
  Scoring: "#f15e49",
  Creation: "#f39a3f",
  Passing: "#d9b44a",
  Progression: "#2eb4a3",
  Defending: "#5f89de",
  "Shot stopping": "#8f6fd0",
  Distribution: "#4c9f7b",
  Sweeping: "#6f9bae",
};

const PHASE_ORDER = [
  "Scoring",
  "Creation",
  "Passing",
  "Progression",
  "Defending",
  "Shot stopping",
  "Distribution",
  "Sweeping",
];

function roleLabel(value?: string | null) {
  if (!value) return "Role unavailable";

  return value
    .toLowerCase()
    .split("_")
    .map(
      (part) =>
        `${part[0]?.toUpperCase() ?? ""}${part.slice(1)}`,
    )
    .join(" ");
}

function metricValue(value?: number | null) {
  if (value == null || !Number.isFinite(Number(value))) {
    return "—";
  }

  const numeric = Number(value);
  const absolute = Math.abs(numeric);

  if (absolute >= 100) return numeric.toFixed(0);
  if (absolute >= 10) return numeric.toFixed(1);
  return numeric.toFixed(2);
}

function splitLabel(label: string): string[] {
  const words = label.trim().split(/\s+/);

  if (label.length <= 18 || words.length === 1) {
    return [label];
  }

  let bestIndex = 1;
  let bestDifference = Number.POSITIVE_INFINITY;

  for (let index = 1; index < words.length; index += 1) {
    const first = words.slice(0, index).join(" ");
    const second = words.slice(index).join(" ");
    const difference = Math.abs(first.length - second.length);

    if (difference < bestDifference) {
      bestDifference = difference;
      bestIndex = index;
    }
  }

  return [
    words.slice(0, bestIndex).join(" "),
    words.slice(bestIndex).join(" "),
  ];
}

function polar(
  centerX: number,
  centerY: number,
  radius: number,
  angle: number,
) {
  return {
    x: centerX + radius * Math.cos(angle),
    y: centerY + radius * Math.sin(angle),
  };
}

function wedgePath(
  centerX: number,
  centerY: number,
  startAngle: number,
  endAngle: number,
  innerRadius: number,
  outerRadius: number,
) {
  const startOuter = polar(
    centerX,
    centerY,
    outerRadius,
    startAngle,
  );
  const endOuter = polar(
    centerX,
    centerY,
    outerRadius,
    endAngle,
  );
  const startInner = polar(
    centerX,
    centerY,
    innerRadius,
    startAngle,
  );
  const endInner = polar(
    centerX,
    centerY,
    innerRadius,
    endAngle,
  );
  const largeArc = endAngle - startAngle > Math.PI ? 1 : 0;

  return [
    `M ${startOuter.x} ${startOuter.y}`,
    `A ${outerRadius} ${outerRadius} 0 ${largeArc} 1 ${endOuter.x} ${endOuter.y}`,
    `L ${endInner.x} ${endInner.y}`,
    `A ${innerRadius} ${innerRadius} 0 ${largeArc} 0 ${startInner.x} ${startInner.y}`,
    "Z",
  ].join(" ");
}

function PizzaChart({ profile }: { profile: ProfileData }) {
  const metrics = [...profile.metrics].sort(
    (left, right) =>
      (left.display_order ?? 0) -
      (right.display_order ?? 0),
  );

  const count = metrics.length;
  const centerX = 600;
  const centerY = 520;
  const innerRadius = 116;
  const outerRadius = 330;
  const percentileRadius = outerRadius + 24;
  const leaderStartRadius = outerRadius + 44;
  const leaderBendRadius = outerRadius + 76;
  const labelRadius = outerRadius + 132;
  const step = (Math.PI * 2) / Math.max(count, 1);
  const startOffset = -Math.PI / 2 - step / 2;

  return (
    <div className={styles.chartShell}>
      <div className={styles.chartHeading}>
        <strong>Same-role EPL percentiles</strong>
        <span>Percentile rank: 0 is lowest, 100 is highest</span>
      </div>

      <svg
        className={styles.pizza}
        viewBox="0 0 1200 1080"
        role="img"
        aria-label={`${profile.player_name ?? "Player"} role-relative performance profile`}
      >
        <circle
          cx={centerX}
          cy={centerY}
          r={outerRadius + 40}
          className={styles.chartBackdrop}
        />

        {[20, 40, 60, 80, 100].map((level, index) => {
          const radius =
            innerRadius +
            ((outerRadius - innerRadius) * level) / 100;

          return (
            <circle
              key={level}
              cx={centerX}
              cy={centerY}
              r={radius}
              className={
                index % 2 === 0
                  ? styles.ringLight
                  : styles.ringDark
              }
            />
          );
        })}

        {metrics.map((metric, index) => {
          const start =
            startOffset + index * step + step * 0.035;
          const end =
            startOffset +
            (index + 1) * step -
            step * 0.035;
          const middle = (start + end) / 2;
          const percentile = Math.max(
            0,
            Math.min(100, Number(metric.percentile ?? 0)),
          );
          const valueRadius =
            innerRadius +
            ((outerRadius - innerRadius) * percentile) /
              100;
          const phaseColor =
            PHASE_COLORS[metric.phase] ?? "#2eb4a3";

          const valuePoint = polar(
            centerX,
            centerY,
            Math.max(innerRadius + 42, valueRadius - 30),
            middle,
          );
          const percentilePoint = polar(
            centerX,
            centerY,
            percentileRadius,
            middle,
          );
          const leaderStart = polar(
            centerX,
            centerY,
            leaderStartRadius,
            middle,
          );
          const leaderBend = polar(
            centerX,
            centerY,
            leaderBendRadius,
            middle,
          );

          const rightSide = Math.cos(middle) >= 0;
          const labelX = rightSide
            ? centerX + labelRadius
            : centerX - labelRadius;
          const labelY = leaderBend.y;
          const elbowX = rightSide
            ? labelX - 26
            : labelX + 26;
          const textAnchor = rightSide ? "start" : "end";
          const lines = splitLabel(metric.label);

          return (
            <g key={`${metric.key}-${index}`}>
              <path
                d={wedgePath(
                  centerX,
                  centerY,
                  start,
                  end,
                  innerRadius,
                  valueRadius,
                )}
                fill={phaseColor}
                className={styles.metricWedge}
              >
                <title>
                  {`${metric.label}: ${metricValue(metric.value)} · ${Math.round(percentile)}th percentile`}
                </title>
              </path>

              <path
                d={wedgePath(
                  centerX,
                  centerY,
                  start,
                  end,
                  outerRadius + 8,
                  outerRadius + 18,
                )}
                fill={phaseColor}
                className={styles.phaseBand}
              />

              <line
                x1={
                  polar(
                    centerX,
                    centerY,
                    innerRadius,
                    middle,
                  ).x
                }
                y1={
                  polar(
                    centerX,
                    centerY,
                    innerRadius,
                    middle,
                  ).y
                }
                x2={
                  polar(
                    centerX,
                    centerY,
                    outerRadius,
                    middle,
                  ).x
                }
                y2={
                  polar(
                    centerX,
                    centerY,
                    outerRadius,
                    middle,
                  ).y
                }
                className={styles.spoke}
              />

              <text
                x={valuePoint.x}
                y={valuePoint.y}
                textAnchor="middle"
                dominantBaseline="middle"
                className={styles.metricNumber}
              >
                {metricValue(metric.value)}
              </text>

              <text
                x={percentilePoint.x}
                y={percentilePoint.y}
                textAnchor="middle"
                dominantBaseline="middle"
                className={styles.percentileNumber}
                style={{ fill: phaseColor }}
              >
                {Math.round(percentile)}
              </text>

              <polyline
                points={[
                  `${leaderStart.x},${leaderStart.y}`,
                  `${leaderBend.x},${leaderBend.y}`,
                  `${elbowX},${labelY}`,
                ].join(" ")}
                className={styles.leaderLine}
              />

              <text
                x={labelX}
                y={labelY - (lines.length > 1 ? 7 : 0)}
                textAnchor={textAnchor}
                className={styles.metricLabel}
              >
                {lines.map((line, lineIndex) => (
                  <tspan
                    key={`${metric.key}-${lineIndex}`}
                    x={labelX}
                    dy={lineIndex === 0 ? 0 : 17}
                  >
                    {line}
                  </tspan>
                ))}
              </text>
            </g>
          );
        })}

        <circle
          cx={centerX}
          cy={centerY}
          r={innerRadius - 2}
          className={styles.chartCore}
        />
        <text
          x={centerX}
          y={centerY - 20}
          textAnchor="middle"
          className={styles.coreTitle}
        >
          {roleLabel(profile.primary_role)}
        </text>
        <text
          x={centerX}
          y={centerY + 14}
          textAnchor="middle"
          className={styles.coreMeta}
        >
          {profile.minutes == null
            ? "—"
            : `${Math.round(
                profile.minutes,
              ).toLocaleString()} min`}
        </text>
        <text
          x={centerX}
          y={centerY + 44}
          textAnchor="middle"
          className={styles.coreFootnote}
        >
          same-role EPL percentiles
        </text>
      </svg>

      <p className={styles.chartNote}>
        Wedge length shows the same-role EPL percentile. The
        value inside each wedge is the underlying per-90 or
        percentage statistic.
      </p>
    </div>
  );
}

export default function PlayerProfileSection({
  playerId,
  onSelectPlayer,
}: Props) {
  const [profile, setProfile] =
    useState<ProfileData | null>(null);
  const [similar, setSimilar] = useState<SimilarPlayer[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);

    Promise.all([
      fetch(`${API_URL}/api/players/${playerId}/profile`),
      fetch(
        `${API_URL}/api/players/${playerId}/similar-players?limit=10`,
      ),
    ])
      .then(
        async (
          [profileResponse, similarResponse],
        ): Promise<
          [ProfileData | null, SimilarPlayer[]]
        > => {
          if (!profileResponse.ok) {
            if (profileResponse.status === 404) {
              return [null, []];
            }

            throw new Error(
              `Profile API returned ${profileResponse.status}`,
            );
          }

          const profilePayload =
            await profileResponse.json();
          const similarPayload = similarResponse.ok
            ? await similarResponse.json()
            : { data: [] };

          return [
            profilePayload.data as ProfileData,
            (similarPayload.data ?? []) as SimilarPlayer[],
          ];
        },
      )
      .then(([nextProfile, nextSimilar]) => {
        if (cancelled) return;

        setProfile(nextProfile);
        setSimilar(nextSimilar);
      })
      .catch(() => {
        if (cancelled) return;

        setProfile(null);
        setSimilar([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [playerId]);

  const phases = useMemo(() => {
    if (!profile) return [];

    const available = new Set(
      profile.metrics.map((metric) => metric.phase),
    );

    return PHASE_ORDER.filter((phase) =>
      available.has(phase),
    );
  }, [profile]);

  if (loading) {
    return (
      <article
        className={`sectionCard wide ${styles.profileCard}`}
      >
        <div className={styles.loading}>
          Building player profile…
        </div>
      </article>
    );
  }

  if (!profile) return null;

  const benchmarkMinutes =
    profile.benchmark?.minimum_minutes ?? 900;

  return (
    <>
      <article
        className={`sectionCard wide ${styles.profileCard}`}
      >
        <div className={styles.profileHeader}>
          <div>
            <p className={styles.season}>
              {profile.season ?? "Current season"}
            </p>
            <h3>{profile.player_name}</h3>
            <p>
              {roleLabel(profile.primary_role)}
              {profile.secondary_role
                ? ` · Secondary: ${roleLabel(
                    profile.secondary_role,
                  )}`
                : ""}
            </p>
          </div>

          <span className={styles.roleBadge}>
            {roleLabel(profile.primary_role)}
          </span>
        </div>

        <div className={styles.factGrid}>
          <div>
            <span>Minutes</span>
            <b>
              {profile.minutes == null
                ? "—"
                : Math.round(
                    profile.minutes,
                  ).toLocaleString()}
            </b>
          </div>
          <div>
            <span>Appearances</span>
            <b>{profile.appearances ?? "—"}</b>
          </div>
          <div>
            <span>Primary-role share</span>
            <b>
              {profile.primary_role_share == null
                ? "—"
                : `${Math.round(
                    profile.primary_role_share * 100,
                  )}%`}
            </b>
          </div>
          <div>
            <span>Percentile benchmark</span>
            <b>{benchmarkMinutes}+ min</b>
          </div>
        </div>

        <div className={styles.phaseLegend}>
          {phases.map((phase) => (
            <span key={phase}>
              <i
                style={{
                  background:
                    PHASE_COLORS[phase] ?? "#2eb4a3",
                }}
              />
              {phase}
            </span>
          ))}
        </div>

        <PizzaChart profile={profile} />
      </article>

      {similar.length > 0 && (
        <section className={styles.similarChapter}>
          <div className={styles.chapterHeader}>
            <span>04</span>
            <div>
              <p className="label">PLAYER SIMILARITY</p>
              <h2>Comparable same-role profiles</h2>
              <p>
                The closest standardized season profiles using
                the same role-specific metrics.
              </p>
            </div>
          </div>

          <article
            className={`sectionCard wide ${styles.similarCard}`}
          >
            <div className={styles.similarHeader}>
              <div>
                <h3>Closest performance profiles</h3>
                <p>
                  Similarity is a profile-distance score, not
                  a probability that two players are identical.
                </p>
              </div>
              <span>Top {similar.length}</span>
            </div>

            <div className={styles.similarGrid}>
              {similar.map((candidate, index) => {
                const candidateId =
                  candidate.similar_player_id ??
                  candidate.whoscored_player_id ??
                  candidate.player_id;
                const name =
                  candidate.similar_player_name ??
                  candidate.player_name ??
                  `Player ${candidateId ?? index + 1}`;
                const score =
                  candidate.similarity ??
                  candidate.profile_similarity ??
                  0;

                return (
                  <button
                    type="button"
                    className={styles.similarRow}
                    key={`${candidateId ?? name}-${index}`}
                    onClick={() => {
                      if (candidateId != null) {
                        onSelectPlayer?.(candidateId);
                      }
                    }}
                    disabled={candidateId == null}
                  >
                    <span className={styles.similarRank}>
                      {candidate.rank ?? index + 1}
                    </span>

                    <span className={styles.similarIdentity}>
                      <b>{name}</b>
                      <small>
                        {roleLabel(
                          candidate.primary_role,
                        )}{" "}
                        ·{" "}
                        {candidate.minutes == null
                          ? "minutes unavailable"
                          : `${Math.round(
                              candidate.minutes,
                            ).toLocaleString()} min`}
                      </small>
                    </span>

                    <span className={styles.similarScore}>
                      <strong>{score.toFixed(1)}</strong>
                      <small>similarity</small>
                    </span>

                    <span className={styles.similarBar}>
                      <i
                        style={{
                          width: `${Math.max(
                            4,
                            Math.min(100, score),
                          )}%`,
                        }}
                      />
                    </span>
                  </button>
                );
              })}
            </div>
          </article>
        </section>
      )}
    </>
  );
}
