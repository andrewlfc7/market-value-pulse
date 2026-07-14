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
  Scoring: "#ef684c",
  Creation: "#e99a55",
  Passing: "#c6a83c",
  Progression: "#299c8e",
  Defending: "#567ec2",
  "Shot stopping": "#8768b2",
  Distribution: "#3f9470",
  Sweeping: "#6b95a6",
};

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

function splitLabel(label: string) {
  const words = label.split(" ");

  if (label.length <= 18 || words.length === 1) {
    return [label];
  }

  let bestIndex = 1;
  let smallestDifference = Number.POSITIVE_INFINITY;

  for (let index = 1; index < words.length; index += 1) {
    const first = words.slice(0, index).join(" ");
    const second = words.slice(index).join(" ");
    const difference = Math.abs(first.length - second.length);

    if (difference < smallestDifference) {
      smallestDifference = difference;
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
  const centerX = 450;
  const centerY = 345;
  const innerRadius = 68;
  const outerRadius = 220;
  const labelRadius = 286;
  const step = (Math.PI * 2) / Math.max(count, 1);
  const startOffset = -Math.PI / 2 - step / 2;

  return (
    <div className={styles.chartShell}>
      <svg
        className={styles.pizza}
        viewBox="0 0 900 700"
        role="img"
        aria-label={`${profile.player_name ?? "Player"} role-relative performance profile`}
      >
        <circle
          cx={centerX}
          cy={centerY}
          r={outerRadius + 30}
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
            startOffset + index * step + step * 0.04;
          const end =
            startOffset +
            (index + 1) * step -
            step * 0.04;
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
            PHASE_COLORS[metric.phase] ?? "#299c8e";

          const valuePoint = polar(
            centerX,
            centerY,
            Math.max(innerRadius + 27, valueRadius - 16),
            middle,
          );
          const percentilePoint = polar(
            centerX,
            centerY,
            outerRadius + 15,
            middle,
          );
          const labelPoint = polar(
            centerX,
            centerY,
            labelRadius,
            middle,
          );

          const horizontalOffset = labelPoint.x - centerX;
          const textAnchor =
            horizontalOffset > 36
              ? "start"
              : horizontalOffset < -36
                ? "end"
                : "middle";
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
                  outerRadius + 5,
                  outerRadius + 11,
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
              >
                {Math.round(percentile)}
              </text>

              <text
                x={labelPoint.x}
                y={labelPoint.y}
                textAnchor={textAnchor}
                className={styles.metricLabel}
              >
                {lines.map((line, lineIndex) => (
                  <tspan
                    key={`${metric.key}-${lineIndex}`}
                    x={labelPoint.x}
                    dy={lineIndex === 0 ? 0 : 13}
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
          y={centerY - 13}
          textAnchor="middle"
          className={styles.coreTitle}
        >
          {roleLabel(profile.primary_role)}
        </text>
        <text
          x={centerX}
          y={centerY + 11}
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
          y={centerY + 29}
          textAnchor="middle"
          className={styles.coreFootnote}
        >
          same-role percentile
        </text>
      </svg>
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

    return Array.from(
      new Set(
        profile.metrics.map((metric) => metric.phase),
      ),
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
            <p className="label">SEASON PROFILE</p>
            <h3>How the player produces value</h3>
            <p>
              Same-role EPL percentiles across scoring,
              creation, progression and defending.
            </p>
          </div>

          <span className={styles.roleBadge}>
            {roleLabel(profile.primary_role)}
          </span>
        </div>

        <div className={styles.profileLayout}>
          <aside className={styles.profileSummary}>
            <div className={styles.profileIdentity}>
              <span>
                {profile.season ?? "Current season"}
              </span>
              <strong>{profile.player_name}</strong>
              {profile.secondary_role && (
                <small>
                  Secondary role:{" "}
                  {roleLabel(profile.secondary_role)}
                </small>
              )}
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
                <span>Role share</span>
                <b>
                  {profile.primary_role_share == null
                    ? "—"
                    : `${Math.round(
                        profile.primary_role_share * 100,
                      )}%`}
                </b>
              </div>
              <div>
                <span>Benchmark</span>
                <b>{benchmarkMinutes}+ min</b>
              </div>
            </div>

            <div className={styles.phaseLegend}>
              {phases.map((phase) => (
                <span key={phase}>
                  <i
                    style={{
                      background:
                        PHASE_COLORS[phase] ?? "#299c8e",
                    }}
                  />
                  {phase}
                </span>
              ))}
            </div>

            <p className={styles.readingNote}>
              The number inside a wedge is the actual per-90
              or percentage statistic. Wedge length and the
              outer number show the percentile.
            </p>
          </aside>

          <PizzaChart profile={profile} />
        </div>
      </article>

      {similar.length > 0 && (
        <section className={styles.similarChapter}>
          <div className={styles.chapterHeader}>
            <span>03</span>
            <div>
              <p className="label">PLAYER SIMILARITY</p>
              <h2>Who plays in a comparable way?</h2>
              <p>
                The closest same-role profiles using the same
                standardized season metrics.
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
