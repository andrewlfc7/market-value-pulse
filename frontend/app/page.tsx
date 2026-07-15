"use client";

import { useEffect, useMemo, useState } from "react";
import PlayerDashboard from "./components/PlayerDashboard";
import {
  fullDate,
  initials,
  money,
} from "./lib/player-format";
import type {
  PlayerDetail,
  PlayerSummary,
  View,
} from "./lib/player-types";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function JsonPanel({
  title,
  value,
}: {
  title: string;
  value: unknown;
}) {
  return (
    <article className="technicalPanel">
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
  const [watchlist, setWatchlist] = useState<Set<number>>(
    new Set(),
  );
  const [catalog, setCatalog] = useState<unknown>(null);
  const [lineage, setLineage] = useState<unknown>(null);
  const [runs, setRuns] = useState<Record<string, unknown>[]>([]);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API_URL}/api/players`)
      .then((response) => {
        if (!response.ok) {
          throw new Error(
            `Player API returned ${response.status}`,
          );
        }
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
        if (!response.ok) {
          throw new Error(
            `Player detail API returned ${response.status}`,
          );
        }
        return response.json();
      })
      .then((payload) => setPlayer(payload.data as PlayerDetail))
      .catch((reason) => setError(String(reason)))
      .finally(() => setDetailLoading(false));
  }, [selectedId]);

  useEffect(() => {
    if (view === "catalog" && catalog == null) {
      Promise.all([
        fetch(`${API_URL}/api/catalog`).then((response) =>
          response.json(),
        ),
        fetch(`${API_URL}/api/lineage`).then((response) =>
          response.json(),
        ),
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

  const watchedPlayers = players.filter((candidate) =>
    watchlist.has(candidate.player_id),
  );

  function toggleWatchlist() {
    if (!player) return;

    setWatchlist((current) => {
      const next = new Set(current);
      if (next.has(player.player_id)) {
        next.delete(player.player_id);
      } else {
        next.add(player.player_id);
      }
      return next;
    });
  }

  function selectPlayer(playerId: number) {
    setSelectedId(playerId);
    setView("player");
    const selected = players.find(
      (candidate) => candidate.player_id === playerId,
    );
    if (selected) setQuery(selected.display_name);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function submitPlayerSearch(
    event: React.FormEvent<HTMLFormElement>,
  ) {
    event.preventDefault();
    const normalized = query.trim().toLowerCase();
    if (!normalized) return;

    const candidate =
      players.find(
        (item) =>
          item.display_name.toLowerCase() === normalized,
      ) ?? filteredPlayers[0];

    if (candidate) selectPlayer(candidate.player_id);
  }

  const viewTitle =
    view === "player"
      ? "Player outlook"
      : view === "watchlist"
        ? "Squad watchlist"
        : view === "catalog"
          ? "Data catalog & lineage"
          : "Pipeline health";

  return (
    <main>
      <aside>
        <div className="brand">
          <span>MV</span>
          <div>
            Market Value Pulse
            <small>Football intelligence</small>
          </div>
        </div>

        <nav aria-label="Primary navigation">
          <button
            className={view === "player" ? "active" : ""}
            onClick={() => setView("player")}
          >
            Player outlook
          </button>
          <button
            className={view === "watchlist" ? "active" : ""}
            onClick={() => setView("watchlist")}
          >
            Watchlist <small>{watchlist.size}</small>
          </button>
          <button
            className={view === "catalog" ? "active" : ""}
            onClick={() => setView("catalog")}
          >
            Data & lineage
          </button>
          <button
            className={view === "health" ? "active" : ""}
            onClick={() => setView("health")}
          >
            Pipeline health
          </button>
        </nav>

        <div className="source">
          <i className={error ? "offline" : ""} />
          {error ? "API unavailable" : "Data service connected"}
          <br />
          <b>{players.length} mapped players</b>
        </div>
      </aside>

      <section className="content">
        <header>
          <div>
            <p className="eyebrow">MARKET VALUE INTELLIGENCE</p>
            <h1>{viewTitle}</h1>
          </div>

          {view === "player" && (
            <form className="search" onSubmit={submitPlayerSearch}>
              <label htmlFor="player-search">Find a player</label>
              <div className="searchControls">
                <input
                  id="player-search"
                  value={query}
                  onChange={(event: React.ChangeEvent<HTMLInputElement>) => setQuery(event.target.value)}
                  placeholder="Search by name"
                />
                <select
                  aria-label="Select player"
                  value={selectedId ?? ""}
                  onChange={(event: React.ChangeEvent<HTMLSelectElement>) =>
                    selectPlayer(Number(event.target.value))
                  }
                >
                  {filteredPlayers.map((candidate) => (
                    <option
                      key={candidate.player_id}
                      value={candidate.player_id}
                    >
                      {candidate.display_name}
                    </option>
                  ))}
                </select>
                <button
                  type="submit"
                  disabled={filteredPlayers.length === 0}
                >
                  View
                </button>
              </div>
            </form>
          )}
        </header>

        {error && (
          <div className="error">
            <b>We could not load the latest data.</b>
            <span>{error}</span>
          </div>
        )}

        {(loading || detailLoading) && view === "player" && (
          <div className="loading">
            <i />
            <span>Loading player outlook…</span>
          </div>
        )}

        {view === "player" &&
          !loading &&
          !detailLoading &&
          !player &&
          !error && (
            <div className="empty">
              No serving data yet. Run the serving build and database
              load.
            </div>
          )}

        {view === "player" && player && !detailLoading && (
          <PlayerDashboard
            player={player}
            watched={watchlist.has(player.player_id)}
            onToggleWatchlist={toggleWatchlist}
            onSelectPlayer={selectPlayer}
          />
        )}

        {view === "watchlist" && (
          <div className="cards">
            {watchedPlayers.length ? (
              watchedPlayers.map((candidate) => (
                <button
                  className="playerCard"
                  key={candidate.player_id}
                  onClick={() => selectPlayer(candidate.player_id)}
                >
                  <span>
                    {candidate.position ?? "Position unavailable"}
                  </span>
                  <div className="playerCardIdentity">
                    <i>{initials(candidate.display_name)}</i>
                    <b>{candidate.display_name}</b>
                  </div>
                  <div>
                    <strong>
                      {money(candidate.current_market_value_eur)}
                    </strong>
                    <small>Published value</small>
                  </div>
                  <div>
                    <strong>
                      {candidate.current_form_rating?.toFixed(2) ??
                        "—"}
                    </strong>
                    <small>Current form</small>
                  </div>
                </button>
              ))
            ) : (
              <div className="empty">
                Add players from Player Outlook to build a local
                shortlist.
              </div>
            )}
          </div>
        )}

        {view === "catalog" && (
          <div className="technicalGrid">
            <JsonPanel
              title="DATA CATALOG"
              value={catalog ?? "Loading…"}
            />
            <JsonPanel
              title="LINEAGE"
              value={lineage ?? "Loading…"}
            />
          </div>
        )}

        {view === "health" && (
          <article className="pipelineCard">
            <p className="label">RECENT PIPELINE RUNS</p>
            <div className="runs">
              {runs.length ? (
                runs.map((run, index) => (
                  <div
                    className="run"
                    key={String(run.run_id ?? index)}
                  >
                    <b>{String(run.pipeline ?? "pipeline")}</b>
                    <span
                      className={String(run.status ?? "unknown")}
                    >
                      {String(run.status ?? "unknown")}
                    </span>
                    <small>{String(run.started_at ?? "")}</small>
                  </div>
                ))
              ) : (
                <div className="empty">
                  No database pipeline runs have been recorded yet.
                </div>
              )}
            </div>
          </article>
        )}

        {view !== "player" && (
          <footer>
            Data refreshed through the active serving layer ·{" "}
            {fullDate(new Date().toISOString())}
          </footer>
        )}
      </section>
    </main>
  );
}
