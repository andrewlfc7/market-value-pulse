# Market Value Pulse architecture

## End-to-end system

```mermaid
flowchart LR
    subgraph Sources
        WS[WhoScored<br/>match events and lineups]
        TM[Transfermarkt<br/>rosters and valuation history]
    end

    subgraph Acquisition
        WI[WhoScored ingestion<br/>discovery, retries and manifests]
        TI[Transfermarkt ingestion<br/>rate limiting, retries and manifests]
    end

    subgraph DataLayer[Storage and validation]
        RAW[(Immutable raw responses)]
        NORM[(Normalized Parquet<br/>schema and quality checks)]
    end

    subgraph FootballFeatures[Performance features]
        FE[Event enrichment<br/>xG, xGOT, xA, xT and xPV]
        RT[Position-aware<br/>post-match ratings]
        ST[(Incremental form state<br/>EWM, rolling 3 and rolling 20)]
        ER[Entity resolution<br/>exact mapping and review queue]
    end

    subgraph Valuation[Valuation modeling]
        DS[Leakage-safe<br/>valuation intervals]
        BM[Hierarchical Bayesian<br/>Student-t regression]
        QC[Chronological holdout<br/>and promotion checks]
        AM[(Active model)]
        CS[Current player scoring<br/>estimate, 90% range and probability]
    end

    subgraph Application[Serving and application]
        SP[(Serving Parquet)]
        PG[(PostgreSQL)]
        API[FastAPI]
        UI[Next.js dashboard]
    end

    WS --> WI --> RAW
    TM --> TI --> RAW
    RAW --> NORM
    NORM --> FE --> RT --> ST
    NORM --> ER
    RT --> DS
    ER --> DS
    NORM --> DS
    DS --> BM --> QC --> AM
    AM --> CS
    ST --> CS
    ER --> CS
    CS --> SP
    NORM --> SP
    RT --> SP
    SP --> PG --> API --> UI
    SP -. local fallback .-> API
```

## Incremental update path

```mermaid
flowchart LR
    A[New completed match]
    B[Ingest and normalize]
    C{Content hash<br/>new or changed?}
    D[Skip unchanged partition]
    E[Enrich affected match]
    F[Append player ratings]
    G[Refresh rolling form state]
    H[Build current features<br/>for affected players]
    I[Score with active<br/>valuation model]
    J[Write estimate,<br/>90% range and probability]
    K[Update serving Parquet<br/>and PostgreSQL]
    L[Dashboard reads<br/>refreshed player outlook]

    A --> B --> C
    C -- No --> D
    C -- Yes --> E --> F --> G --> H --> I --> J --> K --> L
```

A completed match is the unit of incremental work. Content hashes make retries
idempotent, unchanged partitions are skipped, and only affected player state and
forecasts need to be refreshed.
