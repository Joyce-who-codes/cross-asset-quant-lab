# High-Frequency Data

## Data sources and overall organization

The high-frequency execution component of this study is built on Binance futures market data obtained from Tardis. The raw data are organized at the daily level and consist of three complementary streams: incremental level-2 order book updates, transaction-level trade prints, and 25-level full order book snapshots. These three streams play different but tightly connected roles in the execution simulation. The incremental order book stream describes how the visible bid and ask queues evolve over time. The trade stream records realized market transactions and is used to characterize recent order-flow pressure. The snapshot stream provides a full cross-section of the order book at a given instant and is used to initialize replay efficiently near an arbitrary starting time.

At the storage level, the project uses a two-layer design. The first layer is the raw daily Tardis data under the symbol-specific folders for `incremental_book_L2`, `trades`, and `snapshot_25`. The second layer is a replay-optimized chunk store in parquet format. The raw daily files are parsed, normalized, and then partitioned into fixed-length chunks, which are subsequently used by the reinforcement learning environment for training and evaluation. This design separates archival storage from computational storage and substantially reduces repeated parsing and replay overhead.

Table 1 summarizes the three raw high-frequency inputs.

| Data stream | Economic meaning | Typical raw fields | Role in replay |
| --- | --- | --- | --- |
| Incremental level-2 book | Changes in visible order book depth at each price level | `timestamp`, `local_timestamp`, `side`, `price`, `amount`, `is_snapshot` | Drives dynamic updates of bid and ask queues |
| Trades | Realized transactions in the market | `timestamp`, `local_timestamp`, `id`, `side`, `price`, `amount` | Provides recent trade imbalance and signed volume features |
| Snapshot 25 | Full 25-level order book cross-section at a point in time | `asks[i].price`, `asks[i].amount`, `bids[i].price`, `bids[i].amount` | Fast bootstrapping of the order book before replay |

## Raw data structure

The incremental level-2 order book file records event-by-event changes in displayed liquidity. Each row corresponds to a single price-level update on either the bid or ask side. In the raw format, the key fields are `exchange`, `symbol`, `timestamp`, `local_timestamp`, `is_snapshot`, `side`, `price`, and `amount`. Here, `side` identifies whether the updated level belongs to the bid or ask book, while `amount` denotes the standing size at that price level after the update rather than a transacted quantity. Conceptually, a level-2 book row answers the question: at this time, what is the updated visible quantity at this exact price level?

The trade file has a different interpretation. Each row represents an actual market transaction, with the main fields `exchange`, `symbol`, `timestamp`, `local_timestamp`, `id`, `side`, `price`, and `amount`. In this case, `amount` is a realized traded quantity and `side` denotes the direction of the aggressive initiator. A trade row therefore records completed market activity rather than a change in standing depth.

The snapshot file contains a dense representation of the book at a particular time. It stores 25 ask levels and 25 bid levels simultaneously, with price and amount for each level. Rather than describing change, the snapshot gives the state of the order book itself. This distinction is important: the snapshot is static, while the level-2 book is dynamic.

Table 2 provides simplified examples of the three raw formats using BTCUSDT.

| Type | Example fields | Interpretation |
| --- | --- | --- |
| Raw book row | `timestamp=1766361600432000`, `side=ask`, `price=88621.3`, `amount=3.417`, `is_snapshot=True` | The ask queue at 88621.3 has standing size 3.417 after this update |
| Raw trade row | `timestamp=1766361600026000`, `side=buy`, `price=88621.3`, `amount=0.005`, `id=7039226776` | A buy-initiated trade of size 0.005 occurred at 88621.3 |
| Raw snapshot row | `asks[0].price=88621.3`, `asks[0].amount=3.417`, `bids[0].price=88621.2`, `bids[0].amount=3.859` | The best ask is 88621.3 with size 3.417 and the best bid is 88621.2 with size 3.859 |

These examples illustrate the economic distinction between the three streams. The snapshot provides a complete picture of the book at one instant, the book updates describe how that picture changes, and the trade prints summarize what was actually executed in the market.

## Data normalization and unified event representation

To make replay computationally tractable, the project first maps the raw book and trade streams into a unified event schema. This normalization is implemented in `src/data/parse_tardis_csv.py`. The parser converts raw timestamps into the common columns `exch_ts` and `local_ts`, standardizes side labels into lowercase strings, coerces `price` and `amount` into numeric types, and introduces a shared event representation with fields `event_type`, `exchange`, `symbol`, `exch_ts`, `local_ts`, `is_snapshot`, `side`, `price`, `amount`, and `trade_id`.

In this unified representation, a book update is stored as `event_type = book` and a trade print is stored as `event_type = trade`. Book events carry `trade_id = NA`, while trade events retain the raw trade identifier. The parser then adds compact encoded variables, namely `event_code` and `side_code`, which are used later by the replay engine for efficient array-based processing.

After normalization, the book and trade tables are concatenated and sorted into a single time-ordered event stream. The sorting logic gives book events priority over trade events at the same timestamp and then breaks ties using local timestamps. This ordering ensures that the state of the book is updated before same-timestamp trade information is consumed by the replay engine.

Table 3 illustrates the processed event format.

| event_type | exch_ts | local_ts | side | price | amount | trade_id | event_code | side_code |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `trade` | 1766361600026000 | 1766361600031910 | `buy` | 88621.3 | 0.005 | 7039226776 | 1 | 2 |
| `trade` | 1766361600138000 | 1766361600141634 | `sell` | 88621.2 | 0.018 | 7039226780 | 1 | 3 |
| `book` | 1766361600432000 | 1766361600884505 | `ask` | 88621.3 | 3.417 | NA | 0 | 1 |

This event representation is the main input to the replay engine.

## Order book reconstruction and the meaning of the market state

The replay engine reconstructs the order book using an array-ladder representation implemented in `src/backtest/order_book_replay.py`. Instead of storing the book as nested dictionaries keyed by price, the engine uses fixed-size arrays for bid and ask quantities. Each valid price level is mapped to a position in the array according to the tick size and the predefined region of interest. The engine continuously maintains the current best bid and best ask indices, allowing near-constant-time updates and efficient extraction of top-of-book features.

In this context, the term *order book* refers to the visible supply and demand schedule at different prices. The *bid* side represents standing buy interest, and the *ask* side represents standing sell interest. The best bid is the highest currently quoted buy price, and the best ask is the lowest currently quoted sell price. Together they define the current spread. The replayed market state therefore contains not only the best bid and ask but also several levels of surrounding depth. This state forms the observable market environment presented to the reinforcement learning agent.

The snapshot parser in `src/data/parse_tardis_snapshot.py` converts one snapshot row into two dictionaries, one for bids and one for asks. These dictionaries are then written into the replay engine’s array structure, thereby creating a full and internally consistent initial book state.

## Why both snapshots and incremental book data are needed

Although snapshots are sufficient to reconstruct the order book at one particular instant, they are not sufficient to describe how the market evolves afterward. A snapshot is a static cross-section. It does not reveal which queues are added to, cancelled, partially depleted, or moved after that moment. By contrast, the incremental level-2 book stream provides exactly this temporal evolution. The replay engine therefore uses snapshots for initialization and level-2 book events for dynamic updating.

This distinction is critical for execution simulation. The reinforcement learning agent does not trade once at the snapshot time and stop; it enters an episode, places passive or aggressive orders, waits as the market evolves, and then reacts to the next observed market state. Such sequential interaction requires a continuously evolving order book. The only way to obtain that evolution is through the incremental book events.

The trade stream is also necessary, but for a different reason. In the current implementation, trades do not directly overwrite book depth arrays. Instead, they are used to update rolling microstructure features such as recent signed volume and trade imbalance. These features help the agent infer whether recent flow is buy-dominated or sell-dominated, thereby enriching the state representation beyond static depth alone.

Accordingly, the three inputs play complementary roles. Snapshots provide efficient initialization, level-2 book events evolve the order book, and trade events summarize local execution flow.

## How snapshots accelerate replay

The main computational challenge in historical order book replay is reset cost. If the engine wishes to start from an arbitrary event index deep inside the day, the naive approach would be to begin from the start of the daily file and replay all prior order book updates before the chosen starting point. For dense instruments such as BTCUSDT, this is prohibitively expensive because a single 6-hour chunk may contain tens of millions of book updates.

The replay engine addresses this problem by using the latest available snapshot prior to the requested start time. Upon reset, it searches for the most recent 25-level snapshot, reconstructs the book immediately from that snapshot, and then replays only the incremental events between the snapshot time and the target start index. If an external snapshot file is unavailable, the engine falls back to embedded snapshot blocks inside the book stream through the `is_snapshot` marker. This design drastically shortens the replay prefix needed at each reset and therefore accelerates repeated reinforcement learning episodes.

In other words, snapshots do not replace level-2 book data; they reduce the amount of level-2 history that must be replayed before the episode can begin.

## Chunking and parquet conversion

Because the raw Tardis files are stored at the daily level and may contain extremely large numbers of rows, the project introduces an intermediate chunked parquet layer through `scripts/build_tardis_chunks.py`. The script first identifies valid daily raw files for a symbol using `src/utils/tardis_daily.py`. It then loads the normalized book, trade, and snapshot tables for each day and slices them into fixed-width windows. In this project, the default chunk size is six hours, which partitions each trading day into four replay units.

Each chunk is written into a dedicated directory containing `book.parquet`, `trades.parquet`, `snapshot.parquet`, and a lightweight `meta.json`. The parquet files store only the normalized columns needed by replay, and the metadata file records the symbol, chunk name, start and end timestamps, and the row counts for each stream. This chunk layer is later enumerated by `src/utils/tardis_chunk.py`, which supplies the training and evaluation pipeline with only complete chunk directories.

The conversion from daily CSV to chunked parquet is not merely a storage convenience. It directly improves training efficiency by reducing I/O, avoiding repeated CSV parsing, shrinking the amount of data loaded per episode, and making random-start replay feasible at scale. The effect is especially important for BTCUSDT. For example, the chunk `2025-12-22_12` contains 42,236,963 book events, 1,566,250 trades, and 356,723 snapshots, as recorded in its metadata file. Without chunking and parquet conversion, repeated experimentation on such dense data would be prohibitively slow.

Table 4 summarizes the chunk structure.

| File | Content | Function in pipeline |
| --- | --- | --- |
| `book.parquet` | Normalized level-2 book events | Updates the replayed order book |
| `trades.parquet` | Normalized trade events | Updates recent flow features |
| `snapshot.parquet` | 25-level snapshots | Fast initialization before replay |
| `meta.json` | Chunk-level metadata | Time range selection and integrity checks |

## Interaction between replayed data and the RL agent

The reinforcement learning agent interacts with the replayed market through the wrapper defined in `src/backtest/tardis_wrapper.py`. The wrapper exposes the replayed best bid, best ask, top-of-book depth, and rolling trade-flow features as the market state. At each step, the agent may choose to hold, place passive buy or sell orders relative to the current best quotes, submit a small market order, or cancel outstanding orders. The subsequent historical order book updates determine whether passive orders would have been filled under the simplified fill logic. Aggressive orders transact immediately at the current best opposite quote.

The key point is that the agent does not alter the historical raw data itself. Instead, the historical market data define the external environment against which the agent’s decisions are evaluated. The replay engine reconstructs the path of the market, while the wrapper maps the agent’s actions into simulated fills, inventory changes, cash flows, and mark-to-market equity. In this sense, the market data and the reinforcement learning policy interact through a one-way historical replay: the market path is fixed, but the realized execution outcome depends on the strategy’s sequence of actions.

## Summary

The high-frequency execution dataset combines static snapshots, dynamic level-2 book updates, and transaction-level trade prints into a coherent replay framework. The data pipeline first standardizes heterogeneous raw feeds into a unified event schema, then converts them into chunked parquet files for efficient access, and finally uses snapshots to bootstrap the order book close to arbitrary starting points. This architecture makes it possible to simulate realistic sequential execution on dense historical order book data while keeping the computational burden manageable. From a methodological perspective, snapshots provide fast initialization, book events supply the true path of displayed liquidity, and trade events enrich the state with recent flow information. Together, these components form the empirical foundation of the high-frequency reinforcement learning execution module.
