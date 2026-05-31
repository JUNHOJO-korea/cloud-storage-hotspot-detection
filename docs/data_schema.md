# Data Schema

The original industrial raw data are private, but the code expects the following schemas.

## Base Long Table

Required columns:

- `timestamp`: observation timestamp
- `stg_ip`: storage system identifier
- `level`: one of `system`, `node`, `port`
- `entity_id`: entity identifier within the level
- `rw`: one of `read`, `write`, `cpu`, `mix`
- `metric`: metric family name
- `value`: numeric observation

## Expected Raw Parquet Inputs

### `data_system.parquet`

- `timestamp`
- `stg_ip`
- `metric`
- `value`

### `data_node.parquet`

- `timestamp`
- `stg_ip`
- `node_name`
- `metric`
- `value`

### `data_port.parquet`

- `timestamp`
- `stg_ip`
- `node_id`
- `port_id`
- `metric`
- `value`

## Derived Tables

### `imb_long`

- all base long keys
- `imb_metric`
- `src_metric`
- `n_entities`
- `active_entities`
- `group_total`

### `segments_final`

- context keys: `stg_ip`, `level`, `rw`, `src_metric`
- time keys: `start_ts`, `end_ts`, `peak_ts`
- event fields: `duration_bins`, `peak_value`, `hotspot_tier`, `hotspot_score`

### `point_labels`

- `timestamp`
- `stg_ip`
- `level`
- `rw`
- `src_metric`
- `segment_id`
- `hotspot_tier`
- `hotspot_score`

## Open-Source Recommendation

If the industrial dataset cannot be released, provide:

- a synthetic sample dataset with the same schema
- a schema validator
- an artifact manifest for expected derived outputs

