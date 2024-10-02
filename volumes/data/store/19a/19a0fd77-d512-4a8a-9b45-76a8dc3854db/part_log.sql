ATTACH TABLE _ UUID 'fae5a7b3-a514-4d5f-a988-35c441b71fcc'
(
    `query_id` String,
    `event_type` Enum8('NewPart' = 1, 'MergeParts' = 2, 'DownloadPart' = 3, 'RemovePart' = 4, 'MutatePart' = 5, 'MovePart' = 6, 'DownloadVectorIndex' = 7),
    `merge_reason` Enum8('NotAMerge' = 1, 'RegularMerge' = 2, 'TTLDeleteMerge' = 3, 'TTLRecompressMerge' = 4),
    `merge_algorithm` Enum8('Undecided' = 0, 'Vertical' = 1, 'Horizontal' = 2),
    `event_date` Date,
    `event_time` DateTime,
    `event_time_microseconds` DateTime64(6),
    `duration_ms` UInt64,
    `database` String,
    `table` String,
    `table_uuid` UUID,
    `part_name` String,
    `partition_id` String,
    `part_type` String,
    `disk_name` String,
    `path_on_disk` String,
    `rows` UInt64,
    `size_in_bytes` UInt64,
    `merged_from` Array(String),
    `bytes_uncompressed` UInt64,
    `read_rows` UInt64,
    `read_bytes` UInt64,
    `peak_memory_usage` UInt64,
    `error` UInt16,
    `exception` String,
    `ProfileEvents` Map(String, UInt64),
    `ProfileEvents.Names` Array(String) ALIAS mapKeys(ProfileEvents),
    `ProfileEvents.Values` Array(UInt64) ALIAS mapValues(ProfileEvents)
)
ENGINE = MergeTree
PARTITION BY toYYYYMM(event_date)
ORDER BY (event_date, event_time)
TTL event_date + toIntervalDay(30)
SETTINGS index_granularity = 8192
