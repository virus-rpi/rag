ATTACH TABLE _ UUID 'fbbfeaa3-0ade-4d48-addc-fbb00f809bd0'
(
    `id` Int64,
    `page_content` String,
    `embeddings` Array(Float32),
    VECTOR INDEX vector_index embeddings TYPE SCANN,
    CONSTRAINT check_data_length CHECK length(embeddings) = 768
)
ENGINE = MergeTree
ORDER BY id
SETTINGS index_granularity = 8192
