ATTACH TABLE _ UUID '5621aa05-7030-4c38-b4a7-50450de35f5d'
(
    `value` UInt64
)
ENGINE = MergeTree
ORDER BY value
SETTINGS index_granularity = 8192
