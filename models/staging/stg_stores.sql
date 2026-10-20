-- Store master data. Source: Rossmann store.csv
select
    cast(Store as integer)                          as store_id,
    lower(cast(StoreType as varchar))               as store_type,
    lower(cast(Assortment as varchar))              as assortment,
    cast(CompetitionDistance as double)             as competition_distance,
    cast(Promo2 as integer)                         as promo2
from {{ source('rossmann', 'store') }}
