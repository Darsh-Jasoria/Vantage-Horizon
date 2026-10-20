-- Planned calendar for future days (promos, closures, holidays). Source: Rossmann test.csv
select
    cast(Store as integer)                  as store_id,
    cast("Date" as date)                    as sale_date,
    cast(DayOfWeek as integer)              as day_of_week,
    cast(null as double)                    as sales,
    cast(null as double)                    as customers,
    cast(coalesce("Open", 1) as integer)    as is_open,
    cast(Promo as integer)                  as is_promo,
    cast(StateHoliday as varchar)           as state_holiday,
    cast(SchoolHoliday as integer)          as is_school_holiday,
    'planned'                               as row_type
from {{ source('rossmann', 'test') }}
