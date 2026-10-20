-- Staged daily store sales (actuals). Source: Rossmann train.csv
select
    cast(Store as integer)          as store_id,
    cast("Date" as date)            as sale_date,
    cast(DayOfWeek as integer)      as day_of_week,
    cast(Sales as double)           as sales,
    cast(Customers as double)       as customers,
    cast("Open" as integer)         as is_open,
    cast(Promo as integer)          as is_promo,
    cast(StateHoliday as varchar)   as state_holiday,
    cast(SchoolHoliday as integer)  as is_school_holiday,
    'actual'                        as row_type
from {{ source('rossmann', 'train') }}
