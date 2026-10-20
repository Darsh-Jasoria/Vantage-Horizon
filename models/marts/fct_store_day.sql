-- One row per store per day: actuals plus the planned future calendar.
-- This is the grain the daily LightGBM model trains and scores on.
with days as (
    select * from {{ ref('stg_sales') }}
    union all
    select * from {{ ref('stg_plan') }}
    where sale_date > (select max(sale_date) from {{ ref('stg_sales') }})
)
select
    d.*,
    s.store_type,
    s.assortment,
    s.competition_distance,
    s.promo2
from days d
join {{ ref('stg_stores') }} s using (store_id)
