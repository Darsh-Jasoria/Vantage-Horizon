-- The one monthly KPI table every visual reads from.
-- Actual months carry sales/customers; planned months carry only the calendar drivers.
-- Forecast rows are appended by the Python job into fct_kpi_forecast.
select
    date_trunc('month', sale_date)::date                        as month,
    coalesce(store_type, 'All')                                 as segment,
    case when max(row_type) = 'planned' then 'planned' else 'actual' end as row_type,
    sum(sales)                                                  as sales,
    sum(customers)                                              as customers,
    sum(sales) / nullif(sum(customers), 0)                      as avg_ticket,
    sum(is_open)                                                as open_store_days,
    sum(is_promo * is_open)::double / nullif(sum(is_open), 0)   as promo_share,
    count(distinct store_id)                                    as stores
from {{ ref('fct_store_day') }}
group by grouping sets ((date_trunc('month', sale_date), store_type),
                        (date_trunc('month', sale_date)))
order by month, segment
