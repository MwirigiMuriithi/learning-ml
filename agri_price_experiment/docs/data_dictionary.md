# Data Dictionary

## Panel columns (`data/synthetic_panel_<scenario>.csv`)

| column                   | type / unit                                                               | description                                                                                                                                                    | may be missing   | availability at forecast origin                                                           |
|:-------------------------|:--------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|:------------------------------------------------------------------------------------------|
| week_start               | date (ISO)                                                                | Monday of the reporting week                                                                                                                                   | no               | model input only via calendar arithmetic                                                  |
| week_index               | int 1-156                                                                 | Consecutive week number; partitions are defined on it                                                                                                          | no               | known                                                                                     |
| market_id                | MKT-01..05                                                                | Market identifier (Nairobi, Nakuru, Eldoret, Kisumu, Mombasa - illustrative names)                                                                             | no               | known                                                                                     |
| crop_id                  | CRP-01..04                                                                | Crop identifier (maize, tomatoes, Irish potatoes, onions)                                                                                                      | no               | known                                                                                     |
| wholesale_price_kes_kg   | KES/kg                                                                    | OBSERVED weekly wholesale price = latent price x measurement noise, possibly corrupted or missing. **Forecast target.**                                        | yes              | week w known at the end of week w                                                         |
| latent_price_kes_kg      | KES/kg                                                                    | True generator price before noise/corruption. **Audit only - never a feature, never a training target.**                                                       | no               | NEVER available to models                                                                 |
| market_arrivals_tonnes   | tonnes/week                                                               | Quantity arriving at the market (supply)                                                                                                                       | yes              | week w known at w+report_delay_weeks                                                      |
| estimated_demand_index   | index (100 = market baseline x level)                                     | Relative demand index incl. seasonal, holiday and surge effects                                                                                                | yes              | as above                                                                                  |
| rainfall_mm              | mm/week                                                                   | Rainfall in the market's supply catchment                                                                                                                      | yes              | on time                                                                                   |
| temperature_c            | deg C                                                                     | Mean weekly temperature in the catchment                                                                                                                       | yes              | on time                                                                                   |
| drought_index            | 0-4                                                                       | Rainfall-deficit index over the previous 8 weeks (0 = normal)                                                                                                  | yes              | on time                                                                                   |
| fuel_cost_index          | index (100 = baseline)                                                    | National fuel cost index                                                                                                                                       | yes              | on time (missing weeks hit all markets)                                                   |
| transport_cost_kes_tonne | KES/tonne                                                                 | Cost to move the crop to the market (distance x rate x fuel x disruption)                                                                                      | yes              | week w known at w+report_delay_weeks                                                      |
| regional_supply_index    | ratio (1 = expected)                                                      | Weighted supply of connected source markets in the PREVIOUS week relative to their expectation                                                                 | yes              | week w known at w+report_delay_weeks                                                      |
| holiday_or_event_flag    | 0/1                                                                       | 1 if holiday week (Christmas window, week before Easter) or a documented demand-surge event applies                                                            | no               | known for week w at the end of w; only the pure holiday calendar is used for FUTURE weeks |
| data_quality_flag        | clean / missing / suspect / synthetic_shock (/ corrected in cleaned data) | Row quality label. Precedence missing > suspect > synthetic_shock > clean. `suspect` is an audit label of an injected recording error, **never a model input** | no               | audit only                                                                                |
| source_market_id         | MKT-xx                                                                    | Dominant source market (highest network weight)                                                                                                                | no               | static metadata                                                                           |
| report_delay_weeks       | 0 or 1                                                                    | 1 = indicator record for this week is delivered one week late                                                                                                  | no               | known at delivery; used to build the as-known view                                        |
| event_ids                | string                                                                    | Documented events whose effect window covers the row (separate from quality labels)                                                                            | no               | audit only                                                                                |
| event_affected           | bool                                                                      | Row lies in an event window (excluding the lasting regime shift)                                                                                               | no               | audit/evaluation only                                                                     |
| regime_shift_flag        | bool                                                                      | Row lies after the lasting transport-cost regime shift                                                                                                         | no               | audit/evaluation only                                                                     |
| scenario                 | clean / realistic / stressed                                              | Data-quality scenario of the file                                                                                                                              | no               | metadata                                                                                  |

Rows: 5 markets x 4 crops x 156 weeks = 3,120 per scenario. Key = (`market_id`, `crop_id`, `week_start`).

## Other data files

- `data/events_schedule.csv` - the documented event calendar (type, markets, crops, weeks, intensity, partition, `new_in_test`).
- `data/data_quality_annotations.csv` - every injected data-quality issue (missing price/indicator, outlier, outage, delay) per scenario.
- `data/clean_world_audit.csv` - the clean latent world (audit only).
- `data/manifest.json` - seed, config hash and file checksums.

## Engineered features (`src/agriprice/features.py`)

Feature sets: **E0** = E0 price-only; **E1** = E1 +supply/demand; **E2** = E2 +weather; **E3** = E3 +logistics; **E4** = E4 full. Every set also contains `market_id` and `crop_id`.

| feature            | group         | enters at   | definition                                                                                             |
|:-------------------|:--------------|:------------|:-------------------------------------------------------------------------------------------------------|
| tgt_week_of_year   | calendar      | E0          | ISO week of the TARGET date                                                                            |
| tgt_month          | calendar      | E0          | month of target                                                                                        |
| tgt_quarter        | calendar      | E0          | quarter of target                                                                                      |
| tgt_sin1           | calendar      | E0          | sin(2*pi*doy/365.25) of target                                                                         |
| tgt_cos1           | calendar      | E0          | cos of target (annual)                                                                                 |
| tgt_sin2           | calendar      | E0          | second harmonic sin                                                                                    |
| tgt_cos2           | calendar      | E0          | second harmonic cos                                                                                    |
| tgt_holiday        | calendar      | E0          | holiday calendar flag for the target week (calendar only)                                              |
| org_week_of_year   | calendar      | E0          | ISO week of the origin                                                                                 |
| org_holiday        | calendar      | E0          | holiday flag of the origin week                                                                        |
| lp0                | price         | E0          | log of latest available (cleaned) price                                                                |
| lr_1               | price         | E0          | log change over last 1 week (origin vs origin-1)                                                       |
| lr_2               | price         | E0          | ... 2 weeks                                                                                            |
| lr_3               | price         | E0          | ... 3 weeks                                                                                            |
| lr_4               | price         | E0          | ... 4 weeks                                                                                            |
| lr_8               | price         | E0          | ... 8 weeks                                                                                            |
| lr_13              | price         | E0          | ... 13 weeks                                                                                           |
| lr_52              | price         | E0          | ... 52 weeks                                                                                           |
| dev_mean_4         | price         | E0          | log price minus mean log price of last 4 weeks                                                         |
| dev_mean_8         | price         | E0          | ... 8 weeks                                                                                            |
| dev_mean_13        | price         | E0          | ... 13 weeks                                                                                           |
| dev_mean_26        | price         | E0          | ... 26 weeks                                                                                           |
| vol_4              | price         | E0          | std of weekly log changes, last 4 weeks                                                                |
| vol_8              | price         | E0          | std of weekly log changes, last 8 weeks                                                                |
| seasonal_ref_ratio | price         | E0          | log(price one year before the TARGET week) minus log(latest price) - horizon-specific, known at origin |
| price_stale_weeks  | price         | E0          | weeks since the last accepted price (handling on only)                                                 |
| arr_l0             | supply_demand | E1          | log arrivals, latest delivered value                                                                   |
| arr_l1             | supply_demand | E1          | log arrivals, one week earlier                                                                         |
| arr_mean4          | supply_demand | E1          | log mean arrivals, last 4 delivered weeks                                                              |
| arr_anom13         | supply_demand | E1          | latest arrivals vs 13-week mean (log ratio)                                                            |
| dem_l0             | supply_demand | E1          | log demand index, latest                                                                               |
| dem_l1             | supply_demand | E1          | log demand index, one week earlier                                                                     |
| dem_mean4          | supply_demand | E1          | log mean demand, 4 weeks                                                                               |
| event_flag_l0      | supply_demand | E1          | holiday/surge flag of the origin week                                                                  |
| rain_l0            | weather       | E2          | latest rainfall                                                                                        |
| rain_mean4         | weather       | E2          | mean rainfall 4 wks                                                                                    |
| rain_mean8         | weather       | E2          | mean rainfall 8 wks                                                                                    |
| rain_sum13         | weather       | E2          | cumulative rainfall 13 wks                                                                             |
| temp_l0            | weather       | E2          | latest temperature                                                                                     |
| temp_mean4         | weather       | E2          | mean temperature 4 wks                                                                                 |
| drought_l0         | weather       | E2          | latest drought index                                                                                   |
| drought_l2         | weather       | E2          | drought index 2 weeks earlier                                                                          |
| drought_mean4      | weather       | E2          | mean drought index 4 wks                                                                               |
| drought_max8       | weather       | E2          | max drought index 8 wks                                                                                |
| fuel_l0            | logistics     | E3          | log fuel index, latest                                                                                 |
| fuel_chg4          | logistics     | E3          | log change of fuel index over 4 weeks                                                                  |
| trans_l0           | logistics     | E3          | log transport cost, latest delivered                                                                   |
| trans_mean4        | logistics     | E3          | log mean transport cost 4 wks                                                                          |
| trans_chg4         | logistics     | E3          | log change of transport cost over 4 wks                                                                |
| conn_gap0          | cross_market  | E4          | weighted log price of source markets minus own log price (same crop)                                   |
| conn_chg1          | cross_market  | E4          | weighted 1-week log price change of source markets                                                     |
| conn_chg4          | cross_market  | E4          | weighted 4-week log price change of source markets                                                     |
| regsup_l0          | cross_market  | E4          | log regional supply index, latest                                                                      |
| regsup_mean4       | cross_market  | E4          | log mean regional supply, 4 wks                                                                        |
| regsup_chg1        | cross_market  | E4          | 1-week log change of regional supply index                                                             |
