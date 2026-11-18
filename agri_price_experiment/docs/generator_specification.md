# Synthetic Data Generator Specification

Configuration: `config/experiment_config.yaml` (seed **42**). Code: `src/agriprice/generator.py` (clean world) and `src/agriprice/degrade.py` (data quality).
The data are **fully synthetic and illustrative** - they mimic the *structure* of agricultural wholesale markets, not any real market.

## 1. Structure

156 weeks starting 2023-01-02, 5 markets x 4 crops = 20 series, 60 burn-in weeks discarded.
Weekly log price of crop *c* in market *m*:

```
log P[m,c,t] = log(base_c) + alpha[m,c] + S[c,t] + z[m,c,t]

S[c,t]   = -A1_c cos(2*pi (doy_t - peak_c)/365.25) + A2_c cos(4*pi (doy_t - peak_c - 60)/365.25)        # crop-specific seasonality
z[m,c,t] = phi_c * [ (1-delta_c) z[m,c,t-1] + delta_c * sum_j W[m,j] z[j,c,t-1] ]                      # persistence + cross-market spillover
         + (1-phi_c) * [ beta_s f(s_eff) + beta_d d_eff + beta_tr rw_m tau + beta_reg log RSI + gamma_dr drought ]   # adjustment to drivers
         + sigma_c * noise_m * eps_t                                                                     # random variation (Student-t, df 5)
```

This is the design equation of the specification in *partial-adjustment form*: the lagged-price term `phi*logP[t-1]` is written on deviations from the
deterministic level (`mu + alpha + S`) and the driver coefficients are long-run elasticities (weekly loading `1-phi`). Perishable tomatoes (phi 0.55) react
quickly; storable maize and onions (phi 0.90-0.92) adjust slowly.

* `s_eff` - supply anomaly `log(arrivals/expected)`, smoothed with `storage_lambda` for storable crops. `f(s) = s (1 + kappa min(max(-s,0),0.6))` makes **scarcity hit prices harder than glut relief** (nonlinear).
* `d_eff` - demand anomaly, amplified when supply is short (interaction).
* `tau` - log transport-cost deviation (fuel x disruption), scaled by route length `rw_m`.
* `RSI` - regional supply index: weighted arrivals of source markets in the **previous week** relative to expectation (also feeds back into the market's own arrivals, `kappa=0.35`) - this is the one-week-lag cross-market transmission.
* `drought` - 8-week rainfall-deficit index (small direct anticipatory price effect).
* Arrivals = `expected (harvest seasonality) x weather factor x regional response x event multipliers x AR(1) noise`. The **weather factor depends on rainfall, drought and (tomatoes) heat with a crop-specific delay** (tomatoes 3 weeks, maize 10 weeks), so weather history carries information about *future* supply.
* Weather: bimodal rainfall climatology (long rains Mar-May, short rains Oct-Dec) with spatially correlated lognormal anomalies; temperature seasonal + AR(1).
* Demand: market level x seasonal x holiday boost (Christmas, Easter) x surge events x AR(1) noise.
* Logistics: AR(1) fuel index, step/regime events; `transport = distance x rate x fuel/100 x disruption x noise`.

## 2. Parameters

### Crops
| crop           |   base KES/kg |   phi (persistence) |   sigma (weekly) |   annual amp |   harvest peak doy |   storage lambda |   beta supply |   beta demand |   beta transport |   beta regional |   gamma drought |   weather lag (wks) |   spillover delta |
|:---------------|--------------:|--------------------:|-----------------:|-------------:|-------------------:|-----------------:|--------------:|--------------:|-----------------:|----------------:|----------------:|--------------------:|------------------:|
| Maize          |        50.000 |               0.920 |            0.014 |        0.130 |                320 |            0.700 |        -0.160 |         0.100 |            0.220 |          -0.100 |           0.050 |                  10 |             0.250 |
| Tomatoes       |        72.000 |               0.550 |            0.060 |        0.150 |                190 |            0.000 |        -0.360 |         0.280 |            0.220 |          -0.180 |           0.030 |                   3 |             0.200 |
| Irish potatoes |        44.000 |               0.800 |            0.030 |        0.140 |                215 |            0.400 |        -0.300 |         0.150 |            0.300 |          -0.140 |           0.030 |                   6 |             0.220 |
| Onions         |        78.000 |               0.900 |            0.022 |        0.110 |                250 |            0.500 |        -0.260 |         0.120 |            0.200 |          -0.120 |           0.040 |                   6 |             0.200 |

### Markets
| id     | market   |   demand level |   distance km |   log premium |   noise mult |   shock sensitivity |   rain mult |
|:-------|:---------|---------------:|--------------:|--------------:|-------------:|--------------------:|------------:|
| MKT-01 | Nairobi  |          1.250 |           160 |         0.100 |        0.850 |               0.900 |       1.000 |
| MKT-02 | Nakuru   |          0.850 |            90 |         0.020 |        1.000 |               1.000 |       1.000 |
| MKT-03 | Eldoret  |          0.800 |            60 |        -0.090 |        1.100 |               1.300 |       1.300 |
| MKT-04 | Kisumu   |          0.950 |           220 |         0.040 |        1.150 |               1.100 |       1.400 |
| MKT-05 | Mombasa  |          1.100 |           480 |         0.140 |        1.000 |               0.900 |       1.100 |

### Market network (weights of source markets; row sums = 1)
| receiving market   |   MKT-02 |   MKT-03 |   MKT-04 |   MKT-01 |
|:-------------------|---------:|---------:|---------:|---------:|
| MKT-01             |     0.50 |     0.35 |     0.15 |     0.00 |
| MKT-02             |     0.00 |     0.75 |     0.25 |     0.00 |
| MKT-03             |     0.60 |     0.00 |     0.40 |     0.00 |
| MKT-04             |     0.40 |     0.60 |     0.00 |     0.00 |
| MKT-05             |     0.30 |     0.25 |     0.00 |     0.45 |

## 3. Documented event schedule

Week numbers are 1-based. Partitions: train 1-104, validation 105-130, test 131-156.
`new_in_test = True` marks shocks of a new type or greater strength than seen in training (EV12 longer/stronger drought, EV13 lasting transport-cost regime shift, EV14/EV15 stronger glut/import shock).

| event_id   | event_type                 | markets                            | crops                       |   start_week |   end_week |   effect_tail_weeks |   intensity | partition   | new_in_test   | description                                                      |
|:-----------|:---------------------------|:-----------------------------------|:----------------------------|-------------:|-----------:|--------------------:|------------:|:------------|:--------------|:-----------------------------------------------------------------|
| EV01       | harvest_glut               | MKT-02|MKT-03                      | CRP-01                      |           38 |         42 |                   2 |       0.600 | train       | False         | Maize harvest glut in production corridor                        |
| EV02       | harvest_glut               | MKT-02|MKT-03                      | CRP-02                      |           70 |         72 |                   2 |       0.800 | train       | False         | Tomato glut after unusually high arrivals                        |
| EV03       | drought                    | MKT-02|MKT-03|MKT-04               | CRP-01|CRP-02|CRP-03|CRP-04 |           55 |         62 |                   8 |       0.700 | train       | False         | Dry-season drought; arrivals fall with a lag                     |
| EV04       | flood                      | MKT-04|MKT-05                      | CRP-01|CRP-02|CRP-03|CRP-04 |           16 |         18 |                   3 |       1.000 | train       | False         | Heavy rainfall, route disruption                                 |
| EV05       | fuel_step                  | MKT-01|MKT-02|MKT-03|MKT-04|MKT-05 | CRP-01|CRP-02|CRP-03|CRP-04 |           85 |         85 |                  15 |       0.250 | train       | False         | Fuel-price increase, gradual easing                              |
| EV06       | demand_surge               | MKT-01|MKT-05                      | CRP-01|CRP-02|CRP-03|CRP-04 |           49 |         51 |                   1 |       0.250 | train       | False         | Pre-festive demand surge                                         |
| EV07       | import_supply_shock        | MKT-01|MKT-02|MKT-03|MKT-04|MKT-05 | CRP-04                      |           90 |         93 |                   3 |       0.500 | train       | False         | Onion import surge                                               |
| EV08       | cross_market_disruption    | MKT-03                             | CRP-01|CRP-02|CRP-03|CRP-04 |          112 |        115 |                   2 |       0.600 | validation  | False         | Eldoret route/source unavailable; spillover to connected markets |
| EV09       | demand_surge               | MKT-01                             | CRP-01|CRP-02|CRP-03|CRP-04 |          118 |        120 |                   1 |       0.300 | validation  | False         | Demand surge in Nairobi                                          |
| EV10       | flood                      | MKT-04|MKT-02                      | CRP-01|CRP-02|CRP-03|CRP-04 |          121 |        123 |                   3 |       0.800 | validation  | False         | Flood in validation window                                       |
| EV11       | harvest_glut               | MKT-02|MKT-03                      | CRP-03                      |          124 |        127 |                   2 |       0.700 | validation  | False         | Potato glut                                                      |
| EV12       | drought                    | MKT-02|MKT-03|MKT-04               | CRP-01|CRP-02|CRP-03|CRP-04 |          134 |        145 |                   8 |       0.950 | test        | True          | NEW stronger and longer drought in test window                   |
| EV13       | structural_transport_shift | MKT-01|MKT-02|MKT-03|MKT-04|MKT-05 | CRP-01|CRP-02|CRP-03|CRP-04 |          140 |        156 |                   0 |       0.350 | test        | True          | Lasting transport-cost regime shift (+35%, no easing)            |
| EV14       | harvest_glut               | MKT-03|MKT-02                      | CRP-02                      |          147 |        149 |                   2 |       1.000 | test        | True          | Stronger tomato glut in test window                              |
| EV15       | import_supply_shock        | MKT-01|MKT-02|MKT-03|MKT-04|MKT-05 | CRP-04                      |          150 |        152 |                   3 |       0.900 | test        | True          | Stronger onion import shock                                      |
| EV16       | demand_surge               | MKT-01|MKT-05                      | CRP-01|CRP-02|CRP-03|CRP-04 |          154 |        156 |                   0 |       0.250 | test        | False         | Festive demand surge                                             |

Event labels (`event_ids`, `event_affected`) are deliberately separate from data-quality labels so that cleaning rules cannot erase genuine market behaviour.

## 4. Data-quality scenarios (applied to the *same* latent world)

| scenario   |   price noise sd |   indicator noise sd |   price missing (random) |   price missing (gaps) | gap length (wks)   |   outlier rate |   late-delivery prob | late markets                   | outages                          |
|:-----------|-----------------:|---------------------:|-------------------------:|-----------------------:|:-------------------|---------------:|---------------------:|:-------------------------------|:---------------------------------|
| clean      |            0.004 |                0.000 |                    0.000 |                  0.000 | [2, 3]             |          0.000 |                0.000 | -                              | -                                |
| realistic  |            0.020 |                0.030 |                    0.022 |                  0.015 | [2, 3]             |          0.007 |                0.300 | MKT-04, MKT-05                 | -                                |
| stressed   |            0.035 |                0.050 |                    0.025 |                  0.020 | [4, 8]             |          0.010 |                0.450 | MKT-04, MKT-05, MKT-02, MKT-03 | MKT-04 w120-127; MKT-05 w141-147 |

Indicator missingness by field (realistic): {'market_arrivals_tonnes': 0.08, 'estimated_demand_index': 0.06, 'rainfall_mm': 0.05, 'temperature_c': 0.05, 'drought_index': 0.05, 'fuel_cost_index': 0.05, 'transport_cost_kes_tonne': 0.07, 'regional_supply_index': 0.06}; stressed: {'market_arrivals_tonnes': 0.12, 'estimated_demand_index': 0.1, 'rainfall_mm': 0.08, 'temperature_c': 0.08, 'drought_index': 0.08, 'fuel_cost_index': 0.08, 'transport_cost_kes_tonne': 0.1, 'regional_supply_index': 0.1}.
Reporting errors multiply the true value by 10, 0.1, 2.5 or 0.4, or report 0 (an impossible value).
Late-delivered records: supply, demand, transport and regional-supply indicators of selected markets reach the forecaster one week after the period ends.

## 5. Acceptance criteria

See `reports/DATA_AUDIT.md` (coverage, uniqueness, validity, seasonality, market differences, shock behaviour, persistence, missingness, no leakage, reproducibility).
