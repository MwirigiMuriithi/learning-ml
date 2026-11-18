# Model Card: <project name>
*(Adapted from Mitchell et al., 2019, "Model Cards for Model Reporting". Keep it to ~1 page. Replace every <…>.)*

## Model details
- **Developer / date / version:** <…>
- **Model type & architecture:** <e.g. 3-block CNN, 35k parameters; 3-seed LSTM ensemble>
- **Framework & training recipe:** <PyTorch 2.x; Adam lr…, epochs…, augmentation…, seeds…>

## Intended use
- **Primary intended use:** <who, what decision, in what setting>
- **Out-of-scope uses:** <e.g. replacing an agronomist; commercial trading decisions; diseases/crops not covered>

## Data
- **Training / validation / test data:** <source, size, collection method, splits (chronological? by field?), preprocessing>
- **Known gaps & biases:** <backgrounds, devices, regions, seasons, class balance, synthetic vs real>

## Metrics
| Metric | Value | Notes |
|:--|:-:|:--|
| Test accuracy | | |
| Recall per class | | worst class: |
| Price MAE (4 w) vs naive | | relative MAE: |
| Interval coverage (nominal 80 %) | | |
- **Uncertainty handling:** <thresholds for diagnose/retake/refer; how intervals are calibrated>
- **Statistical caveats:** <test-set sizes, seeds/variance>

## Limitations
1. <specific, evidence-based>
2. <…>
3. <…>

## Ethical considerations & risks
- **Harms from wrong output:** <who, how severe>
- **Who might be excluded:** <connectivity, language, literacy, devices>
- **Data consent & ownership:** <…>
- **Mitigations:** <abstention, human-in-the-loop, monitoring, feedback channel>

## Maintenance
- **How drift is monitored:** <…>  **Update plan:** <…>  **Contact / feedback:** <…>
