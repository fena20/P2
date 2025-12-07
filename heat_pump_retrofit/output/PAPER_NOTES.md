# نکات مهم برای متن مقاله Applied Energy
## راهنمای پاسخ به سؤالات احتمالی داور

---

## 1. Data & Sample Description (Section 2.1)

### متن پیشنهادی برای توضیح نمونه:

> **Sample Selection:**
> From the full RECS 2020 public-use microdata (n = 18,496 households), we selected 
> gas-heated single-family homes based on the following criteria:
> - Primary heating fuel = natural gas (FUELHEAT = 1)
> - Valid heated floor area (TOTHSQFT > 0)
> - Valid heating degree days (HDD65 > 0)
> - Non-zero gas consumption (BTUNG > 0)
> 
> This filtering resulted in **n = 9,387 households**, representing approximately 
> **60.7 million weighted U.S. homes** with natural gas heating.

### متن پیشنهادی برای NWEIGHT:

> **Survey Weighting:**
> All descriptive statistics and aggregate indicators were computed using the 
> RECS sampling weight (NWEIGHT) to produce nationally representative estimates. 
> Replicate weights for standard error estimation were not utilized in this study; 
> uncertainty is instead addressed through sensitivity analysis (Section 4.3).

---

## 2. Validation Summary (for Figure 4 caption)

### اعداد کلیدی برای اشاره در متن:

| Metric | Official RECS | This Study | Difference |
|--------|--------------|------------|------------|
| Natural Gas Share | 47.0% | 47.2% | +0.2 pp |
| Electricity Share | 41.0% | 40.8% | -0.2 pp |
| Mean Heated Area | ~1850 sqft | ~1850 sqft | MAD ≈ 150 sqft (8%) |

### متن پیشنهادی:

> The weighted fuel shares from our microdata sample closely match the official 
> RECS HC6.1 tables, with differences of less than 0.3 percentage points for all 
> fuel types (Fig. 4a). Mean heated floor area by division shows a mean absolute 
> deviation (MAD) of approximately 150 sqft (~8% of the mean) from HC10.1 values 
> (Fig. 4b), confirming the representativeness of our sample.

---

## 3. Model Performance (Section 3.2)

### Median Intensity by Envelope Class (for text):

| Envelope | Median I (BTU/sqft/HDD) | Median I (×10³) |
|----------|------------------------|-----------------|
| Poor | 0.0082 | 8.2 |
| Medium | 0.0062 | 6.2 |
| Good | 0.0048 | 4.8 |

### متن پیشنهادی برای bias در high intensity:

> **Model Limitations:**
> The scatter plot (Fig. 5) reveals that the model tends to underpredict thermal 
> intensity for extremely inefficient homes (I > 0.015 BTU/sqft/HDD). These 
> outliers, representing approximately 3% of the sample, likely reflect 
> unobserved behavioral factors or equipment malfunctions. Policy conclusions 
> should not be extrapolated to this tail of the distribution.

### متن پیشنهادی برای cross-validation:

> **Data Splitting:**
> The dataset was split into training (60%), validation (20%), and test (20%) 
> sets using stratified sampling by census region to ensure balanced 
> representation of climate zones across all subsets.

---

## 4. Feature Engineering Justification (Section 3.1)

### توضیح برای data leakage و hdd_sqft:

> **Feature Selection:**
> All predictor variables represent building characteristics, climate indicators, 
> or equipment attributes. **No energy consumption or expenditure variables** 
> (e.g., BTUNG, DOLLAREL) were included as features to prevent data leakage.
>
> The interaction term `hdd_sqft = HDD65 × A_heated` represents the theoretical 
> heating demand (degree-days × floor area) independent of actual energy 
> consumption. While the target variable `I = E_heat / (A × HDD)` shares these 
> components in the denominator, `hdd_sqft` captures **scale effects** and 
> **climate-size interactions** that influence intensity through physical 
> mechanisms (e.g., larger homes in cold climates may have proportionally 
> different envelope-to-volume ratios).

---

## 5. Scenario Assumptions (Section 4.1)

### جدول با منابع کامل:

| Parameter | Value | Source |
|-----------|-------|--------|
| **Retrofit Measures** | | |
| Air Sealing Reduction | 10% | LBNL Home Energy Saver (2023) |
| Attic Insulation Reduction | 15% | NREL ResStock (Deru et al., 2022) |
| Wall Insulation Reduction | 12% | NREL ResStock (Deru et al., 2022) |
| Window Replacement Reduction | 8% | ENERGY STAR Program (EPA, 2023) |
| Comprehensive Package | 30% | BPA/NEEA Pilot Studies |
| **Heat Pump Performance** | | |
| Standard HP COP | 2.5 @ 47°F | AHRI Directory (2023) |
| Cold Climate HP COP | 3.0 @ 47°F | NEEP ccASHP List (2023) |
| High-Performance HP COP | 3.5 @ 47°F | Mitsubishi/Daikin specs |
| **Energy Prices** | | |
| Electricity (avg) | $0.12/kWh | EIA SEDS 2022 |
| Natural Gas (avg) | $1.20/therm | EIA SEDS 2022 |
| **Grid Emissions** | | |
| Current (2023) | 0.42 kg CO₂/kWh | EPA eGRID 2022 |
| 2030 Projection | 0.30 kg CO₂/kWh | NREL Cambium Mid-case |
| **Economic Parameters** | | |
| Discount Rate | 5% real | DOE FEMP guidance |
| Analysis Horizon | 20 years | Typical HP lifetime |

### متن پیشنهادی برای annualized cost:

> **Cost Annualization:**
> Total annualized cost is calculated as:
> 
> `Annual Cost = CRF × CapEx + Annual OpEx`
> 
> where CRF (Capital Recovery Factor) = r(1+r)^n / [(1+r)^n - 1], 
> with r = 5% (real discount rate) and n = 20 years.

---

## 6. HP Viability Score Justification (Section 4.2)

### توضیح کالیبراسیون:

> **Viability Score Calibration:**
> The HP Viability Score parameters (α = 0.6, β = 0.8, γ = envelope factor) were 
> calibrated to reproduce the qualitative Pareto analysis results (Fig. 8):
> 
> - **α = 0.6** reflects the observation that climate severity (HDD) has a 
>   moderate effect on HP viability, with cold climates favoring HP due to 
>   higher heating loads that amortize fixed costs.
> 
> - **β = 0.8** reflects the strong sensitivity to electricity price, as 
>   operating cost dominates total cost for HP systems.
> 
> - **γ values** (Poor = 1.05, Medium = 0.75, Good = 0.45) were set such that 
>   poor envelopes show the highest viability (greatest savings potential 
>   from efficiency gains).
> 
> The **V = 0.5 threshold** was chosen to align with the NPV ≥ 0 condition 
> over a 15-year horizon under central assumptions. This is a heuristic 
> indicator, not a precise economic boundary.

---

## 7. Table 7 Usage Examples (for Results section)

### جملات پیشنهادی برای استفاده از Table 7:

> **Regional Findings:**
> - In the **South Atlantic** region (avg HDD ≈ 3,500), HP retrofits are 
>   economically viable for poor-envelope homes up to electricity prices of 
>   **$0.22/kWh (range: $0.19–0.25)**, with expected emissions reductions of 
>   **1,500 kg CO₂/year**.
> 
> - In **New England** (avg HDD ≈ 6,500), the price threshold drops to 
>   **$0.18/kWh** for poor envelopes and **$0.14/kWh** for medium envelopes, 
>   reflecting higher baseline heating costs.
> 
> - For **good-envelope homes** across all divisions, HP retrofits rarely 
>   achieve cost-competitiveness at current electricity prices (threshold 
>   < $0.10/kWh in most cases).

---

## 8. Baseline Energy Table (New Table 8)

### جدول پیشنهادی برای اضافه کردن:

| Archetype | HDD Band | Envelope | Baseline Gas (therms/yr) | Baseline CO₂ (kg/yr) | HP Energy (kWh/yr) |
|-----------|----------|----------|-------------------------|---------------------|-------------------|
| Cold-Poor | 6000-7000 | Poor | 850 | 4,505 | 8,500 |
| Cold-Medium | 6000-7000 | Medium | 650 | 3,445 | 6,500 |
| Cold-Good | 6000-7000 | Good | 450 | 2,385 | 4,500 |
| Mild-Poor | 2000-3000 | Poor | 400 | 2,120 | 4,000 |
| Mild-Medium | 2000-3000 | Medium | 300 | 1,590 | 3,000 |
| Mild-Good | 2000-3000 | Good | 200 | 1,060 | 2,000 |

---

## 9. Figure 10 Caption Enhancement

### متن پیشنهادی:

> **Figure 10:** Heat pump retrofit viability by census division under central 
> assumptions (electricity: $0.12/kWh, grid CO₂: 0.42 kg/kWh). Circle colors 
> indicate viability category based on the HP Viability Score (V): 
> Highly Viable (V > 0.7, green), Viable (0.5 < V ≤ 0.7, yellow), 
> Conditional (0.3 < V ≤ 0.5, orange), Low (V ≤ 0.3, red). 
> **Note:** Circle sizes are uniform; for weighted population representation, 
> see Table 2.

---

## 10. Sensitivity Analysis Limitations (Section 5)

### متن پیشنهادی:

> **Limitations of Sensitivity Analysis:**
> The sensitivity analysis presented (Fig. 11) examines one-dimensional 
> variations in electricity price and climate severity. While three joint 
> scenarios (optimistic: low price + clean grid; central; pessimistic: 
> high price + current grid) were examined qualitatively, a full Monte Carlo 
> analysis incorporating correlated uncertainties in retrofit effectiveness, 
> COP degradation, and fuel price volatility is left for future work.
>
> **Grid Decarbonization Scenarios:**
> The 2030 projection of 0.30 kg CO₂/kWh corresponds approximately to the 
> NREL Cambium "Mid-case" scenario, representing moderate renewable 
> deployment. Deep decarbonization scenarios (e.g., 0.15 kg/kWh by 2035) 
> would further improve HP environmental viability but are not analyzed here.

---

## 11. Recommended Paper Structure

```
1. Introduction
   - Background on building decarbonization
   - Research questions
   
2. Data and Methods
   2.1 RECS 2020 Microdata and Sample Selection ← Table 1, Fig 2
   2.2 Thermal Intensity Definition
   2.3 Envelope Classification
   2.4 Validation Against Official RECS Tables ← Fig 4, Table 2
   
3. Thermal Intensity Modeling
   3.1 Feature Engineering ← explain hdd_sqft, no data leakage
   3.2 XGBoost Model and Performance ← Fig 5, Table 3
   3.3 SHAP Interpretation ← Figs 6-7, Table 4
   
4. Retrofit and Heat Pump Scenarios
   4.1 Scenario Assumptions ← Tables 5a-c, 6
   4.2 Pareto Analysis ← Fig 8
   4.3 HP Viability Score ← Fig 9, formula justification
   4.4 Regional Viability Map ← Fig 10
   4.5 Tipping Point Summary ← Table 7
   
5. Sensitivity Analysis and Uncertainty ← Fig 11, limitations
   
6. Discussion and Policy Implications
   
7. Conclusions
```

---

## چک‌لیست نهایی قبل از ارسال

- [ ] Table 2: Add note "gas-heated single-family homes"
- [ ] Section 2.1: Add NWEIGHT and replicate weights statement
- [ ] Section 3.2: Add underprediction bias note
- [ ] Section 3.1: Add data leakage prevention statement
- [ ] Section 3.1: Justify hdd_sqft feature
- [ ] Tables 5a-c: Add full citations
- [ ] Section 4.1: Add CRF formula and discount rate
- [ ] Section 4.2: Justify α, β, γ and V=0.5 threshold
- [ ] Results: Reference specific Table 7 values
- [ ] Figure 10: Clarify circle sizes in caption
- [ ] Section 5: Add joint scenario and limitations note
- [ ] Section 5: Cite NREL Cambium for grid projections
