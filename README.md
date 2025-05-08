# Machine Learning in Women's Reproductive Health Research: From Risk Factor Analysis to Clinical Outcome Prediction

This repository contains code and resources for analyzing and predicting the risk of genirourinary infections in HIV-infected women using machine learning and Monte Carlo simulations

## Table of Contents

## Table of Contents
- [Background](#background)
- [Project Structure](#project-structure)
- [Data Sources](#data-sources)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Background

Genital infections are among the most common infections affecting women, with a lifetime prevalence of 50–60% in adult women. The incidence of urinary tract infections increases with age, and certain factors, such as sexual activity, anatomical differences, and hormonal changes, contribute to this increased risk. In HIV-infected women, the risk of infection may be increased by immunosuppression, which complicates the body's ability to fight infection. Despite the high prevalence and significant impact of genitourinary diseases, research on their specific risk and treatment in HIV-infected women remains limited. Historically, medical research has often overlooked women's health, treating women's bodies as atypical and using men's bodies as the standard for research. This bias has led to a significant gap in understanding how diseases affect women differently, including the knowledge and treatment of genitourinary infections. Although legal changes in the 1990s began to address these disparities, women, especially those with intersecting vulnerabilities such as HIV, remain underrepresented in clinical trials. This underrepresentation impacts the quality of care women receive, as treatments and interventions are often not tailored to their specific needs.

In recent years, the application of machine learning to health care has shown great promise in predicting clinical risk and patient outcomes. Machine learning algorithms can analyze complex data sets to identify patterns and risk factors that may not be apparent using traditional statistical methods. In the case of genitourinary infections in HIV-infected women, machine learning can help predict individual risk profiles by analyzing multiple variables, including demographic, clinical, and behavioral data.

Monte Carlo simulations offer another tool for risk modeling. These simulations use random sampling to understand the impact of risk and uncertainty in predictive models. In the context of genitourinary diseases, Monte Carlo methods can simulate different scenarios to predict the likelihood of infection under different conditions, providing a probabilistic framework that complements the deterministic nature of machine learning models. The integration of machine learning and Monte Carlo techniques in this study aims to fill a gap in the literature by providing a more nuanced understanding of the risk of infection in HIV-infected women. Using these advanced computational methods, the analysis aims to identify key risk factors and develop predictive models that can inform clinical decision-making and improve patient outcomes. This approach not only addresses the historical neglect of women’s health in research, but also contributes to the broader field of precision medicine by tailoring healthcare strategies to the unique needs of women living with HIV.

## Installation

1. Clone the repository:
```
git clone https://github.com/werkakieliszek/XGBoost-for-womens-health.git
cd XGBoost-for-womens-health
```
2. Create and activate a virtual environment:
```
python -m venv .venv
source .venv/bin/activate
```
3. Install dependencies:
```
pip install -r requirements.txt
```
## Usage

### Preprocess and Impute Data
```
python src/data_preprocessing.py
python src/imputation.py
```
### Train Models
```
python src/training.py
```
Artifacts are saved in `model_artifacts/YYYYMMDD_HHMMSS/`.

### Evaluate Models
```
python src/evaluation.py
```
### Generate Synthetic Data
```
python src/generate_synthetic_data.py
```
## Results

- Multiple ML models trained for trichomoniasis, bacterial vaginosis, chlamydia, and gonorrhea.
- Model artifacts, selected features, and hyperparameters are versioned and stored for reproducibility.
- Synthetic data enables robust scenario analysis via Monte Carlo simulation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Acknowledgments

- Clinical collaborators and data providers
- [SDV](https://sdv.dev/) for synthetic data generation
- Open-source ML libraries: scikit-learn, XGBoost, imbalanced-learn, etc.