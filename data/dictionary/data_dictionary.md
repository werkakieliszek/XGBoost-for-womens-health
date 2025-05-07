# Data Dictionary

## Data Collection

In my quest to examine the impact of contraceptive methods on HIV-infected women, I chose to analyze the dataset from clinical trial NCT01721798, entitled “A Comparison of Two Intrauterine Devices (IUDs) Among HIV-Infected Women in Cape Town” (Todd, 2020). This dataset is particularly valuable because it provides detailed information on the safety and
efficacy of two types of IUDs—the 52 mg levonorgestrel intrauterine device (LNG-IUD) and the T380-A copper intrauterine device (C-IUD)—in a cohort of HIV-infected women.

I found this dataset on ClinicalTrials.gov, a database of privately and publicly funded clinical trials conducted worldwide. The study was first published on February 5, 2024. The dataset includes
comprehensive data such as HIV viral load, CD4 count, and infection outcomes,
which are key to analyzing the interaction between HIV progression and contraceptive use.
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NTN7KY

The selection of this dataset is consistent with my research goals of
understanding how different contraceptives affect the health of HIV-infected women. The structured data from this study provide a solid basis
for detailed statistical analysis and drawing conclusions.

## Data Source

The dataset from the clinical trial NCT01721798 entitled “Comparison of two
intrauterine devices (IUDs) among HIV-infected women in Cape Town” is of key importance for researchers examining the impact of these contraceptive methods on HIV-infected populations, particularly in terms of HIV viral load and CD4 cell counts, as well as overall safety and IUD continuation rates.
- Study participants: The dataset includes data from 199 participants,
grouped by type of IUD inserted. Participants average approximately 31.4 years of age, indicating a relatively young cohort.
- Baseline characteristics: Contains detailed baseline characteristics such as age, gender distribution (in this case, females only), and other relevant health indicators. This information is critical to understanding the context and generalizability of the study results.
- Panel data: The dataset tracks participants at multiple time points—enrollment, 6 months, 12 months, 18 months, and study completion. The data allow for analysis of outcomes over time.
- Outcome measures: Primary outcomes include detectable HIV RNA levels in genital tracts and plasma, which provide insight into the potential impact of IUD use on HIV viral load. Safety and efficacy outcomes of IUD use are also included, providing a picture of the benefits and risks of each type of IUD. 
- Reasons for study discontinuation: Information is provided on reasons why participants did not complete the study, including death, loss to follow-up, pregnancy, withdrawal at the participant’s discretion, and ineligibility for the 24-month visit. These data are necessary to assess the rate of study attrition and its potential impact on the results.

This dataset offers several analytical possibilities, such as comparing the safety profiles of LNG-IUDs with C-IUDs in HIV-infected women, assessing the impact of IUD use on HIV viral load dynamics, and assessing rates of IUD continuation and causes of discontinuation in this population. The nature of the dataset allows for detailed analysis of the interrelationships between contraceptive methods and HIV treatment, contributing insights to reproductive health in HIV-infected populations.

## Description of the dataset

The dataset obtained from the NCT01721798 clinical trial includes a large set of 1194 observations, in a total of 55 columns. These columns capture a wide range of data types, including integers, floating point numbers, datetime objects, and strings, reflecting the multifaceted nature of the data collection efforts in this study.

Overview of key components:
1. Participant identification and time frame: Each record is uniquely identified by an identifier and includes a month column indicating the time of data collection, from initial enrollment through follow-up visits.
2. Health and pregnancy data: The dataset includes detailed health-related information, such as the presence of pelvic inflammatory disease (PID), pregnancy status (preg), ectopic pregnancy (ECTOP), and intrauterine device expulsion events, along with the relevant dates to which it applies. 
3. IUD specific information: Information on, among others, IUD interruption (IUDDC), IUD type (arm), and reason for removal (iud_reason) provides insight into the comparative analysis of LNG-IUD and C-IUD devices. 
4. HIV and infection data: Key variables include genital and plasma HIV viral load (gvl, pvl) and quantity (gvlquant, pvlquant), as well as data on other sexually transmitted diseases such as trichomonas vaginalis (trich), bacterial vaginosis (bv), syphilis (sy), Chlamydia trachomatis (ct), and gonorrhea (gc). 
5. Treatment and demographic data: The dataset also includes variables related to
antiretroviral therapy (art, artgroup), demographic information (age,
education, employed), and sexual health history (everpreg, sexparts, sexfreq).
6. Dates and analysis population: Important dates such as HIV diagnosis date (hivdate), enrollment date (enrollment date), and enddate of the study (enddate) are recorded,
along with inclusion in the treatment analysis population (ATPOP).

## Description of columns and measures

| Variable Name      | Label                                | Description                                                                                   |
|--------------------|--------------------------------------|-----------------------------------------------------------------------------------------------|
| fakeid             | Artificial identifier                | Unique identifier assigned to each study participant to protect her privacy.                  |
| month              | Month of visit                       | Month of the study visit, possible values: 0, 3, 6, 12, 18, and 24 months.                    |
| PID                | Pelvic inflammatory disease          | Indicates if the participant had pelvic inflammatory disease; 0 = No, 1 = Yes.                |
| preg               | Pregnancy at visit month             | Indicates if the participant was pregnant at the time of the visit; 0 = No, 1 = Yes.          |
| ECTOP              | Ectopic pregnancy                    | Indicates if the participant experienced an ectopic pregnancy; 0 = No, 1 = Yes.               |
| expl               | IUD expulsion                        | Indicates if IUD expulsion occurred; 1 = Yes.                                                 |
| expldate           | Date of IUD expulsion                | Date when the participant's IUD expulsion occurred.                                           |
| IUDDC              | IUD discontinuation                  | Indicates if the participant discontinued IUD use; 0 = No, 1 = Yes.                           |
| iuddcdate          | Date of IUD discontinuation          | Date when the participant discontinued IUD use.                                               |
| arm                | Treatment group                      | Treatment group assigned: "LNG IUD" or "C-IUD".                                               |
| gvl                | Detectable genital tract HIV VL      | Indicates if genital tract HIV viral load was detectable; 0 = No, 1 = Yes.                    |
| gvlquant           | Genital tract HIV viral load         | Quantitative value of the participant's genital tract HIV viral load.                         |
| gvldate            | Date of genital tract HIV VL         | Date when genital tract HIV viral load was measured.                                          |
| pvl                | Detectable plasma viral load         | Indicates if plasma viral load was detectable; 0 = No, 1 = Yes.                               |
| pvlquant           | Plasma viral load                    | Quantitative value of the participant's plasma viral load.                                    |
| pvldate            | Date of plasma viral load            | Date when plasma viral load was measured.                                                     |
| trich              | Trichomonas vaginalis (TV)           | Indicates if participant tested positive for TV; 0 = No, 1 = Yes.                             |
| trichdate          | Date of TV testing                   | Date when participant was tested for TV.                                                      |
| trichtreatdate     | Date TV treated                      | Date when participant was treated for TV.                                                     |
| bv                 | Sialidase-positive bacterial vaginosis (BV) | Indicates if participant tested positive for BV; 0 = No, 1 = Yes.                      |
| bvdate             | Date of BV testing                   | Date when participant was tested for BV.                                                      |
| bvtreatdate        | Date BV treated                      | Date when participant was treated for BV.                                                     |
| sy                 | Syphilis                             | Indicates if participant tested positive for syphilis; 0 = No, 1 = Yes.                       |
| sy_rapiddate       | Date of syphilis testing             | Date when participant was tested for syphilis.                                                |
| ct                 | Chlamydia trachomatis (CT)           | Indicates if participant tested positive for CT; 0 = No, 1 = Yes.                             |
| ctdate             | Date of CT testing                   | Date when participant was tested for CT.                                                      |
| cttreatdate        | Date CT treated                      | Date when participant was treated for CT.                                                     |
| gc                 | Gonorrhea                            | Indicates if participant tested positive for gonorrhea; 0 = No, 1 = Yes.                      |
| gcdate             | Date of gonorrhea testing            | Date when participant was tested for gonorrhea.                                               |
| gctreatdate        | Date gonorrhea treated               | Date when participant was treated for gonorrhea.                                              |
| anyrti             | Any reproductive tract infection     | Indicates if participant had any reproductive tract infection; 0 = No, 1 = Yes.               |
| hbg                | Hemoglobin                           | Participant's hemoglobin level.                                                               |
| cd4                | CD4 count                            | Participant's CD4 lymphocyte count.                                                           |
| cd4date            | Date of CD4 count measured           | Date when CD4 count was measured.                                                             |
| art                | Received ART                         | Indicates if participant received antiretroviral therapy (ART); 0 = No, 1 = Yes.              |
| iud_remove         | IUD removal                          | Indicates if IUD was removed; 1 = Yes.                                                        |
| iud_removedate     | Date of IUD removal                  | Date when IUD was removed.                                                                    |
| iud_reason         | Reason for IUD removal               | Reason for IUD removal.                                                                       |
| iud_expulsion      | IUD expulsion                        | Indicates if IUD expulsion occurred; 1 = Yes.                                                 |
| iud_expulsiondate  | Date of IUD expulsion                | Date when IUD expulsion occurred.                                                             |
| iud_replaced       | IUD replacement                      | Indicates if IUD was replaced; 0 = No, 1 = Yes.                                               |
| iud_replacedate    | Date of IUD replacement              | Date when IUD was replaced.                                                                   |
| iud_nonelect       | Not elected IUD removal              | Indicates if IUD removal was not participant's choice; 0 = No, 1 = Yes.                       |
| artgroup           | ART group                            | ART group: 1 = pre-ART, 2 = ART using.                                                        |
| enddate            | End of study date                    | Date when participant finished the study.                                                     |
| age                | Age at enrollment                    | Participant's age at study enrollment.                                                        |
| education          | Completed secondary education        | Indicates if participant completed secondary education; 0 = No, 1 = Yes.                      |
| employed           | Currently employed                   | Indicates if participant was currently employed; 0 = No, 1 = Yes.                             |
| everpreg           | Ever pregnant                        | Indicates if participant was ever pregnant; 0 = No, 1 = Yes.                                  |
| sexparts           | Number of sex partners (past 12 mo.) | Number of sexual partners in the past 12 months.                                              |
| sexfreq            | Sexual frequency (last 3 mo.)        | Frequency of sexual intercourse in the last 3 months.                                         |
| hivdate            | Date of HIV diagnosis                | Date when participant was diagnosed with HIV.                                                 |
| enrolldate         | Date of enrollment                   | Date when participant was enrolled in the study.                                              |
| ATPOP              | Included in treated analysis pop.    | Indicates if participant was included in the per-protocol analysis; 0 = No, 1 = Yes.          |
