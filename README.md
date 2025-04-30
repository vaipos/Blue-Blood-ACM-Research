# Blue Blood: Predicting Medication Impact on Blood Biomarkers

![Blue-Blood-Poster](https://github.com/user-attachments/assets/0b0386fe-f6c7-4fe9-9146-e842be9f13a5)

## Introduction and Problem Motivation

Understanding how drug compounds alter the human body is vital for advancing personalized healthcare. This project leverages machine learning (ML) to analyze Complete Blood Count (CBC) profiles, uncovering subtle systemic patterns that traditional methods may neglect. 

Nearly 50% of patients with chronic diseases do not respond effectively to first‐line medications, resulting in adverse drug reactions (ADRs) that cost the U.S. healthcare system over $136 billion annually[2]. Hence, understanding how prescribed medications impact blood profiles is paramount for developing safer, more personalized treatments.

Current personalized medicine approaches, such as pharmacogenomics (e.g., 23andMe), face cost and real‐time biomarker capture limitations. Meanwhile, ML efforts in drug discovery (e.g., Insilico) and EHR‐based models suffer from missing data and a lack of real-time biological signals. By leveraging synthetic data and deep learning, our project aims to address these gaps, improving diagnosis and therapeutic decision‐making.

**Goal:**  
BlueBlood seeks to model and predict drug impact on physiological biomarkers, offering safer, more personalized treatments.

## Data Sourcing

The dataset is derived from MIT’s MIMIC‐III clinical database, which houses over 150,000 clinical records spanning billed prescriptions, hourly vital signs, lab test results, diagnostic codes (ICD‐9), procedure records, microbiology reports, and clinical notes from healthcare providers. To focus our analysis, we extracted and restructured the raw data using Google BigQuery, narrowing the scope to Complete Blood Count (CBC) profiles — a standardized panel of 25 biomarkers that provide a comprehensive snapshot of a patient’s blood health — alongside corresponding prescription records, enabling a targeted investigation of drug impacts through pre‐ and post‐treatment CBC measurements.

## Data Preprocessing
Data preprocessing involved two parallel pipelines:

- **Numerical Data (CBC Profiles):**
  - Scaling
  - Normalization
  - Imputation of missing values

- **Categorical Data (Prescriptions):**
  - Encoding using **BERT embeddings** to capture semantic similarities between medications

Processed data of about ~2,182 samples was stored in **AWS S3** for scalable cloud access.

## Model Methodology

**Synthetic Data Generation:** CTGAN, developed by MIT, is a generative adversarial network (GAN) specialized for tabular data synthesis. Unlike traditional GANs, CTGAN introduces a conditional generator that samples a discrete variable first, ensuring better handling of imbalanced categorical features. Mode‐specific normalization further stabilizes training by centering numerical columns around their most frequent values [1]. This architecture enables CT‐GAN to effectively model numerical and categorical features, overcoming limitations faced by traditional GANs when applied to structured healthcare datasets.

  **CTGAN Metrics:**
  - Validity: **93.30%**
  - Quality: **84.66%**


**Model Architecture:** To capture temporal relationships between complete blood count (CBC) profiles before and after prescription
administration, we utilized a time‐series‐based Long Short‐Term Memory (LSTM) network, a type of recurrent neural network.

The input consists of padded sequences combining pre‐treatment CBC panels and prescription feature embeddings, ensuring consistent sequence lengths across all samples.

The model predicts post‐treatment CBC blood profiles as continuous vectors, trained using mean squared error (MSE) loss with the Adam optimizer. Performance was evaluated based on MSE, root mean squared error (RMSE), and mean absolute error (MAE) between predicted and actual CBC measurements. It utilized a cloud‐based training and evaluation pipeline using AWS SageMaker and an 80‐20 training-validation split.

## Results

Despite showing no signs of overfitting, statistical evaluations reveal notable limitations. Predictions for each of the 25 CBC biomarkers deviate by an average of 39.45%, while per‐sample predictions differ by 20.88%. The relative error distribution (Figure 5) shows that 5 to 12 biomarkers per sample deviate by more than 25%.

Reliable predictions remain challenging due to dataset limitations. Imputed (NULL) fields introduce uncertainty, while noisy features, prescription variability, and outliers disrupt learning patterns. Future work should prioritize advanced imputation, outlier detection, and finer treatment encoding to enhance model reliability.

## Conclusion

This project marks an early step toward improving patient outcomes by providing data‐driven insights into how prescribed medications impact physiological biomarkers, addressing gaps in current treatment plans prone to adverse drug reactions (ADRs). Future work could explore advanced architectures like autoencoders, leverage transfer learning, and refine preprocessing and model tuning to better manage clinical data variability and enhance personalized treatment strategies.

## References

[1] Borisov, V., Leemann, T., Seßler, K., Haug, J., Pawelczyk, M., and Kasneci, G. (2021). Deep neural networks and tabular data: A survey. arXiv.org. https://arxiv.org/abs/2110.01889

[2] Johnson, J. A., and Bootman, J. L. (1995). Drug‐related morbidity and mortality: A cost‐of‐illness model. Archives of Internal Medicine, 155(18), 1949–1956.

[3] Johnson, A. E. W., Pollard, T. J., Shen, L., Lehman, L. H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L. A., and Mark, R. G. (2016). MIMIC‐III, a freely accessible critical care database. Scientific Data, 3, 160035.

[4] Koroteev, M. V. (2021, March 22). BERT: A review of applications in natural language processing and understanding. arXiv.org. https://arxiv.org/abs/2103.11943

## Contributors

- Ubaid Mohammed
- Advay Chandramouli
- Neha Senthil Kumar
- Poojasri Sundaresan
- **Research Lead:** Vaishnavi Pasumarthi
- **Faculty Advisors:** Dr. Sriraam Natarajan, Dr. Anurag Nagar


## License

This project is licensed under the APACHE License - see the [LICENSE](LICENSE) file for details.








