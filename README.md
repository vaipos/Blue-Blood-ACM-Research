# BlueBlood: Predicting Medication Impact on Blood Biomarkers
BlueBlood is a machine learning pipeline that predicts how prescribed medications alter Complete Blood Count (CBC) profiles. It combines clinical records data from MIMIC-III, CTGAN-based synthetic data generation, and LSTM forecasting models. The entire workflow is built on a scalable AWS stack using S3 for storage, SageMaker for training, and EC2 for compute.

![Blue-Blood-Poster](https://github.com/user-attachments/assets/0b0386fe-f6c7-4fe9-9146-e842be9f13a5)


## Data Sourcing

The dataset is derived from MIT’s MIMIC‐III clinical database [[3]](#references), which houses over 150,000 clinical records spanning billed prescriptions, hourly vital signs, lab test results, diagnostic codes (ICD‐9), procedure records, microbiology reports, and clinical notes from healthcare providers. To focus our analysis, we extracted and restructured the raw data using Google BigQuery, narrowing the scope to Complete Blood Count (CBC) profiles — a standardized panel of 25 biomarkers that provide a comprehensive snapshot of a patient’s blood health — alongside corresponding prescription records, enabling a targeted investigation of drug impacts through pre‐ and post‐treatment CBC measurements.

## Data Preprocessing
Data preprocessing involved two parallel pipelines:

- **Numerical Data (CBC Profiles):**
  - Scaling
  - Normalization
  - Imputation of missing values

- **Categorical Data (Prescriptions):**
  - Encoding using **BERT embeddings** [[4]](#references) to capture semantic similarities between medications

Processed data of about ~2,182 samples was stored in **AWS S3** for scalable cloud access.

## Model Architecture

### Overall Workflow
![ACM Final Model Architecture](https://github.com/user-attachments/assets/6ea43330-4a55-47b8-ae9f-4f9c568f1c01)

### Synthetic Data Generation

CTGAN, developed by MIT, is a generative adversarial network (GAN) specialized for tabular data synthesis [[5]](#references). Unlike traditional GANs, CTGAN introduces a conditional generator that samples a discrete variable first, ensuring better handling of imbalanced categorical features. Mode‐specific normalization further stabilizes training by centering numerical columns around their most frequent values. This architecture enables CTGAN to effectively model numerical and categorical features, overcoming limitations faced by traditional GANs when applied to structured healthcare datasets [[1]](#references).

**CTGAN Metrics:**
- Validity: **93.30%**
- Quality: **84.66%**

<div style="display:flex;justify-content:space-between;gap:4rem;margin:0;padding:0;">
  <img src="https://github.com/user-attachments/assets/c95bea25-e8d7-47fa-a1a9-905d58b346a9" alt="CTGAN Metric Plot 1" width="46%"/>
  <img src="https://github.com/user-attachments/assets/99dfe43d-d765-419f-af61-f52984ec3325" alt="CTGAN Metric Plot 2" width="46%"/>
</div>

### Predictive Modeling
To capture temporal relationships between complete blood count (CBC) profiles before and after prescription administration, we utilized a time‐series‐based Long Short‐Term Memory (LSTM) network, a type of recurrent neural network.

The input consists of padded sequences combining pre‐treatment CBC panels and prescription feature embeddings, ensuring consistent sequence lengths across all samples.

The model predicts post‐treatment CBC blood profiles as continuous vectors, trained using mean squared error (MSE) loss with the Adam optimizer. Performance was evaluated based on MSE, root mean squared error (RMSE), and mean absolute error (MAE) between predicted and actual CBC measurements. It utilized a cloud‐based training and evaluation pipeline using AWS SageMaker and an 80‐20 training-validation split.


## Results

Despite showing no signs of overfitting, statistical evaluations reveal notable limitations. Predictions for each of the 25 CBC biomarkers deviate by an average of 39.45%, while per‐sample predictions differ by 20.88%. The relative error distribution (Figure 5) shows that 5 to 12 biomarkers per sample deviate by more than 25%.

Reliable predictions remain challenging due to dataset limitations. Imputed (NULL) fields introduce uncertainty, while noisy features, prescription variability, and outliers disrupt learning patterns. Future work should prioritize advanced imputation, outlier detection, and finer treatment encoding to enhance model reliability.

<div style="display:flex;justify-content:space-between;gap:4rem;margin:0;padding:0;">
  <img src="https://github.com/user-attachments/assets/04185ee3-c275-4c32-8bfc-c934ac6627c1" alt="Training vs Validation Loss" width="46%"/>
  <img src="https://github.com/user-attachments/assets/e73bf771-01ea-4336-8f48-4d776eaffd65" alt="Relative Error Distribution" width="46%"/>
</div>

## Conclusion

This project marks an early step toward improving patient outcomes by providing data‐driven insights into how prescribed medications impact physiological biomarkers, addressing gaps in current treatment plans prone to adverse drug reactions (ADRs). Future work could explore advanced architectures like autoencoders, leverage transfer learning, and refine preprocessing and model tuning to better manage clinical data variability and enhance personalized treatment strategies.

## References

[1] Borisov, V., Leemann, T., Seßler, K., Haug, J., Pawelczyk, M., and Kasneci, G. (2021). Deep neural networks and tabular data: A survey. [arXiv:2110.01889](https://arxiv.org/abs/2110.01889)
 
 [2] Johnson, J.A., & Bootman, J.L. (1995). Drug-related morbidity and mortality: A cost-of-illness model. *Archives of Internal Medicine*, 155(18), 1949–1956.
 
 [3] Johnson, A.E.W., Pollard, T.J., Shen, L., Lehman, L.H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L.A., & Mark, R.G. (2016). MIMIC-III, a freely accessible critical care database. *Scientific Data*, 3, 160035.

[4] Koroteev, M.V. (2021). BERT: A review of applications in natural language processing and understanding. [arXiv:2103.11943](https://arxiv.org/abs/2103.11943)

[5] Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling Tabular Data Using Conditional GAN. *NeurIPS Workshop on Synthetic Data Generation*, 2019. [arXiv:1907.00503](https://arxiv.org/abs/1907.00503)


## Contributors

- Ubaid Mohammed
- Advay Chandramouli
- Neha Senthil Kumar
- Poojasri Sundaresan
- **Research Lead:** Vaishnavi Pasumarthi
- **Faculty Advisors:** Dr. Sriraam Natarajan, Dr. Anurag Nagar


## License

This project is licensed under the APACHE License - see the [LICENSE](LICENSE) file for details.








