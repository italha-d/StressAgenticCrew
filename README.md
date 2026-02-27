## StressAgenticCrew
This code is related to the publication "Agentic AI System for Stress Monitoring: A Healthcare Multi-agent Crew" built with:

- Dash for interactive dashboard
- scikit-learn for stress classification
- Ollama + Mistral AI models for clinical report generation
- Matplotlib & Seaborn for visualization
- ReportLab for PDF report creation

In this repo, the "Adaboost_Reports_Abnormal_1.rar", "Adaboost_Reports_Abnormal_2.rar", "RF_Report_Abnormal_1.rar", "RF_Report_Abnormal_2.rar", "RF_Report_Abnormal_3.rar", "RF_Report_Abnormal_4.rar", "RF_Report_Abnormal_5.rar", and "RF_Report_NOAbnormal.rar" are the Supplementary Materials for the publication.

## Libraries

The following libraries were used:

- dash
- pandas
- scikit-learn
- matplotlib
- seaborn
- reportlab
- ollama

## Expected Dataset Format

CSV should contain:
- participant_id (or ID)
- time
- HR (heart rate)
- respr (respiratory rate)
- Label_B (stress label)

## References

**When using this dataset, please cite the following:**

1 - Talha Iqbal, Andrew J. Simpkin, Davood Roshan, Nicola Glynn, John Killilea, Jane Walsh, Gerard Molloy, Sandra Ganly, Hannah Ryman, Eileen Coen, Adnan Elahi, William Wijns, and Atif Shahzad. 2022. "Stress Monitoring Using Wearable Sensors: A Pilot Study and Stress-Predict Dataset", Sensors 22, no. 21: 8135. https://doi.org/10.3390/s22218135

2 - Talha Iqbal, Edward Curry, and Ihsan Ullah. "Agentic AI System for Stress Monitoring: A Multi-agent Healthcare Crew." International Journal of Computational Intelligence Systems (2026). https://doi.org/10.1007/s44196-026-01215-0
