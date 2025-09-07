# Medicine_Recommendation
An AI-powered system that predicts diseases based on user-input symptoms and provides personalized medical recommendations including medications, diet plans, workouts, and preventive measures.

##  Project Overview
This system uses multiple machine learning models to classify diseases from symptoms and generates relevant health recommendations. It aims to promote preventive healthcare and assist users in early diagnosis.

##  Features
- Symptom-based disease prediction
- Personalized medical advice
- Recommendations include:
  - Disease description
  - Precautions
  - Medications
  - Workout plans
  - Diet plans
- High accuracy using machine learning models
- Designed for accessibility and preventive healthcare

##  Machine Learning Models Used
- **Support Vector Classifier (SVC)** — selected for implementation due to its high generalization and 100% accuracy
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Multinomial Naïve Bayes**

All models were tested and evaluated using precision, recall, F1-score, and confusion matrices.

##  Performance
-  100% Accuracy
-  0% False Positives
-  0% False Negatives
-  1.0 Confidence Score (SVC)
-  Best Model: SVC (Support Vector Classifier)

##  System Architecture
1. **User Input**: Symptoms entered by the user
2. **Data Processing**: Symptoms are encoded into machine-readable format
3. **Prediction**: Trained model classifies the disease
4. **Recommendation**: System provides actionable advice

##  Technologies Used
- Python
- Scikit-learn
- Pandas / NumPy
- Jupyter Notebook

##  Future Enhancements
- Integration of deep learning models
- Expansion of datasets for more diverse predictions
- Mobile application for real-time accessibility

##  Limitations
- Limited dataset affecting generalization
- Symptom ambiguity across multiple diseases
- Not a substitute for professional medical advice
