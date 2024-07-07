# Weather Prediction App

## Motivation

This project aims to build a weather prediction app tailored for farmers, providing them with accurate and timely weather forecasts to help them make better-informed decisions. The current implementation is an example or prototype of the final vision.

## Overview

This project is divided into two main parts:

1. **Model Training and Testing**: The initial code was developed and tested in a Google Colab notebook, where we trained a Ridge Regression model to predict the maximum temperature (TMAX) for the next day using historical weather data.
2. **Web Application**: To make the model accessible to end-users, a compact version of the model was created, and a web app was built using Streamlit. The web app allows users to input weather data and get predictions for TMAX.

## Technologies Used

- **Python**: Programming language used for developing the app.
- **Pandas**: For data manipulation and analysis.
- **Scikit-Learn**: For building and training the regression model.
- **Streamlit**: For building the interactive web app.
- **Google Colab**: For initial model training and testing.
- **Matplotlib**: For data visualization.

## Installation and Usage

### Prerequisites

- Python 3.x installed on your system.
- `pip` (Python package installer).

### Steps to Install and Run the App

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/weather-prediction-app.git
   cd weather-prediction-app
2. **Install required packages:**
   ```bash
   pip install -r requirements.txt
3. **Run the Streamlit app:**
   ```bash
   run predict.py
   streamlit run app.py
4. **Open the app:**
   - The app will automatically open in your default web browser. If not, you can access it at `http://localhost:8501`.

### Usage

1. Enter the required weather parameters in the input fields provided in the web app.
2. Click the "Predict" button to get the predicted TMAX for the next day.

## Model Training

The model was initially trained using a Google Colab notebook with the following steps:

1. **Load and preprocess the data**:
   - Handle missing values.
   - Add necessary features and target variable.
2. **Train the Ridge Regression model**.
3. **Evaluate the model**:
   - Perform backtesting to calculate the Mean Absolute Error (MAE).
   - Visualize prediction differences.

The trained model was then saved using `pickle` and loaded into a compact version for the web app.

### Key Files

- `predict.py`: Contains the code for loading the data, training the model, and saving the trained model.
- `app.py`: Contains the code for the Streamlit web app, which loads the trained model and makes predictions based on user input.

## Embedding a Video

### [![Watch the video]()](https://youtu.be/0DvVL-GU_DE)

*Click the image to watch the demo video of the working app.*

## Conclusion

Through this project, I learned the end-to-end process of developing a machine learning application, from data preprocessing and model training to building an interactive web application. The key takeaways include:

- Understanding the importance of data preprocessing and handling missing values.
- Learning how to train and evaluate a Ridge Regression model using Scikit-Learn.
- Gaining experience in backtesting and model validation techniques.
- Building a web application using Streamlit to make the machine learning model accessible to end-users.
- Realizing the practical aspects of deploying a machine learning model and ensuring the consistency of features between training and prediction phases.

By implementing this project, the motivation to create a useful tool for farmers has been achieved. The current prototype serves as a foundation for further development and enhancement.
