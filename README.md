# Titanic Survival Predictor

Imagine you're transported back to 1912, aboard the RMS Titanic. As a data scientist, you're given an unusual task: can you use the limited information available... like a passengers age, gender, travel class, and ticket fare to predict whether they would survive the disaster???

This project is a simple attempt to recreate that very challenge using real Titanic passenger data. With the help of basic machine learning techniques, we're training a model to make survival predictions based on some key passenger attributes.

It’s an ideal project for beginners who want to get hands-on experience with data cleaning, feature selection, and basic classification algorithms—all wrapped in a practical and historical context.

## About the Project

This machine learning project uses logistic regression to predict survival outcomes from Titanic's passenger dataset. It’s a lightweight project that focuses on understanding the endtoend ML pipeline from reading data to making predictions and optionally deploying the model using Streamlit.


## Tech Stack

- Language: Python
- Libraries:Pandas,Scikit-learn
- Model: Logistic Regression
- Interface: Streamlit
- Dev Env: VS Code

## Dataset

The dataset is based on real passenger data from the Titanic and includes details like:
- Age
- Sex
- Passenger class (Pclass)
- Fare paid
- Number of siblings/spouses or parents/children aboard
- Port of embarkation

This dataset is sourced from [Kaggle's Titanic Competition](https://www.kaggle.com/c/titanic/data).


## Project Structure



titanic-survival-predictor/
├── titanic_proj.py               # ML model and prediction logic
├── train.csv                    # Passenger dataset
├── requirements.txt             # Required Python packages
└── README.md                     # Project documentation

## Features Used for Prediction

The model considers the following attributes:
- `Pclass`: Ticket class (1st, 2nd, 3rd)
- `Sex`: Gender of the passenger
- `Age`: Age in years
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Fare`: Amount paid for the ticket
- `Embarked`: Port where the passenger boarded (C, Q, S)

---

## How to Run Locally

Before running the project, ensure Python 3.7+ is installed on your system.

### 1. Clone the repository

```bash
git clone https://github.com/Prayagi/titanic-survival-predictor.git
cd titanic-survival-predictor
````

### 2. Install the dependencies
pip install -r requirements.txt

### 3. Run the script (terminal version)

python titanic_proj.py

### 4. (Optional) Run with Streamlit for a web interface
streamlit run titanic_proj.py


## made by

This project was made by **Prayagi Sahajwani** as part of a hands-on learning journey into machine learning. It’s one of the first steps toward building more real-world, relatable ML applications.

Feel free to fork the repo, use it as a template, or contribute to it!
