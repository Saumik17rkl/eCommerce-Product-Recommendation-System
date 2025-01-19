# eCommerce-Product-Recommendation-System

This project is an eCommerce Product Recommendation System designed to provide personalized product recommendations to users based on their historical behavior using machine learning techniques. The system uses collaborative filtering to suggest relevant products to users.

Table of Contents
Overview
Technologies Used
Installation
Usage
File Structure
Dataset
Model
License
Overview
This project is designed to recommend products to users in an e-commerce platform. It uses data such as product views, user behavior, and ratings to suggest personalized recommendations. The system aims to enhance the user experience by making relevant suggestions, thereby increasing engagement and sales.

Key Features:

Personalized product recommendations
Collaborative filtering-based algorithm
Data preprocessing and feature extraction
Simple web app interface using Streamlit
Technologies Used
Python: Programming language for implementing the recommendation algorithm and the web interface.
pandas: Data manipulation and analysis.
NumPy: Numerical computations.
Scikit-learn: For machine learning and model evaluation.
Streamlit: For building the web app interface.
SciPy: For sparse matrix operations (e.g., Singular Value Decomposition).
Matplotlib/Seaborn: Data visualization for understanding the dataset and model performance.
Parquet: Data format used for storing and reading the dataset files.
Installation
Clone this repository:

bash
Copy
Edit
git clone https://github.com/Saumik17rkl/eCommerce-Product-Recommendation-System.git
cd eCommerce-Product-Recommendation-System
Create a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv env
source env/bin/activate  # For Linux/macOS
.\env\Scripts\activate   # For Windows
Install the necessary dependencies:

bash
Copy
Edit
pip install -r requirements.txt
If you don't have a requirements.txt yet, run the following:

bash
Copy
Edit
pip freeze > requirements.txt
Usage
Running the Recommendation Model:

You can run the model locally on your machine by executing:
bash
Copy
Edit
python recommendation_model.py
This will train the recommendation model using the dataset and output the recommendations.
Starting the Web Application:

After training the model, you can run the web app using Streamlit:
bash
Copy
Edit
streamlit run app.py
This will launch the web app where you can input a user ID to receive product recommendations.
Testing the Model:

Test the recommendation system by interacting with the web app to see how recommendations change based on user input.
File Structure
The project folder contains the following files and directories:

bash
Copy
Edit
eCommerce-Product-Recommendation-System/
│
├── data/
│   ├── train.parquet
│   ├── test.parquet
│   └── val.parquet
│
├── app.py                  # Streamlit web app
├── recommendation_model.py  # Code for building and evaluating the recommendation system
├── requirements.txt         # List of Python dependencies
└── README.md                # This file
data/: Contains the dataset files (train.parquet, test.parquet, val.parquet) used for training and testing the recommendation model.
app.py: The Streamlit application used for building the web interface.
recommendation_model.py: The main code where the recommendation algorithm is implemented and trained.
Dataset
The dataset used in this project is the Recsys 2020 eCommerce Dataset from Kaggle. You can find the dataset here.

The dataset includes:

UserID: The ID of the user.
ProductID: The ID of the product.
Rating/Interaction: The rating or interaction (e.g., click or purchase).
Timestamp: The time when the interaction took place.
The dataset is available in Parquet format, which is used for efficient storage and reading in the project.

Model
The recommendation system is based on Collaborative Filtering using Singular Value Decomposition (SVD). The model works by:

Creating a user-product interaction matrix.
Applying SVD to decompose the matrix into latent factors.
Using the latent factors to predict missing values and recommend products.
License
This project is open-source and available under the MIT License.

Example of Requirements.txt:
makefile
Copy
Edit
pandas==1.5.3
numpy==1.23.4
scipy==1.9.3
streamlit==1.14.0
scikit-learn==1.1.3
matplotlib==3.6.2
seaborn==0.11.2
With this update, the dataset section now includes the link to the Kaggle dataset you used. Let me know if you need any further adjustments!
