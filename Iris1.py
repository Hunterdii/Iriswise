# import pandas as pd
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import pickle
# import streamlit as st
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset
# dataset = pd.read_csv("Iris.csv")

# # Prepare the features and target variable
# x = dataset.drop(["Species", "Id"], axis=1)
# y = dataset["Species"]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# # Initialize and train the KNN classifier
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(x_train, y_train)

# # Save the model
# pickle_out = open("classifier.pkl", "wb")
# pickle.dump(knn, pickle_out)
# pickle_out.close()

# # Streamlit UI
# st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="wide")
# st.title("🌼 Iris Species Prediction App")

# st.markdown("""
#     Welcome to the Iris Species Prediction app! Enter the details below to predict the species of an Iris flower based on its features.
#     This application uses a K-Nearest Neighbors classification model to predict the species.
# """)

# # Input form
# with st.form("prediction_form"):
#     Sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.0, step=0.1)
#     Sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.0, step=0.1)
#     Petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.5, step=0.1)
#     Petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2, step=0.1)
#     submit_button = st.form_submit_button(label='Predict Species')
    
# # Prediction and output
# if submit_button:
#     model = joblib.load("classifier.pkl")
#     x = np.array([Sepal_length, Sepal_width, Petal_length, Petal_width])
#     if any(x <= 0):
#         st.warning("⚠️ Input values must be greater than 0")
#     else:
#         prediction = model.predict([x])
#         st.success(f"🌷 Predicted Species: **{prediction[0]}**")

# # Data visualization
# st.markdown("### 🌸 Dataset Overview & Visualization")
# st.write(dataset.head())

# # Pairplot
# st.markdown("#### Pairplot of Iris Features")
# sns.pairplot(dataset, hue="Species", palette="viridis")
# st.pyplot(plt)

# # Model performance metrics
# st.markdown("### 📈 Model Performance")
# st.markdown(f"**Train Set Accuracy:** {knn.score(x_train, y_train):.2f}")
# st.markdown(f"**Test Set Accuracy:** {knn.score(x_test, y_test):.2f}")



# import pandas as pd
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import pickle
# import streamlit as st
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the dataset
# dataset = pd.read_csv("Iris.csv")

# # Prepare the features and target variable
# x = dataset.drop(["Species", "Id"], axis=1)
# y = dataset["Species"]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# # Initialize and train the KNN classifier
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(x_train, y_train)

# # Save the model
# pickle_out = open("classifier.pkl", "wb")
# pickle.dump(knn, pickle_out)
# pickle_out.close()

# # Streamlit UI
# st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="wide")
# st.title("🌼 Iris Species Prediction App")

# st.markdown("""
#     Welcome to the Iris Species Prediction app! 🌺
#     This app predicts the species of an Iris flower based on its features. 
#     The model behind this app is a K-Nearest Neighbors classifier trained on the classic Iris dataset.
# """)

# # Sidebar for additional options
# with st.sidebar:
#     st.header("🔧 Settings")
#     st.markdown("Adjust the app settings below:")

#     n_neighbors = st.slider("Number of Neighbors (K in KNN)", 1, 15, 3)
#     knn.n_neighbors = n_neighbors
#     knn.fit(x_train, y_train)  # Re-train with new K
    
#     show_dataset = st.checkbox("Show Dataset Overview", value=True)
#     show_pairplot = st.checkbox("Show Pairplot Visualization", value=True)
#     show_performance = st.checkbox("Show Model Performance", value=True)

# # Input form
# with st.form("prediction_form"):
#     st.subheader("🔍 Enter Flower Features")
#     Sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.0, step=0.1)
#     Sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.0, step=0.1)
#     Petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.5, step=0.1)
#     Petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2, step=0.1)
#     submit_button = st.form_submit_button(label='🌟 Predict Species')
    
# # # Prediction and output
# # if submit_button:
# #     model = joblib.load("classifier.pkl")
# #     x = np.array([Sepal_length, Sepal_width, Petal_length, Petal_width])
# #     if any(x <= 0):
# #         st.warning("⚠️ Input values must be greater than 0")
# #     else:
# #         prediction = model.predict([x])
# #         species_images = {
# #             'setosa': 'Irissetosa1.jpg',
# #             'versicolor': 'Versicolor.webp',
# #             'virginica': 'virgina.jpg'
# #         }
# #         st.success(f"🎉 Predicted Species: **{prediction[0]}**")
# #         st.image(species_images[prediction[0].lower()], caption=f'Iris {prediction[0]}', use_column_width=True)
# # Prediction and output
# if submit_button:
#     model = joblib.load("classifier.pkl")
#     x = np.array([Sepal_length, Sepal_width, Petal_length, Petal_width])
#     if any(x <= 0):
#         st.warning("⚠️ Input values must be greater than 0")
#     else:
#         prediction = model.predict([x])
        
#         # Corrected dictionary keys to match the exact species names
#         species_images = {
#             'Iris-setosa': 'Irissetosa1.jpg',
#             'Iris-versicolor': 'Versicolor.webp',
#             'Iris-virginica': 'virgina.jpg'
#         }
        
#         # Accessing the species image based on the prediction
#         st.success(f"🎉 Predicted Species: **{prediction[0]}**")
#         st.image(species_images[prediction[0]], caption=f'Iris {prediction[0]}', use_column_width=True)


# # Data visualization
# if show_dataset:
#     st.markdown("### 🌸 Dataset Overview & Visualization")
#     st.write(dataset.head())

# if show_pairplot:
#     st.markdown("#### 📊 Pairplot of Iris Features")
#     sns.pairplot(dataset, hue="Species", palette="viridis")
#     st.pyplot(plt)

# # Model performance metrics
# if show_performance:
#     st.markdown("### 📈 Model Performance")
    
#     # Calculate accuracy as percentages
#     train_accuracy = knn.score(x_train, y_train) * 100
#     test_accuracy = knn.score(x_test, y_test) * 100
    
#     # Display the accuracy with percentage formatting
#     st.markdown(f"**Train Set Accuracy:** {train_accuracy:.2f}")
#     st.markdown(f"**Test Set Accuracy:** {test_accuracy:.2f}")

#     # # Confusion Matrix
#     # st.markdown("#### Confusion Matrix")
#     # from sklearn.metrics import confusion_matrix
#     # cm = confusion_matrix(y_test, knn.predict(x_test))
#     # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
#     # st.pyplot(plt)

# import pandas as pd
# import numpy as np
# import streamlit as st
# import pickle
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix

# # Load the dataset
# dataset = pd.read_csv("Iris.csv")

# # Prepare the features and target variable
# x = dataset.drop(["Species", "Id"], axis=1)
# y = dataset["Species"]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# # Initialize and train the KNN classifier
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(x_train, y_train)

# # Save the model
# with open("classifier.pkl", "wb") as pickle_out:
#     pickle.dump(knn, pickle_out)

# # Streamlit UI
# st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="wide")
# st.title("🌼 Iris Species Prediction App")

# st.markdown("""
#     Welcome to the Iris Species Prediction app! 🌺
#     This app predicts the species of an Iris flower based on its features. 
#     The model behind this app is a K-Nearest Neighbors classifier trained on the classic Iris dataset.
# """)

# # Sidebar for additional options
# with st.sidebar:
#     st.header("🔧 Settings")
#     st.markdown("Adjust the app settings below:")
    
#     # Number of neighbors for KNN
#     n_neighbors = st.slider("Number of Neighbors (K in KNN)", 1, 15, 3)
#     knn.n_neighbors = n_neighbors
#     knn.fit(x_train, y_train)  # Re-train with new K
    
#     # Checkboxes for additional features
#     show_dataset = st.checkbox("Show Dataset Overview", value=True)
#     show_pairplot = st.checkbox("Show Pairplot Visualization", value=True)
#     show_performance = st.checkbox("Show Model Performance", value=True)
#     show_confusion_matrix = st.checkbox("Show Confusion Matrix", value=True)
#     show_model_summary = st.checkbox("Show Model Summary", value=True)

# # Input form
# with st.form("prediction_form"):
#     st.subheader("🔍 Enter Flower Features")
#     Sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.0, step=0.1)
#     Sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.0, step=0.1)
#     Petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.5, step=0.1)
#     Petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2, step=0.1)
#     submit_button = st.form_submit_button(label='🌟 Predict Species')

# # Prediction and output
# if submit_button:
#     model = joblib.load("classifier.pkl")
#     x_input = np.array([Sepal_length, Sepal_width, Petal_length, Petal_width])
#     if any(x_input <= 0):
#         st.warning("⚠️ Input values must be greater than 0")
#     else:
#         prediction = model.predict([x_input])
        
#         # Corrected dictionary keys to match the exact species names
#         species_images = {
#             'Iris-setosa': 'Irissetosa1.jpg',
#             'Iris-versicolor': 'Versicolor.webp',
#             'Iris-virginica': 'virgina.jpg'
#         }
        
#         # Accessing the species image based on the prediction
#         st.success(f"🎉 Predicted Species: **{prediction[0]}**")
#         st.image(species_images[prediction[0]], caption=f'Iris {prediction[0]}', use_column_width=True)

# # Data visualization
# if show_dataset:
#     st.markdown("### 🌸 Dataset Overview & Visualization")
#     st.write(dataset.head())
#     st.markdown("#### Summary Statistics")
#     st.write(dataset.describe())
#     st.markdown("#### Data Distribution")
#     st.bar_chart(dataset["Species"].value_counts())

# if show_pairplot:
#     st.markdown("#### 📊 Pairplot of Iris Features")
#     sns.pairplot(dataset, hue="Species", palette="viridis")
#     st.pyplot(plt)

# # Model performance metrics
# if show_performance:
#     st.markdown("### 📈 Model Performance")
    
#     # Calculate accuracy as percentages
#     train_accuracy = knn.score(x_train, y_train) * 100
#     test_accuracy = knn.score(x_test, y_test) * 100
    
#     # Display the accuracy with percentage formatting
#     st.markdown(f"**Train Set Accuracy:** {train_accuracy:.2f}%")
#     st.markdown(f"**Test Set Accuracy:** {test_accuracy:.2f}%")

#     # Confusion Matrix
#     if show_confusion_matrix:
#         st.markdown("### 🧩 Confusion Matrix")
#         cm = confusion_matrix(y_test, knn.predict(x_test), labels=knn.classes_)
#         fig, ax = plt.subplots(figsize=(8, 6))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=knn.classes_, yticklabels=knn.classes_, ax=ax)
#         plt.xlabel('Predicted Labels')
#         plt.ylabel('True Labels')
#         plt.title('Confusion Matrix')
#         st.pyplot(fig)

# # Model Summary
# if show_model_summary:
#     st.markdown("### 📋 Model Summary")
#     st.write(f"**Number of Neighbors:** {knn.n_neighbors}")
#     st.write(f"**Algorithm:** {knn._fit_method}")
    
#     # Display the distance metric
#     distance_metric = knn.metric
#     if distance_metric == 'minkowski' and knn.p == 2:
#         distance_metric = 'Euclidean (Minkowski with p=2)'
#     elif distance_metric == 'minkowski':
#         distance_metric = f'Minkowski with p={knn.p}'
        
#     st.write(f"**Distance Metric:** {distance_metric}")
#     st.write("This summary provides an overview of the current model configuration.")


# import pandas as pd
# import numpy as np
# import streamlit as st
# import pickle
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import roc_curve, auc, precision_recall_curve
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder

# # Load the dataset
# dataset = pd.read_csv("Iris.csv")

# # Prepare the features and target variable
# x = dataset.drop(["Species", "Id"], axis=1)
# y = dataset["Species"]

# # Encode target variable
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)

# x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.3, random_state=42)

# # Standardize the features
# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)

# # Initialize models
# models = {
#     "KNN": KNeighborsClassifier(),
#     "Random Forest": RandomForestClassifier(n_estimators=100),
#     "SVM": SVC(probability=True),
#     "Logistic Regression": LogisticRegression()
# }

# # Streamlit UI
# st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="wide")
# st.title("🌼 Iris Species Prediction App")

# st.markdown("""
#     Welcome to the Iris Species Prediction app! 🌺
#     This app predicts the species of an Iris flower based on its features. 
#     The model behind this app can be selected from the list below.
# """)

# # Sidebar for additional options
# with st.sidebar:
#     st.header("🔧 Settings")
#     st.markdown("Adjust the app settings below:")

#     # Model selection
#     model_choice = st.selectbox("Choose a Model", list(models.keys()))

#     # Number of neighbors for KNN
#     if model_choice == "KNN":
#         n_neighbors = st.slider("Number of Neighbors (K in KNN)", 1, 15, 3)
#         models[model_choice].n_neighbors = n_neighbors

#     # Fit the selected model
#     model = models[model_choice]
#     model.fit(x_train_scaled, y_train)

#     # Checkboxes for additional features
#     show_dataset = st.checkbox("Show Dataset Overview", value=True)
#     show_pairplot = st.checkbox("Show Pairplot Visualization", value=True)
#     show_performance = st.checkbox("Show Model Performance", value=True)
#     show_decision_boundary = st.checkbox("Show Decision Boundary", value=True)
#     show_roc_curve = st.checkbox("Show ROC Curve", value=True)
#     show_precision_recall = st.checkbox("Show Precision-Recall Curve", value=True)
#     show_confidence_score = st.checkbox("Show Confidence Score", value=True)

# # Input form
# with st.form("prediction_form"):
#     st.subheader("🔍 Enter Flower Features")
#     Sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.0, step=0.1)
#     Sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.0, step=0.1)
#     Petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.5, step=0.1)
#     Petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2, step=0.1)
#     submit_button = st.form_submit_button(label='🌟 Predict Species')

# # Prediction and output
# if submit_button:
#     x_input = np.array([Sepal_length, Sepal_width, Petal_length, Petal_width])
#     x_input_scaled = scaler.transform([x_input])

#     if any(x_input <= 0):
#         st.warning("⚠️ Input values must be greater than 0")
#     else:
#         prediction = model.predict(x_input_scaled)
#         probability = model.predict_proba(x_input_scaled) if hasattr(model, "predict_proba") else None
#         predicted_species = label_encoder.inverse_transform(prediction)[0]

#         species_images = {
#             'Iris-setosa': 'Irissetosa1.jpg',
#             'Iris-versicolor': 'Versicolor.webp',
#             'Iris-virginica': 'virgina.jpg'
#         }

#         st.success(f"🎉 Predicted Species: **{predicted_species}**")
#         st.image(species_images.get(predicted_species, 'default.jpg'), caption=f'Iris {predicted_species}', use_column_width=True)

#         if show_confidence_score and probability is not None:
#             st.info(f"🧠 Model Confidence: **{np.max(probability) * 100:.2f}%**")

# # Data visualization
# if show_dataset:
#     st.markdown("### 🌸 Dataset Overview & Visualization")
#     st.write(dataset.head())

# if show_pairplot:
#     st.markdown("#### 📊 Pairplot of Iris Features")
#     sns.pairplot(dataset, hue="Species", palette="viridis")
#     st.pyplot(plt)

# # Model performance metrics
# if show_performance:
#     st.markdown("### 📈 Model Performance")

#     train_accuracy = model.score(x_train_scaled, y_train) * 100
#     test_accuracy = model.score(x_test_scaled, y_test) * 100

#     st.markdown(f"**Train Set Accuracy:** {train_accuracy:.2f}%")
#     st.markdown(f"**Test Set Accuracy:** {test_accuracy:.2f}%")

# if show_decision_boundary:
#     st.markdown("### 🗺️ Decision Boundary Visualization")

#     # Choose two features for visualization
#     feature_indices = [0, 1]  # Indexes for Sepal Length and Sepal Width
#     feature_names = ['Sepal Length (cm)', 'Sepal Width (cm)']
    
#     # Use only two features for the training data
#     x_train_2d = x_train_scaled[:, feature_indices]
#     x_test_2d = x_test_scaled[:, feature_indices]
    
#     # Define mesh grid for the decision boundary
#     x_min, x_max = x_train_2d[:, 0].min() - 1, x_train_2d[:, 0].max() + 1
#     y_min, y_max = x_train_2d[:, 1].min() - 1, x_train_2d[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    
#     # Flatten the mesh grid for prediction
#     grid_points = np.c_[xx.ravel(), yy.ravel()]
    
#     # Create a full feature set for prediction (with zeros for the missing features)
#     grid_points_full = np.zeros((grid_points.shape[0], x_train_scaled.shape[1]))
#     grid_points_full[:, feature_indices] = grid_points
    
#     # Predict using the model
#     try:
#         Z = model.predict(grid_points_full)
#         Z = np.array(Z)  # Ensure Z is a NumPy array
#     except Exception as e:
#         st.error(f"Prediction error: {e}")
#         Z = np.zeros(xx.shape, dtype=int)  # Fallback to avoid further errors
    
#     # Ensure Z is valid and reshaped correctly
#     Z = Z.reshape(xx.shape)
    
#     # Convert to numeric type if necessary
#     if not np.issubdtype(Z.dtype, np.number):
#         Z = np.where(np.array(Z) == 0, 0, np.where(np.array(Z) == 1, 1, 2))
    
#     plt.figure(figsize=(10, 6))
#     plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
#     plt.scatter(x_train_2d[:, 0], x_train_2d[:, 1], c=y_train, edgecolor='k', s=20, cmap=plt.cm.coolwarm)
#     plt.title(f"Decision Boundary with {model_choice}")
#     plt.xlabel(feature_names[0])
#     plt.ylabel(feature_names[1])
#     st.pyplot(plt)

# if show_roc_curve and hasattr(model, "predict_proba"):
#     st.markdown("### 🚀 ROC Curve")
#     plt.figure(figsize=(10, 6))
#     for i, species in enumerate(label_encoder.classes_):
#         fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_test_scaled)[:, i], pos_label=i)
#         roc_auc = auc(fpr, tpr)
#         plt.plot(fpr, tpr, lw=2, label=f'{species} (AUC = {roc_auc:.2f})')

#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
#     plt.legend(loc="lower right")
#     st.pyplot(plt)

# if show_precision_recall and hasattr(model, "predict_proba"):
#     st.markdown("### 🔍 Precision-Recall Curve")
#     plt.figure(figsize=(10, 6))
    
#     for i, species in enumerate(label_encoder.classes_):
#         precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(x_test_scaled)[:, i], pos_label=i)
#         plt.plot(recall, precision, lw=2, label=f'{species}')

#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.legend(loc="best")
#     st.pyplot(plt)

# import streamlit as st
# from streamlit_lottie import st_lottie
# import json
# import joblib
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix

# # Load Lottie Animation
# def load_lottiefile(filepath: str):
#     with open(filepath, "r") as f:
#         return json.load(f)

# # Load animations
# success_animation = load_lottiefile("success_animation.json")


# # Streamlit UI
# st.set_page_config(page_title="Test Lottie Animation", page_icon="🌸")
# st.title("🌼 Lottie Animation Test")

# # Display Lottie animation
# st_lottie(success_animation, speed=1, reverse=False, loop=True, height="300px", width="300px")

# st.markdown("If you see the animation above, the Lottie file is working correctly.")

# # Load the dataset
# dataset = pd.read_csv("Iris.csv")

# # Prepare the features and target variable
# x = dataset.drop(["Species", "Id"], axis=1)
# y = dataset["Species"]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# # Initialize and train the KNN classifier
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(x_train, y_train)

# # Save the model
# with open("classifier.pkl", "wb") as pickle_out:
#     joblib.dump(knn, pickle_out)

# # Streamlit UI
# st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="wide")
# st.title("🌼 Iris Species Prediction App")

# st.markdown("""
#     <div style="background-color:#black; padding: 10px; border-radius: 10px;">
#     <h2 style="color:#4B0082;">Welcome to the Iris Species Prediction app! 🌺</h2>
#     <p>This app predicts the species of an Iris flower based on its features. 
#     The model behind this app is a K-Nearest Neighbors classifier trained on the classic Iris dataset.</p>
#     </div>
# """, unsafe_allow_html=True)

# # Sidebar for additional options with icons
# with st.sidebar:
#     st_lottie(success_animation, speed=1, reverse=False, loop=True, height="150px", width="100%")
#     st.header("🔧 Settings")
#     st.markdown("Adjust the app settings below:")
    
#     # Number of neighbors for KNN
#     n_neighbors = st.slider("Number of Neighbors (K in KNN)", 1, 15, 3)
#     knn.n_neighbors = n_neighbors
#     knn.fit(x_train, y_train)  # Re-train with new K
    
#     # Checkboxes for additional features with icons
#     show_dataset = st.checkbox("Show Dataset Overview 🗂️", value=True)
#     show_pairplot = st.checkbox("Show Pairplot Visualization 📊", value=True)
#     show_performance = st.checkbox("Show Model Performance 📈", value=True)
#     show_confusion_matrix = st.checkbox("Show Confusion Matrix 🧩", value=True)
#     show_model_summary = st.checkbox("Show Model Summary 📝", value=True)

# # Input form
# with st.form("prediction_form"):
#     st.subheader("🔍 Enter Flower Features")
#     Sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.0, step=0.1)
#     Sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.0, step=0.1)
#     Petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.5, step=0.1)
#     Petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2, step=0.1)
#     submit_button = st.form_submit_button(label='🌟 Predict Species')

# # Prediction and output
# if submit_button:
#     model = joblib.load("classifier.pkl")
#     x_input = np.array([Sepal_length, Sepal_width, Petal_length, Petal_width])
#     if any(x_input <= 0):
#         st.warning("⚠️ Input values must be greater than 0")
#     else:
#         prediction = model.predict([x_input])
        
#         species_images = {
#             'Iris-setosa': 'Irissetosa1.jpg',
#             'Iris-versicolor': 'Versicolor.webp',
#             'Iris-virginica': 'virgina.jpg'
#         }
        
#         st.success(f"🎉 Predicted Species: **{prediction[0]}**")
#         # st_lottie(success_animation, speed=1, reverse=False, loop=False, height="200px", width="100%")
#         st.image(species_images[prediction[0]], caption=f'Iris {prediction[0]}', use_column_width=True)

# # Data visualization with expanders
# if show_dataset:
#     with st.expander("🌸 Dataset Overview & Visualization"):
#         st.write(dataset.head())
#         st.markdown("#### Summary Statistics")
#         st.write(dataset.describe())
#         st.markdown("#### Data Distribution")
#         st.bar_chart(dataset["Species"].value_counts())

# if show_pairplot:
#     with st.expander("📊 Pairplot of Iris Features"):
#         sns.pairplot(dataset, hue="Species", palette="viridis")
#         st.pyplot(plt)

# # Model performance metrics
# if show_performance:
#     with st.expander("📈 Model Performance"):
#         train_accuracy = knn.score(x_train, y_train) * 100
#         test_accuracy = knn.score(x_test, y_test) * 100
        
#         st.markdown(f"**Train Set Accuracy:** {train_accuracy:.2f}%")
#         st.markdown(f"**Test Set Accuracy:** {test_accuracy:.2f}%")

#         if show_confusion_matrix:
#             st.markdown("### 🧩 Confusion Matrix")
#             cm = confusion_matrix(y_test, knn.predict(x_test), labels=knn.classes_)
#             fig, ax = plt.subplots(figsize=(8, 6))
#             sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=knn.classes_, yticklabels=knn.classes_, ax=ax)
#             plt.xlabel('Predicted Labels')
#             plt.ylabel('True Labels')
#             plt.title('Confusion Matrix')
#             st.pyplot(fig)

# # Model Summary
# if show_model_summary:
#     with st.expander("📋 Model Summary"):
#         st.write(f"**Number of Neighbors:** {knn.n_neighbors}")
#         st.write(f"**Algorithm:** {knn._fit_method}")
        
#         distance_metric = knn.metric
#         if distance_metric == 'minkowski' and knn.p == 2:
#             distance_metric = 'Euclidean (Minkowski with p=2)'
#         elif distance_metric == 'minkowski':
#             distance_metric = f'Minkowski with p={knn.p}'
            
#         st.write(f"**Distance Metric:** {distance_metric}")
#         st.write("This summary provides an overview of the current model configuration.")

import streamlit as st
from streamlit_lottie import st_lottie
import json
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Set page configuration
st.set_page_config(page_title="Iris Species Predictor", page_icon="🌸", layout="wide")

# Load Lottie Animation
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load animations
success_animation = load_lottiefile("success_animation.json")

# Streamlit UI
st.title("🌼 Iris Species Prediction App")

st.markdown("""
    <div style="background-color:#black; padding: 10px; border-radius: 10px;">
    <h2 style="color:#4B0082;">Welcome to the Iris Species Prediction app! 🌺</h2>
    <p>This app predicts the species of an Iris flower based on its features. 
    The model behind this app is a K-Nearest Neighbors classifier trained on the classic Iris dataset.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar for additional options with icons
with st.sidebar:
    st_lottie(success_animation, speed=1, reverse=False, loop=True, height="150px", width="100%")
    st.header("🔧 Settings")
    st.markdown("Adjust the app settings below:")
    
    # Number of neighbors for KNN
    n_neighbors = st.slider("Number of Neighbors (K in KNN)", 1, 15, 3)
    
    # Load the dataset
    dataset = pd.read_csv("Iris.csv")
    x = dataset.drop(["Species", "Id"], axis=1)
    y = dataset["Species"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    
    # Initialize and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    
    # Checkboxes for additional features with icons
    show_dataset = st.checkbox("Show Dataset Overview 🗂️", value=True)
    show_pairplot = st.checkbox("Show Pairplot Visualization 📊", value=True)
    show_performance = st.checkbox("Show Model Performance 📈", value=True)
    show_confusion_matrix = st.checkbox("Show Confusion Matrix 🧩", value=True)
    show_model_summary = st.checkbox("Show Model Summary 📝", value=True)

# Input form
with st.form("prediction_form"):
    st.subheader("🔍 Enter Flower Features")
    Sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, value=5.0, step=0.1)
    Sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, value=3.0, step=0.1)
    Petal_length = st.number_input("Petal Length (cm)", min_value=0.0, value=1.5, step=0.1)
    Petal_width = st.number_input("Petal Width (cm)", min_value=0.0, value=0.2, step=0.1)
    submit_button = st.form_submit_button(label='🌟 Predict Species')

# Prediction and output
if submit_button:
    model = joblib.load("classifier.pkl")
    x_input = np.array([Sepal_length, Sepal_width, Petal_length, Petal_width])
    if any(x_input <= 0):
        st.warning("⚠️ Input values must be greater than 0")
    else:
        prediction = model.predict([x_input])
        
        species_images = {
            'Iris-setosa': 'Irissetosa1.jpg',
            'Iris-versicolor': 'Versicolor.webp',
            'Iris-virginica': 'virgina.jpg'
        }
        
        st.success(f"🎉 Predicted Species: **{prediction[0]}**")
        st.image(species_images[prediction[0]], caption=f'Iris {prediction[0]}', use_column_width=True)

# Data visualization with expanders
if show_dataset:
    with st.expander("🌸 Dataset Overview & Visualization"):
        st.write(dataset.head())
        st.markdown("#### Summary Statistics")
        st.write(dataset.describe())
        st.markdown("#### Data Distribution")
        st.bar_chart(dataset["Species"].value_counts())

if show_pairplot:
    with st.expander("📊 Pairplot of Iris Features"):
        sns.pairplot(dataset, hue="Species", palette="viridis")
        st.pyplot(plt)

# Model performance metrics
if show_performance:
    with st.expander("📈 Model Performance"):
        train_accuracy = knn.score(x_train, y_train) * 100
        test_accuracy = knn.score(x_test, y_test) * 100
        
        st.markdown(f"**Train Set Accuracy:** {train_accuracy:.2f}%")
        st.markdown(f"**Test Set Accuracy:** {test_accuracy:.2f}%")

        if show_confusion_matrix:
            st.markdown("### 🧩 Confusion Matrix")
            cm = confusion_matrix(y_test, knn.predict(x_test), labels=knn.classes_)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=knn.classes_, yticklabels=knn.classes_, ax=ax)
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            st.pyplot(fig)

# Model Summary
if show_model_summary:
    with st.expander("📋 Model Summary"):
        st.write(f"**Number of Neighbors:** {knn.n_neighbors}")
        st.write(f"**Algorithm:** {knn._fit_method}")
        
        distance_metric = knn.metric
        if distance_metric == 'minkowski' and knn.p == 2:
            distance_metric = 'Euclidean (Minkowski with p=2)'
        elif distance_metric == 'minkowski':
            distance_metric = f'Minkowski with p={knn.p}'
            
        st.write(f"**Distance Metric:** {distance_metric}")
        st.write("This summary provides an overview of the current model configuration.")
