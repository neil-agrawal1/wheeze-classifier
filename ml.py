from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # SVM classifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
from processing import process_data;
import pandas as pd

folder_path = 'normal_wavfiles'
output_csv = "output.csv"
dataframes = []

for file_name in os.listdir(folder_path): 
    if file_name.endswith(".wav"): 
        file_path = os.path.join(folder_path, file_name)
        df = process_data(file_path, "NW")
        dataframes.append(df)

folder_path = 'wheeze_wavfiles'

for file_name in os.listdir(folder_path): 
    if file_name.endswith(".wav"): 
        file_path = os.path.join(folder_path, file_name)
        df2 = process_data(file_path, "W")
        dataframes.append(df2)

final_df = pd.concat(dataframes, ignore_index=True)

X = final_df.drop(columns='Label')
y = final_df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = SVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')