import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle


DATA_PATH = "asl_dataset/asl_dataset_a_z.csv"
df = pd.read_csv(DATA_PATH)


X = df.drop('label', axis=1)  
y = df['label']  


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.2f}")


with open('asl_model.pkl', 'wb') as f:
    pickle.dump(model, f)


label_map = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
print("Label mapping:", label_map)

print("Model saved to asl_model.pkl")