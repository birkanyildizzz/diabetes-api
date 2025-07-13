import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# CSV dosyasını oku
df = pd.read_csv("diabetes_data_upload.csv")

label_encoders = {}

# String olan sütunları LabelEncoder ile dönüştür
for column in df.columns:
    if df[column].dtype == object:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# Özellikler ve hedef
X = df.drop("class", axis=1)
y = df["class"]

# Eğitim ve test verilerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli eğit
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model ve label_encoders'ı kaydet
joblib.dump(model, "diabetes_rf_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("✅ Model ve encoder başarıyla eğitildi ve kaydedildi.")
