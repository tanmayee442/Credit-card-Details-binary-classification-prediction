import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

X = pd.read_csv("Credit_card.csv")
y = pd.read_csv("Credit_card_label.csv")
df = pd.merge(X, y, on="Ind_ID")
df.dropna(subset=["label"], inplace=True)
df.drop(columns=["Ind_ID"], inplace=True)

for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0])
    else:
        df[col].fillna(df[col].median())

df_encoded = df.copy()
label_encoders = {}
for col in df_encoded.select_dtypes(include="object"):
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

X_clean = df_encoded.drop("label", axis=1)
y_clean = df_encoded["label"]
X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(" Accuracy:", accuracy)
print("\n Classification Report:\n", report)

plt.figure(figsize=(6, 4))# Class distribution to check imbalance
sns.countplot(x=df['label'])
plt.title("Class Distribution")
plt.xlabel("Label (0 = No Default, 1 = Default)")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 5))#Confusion matrix 
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(10, 5))# To visualize where missing values were originally present (before encoding)
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

plt.figure(figsize=(12, 10))#  To visualize correlation between all numerical and encoded categorical features(after encoding)
sns.heatmap(df_encoded.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

importances = model.feature_importances_# This helps to display the top features that influence the Random Forest model
features = X_clean.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title("Feature Importance graph")
plt.tight_layout()
plt.show()



