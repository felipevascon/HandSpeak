import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ---------------------------
# 1. Carregar dados
# ---------------------------
# Exemplo: vetor de landmarks + label da letra
# dataset: CSV com colunas 'x0', 'y0', ..., 'x20', 'y20', 'label'

# Carregar CSV
df = pd.read_csv('landmarks.csv')

# Verificar as contagens das labels
print(df['label'].value_counts())

# Remover labels com menos de 2 ocorrências
counts = df['label'].value_counts()
labels_to_keep = counts[counts > 1].index
df = df[df['label'].isin(labels_to_keep)]

X = df.drop('label', axis=1).values
y = df['label'].values

# ---------------------------
# 2. Pré-processamento
# ---------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------
# 3. Dividir os dados
# ---------------------------
print(pd.Series(y).value_counts())
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# ---------------------------
# 4. Modelos
# ---------------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', C=1.0, gamma='scale'),
    "MLP (Rede Neural)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
}

# ---------------------------
# 5. Treinamento e Avaliação
# ---------------------------
def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    print(f"🔍 Avaliando: {name}")
    print("Acurácia:", accuracy_score(y_val, y_pred))
    print("Classification Report:\n", classification_report(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred, labels=np.unique(y))
    plt.figure(figsize=(12, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(y), yticklabels=np.unique(y), cmap='Blues')
    plt.title(f'Matriz de Confusão - {name}')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.show()

# Rodar todos os modelos
for name, model in models.items():
    evaluate_model(name, model, X_train, y_train, X_val, y_val)

# ---------------------------
# 6. Teste Final com melhor modelo
# ---------------------------
# Exemplo usando Random Forest como melhor modelo
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)
print("🎯 Teste final - Random Forest")
print("Acurácia no teste:", accuracy_score(y_test, y_test_pred))

# No código de treino, depois de escolher o melhor modelo:
joblib.dump(best_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")