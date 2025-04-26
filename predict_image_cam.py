import cv2
import mediapipe as mp
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# ======================
# CONSTANTES DE ERRO
# ======================
ERROR_CODES = {
    "MODEL_LOAD": "ER001",
    "LANDMARK_EXTRACTION": "ER003",
    "PREDICTION": "ER004",
    "NO_HAND": "ER005",
    "VISUALIZATION": "ER006"
}

# ======================
# 1. CARREGAMENTO DE MODELOS
# ======================
def load_models():
    try:
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        print(f"❌ [{ERROR_CODES['MODEL_LOAD']}] Falha ao carregar modelos - {type(e).__name__}: {str(e)}")
        exit()

model, scaler = load_models()

# ======================
# 2. FUNÇÃO DE EXTRAÇÃO (AGORA FRAME, NÃO ARQUIVO)
# ======================
def extract_hand_landmarks_from_frame(frame):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
    try:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None

        landmarks = results.multi_hand_landmarks[0]
        coords = np.array([coord for lm in landmarks.landmark for coord in [lm.x, lm.y, lm.z]])
        return coords
    except Exception as e:
        print(f"❌ [{ERROR_CODES['LANDMARK_EXTRACTION']}] Erro no processamento - {type(e).__name__}: {str(e)}")
        return None

# ======================
# 3. EXECUÇÃO PRINCIPAL (WEBCAM)
# ======================
if __name__ == "__main__":
    print("\n=== MODO WEBCAM - Análise de Libras ===\n")
    cap = cv2.VideoCapture(0)  # 0 = webcam padrão

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ Erro ao capturar frame")
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Desenhar landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extrair coordenadas
                    coords = np.array([coord for lm in hand_landmarks.landmark for coord in [lm.x, lm.y, lm.z]])

                    # Fazer predição
                    try:
                        features_scaled = scaler.transform([coords])
                        prediction = model.predict(features_scaled)
                        letra = prediction[0]

                        # Exibir o resultado na tela
                        cv2.putText(frame, f"Letra: {letra}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    except Exception as e:
                        print(f"❌ [{ERROR_CODES['PREDICTION']}] Predição falhou - {type(e).__name__}: {str(e)}")

            # Mostrar a imagem
            cv2.imshow("Detecção de Libras - Webcam", frame)

            # Pressione 'q' para sair
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n⏹ Encerrado pelo usuário.")

    cap.release()
    cv2.destroyAllWindows()
    print("\n=== FIM ===")
