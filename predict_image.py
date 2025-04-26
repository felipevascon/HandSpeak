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
    "IMAGE_LOAD": "ER002",
    "LANDMARK_EXTRACTION": "ER003",
    "PREDICTION": "ER004",
    "NO_HAND": "ER005",
    "VISUALIZATION": "ER006"
}

# ======================
# 1. CARREGAMENTO DE MODELOS
# ======================
def load_models():
    """Carrega o modelo e scaler com tratamento de erros"""
    try:
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        print(f"❌ [{ERROR_CODES['MODEL_LOAD']}] Falha ao carregar modelos - {type(e).__name__}: {str(e)}")
        exit()

model, scaler = load_models()

# ======================
# 2. FUNÇÕES DE PROCESSAMENTO
# ======================
def resize_to_square(image):
    """Redimensiona imagem para formato quadrado"""
    try:
        size = max(image.shape[:2])
        new_image = np.zeros((size, size, 3), dtype=np.uint8)
        y_offset = (size - image.shape[0]) // 2
        x_offset = (size - image.shape[1]) // 2
        new_image[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
        return new_image
    except Exception as e:
        print(f"❌ [{ERROR_CODES['IMAGE_LOAD']}] Falha no redimensionamento - {type(e).__name__}: {str(e)}")
        return None

def extract_hand_landmarks(image_path, visualize=False):
    """Extrai landmarks da mão com tratamento completo de erros"""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils

    try:
        # Carrega e verifica imagem
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ [{ERROR_CODES['IMAGE_LOAD']}] Imagem não encontrada ou inválida")
            return None
            
        # Processamento da imagem
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb_square = resize_to_square(image_rgb)
        results = hands.process(image_rgb_square)

        if not results.multi_hand_landmarks:
            if visualize:
                print(f"⚠️ [{ERROR_CODES['NO_HAND']}] Nenhuma mão detectada na imagem")
            return None

        # Extração de landmarks
        landmarks = results.multi_hand_landmarks[0]
        coords = np.array([coord for lm in landmarks.landmark for coord in [lm.x, lm.y, lm.z]])

        # Visualização (se solicitado)
        if visualize:
            try:
                annotated_image = cv2.cvtColor(image_rgb_square, cv2.COLOR_RGB2BGR).copy()
                mp_drawing.draw_landmarks(annotated_image, landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.imshow("Detecção de Mão", annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"⚠️ [{ERROR_CODES['VISUALIZATION']}] Erro na visualização - {type(e).__name__}: {str(e)}")

        return coords
        
    except Exception as e:
        print(f"❌ [{ERROR_CODES['LANDMARK_EXTRACTION']}] Erro no processamento - {type(e).__name__}: {str(e)}")
        return None

# ======================
# 3. FUNÇÃO DE PREDIÇÃO
# ======================
def predict_from_image(image_path):
    """Executa a predição com tratamento completo de erros"""
    features = extract_hand_landmarks(image_path, visualize=False)
    
    if features is None:
        return None

    try:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        return prediction[0]
    except Exception as e:
        print(f"❌ [{ERROR_CODES['PREDICTION']}] Erro na predição - {type(e).__name__}: {str(e)}")
        return None

# ======================
# 4. EXECUÇÃO PRINCIPAL
# ======================
if __name__ == "__main__":
    # Configuração do caminho da imagem
    image_path = "C:/Project_HandSpeak/training/C/1.png"
    
    print("\n=== ANÁLISE DE LINGUAGEM DE SINAIS ===")
    print(f"Processando a imagem localizada no endereço: {image_path}\n")
    
    # Processamento principal
    try:
        # Etapa 1: Extração de características
        features = extract_hand_landmarks(image_path, visualize=False)
        
        if features is None:
            print(f"❌ [{ERROR_CODES['NO_HAND']}] Análise não pode ser concluída - Nenhuma mão detectada")
        else:
            # Etapa 2: Predição
            prediction = predict_from_image(image_path)
            
            if prediction is not None:
                print(f"✅ RESULTADO: Letra '{prediction}' identificada com sucesso")
            else:
                print(f"❌ [{ERROR_CODES['PREDICTION']}] A mão foi detectada, mas a predição falhou")
            
            # Etapa 3: Visualização opcional
            input("\nPressione Enter para visualizar a detecção (ou Ctrl+C para sair)...")
            extract_hand_landmarks(image_path, visualize=True)
            
    except KeyboardInterrupt:
        print("\n⏹ Processo interrompido pelo usuário")
    except Exception as e:
        print(f"❌ [ER000] Erro não esperado - {type(e).__name__}: {str(e)}")
    
    print("\n=== FIM DA ANÁLISE ===")
