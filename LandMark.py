import os
import csv
import psycopg2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests
from dotenv import load_dotenv
import cv2

# Carrega variáveis de ambiente
load_dotenv()

# Configuração do banco de dados
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "5432")

# Caminho para o modelo
MODEL_PATH = 'hand_landmarker.task'
print("📂 Caminho absoluto do modelo:", os.path.abspath(MODEL_PATH))
print("🔍 Existe o arquivo?", os.path.exists(MODEL_PATH))

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Modelo não encontrado em: {os.path.abspath(MODEL_PATH)}")

# Inicializa o detector
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

# Arquivo de saída
OUTPUT_CSV = 'landmarks.csv'

def connect_db():
    """Conecta ao banco de dados PostgreSQL."""
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

def fetch_images_and_letters():
    """Busca 300 imagens de cada letra e associa ao nome da letra."""
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT l.letters_numbers AS letter, i.images_url AS image_url
        FROM letter_number_images i
        INNER JOIN letters_numbers l ON i.letter_number_id::text = l.id::text
    """)
    data = cur.fetchall()
    cur.close()
    conn.close()
    return data

os.makedirs("output_landmarked_images", exist_ok=True)

# Cria arquivo CSV e escreve cabeçalho
with open(OUTPUT_CSV, mode='w', newline='') as f:
    writer = csv.writer(f)
    header = ['label'] + [f'{coord}{i}' for i in range(21) for coord in ['x', 'y', 'z']]
    writer.writerow(header)

    # Busca imagens e letras do banco de dados
    images_and_letters = fetch_images_and_letters()

    for idx, (letter, image_url) in enumerate(images_and_letters):
        # Faz o download da imagem com timeout
        try:
            response = requests.get(image_url, timeout=10)
            if response.status_code != 200:
                print(f"❌ Não foi possível baixar a imagem: {image_url} (Status: {response.status_code})")
                continue
        except requests.exceptions.RequestException as e:
            print(f"❌ Erro ao baixar a imagem: {image_url} ({e})")
            continue

        # Salva a imagem temporariamente
        temp_image_path = f"temp_{letter}_{idx}.jpg"
        with open(temp_image_path, "wb") as img_file:
            img_file.write(response.content)

        # Redimensiona a imagem para ser quadrada
        image = cv2.imread(temp_image_path)
        height, width, _ = image.shape
        size = max(height, width)
        square_image = cv2.copyMakeBorder(
            image,
            (size - height) // 2,
            (size - height + 1) // 2,
            (size - width) // 2,
            (size - width + 1) // 2,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # Carrega a imagem redimensionada no MediaPipe
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(square_image, cv2.COLOR_BGR2RGB))

        # Detecta landmarks
        result = detector.detect(mp_image)
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            row = [letter]
            for landmark in hand:
                row.extend([landmark.x, landmark.y, landmark.z])
            writer.writerow(row)

            # Desenhar os landmarks na imagem
            annotated_image = square_image.copy()
            for landmark in hand:
                x_px = int(landmark.x * size)
                y_px = int(landmark.y * size)
                cv2.circle(annotated_image, (x_px, y_px), 4, (0, 255, 0), -1)  # Pontinhos verdes

            # Conectar pontos com linhas (opcional, se quiser deixar mais bonito)
            # Para conectar corretamente, MediaPipe define uma conexão padrão:
            connections = mp.solutions.hands.HAND_CONNECTIONS
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                start = hand[start_idx]
                end = hand[end_idx]
                start_point = (int(start.x * size), int(start.y * size))
                end_point = (int(end.x * size), int(end.y * size))
                cv2.line(annotated_image, start_point, end_point, (255, 0, 0), 2)  # Linhas azuis

            # Salvar a imagem anotada
            output_filename = os.path.join("output_landmarked_images", f"{letter}_{idx}_landmarked.jpg")
            cv2.imwrite(output_filename, annotated_image)

        else:
            print(f"⚠️ Nenhuma mão detectada na imagem: {image_url}")

        # Remove as imagens temporárias
        os.remove(temp_image_path)

print("🚀 Processamento concluído, landmarks salvos no CSV e imagens salvas com anotações.")