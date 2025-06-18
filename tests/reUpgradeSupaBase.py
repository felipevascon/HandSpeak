import os
import psycopg2
from supabase import create_client, Client
from dotenv import load_dotenv
from storage3.exceptions import StorageApiError

load_dotenv()

# Supabase config
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_FOLDER = "A"
BUCKET_URL_BASE = f"{SUPABASE_URL}/storage/v1/object/public/base-photos/{BUCKET_FOLDER}/"

# Postgres config
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "5432")

def connect_db():
    return psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )

def upload_images_and_register(folder_path, letter_id):
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    conn = connect_db()
    cur = conn.cursor()

    # Corrigir sequência do ID
    cur.execute("""
        SELECT setval(
            pg_get_serial_sequence('letter_number_images', 'id'),
            COALESCE((SELECT MAX(id) FROM letter_number_images), 1)
        );
    """)

    # Verificar quantidade atual de imagens para essa letra
    cur.execute("""
        SELECT COUNT(*) FROM letter_number_images
        WHERE letter_number_id = %s
    """, (letter_id,))
    quantidade_atual = cur.fetchone()[0]

    limite_por_letra = 1500

    if quantidade_atual >= limite_por_letra:
        print(f"⚠️ Letra {letter_id} já possui {quantidade_atual} imagens. Upload cancelado.")
        cur.close()
        conn.close()
        return

    for file_name in os.listdir(folder_path):
        if quantidade_atual >= limite_por_letra:
            print(f"✅ Já atingiu o limite de {limite_por_letra} imagens para a letra/número de índice {letter_id}. Parando upload.")
            break

        file_path = os.path.join(folder_path, file_name)

        if not os.path.isfile(file_path):
            continue

        try:
            folder_name = os.path.basename(folder_path)
            new_file_name = f"{folder_name}_{file_name}"
            storage_path = f"{BUCKET_FOLDER}/{new_file_name}"

            with open(file_path, "rb") as f:
                supabase.storage.from_("base-photos").upload(
                    storage_path, f, {"content-type": "image/png"}
                )

        except StorageApiError as e:
            if "Duplicate" in str(e):
                print(f"⚠️ {new_file_name} já existe no storage. Pulando upload.")                
            else:
                print(f"❌ Erro ao enviar {new_file_name}: {e}")
                continue

        image_url = BUCKET_URL_BASE + new_file_name

        try:
            cur.execute("""
                INSERT INTO letter_number_images (letter_number_id, images_url)
                VALUES (%s, %s)
            """, (letter_id, image_url))
            quantidade_atual += 1
            print(f"✅ {file_name} registrada no banco. Total: {quantidade_atual}/{limite_por_letra}")

        except psycopg2.errors.UniqueViolation:
            print(f"⚠️ {file_name} já está no banco. Pulando insert.")
            conn.rollback()
            continue

    conn.commit()
    cur.close()
    conn.close()
    print("🚀 Upload e registro finalizados.")

if __name__ == "__main__":
    pasta = input("Digite o caminho da pasta com as imagens: ")
    letra_id = input("Digite o ID da letra (letters_number_id): ")
    upload_images_and_register(pasta, letra_id)