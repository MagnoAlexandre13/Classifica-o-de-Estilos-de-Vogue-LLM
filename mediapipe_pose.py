"""
Roda o MediaPipe Pose em um vídeo, desenha o esqueleto e salva as
coordenadas dos 33 pontos do corpo em um CSV (uma linha por frame).

Instalação:  pip install mediapipe opencv-python
Uso:         python mediapipe_pose.py meu_video.mp4

Gera:
  - meu_video_esqueleto.mp4   (vídeo com o esqueleto desenhado)
  - meu_video_pose.csv        (coordenadas por frame)

Na primeira execução, o script baixa o modelo de pose (~6 MB).
"""
import argparse
import csv
import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODELO_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODELO_PATH = "pose_landmarker_lite.task"

# Os 33 pontos do corpo, na ordem em que o MediaPipe os entrega
NOMES_PONTOS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


# ---------------------------------------------------------------
# 1) Baixar o modelo (só na primeira vez) e criar o detector
# ---------------------------------------------------------------
def baixar_modelo():
    if not os.path.exists(MODELO_PATH):
        print("Baixando o modelo de pose...")
        urllib.request.urlretrieve(MODELO_URL, MODELO_PATH)
        print("Modelo salvo em", MODELO_PATH)


def criar_detector():
    baixar_modelo()
    opcoes = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODELO_PATH),
        running_mode=vision.RunningMode.VIDEO,  # modo para vídeos
    )
    return vision.PoseLandmarker.create_from_options(opcoes)


# ---------------------------------------------------------------
# 2) Desenhar o esqueleto sobre o frame
# ---------------------------------------------------------------
def desenhar_esqueleto(frame, pontos):
    """pontos: lista de 33 landmarks com x e y normalizados (0 a 1)."""
    altura, largura = frame.shape[:2]
    coords = [(int(p.x * largura), int(p.y * altura)) for p in pontos]

    for conexao in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        cv2.line(frame, coords[conexao.start], coords[conexao.end],
                 (0, 255, 0), 2)
    for ponto in coords:
        cv2.circle(frame, ponto, 4, (0, 0, 255), -1)
    return frame


# ---------------------------------------------------------------
# 3) Cabeçalho e linhas do CSV
# ---------------------------------------------------------------
def cabecalho_csv():
    colunas = ["frame", "tempo_s", "detectado"]
    for nome in NOMES_PONTOS:
        colunas += [f"{nome}_x", f"{nome}_y", f"{nome}_z", f"{nome}_vis"]
    return colunas


def linha_csv(frame_idx, tempo_s, pontos):
    """Se nenhuma pessoa foi detectada (pontos=None), deixa as células vazias."""
    if pontos is None:
        return [frame_idx, round(tempo_s, 4), 0] + [""] * (len(NOMES_PONTOS) * 4)

    linha = [frame_idx, round(tempo_s, 4), 1]
    for p in pontos:
        linha += [p.x, p.y, p.z, p.visibility]
    return linha


# ---------------------------------------------------------------
# 4) Processar o vídeo inteiro
# ---------------------------------------------------------------
def processar_video(caminho, saida_csv, saida_video=None, detector=None):
    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não consegui abrir o vídeo: {caminho}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    escritor = None
    if saida_video:
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        escritor = cv2.VideoWriter(saida_video, codec, fps, (largura, altura))

    if detector is None:
        detector = criar_detector()

    detectados = 0
    total = 0

    with open(saida_csv, "w", newline="") as arquivo:
        csv_writer = csv.writer(arquivo)
        csv_writer.writerow(cabecalho_csv())

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # O OpenCV lê em BGR; o MediaPipe espera RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imagem_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # O modo VIDEO exige o tempo do frame em milissegundos
            tempo_ms = int(total * 1000 / fps)
            resultado = detector.detect_for_video(imagem_mp, tempo_ms)

            pontos = None
            if resultado.pose_landmarks:  # lista vazia = ninguém detectado
                pontos = resultado.pose_landmarks[0]  # primeira pessoa
                detectados += 1
                frame = desenhar_esqueleto(frame, pontos)

            csv_writer.writerow(linha_csv(total, total / fps, pontos))
            if escritor:
                escritor.write(frame)

            total += 1

    cap.release()
    if escritor:
        escritor.release()
    if hasattr(detector, "close"):
        detector.close()

    print(f"Frames processados: {total}")
    print(f"Frames com pessoa detectada: {detectados}")
    print(f"CSV salvo em: {saida_csv}")
    if saida_video:
        print(f"Vídeo com esqueleto salvo em: {saida_video}")


# ---------------------------------------------------------------
# Execução
# ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="caminho do vídeo de entrada")
    parser.add_argument("--sem-video", action="store_true",
                        help="não gerar o vídeo com esqueleto (mais rápido)")
    args = parser.parse_args()

    base = os.path.splitext(args.video)[0]
    processar_video(
        args.video,
        saida_csv=f"{base}_pose.csv",
        saida_video=None if args.sem_video else f"{base}_esqueleto.mp4",
    )