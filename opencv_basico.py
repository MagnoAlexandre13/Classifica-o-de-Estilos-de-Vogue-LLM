"""
Exercícios básicos de OpenCV com vídeo.
Instalação:  pip install opencv-python
Uso:         python opencv_basico.py meu_video.mp4
"""
import sys
import cv2


# ---------------------------------------------------------------
# 1) Abrir o vídeo e ver as informações básicas
# ---------------------------------------------------------------
def info_video(caminho):
    cap = cv2.VideoCapture(caminho)
    if not cap.isOpened():
        raise FileNotFoundError(f"Não consegui abrir o vídeo: {caminho}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()  # sempre libere o vídeo ao terminar

    print(f"Resolução: {largura}x{altura}")
    print(f"FPS: {fps:.2f}")
    print(f"Total de frames: {total_frames}")
    print(f"Duração: {total_frames / fps:.1f} segundos")
    return fps, largura, altura, total_frames


# ---------------------------------------------------------------
# 2) Percorrer os frames um a um (e mostrar na tela)
# ---------------------------------------------------------------
def percorrer_frames(caminho, mostrar=True):
    cap = cv2.VideoCapture(caminho)
    contador = 0

    while True:
        ok, frame = cap.read()  # ok = False quando o vídeo acaba
        if not ok:
            break

        contador += 1

        if mostrar:
            cv2.imshow("Video (aperte q para sair)", frame)
            # waitKey(25) espera 25 ms; ajuste conforme o FPS do vídeo
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Frames lidos: {contador}")


# ---------------------------------------------------------------
# 3) Salvar um frame específico como imagem
# ---------------------------------------------------------------
def salvar_frame(caminho, numero_frame, saida="frame.jpg"):
    cap = cv2.VideoCapture(caminho)
    cap.set(cv2.CAP_PROP_POS_FRAMES, numero_frame)  # pula direto pro frame
    ok, frame = cap.read()
    cap.release()

    if ok:
        cv2.imwrite(saida, frame)
        print(f"Frame {numero_frame} salvo em {saida}")
    else:
        print("Não consegui ler esse frame (número maior que o vídeo?)")


# ---------------------------------------------------------------
# 4) Salvar um trecho do vídeo (de X até Y segundos)
# ---------------------------------------------------------------
def salvar_trecho(caminho, saida, inicio_s, fim_s):
    cap = cv2.VideoCapture(caminho)
    fps = cap.get(cv2.CAP_PROP_FPS)
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_inicio = int(inicio_s * fps)
    frame_fim = int(fim_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)

    codec = cv2.VideoWriter_fourcc(*"mp4v")
    escritor = cv2.VideoWriter(saida, codec, fps, (largura, altura))

    for _ in range(frame_inicio, frame_fim):
        ok, frame = cap.read()
        if not ok:
            break
        escritor.write(frame)

    cap.release()
    escritor.release()
    print(f"Trecho de {inicio_s}s a {fim_s}s salvo em {saida}")


# ---------------------------------------------------------------
# Execução: rode cada etapa e veja o resultado
# ---------------------------------------------------------------
if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "meu_video.mp4"

    info_video(video)
    salvar_frame(video, numero_frame=30, saida="frame_30.jpg")
    salvar_trecho(video, "trecho.mp4", inicio_s=1, fim_s=4)
    percorrer_frames(video, mostrar=True)