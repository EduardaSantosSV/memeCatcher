import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# API legada "solutions" do MediaPipe (versão fixada no requirements.txt).
try:
    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
except AttributeError:
    from mediapipe.python.solutions import drawing_styles as mp_styles
    from mediapipe.python.solutions import drawing_utils as mp_draw
    from mediapipe.python.solutions import holistic as mp_holistic


# =============================================================================
# Configuracao
# =============================================================================

PASTA_MEMES = Path(__file__).resolve().parent / "assets" / "memes"

INDICES_CAMERA = (0, 1, 2)
LARGURA_CAMERA = 640
ALTURA_CAMERA = 480

HOLISTIC_COMPLEXIDADE = 1       
HOLISTIC_REFINAR_ROSTO = True     
CONFIANCA_MIN_DETECCAO = 0.7
CONFIANCA_MIN_RASTREIO = 0.7

FRACAO_LARGURA_MEME = 0.5

DEBUG_INICIAL = True
 
FATOR_SILENCIO = 0.6   
FATOR_QUEIXO = 0.7     
FATOR_TESTA = 0.9      
FATOR_REZA = 0.5       

SUAVIZACAO_ALFA = 0.4

TEMPO_MIN_EXIBICAO = 0.4
MARGEM_TROCA = 0.15
TEMPO_TROCA = 0.1

GATE_PULSOS_REZA = 2.5

PISO_REZA = 0.15


# =============================================================================
# Indices dos landmarks do MediaPipe
# =============================================================================

# Rosto  
LABIO_SUP = 13    
LABIO_INF = 14   
BOCA_ESQ = 61     
BOCA_DIR = 291    
QUEIXO = 152      
TESTA = 10       

# Mao 
N_PONTOS_MAO = 21
PULSO = 0
MEIO_MCP = 9     
 
DEDOS = {
    "polegar":   (4, 2),
    "indicador": (8, 6),
    "medio":     (12, 10),
    "anelar":    (16, 14),
    "mindinho":  (20, 18),
}

PONTOS_MAO_BOCA = (DEDOS["indicador"][0],)    
PONTOS_MAO_QUEIXO = (4, 8, 12)               
PONTOS_MAO_TESTA = (8, 12, 16, 20, MEIO_MCP)    
PONTAS_SEM_POLEGAR = (8, 12, 16, 20)

ESTILO_ROSTO = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
ESTILO_PONTOS_MAO = mp_styles.get_default_hand_landmarks_style()
ESTILO_CONEXOES_MAO = mp_styles.get_default_hand_connections_style()


# =============================================================================
# Medidas do frame: landmarks convertidos para pixels uma unica vez
# =============================================================================

@dataclass(frozen=True)
class Medidas: 
    mao_esq: np.ndarray | None
    mao_dir: np.ndarray | None
    rosto: dict[str, np.ndarray] | None
    escala: float | None

    @property
    def maos(self):
        return tuple(m for m in (self.mao_dir, self.mao_esq) if m is not None)


def _em_pixels(landmarks, indices, largura, altura):
    pontos = landmarks.landmark
    return np.array([(pontos[i].x, pontos[i].y) for i in indices]) * (largura, altura)


def medir(results, largura, altura):
    def mao(landmarks):
        if landmarks is None:
            return None
        return _em_pixels(landmarks, range(N_PONTOS_MAO), largura, altura)

    rosto = escala = None
    if results.face_landmarks is not None:
        sup, inf, esq, dir_, queixo, testa = _em_pixels(
            results.face_landmarks,
            (LABIO_SUP, LABIO_INF, BOCA_ESQ, BOCA_DIR, QUEIXO, TESTA),
            largura, altura,
        )
        rosto = {"boca": (sup + inf) / 2, "queixo": queixo, "testa": testa}
        largura_boca = math.dist(esq, dir_)
        escala = largura_boca if largura_boca > 0 else None

    return Medidas(
        mao_esq=mao(results.left_hand_landmarks),
        mao_dir=mao(results.right_hand_landmarks),
        rosto=rosto,
        escala=escala,
    )


# =============================================================================
# Geometria dos gestos
# =============================================================================

def dedo_estendido(mao, dedo): 
    ponta, base = DEDOS[dedo]
    pulso = mao[PULSO]
    return math.dist(mao[ponta], pulso) > math.dist(mao[base], pulso)


def mao_apontando(mao): 
    return dedo_estendido(mao, "indicador") and not dedo_estendido(mao, "medio")


def eh_hangloose(mao):
    return (
        dedo_estendido(mao, "polegar")
        and dedo_estendido(mao, "mindinho")
        and not any(dedo_estendido(mao, d) for d in ("indicador", "medio", "anelar"))
    )


def tamanho_mao(mao):
    return math.dist(mao[PULSO], mao[MEIO_MCP])


def razao_mao_no_rosto(alvo, pontos_mao, filtro_mao=None): 
    indices = list(pontos_mao)

    def razao(medidas):
        if medidas.escala is None:
            return None
        ponto = medidas.rosto[alvo]
        return min(
            (
                float(np.linalg.norm(mao[indices] - ponto, axis=1).min()) / medidas.escala
                for mao in medidas.maos
                if filtro_mao is None or filtro_mao(mao)
            ),
            default=None,
        )

    return razao


def razao_rezando(medidas): 
    esq, dire = medidas.mao_esq, medidas.mao_dir
    if esq is None or dire is None:
        return None
    escala = (tamanho_mao(esq) + tamanho_mao(dire)) / 2
    if escala == 0:
        return None

    if math.dist(esq[PULSO], dire[PULSO]) > GATE_PULSOS_REZA * escala:
        return None

    indices = list(PONTAS_SEM_POLEGAR)
    razao = float(np.linalg.norm(esq[indices] - dire[indices], axis=1).mean()) / escala

    return None if razao < PISO_REZA else razao


# =============================================================================
# Imagens: carregar, redimensionar e colar memes
# =============================================================================

def carregar_meme(caminho):
    meme = cv2.imread(str(caminho), cv2.IMREAD_UNCHANGED)
    if meme is None:
        print(f"Aviso: '{caminho}' não encontrado, o overlay será ignorado.")
        return None
    if meme.ndim == 2:  
        meme = cv2.cvtColor(meme, cv2.COLOR_GRAY2BGR)
    return meme


def redimensionar_largura(imagem, largura):
    altura = int(imagem.shape[0] * largura / imagem.shape[1])
    interpolacao = cv2.INTER_AREA if largura < imagem.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(imagem, (largura, altura), interpolation=interpolacao)


def sobrepor_imagem(frame, imagem, x, y, opacidade=1.0): 
    h, w = imagem.shape[:2]
    fh, fw = frame.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, fw), min(y + h, fh)
    if x1 >= x2 or y1 >= y2:
        return frame   

    recorte = imagem[y1 - y:y2 - y, x1 - x:x2 - x]
    fundo = frame[y1:y2, x1:x2]  

    if recorte.shape[2] == 3 and opacidade >= 1.0:
        fundo[:] = recorte 
        return frame

    if recorte.shape[2] == 4:
        alfa = recorte[:, :, 3:4].astype(np.float32) * (opacidade / 255.0)
    else:
        alfa = np.float32(opacidade)
    fundo[:] = (alfa * recorte[:, :, :3] + (1.0 - alfa) * fundo).astype(np.uint8)
    return frame


# =============================================================================
# Gestos
# =============================================================================

class GestoMeme: 

    def __init__(self, nome, arquivo_meme, prioridade=0):
        self.nome = nome 
        self.prioridade = prioridade
        self._meme = carregar_meme(PASTA_MEMES / arquivo_meme)
        self._meme_na_tela = None   

    def confianca(self, medidas):
        raise NotImplementedError

    def desenhar(self, frame, fracao):
        if self._meme is None:
            return
        largura_alvo = int(frame.shape[1] * fracao)
        if self._meme_na_tela is None or self._meme_na_tela.shape[1] != largura_alvo:
            self._meme_na_tela = redimensionar_largura(self._meme, largura_alvo)
        meme = self._meme_na_tela
        x = (frame.shape[1] - meme.shape[1]) // 2
        y = (frame.shape[0] - meme.shape[0]) // 2
        sobrepor_imagem(frame, meme, x, y)


class GestoProximidade(GestoMeme):

    def __init__(self, nome, arquivo_meme, razao_fn, fator, prioridade=0):
        super().__init__(nome, arquivo_meme, prioridade)
        self._razao_fn = razao_fn
        self._fator = fator
        self._razao_suave = None

    def confianca(self, medidas):
        razao = self._razao_fn(medidas)
        if razao is None:
            self._razao_suave = None   
            return 0.0

        if self._razao_suave is None:
            self._razao_suave = razao
        else:
            self._razao_suave = (
                SUAVIZACAO_ALFA * razao + (1 - SUAVIZACAO_ALFA) * self._razao_suave
            )
        return max(0.0, 1.0 - self._razao_suave / self._fator)


class GestoPose(GestoMeme):

    def __init__(self, nome, arquivo_meme, pose_fn, prioridade=0):
        super().__init__(nome, arquivo_meme, prioridade)
        self._pose_fn = pose_fn

    def confianca(self, medidas):
        return 1.0 if any(self._pose_fn(mao) for mao in medidas.maos) else 0.0


def criar_gestos():
    return [
        GestoProximidade("Rezando", "rezando.jpeg", razao_rezando, FATOR_REZA, prioridade=1),
        GestoProximidade(
            "Silêncio", "naogrita.jpeg",
            razao_mao_no_rosto("boca", PONTOS_MAO_BOCA, filtro_mao=mao_apontando),
            FATOR_SILENCIO,
        ),
        GestoProximidade(
            "Pensativo", "pensativo.jpeg",
            razao_mao_no_rosto("queixo", PONTOS_MAO_QUEIXO), FATOR_QUEIXO,
        ),
        GestoProximidade(
            "Mão na cabeça", "maonacabeca.jpeg",
            razao_mao_no_rosto("testa", PONTOS_MAO_TESTA), FATOR_TESTA,
        ),
        GestoPose("Hang loose", "hangloose.jpeg", eh_hangloose),
    ]


# =============================================================================
# Arbitro: escolhe um unico meme por vez, com estabilidade
# =============================================================================

class Arbitro:

    def __init__(self, gestos):
        self.gestos = gestos
        self.vencedor = None
        self.ultimas_confs = {}       
        self._candidato = None         
        self._candidato_desde = 0.0    
        self._vencedor_visto = 0.0    

    def atualizar(self, medidas, agora):
        confs = {g: g.confianca(medidas) for g in self.gestos}
        self.ultimas_confs = confs

        detectados = [g for g in self.gestos if confs[g] > 0]
        melhor = max(detectados, key=lambda g: (g.prioridade, confs[g]), default=None)

        if melhor is not None and melhor is not self.vencedor and self._pode_desafiar(melhor, confs):
            if melhor is not self._candidato:
                self._candidato = melhor
                self._candidato_desde = agora
            if agora - self._candidato_desde >= TEMPO_TROCA:
                self.vencedor = melhor
                self._vencedor_visto = agora
                self._candidato = None
                print(f"{self.vencedor.nome} detectado!")
        else:
            self._candidato = None

        if self.vencedor is not None:
            if confs[self.vencedor] > 0:
                self._vencedor_visto = agora
            elif agora - self._vencedor_visto >= TEMPO_MIN_EXIBICAO:
                self.vencedor = None

        return self.vencedor

    def _pode_desafiar(self, desafiante, confs): 
        if self.vencedor is None or desafiante.prioridade > self.vencedor.prioridade:
            return True
        return confs[desafiante] >= confs[self.vencedor] + MARGEM_TROCA


# =============================================================================
# Desenho do modo depuracao
# =============================================================================

def _texto(frame, texto, posicao, cor):
    cv2.putText(frame, texto, posicao, cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2, cv2.LINE_AA)


def desenhar_landmarks(frame, results):
    if results.face_landmarks is not None:
        mp_draw.draw_landmarks(
            frame, results.face_landmarks,
            connections=None, landmark_drawing_spec=ESTILO_ROSTO,
        )
    for mao in (results.right_hand_landmarks, results.left_hand_landmarks):
        if mao is not None:
            mp_draw.draw_landmarks(
                frame, mao, mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=ESTILO_PONTOS_MAO,
                connection_drawing_spec=ESTILO_CONEXOES_MAO,
            )


def desenhar_debug(frame, arbitro, medidas, fps): 
    y = 24
    for g in arbitro.gestos:
        venceu = g is arbitro.vencedor
        conf = arbitro.ultimas_confs.get(g, 0.0)
        cor = (0, 255, 0) if venceu else (180, 180, 180)
        _texto(frame, f"{'>' if venceu else ' '} {g.nome}: {conf:.2f}", (10, y), cor)
        y += 26

    esq = "L" if medidas.mao_esq is not None else "-"
    dire = "R" if medidas.mao_dir is not None else "-"
    _texto(frame, f"maos: {esq} {dire}   fps: {fps:.0f}", (10, y + 4), (0, 200, 255))


# =============================================================================
# Programa principal
# =============================================================================

def abrir_camera():
    for indice in INDICES_CAMERA:
        cap = cv2.VideoCapture(indice)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, LARGURA_CAMERA)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA_CAMERA)
            print(f"Câmera iniciada no índice {indice}.")
            return cap
        cap.release()
    return None


def main():
    cap = abrir_camera()
    if cap is None:
        print("Erro: Não foi possível acessar a câmera.")
        return 1

    arbitro = Arbitro(criar_gestos())
    debug = DEBUG_INICIAL
    fps = 0.0
    anterior = time.monotonic()
    print("Pressione 'q' para sair e 'd' para ligar/desligar a depuração.")

    try:
        with mp_holistic.Holistic(
            model_complexity=HOLISTIC_COMPLEXIDADE,
            refine_face_landmarks=HOLISTIC_REFINAR_ROSTO,
            min_detection_confidence=CONFIANCA_MIN_DETECCAO,
            min_tracking_confidence=CONFIANCA_MIN_RASTREIO,
        ) as holistic:
            while True:
                ok, image = cap.read()
                if not ok:
                    print("Erro: a câmera parou de enviar imagens.")
                    break

                image = cv2.flip(image, 1)

                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = holistic.process(image_rgb)

                altura, largura = image.shape[:2]
                medidas = medir(results, largura, altura)
                agora = time.monotonic()
                vencedor = arbitro.atualizar(medidas, agora)

                intervalo = agora - anterior
                anterior = agora
                if intervalo > 0:
                    fps = 1 / intervalo if fps == 0 else 0.9 * fps + 0.1 / intervalo

                if debug:
                    desenhar_landmarks(image, results)
                if vencedor is not None:
                    vencedor.desenhar(image, FRACAO_LARGURA_MEME)
                if debug:
                    desenhar_debug(image, arbitro, medidas, fps)

                cv2.imshow("memeCatcher - Rastreamento", image)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla == ord("q"):
                    break
                if tecla == ord("d"):
                    debug = not debug
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
