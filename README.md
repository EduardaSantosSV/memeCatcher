# memeCatcher

O memeCatcher usa a webcam para reconhecer gestos e mostra um meme na tela quando
você faz um deles. O rastreamento de rosto e mãos é feito pelo
[MediaPipe Holistic](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/holistic.md),
e a imagem é exibida com o OpenCV.

## Gestos reconhecidos

| Gesto | Como fazer | Meme |
|---|---|---|
| Rezando | Junte as palmas na frente do corpo | `rezando.jpeg` |
| Silêncio | Encoste o indicador esticado na boca, como quem pede "shhh" | `naogrita.jpeg` |
| Pensativo | Apoie a mão no queixo | `pensativo.jpeg` |
| Mão na cabeça | Ponha a palma na testa | `maonacabeca.jpeg` |
| Hang loose | Estique o polegar e o mindinho e dobre os outros dedos 🤙 | `hangloose.jpeg` |

As imagens `ata.jpeg` e `deboche.jpeg` estão na pasta de memes, mas ainda não têm
gesto associado.

## Instalação

O projeto foi testado com Python 3.12 no Linux.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

O `requirements.txt` fixa o `mediapipe` na versão 0.10.14 porque o código usa a API
legada `mp.solutions`. Não instale o pacote `opencv-python` no mesmo ambiente: o
MediaPipe já traz o `opencv-contrib-python`, e os dois pacotes disputam o mesmo
módulo `cv2`.

## Como usar

```bash
python main.py
```

| Tecla | Ação |
|---|---|
| `q` | Fecha o programa |
| `d` | Liga ou desliga o modo de depuração |

O modo de depuração começa ligado. Ele desenha os pontos do rosto e o esqueleto das
mãos e mostra, no canto da tela:

- a confiança de cada gesto, com o vencedor atual em verde;
- quais mãos o MediaPipe está vendo (`L` e `R`);
- o FPS, útil para comparar ajustes de desempenho.

Desligar o modo de depuração deixa só a imagem da câmera e o meme, e também deixa o
programa um pouco mais rápido.

## Como funciona

A cada frame, o `main.py` faz cinco passos:

1. Lê a imagem da câmera e espelha na horizontal.
2. Passa a imagem pelo MediaPipe Holistic, que encontra rosto e mãos de uma vez.
3. Converte para pixels só os pontos que os gestos usam (função `medir`). Isso
   acontece uma vez por frame, e todos os gestos leem o mesmo resultado.
4. Cada gesto calcula uma **confiança** entre 0 e 1.
5. O **árbitro** escolhe um único gesto vencedor e o meme dele é colado no centro
   da tela.

### Confiança dos gestos

Os gestos de proximidade, como Silêncio, Pensativo e Mão na cabeça, medem a menor
distância entre alguns pontos da mão e um ponto do rosto. Essa distância é dividida
pela largura da boca, para dar o mesmo resultado perto ou longe da câmera. O
resultado é a **razão**.

- A razão passa por uma média móvel exponencial, que suaviza o tremido da detecção.
- A confiança é `1 - razão / fator`. Ela vale 0 quando a mão está no limite de
  ativação e chega perto de 1 quando a mão encosta no rosto.

O gesto Rezando funciona do mesmo jeito, mas compara as pontas dos dedos das duas
mãos entre si e usa o tamanho da mão como escala. Ele é descartado quando os
pulsos estão longe um do outro ou quando as mãos estão coladas demais, o que
costuma indicar a mesma mão detectada duas vezes.

O Hang loose é uma pose de sim ou não: a confiança é 1 se alguma mão faz a pose e
0 se nenhuma faz.

### Árbitro

O árbitro evita que o meme pisque ou fique trocando sem parar.

- Vence o gesto de maior prioridade e, entre gestos de mesma prioridade, o de maior
  confiança. O Rezando tem prioridade maior porque usa as duas mãos.
- Para tomar o lugar do vencedor, o desafiante precisa superar a confiança dele por
  uma margem e manter a liderança por um tempo mínimo.
- Quando o gesto vencedor some, o meme ainda fica na tela por um instante antes de
  desaparecer.

Esses tempos são medidos em segundos, então o comportamento é o mesmo em
computadores rápidos e lentos.

## Configuração

Todas as opções ficam no topo do `main.py`, na seção "Configuração".

| Constante | Padrão | Para que serve |
|---|---|---|
| `INDICES_CAMERA` | `(0, 1, 2)` | Câmeras testadas, em ordem |
| `LARGURA_CAMERA`, `ALTURA_CAMERA` | `640`, `480` | Resolução pedida à câmera |
| `HOLISTIC_COMPLEXIDADE` | `1` | Modelo do Holistic: 0 é mais leve, 2 é mais preciso |
| `HOLISTIC_REFINAR_ROSTO` | `True` | Pontos de lábios e olhos mais precisos, com custo extra |
| `FRACAO_LARGURA_MEME` | `0.5` | Largura do meme em relação à tela |
| `DEBUG_INICIAL` | `True` | Se o modo de depuração começa ligado |
| `FATOR_SILENCIO`, `FATOR_QUEIXO`, `FATOR_TESTA`, `FATOR_REZA` | `0.6`, `0.7`, `0.9`, `0.5` | Distância de ativação de cada gesto. Valor maior dispara de mais longe |
| `SUAVIZACAO_ALFA` | `0.4` | Peso do valor novo na suavização. Valor menor fica mais estável e mais lento |
| `TEMPO_MIN_EXIBICAO` | `0.4` s | Quanto tempo o meme fica depois que o gesto some |
| `MARGEM_TROCA` | `0.15` | Quanto o desafiante precisa superar o vencedor |
| `TEMPO_TROCA` | `0.1` s | Quanto tempo o desafiante precisa liderar para assumir |

## Como adicionar um gesto

1. Coloque a imagem do meme em `assets/memes/`. Prefira nomes sem espaço e sem
   acento.
2. Escreva a função que reconhece o gesto.
3. Adicione uma linha na função `criar_gestos`.

Para uma pose da mão, escreva uma função que recebe os pontos de uma mão e devolve
`True` ou `False`. A função `dedo_estendido` ajuda nisso:

```python
def eh_paz_e_amor(mao):
    """✌️ indicador e médio esticados, anelar e mindinho dobrados."""
    return (
        dedo_estendido(mao, "indicador")
        and dedo_estendido(mao, "medio")
        and not dedo_estendido(mao, "anelar")
        and not dedo_estendido(mao, "mindinho")
    )

# em criar_gestos():
GestoPose("Paz e amor", "pazeamor.jpeg", eh_paz_e_amor),
```

Para um gesto de mão encostando no rosto, use `razao_mao_no_rosto` com o ponto do
rosto (`"boca"`, `"queixo"` ou `"testa"`) e os índices dos pontos da mão:

```python
GestoProximidade(
    "Dedo no queixo", "deboche.jpeg",
    razao_mao_no_rosto("queixo", (8,)), fator=0.5,
),
```

Para usar outro ponto do rosto, como a bochecha, acrescente o índice dele na
função `medir`. O
[mapa de pontos do Face Mesh](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)
mostra o índice de cada ponto.

## Desempenho

O Holistic é a parte mais pesada de cada frame. Para ganhar FPS, ligue o modo de
depuração, observe o número de FPS e teste uma mudança por vez:

- `HOLISTIC_COMPLEXIDADE = 0`;
- `HOLISTIC_REFINAR_ROSTO = False`, que pode deixar o gesto Silêncio menos preciso;
- uma resolução de câmera menor.

Com o modo de depuração desligado, o programa não desenha os pontos do rosto e das
mãos e fica mais rápido.

## Limitações conhecidas

- **Polegar no hang loose.** O polegar é considerado esticado quando a ponta está
  mais longe do pulso do que a junta da base. Um polegar dobrado sobre a palma
  pode passar nesse teste, e o meme pode aparecer com só o mindinho levantado.
- **Suavização por frame.** A média móvel da razão é aplicada a cada frame, então
  ela reage um pouco mais devagar em computadores com FPS baixo.
- **API legada do MediaPipe.** A API `mp.solutions` foi descontinuada. A versão
  fixada no `requirements.txt` funciona, mas uma migração futura deve usar a
  [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe/solutions/guide).

## Estrutura do projeto

```text
memeCatcher/
├── main.py            # todo o programa
├── requirements.txt   # dependências com versões fixadas
└── assets/
    └── memes/         # imagens exibidas para cada gesto
```
