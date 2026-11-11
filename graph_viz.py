"""Visualização interativa de grafos do igraph com Sigma.js + Graphology.

Substitui o Graphistry: gera um único arquivo HTML autocontido (os dados vão embutidos
na página e as bibliotecas JS são carregadas do CDN jsDelivr). O arquivo abre em qualquer
navegador, sem servidor e sem conta em serviço externo, e também pode ser exibido dentro
do notebook (Jupyter, VS Code ou Colab).

Mesmas ferramentas usadas em https://novelo-master.fausel.adv.br/grafo/:
  * Sigma.js 3 – renderização WebGL (dezenas de milhares de arestas sem travar);
  * Graphology – estrutura de dados do grafo no navegador;
  * ForceAtlas2 – layout de forças rodando em um Web Worker.

Uso no notebook:

    import graph_viz
    graph_viz.show(graph, path='camara.html')                  # um grafo

    timelines = {'Geral': {'2023': g1, '2024': g2}, 'Saúde': {'2023': g3, '2024': g4}}
    graph_viz.show(timelines, path='camara.html')              # linhas do tempo

    escalas = {'Ano': {'Geral': {...}, 'Saúde': {...}}, 'Legislatura': {'Geral': {...}, ...}}
    graph_viz.show(escalas, path='camara.html')                # várias escalas de tempo

Com linhas do tempo, cada grafo é um "quadro" (ex.: tema × período). Os vértices são
identificados entre quadros pelo atributo ``name``; ao trocar de quadro, o ForceAtlas2
continua a partir das posições atuais, então o grafo se transforma de forma contínua.
"""

from __future__ import annotations

import base64
import gzip
import html
import io
import json
import math
import random
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

# Versões fixas das bibliotecas JS (carregadas de https://cdn.jsdelivr.net/npm/).
JS_LIBS = {
    "sigma": "3.0.2",
    "graphology": "0.26.0",
    "fa2": "0.10.1",  # graphology-layout-forceatlas2
}

# Parâmetros do ForceAtlas2 (graphology-layout-forceatlas2) ajustáveis em `fa2_settings`
# e ao vivo no painel "⚙ Parâmetros" da página.
FA2_KEYS = (
    "gravity", "scalingRatio", "slowDown", "edgeWeightInfluence", "barnesHutTheta",
    "strongGravityMode", "linLogMode", "outboundAttractionDistribution", "adjustSizes", "barnesHutOptimize",
)

DEFAULT_COLOR_ATTRS = {"Comunidade": "Comunidade", "Partido": "Partido", "UF": "UF"}
DEFAULT_SIZE_ATTRS = {
    "degree": "Grau (nº de conexões)",
    "betweenness": "Intermediação (betweenness)",
    "closeness": "Proximidade (closeness)",
}

_NONE = 65535                 # código de "sem valor" nos atributos categóricos (uint16)
# Votos: 0 = não votou; 1.. = rótulos abaixo (qualquer outro valor vira "Outro")
_VOTE_LABELS = ("Sim", "Não", "Abstenção", "Obstrução", "Artigo 17", "Outro")
_MAX_LAYOUT_EDGES = 150_000   # arestas usadas no layout inicial da união dos quadros
# Valor das arestas quantizado em 10 bits (passo de ~0,1% da faixa de valores, abaixo das 3 casas
# mostradas na página), guardado em dois planos de bytes (altos e baixos), que o gzip comprime bem.
_Q_MAX = 1023


def _clean(value):
    """Converte valores do igraph/numpy/pandas para algo serializável em JSON."""
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()  # numpy -> python
        except (ValueError, TypeError):
            pass
    try:
        if value != value:  # NaN, NaT
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    text = str(value).strip()
    return text or None


def _number(value):
    value = _clean(value)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else math.nan


def _floats(values):
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return np.array([_number(v) for v in values], dtype=float)


def _as_dict(spec, default, available):
    """Aceita dict {atributo: rótulo}, lista de atributos ou None (usa o padrão)."""
    if spec is None:
        spec = default
    if not isinstance(spec, dict):
        spec = {attr: attr for attr in spec}
    return {k: v for k, v in spec.items() if k in available}


def _is_graph(obj):
    return hasattr(obj, "vs") and hasattr(obj, "es")


def _timelines(graphs):
    """Normaliza as entradas aceitas em ``[(escala, linha, [(quadro, Graph)])]``.

    igraph.Graph | {linha: {quadro: Graph}} | {linha: [(quadro, Graph), ...]}
    | {escala: {linha: {quadro: Graph}}} (escala = ``""`` nas formas sem escala).
    """
    if _is_graph(graphs):
        return [("", "", [("", graphs)])]

    def lines(mapping, scale):
        out = []
        for name, frames in mapping.items():
            items = frames.items() if isinstance(frames, dict) else frames
            items = [(str(label), g) for label, g in items if g is not None]
            if items:
                out.append((scale, str(name), items))
        return out

    nested = [v for v in graphs.values() if isinstance(v, dict) and v]
    three_levels = nested and all(isinstance(f, dict) for v in nested for f in v.values())
    if three_levels:
        timelines = [t for scale, mapping in graphs.items() for t in lines(mapping, str(scale))]
    else:
        timelines = lines(graphs, "")
    if not timelines:
        raise ValueError("Nenhum grafo para visualizar.")
    return timelines


def _iso_day(value):
    """Data (str ISO, date, datetime, pandas.Timestamp) -> 'AAAA-MM-DD', ou None."""
    value = _clean(value)
    if not isinstance(value, str):
        return None
    match = re.match(r"\d{4}-\d{2}-\d{2}", value)
    return match.group(0) if match else None


def align_labels(frames, attr="community", new_attr="Comunidade", label="Comunidade {}"):
    """Dá rótulos estáveis a grupos arbitrários (ex.: comunidades do Leiden) ao longo do tempo.

    ``frames`` é um dict {período: Graph} (ou lista de Graph) em ordem temporal. No primeiro
    quadro os grupos são numerados por tamanho (1 = maior). Nos seguintes, cada grupo herda o
    rótulo do grupo do quadro anterior com que compartilha mais vértices (pelo ``name``), em
    ordem decrescente de sobreposição, desde que o número caiba no quadro (um quadro com k grupos
    usa exatamente 1..k); os que sobram recebem os menores números livres, por tamanho. O
    resultado vai para o atributo ``new_attr`` de cada grafo, para a cor de um grupo se manter
    entre quadros.
    Vértices sem grupo (None/NaN, ex.: deputados sem nenhuma aresta) ficam sem rótulo, e um quadro
    sem nenhum grupo não interrompe a sequência (o seguinte é comparado ao último com grupos).
    """
    graphs = list(frames.values()) if isinstance(frames, dict) else list(frames)
    previous = {}
    for g in graphs:
        groups = {}
        if attr in g.vs.attributes():
            for name, c in zip(g.vs["name"], g.vs[attr]):
                if _clean(c) is not None:
                    groups.setdefault(c, set()).add(name)
        pairs = sorted(((len(members & old), c, k) for c, members in groups.items()
                        for k, old in previous.items() if members & old), key=lambda p: -p[0])
        mapping, used = {}, set()
        for _, c, k in pairs:
            if c not in mapping and k not in used and k <= len(groups):
                mapping[c] = k
                used.add(k)
        free = (k for k in range(1, len(groups) + 1) if k not in used)
        for c in sorted(groups, key=lambda c: -len(groups[c])):
            if c not in mapping:
                mapping[c] = next(free)
        g.vs[new_attr] = [label.format(mapping[c]) if c in mapping else None
                          for c in (g.vs[attr] if attr in g.vs.attributes() else [None] * g.vcount())]
        if groups:
            previous = {mapping[c]: members for c, members in groups.items()}
    return frames


# ---------------------------------------------------------------------------
# Cores categóricas fixas (ex.: uma por partido, tiradas dos logos)
# ---------------------------------------------------------------------------
_UA = "vote-network/1.0 (+https://github.com/LucasAlegre/vote-network)"
_LIGHT_L = (0.32, 0.80)  # faixa de claridade (OKLCH) das cores no tema claro
_DARK_L = (0.52, 0.92)   # ... e no tema escuro


def _srgb_to_oklab(rgb):
    rgb = np.asarray(rgb, dtype=float)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    lms = np.cbrt(np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                            [0.2119034982, 0.6806995451, 0.1073969566],
                            [0.0883024619, 0.2817188376, 0.6299787005]]) @ lin)
    return np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                     [1.9779984951, -2.4285922050, 0.4505937099],
                     [0.0259040371, 0.7827717662, -0.8086757660]]) @ lms


def _oklab_to_linear(lab):
    lms = np.array([[1, 0.3963377774, 0.2158037573],
                    [1, -0.1055613458, -0.0638541728],
                    [1, -0.0894841775, -1.2914855480]]) @ np.asarray(lab, dtype=float)
    return np.array([[4.0767416621, -3.3077115913, 0.2309699292],
                     [-1.2684380046, 2.6097574011, -0.3413193965],
                     [-0.0041960863, -0.7034186147, 1.7076147010]]) @ lms ** 3


def _hex_to_lab(color):
    h = color.lstrip("#")
    return _srgb_to_oklab([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def _lab_to_hex(lab):
    lin = np.clip(_oklab_to_linear(lab), 0, 1)
    rgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in rgb)


def _lch_to_lab(L, C, H):
    """OKLCH -> OKLab, reduzindo a saturação (busca binária) até a cor caber no sRGB."""
    lab = lambda c: np.array([L, c * math.cos(H), c * math.sin(H)])  # noqa: E731

    def inside(c):
        lin = _oklab_to_linear(lab(c))
        return lin.min() >= -1e-4 and lin.max() <= 1 + 1e-4

    if inside(C):
        return lab(C)
    lo, hi = 0.0, C
    for _ in range(25):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if inside(mid) else (lo, mid)
    return lab(lo)


def _with_lightness(color, band):
    """Mesma cor (tom e saturação) com a claridade levada para dentro de `band`."""
    L, a, b = _hex_to_lab(color)
    return _lab_to_hex(_lch_to_lab(min(max(L, band[0]), band[1]), math.hypot(a, b), math.atan2(b, a)))


def _dark_variant(color):
    """Versão de uma cor avulsa para fundo escuro: só as cores escuras clareiam; as demais ficam iguais."""
    L, a, b = _hex_to_lab(color)
    L2 = L if L >= 0.58 else 0.58 - (0.58 - L) * 0.45
    return _lab_to_hex(_lch_to_lab(L2, math.hypot(a, b), math.atan2(b, a)))


def _theme_pair(color):
    """'#cor' ou (claro, escuro) -> [claro, escuro]."""
    if isinstance(color, (tuple, list)):
        return [str(color[0]), str(color[1])]
    return [color, _dark_variant(color)]


def _raster_url(url):
    """SVG da Wikimedia -> miniatura PNG (o Pillow não lê SVG)."""
    m = re.match(r"(https://upload\.wikimedia\.org/wikipedia/[^/]+)/(\w)/(\w\w)/([^/?#]+\.svg)$", url)
    return f"{m[1]}/thumb/{m[2]}/{m[3]}/{m[4]}/250px-{m[4]}.png" if m else url


def _dominant_colors(img, max_colors=3):
    """Cores dominantes de uma imagem: pixels opacos e saturados agrupados por tom (24 faixas)."""
    img = img.convert("RGBA")
    img.thumbnail((128, 128))
    px = np.asarray(img, dtype=float).reshape(-1, 4) / 255
    rgb, alpha = px[:, :3], px[:, 3]
    mx, mn = rgb.max(axis=1), rgb.min(axis=1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-9), 0)
    keep = (alpha > 0.5) & (sat > 0.35) & (mx > 0.2)  # sem branco, preto, cinza e transparência
    if keep.sum() < 12:
        return []
    r, g, b = rgb[keep].T
    d = (mx - mn)[keep] + 1e-12
    hue = np.where(mx[keep] == r, ((g - b) / d) % 6, np.where(mx[keep] == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    bins = (hue * 24).astype(int) % 24
    hist = np.bincount(bins, weights=sat[keep], minlength=24)
    colors, used = [], []
    for k in np.argsort(-hist):
        if len(colors) >= max_colors or hist[k] < 0.15 * hist.max():
            break
        if any(min(abs(k - u), 24 - abs(k - u)) <= 1 for u in used):
            continue  # tom vizinho de um já escolhido
        med = np.median(rgb[keep][bins == k], axis=0)
        colors.append("#%02x%02x%02x" % tuple(int(round(v * 255)) for v in med))
        used.append(k)
    return colors


def logo_palette(urls, cache=None, max_colors=3, timeout=30):
    """Baixa cada imagem de ``{chave: url}`` e devolve ``{chave: [cores dominantes]}``.

    As cores vêm em ordem de importância (a primeira é a principal). ``cache`` é um arquivo
    JSON onde o resultado de cada URL fica guardado para as próximas execuções. Imagens que
    não puderem ser baixadas ou lidas ficam com lista vazia. Requer o Pillow.
    """
    from PIL import Image

    cache = Path(cache) if cache else None
    known = json.loads(cache.read_text(encoding="utf-8")) if cache and cache.exists() else {}
    result = {}
    for key, url in urls.items():
        if not isinstance(url, str) or not url:
            result[key] = []
            continue
        if url not in known:
            try:
                req = urllib.request.Request(urllib.parse.quote(_raster_url(url), safe=":/%?=&+"),
                                             headers={"User-Agent": _UA})
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    known[url] = _dominant_colors(Image.open(io.BytesIO(response.read())), max_colors)
            except Exception as err:  # noqa: BLE001 — logo quebrado não deve parar o notebook
                print(f"Sem cor do logo de {key}: {err}")
                result[key] = []
                continue
        result[key] = known[url]
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


def _color_candidates(colors, secondary_penalty, band):
    """Variações (claridade, saturação, tom) de cada cor preferida, ordenadas pelo custo de usá-las.

    O custo pesa pouco a claridade (um tom mais claro/escuro ainda lembra o logo), um pouco mais a
    saturação e bastante o tom (azul que vira violeta deixa de lembrar o partido).
    """
    costs, labs = [], []
    for i, color in enumerate(colors):
        base = _hex_to_lab(_with_lightness(color, band))
        L, C, H = base[0], math.hypot(base[1], base[2]), math.atan2(base[2], base[1])
        for dl in np.arange(-0.2, 0.2001, 0.04):
            if not band[0] <= L + dl <= band[1]:
                continue
            for fc in (1, 0.8, 1.25, 0.6):
                for dh in (0, -6, 6, -12, 12, -20, 20):
                    lab = _lch_to_lab(L + dl, C * fc, H + math.radians(dh))
                    c2, h2 = math.hypot(lab[1], lab[2]), math.atan2(lab[2], lab[1])
                    dhue = math.remainder(h2 - H, 2 * math.pi) * min(C, c2)  # desvio de tom (comprimento do arco)
                    cost = math.sqrt((0.5 * (lab[0] - L)) ** 2 + (0.7 * (c2 - C)) ** 2 + (1.5 * dhue) ** 2)
                    costs.append(cost + secondary_penalty * i)
                    labs.append(lab)
    order = np.argsort(costs, kind="stable")
    return np.asarray(costs)[order], np.asarray(labs).reshape(-1, 3)[order]


def _assign_colors(preferred, weights, band, neutral, neutral_color, min_delta, secondary_penalty, rounds):
    """Núcleo de ``distinct_colors`` para uma faixa de claridade: devolve {chave: cor OKLab}."""
    keys = list(preferred)
    order = sorted(keys, key=lambda k: (-weights.get(k, 0), str(k)))
    w = {k: math.sqrt(max(weights.get(k, 0), 0)) + 1 for k in keys}
    lab_of = {k: _hex_to_lab(neutral_color) for k in keys if k in neutral}
    fixed = set(lab_of)
    grid = np.array([_lch_to_lab(L, C, math.radians(h))
                     for L in np.linspace(band[0] + 0.08, band[1] - 0.08, 3) for C in (0.1, 0.16) for h in range(0, 360, 15)])
    cand = {k: (_color_candidates(preferred[k], secondary_penalty, band) if preferred.get(k)
                else (np.full(len(grid), 1.0), grid)) for k in keys if k not in fixed}

    def min_dist(labs, exclude=()):
        others = [lab for j, lab in lab_of.items() if j not in exclude]
        if not others:
            return np.full(len(labs), np.inf)
        return np.linalg.norm(labs[:, None, :] - np.asarray(others)[None, :, :], axis=2).min(axis=1)

    cost_of = {}
    for k in order:  # escolha gulosa (relaxa o limite se nada couber; o reparo tenta corrigir depois)
        if k in fixed:
            continue
        costs, labs = cand[k]
        d = min_dist(labs)
        if not preferred.get(k):
            i = int(np.argmax(d))
        else:
            i = next((int(np.argmax(d >= t)) for t in (min_delta, 0.8 * min_delta, 0.6 * min_delta, 0.45 * min_delta)
                      if (d >= t).any()), int(np.argmax(d)))
        lab_of[k], cost_of[k] = labs[i], costs[i]

    def repair():  # quem está perto demais de outra cor troca para a candidata viável mais barata
        changed = False
        for k in order:
            if k in fixed or min_dist(lab_of[k][None, :], exclude={k})[0] >= min_delta:
                continue
            costs, labs = cand[k]
            ok = min_dist(labs, exclude={k}) >= min_delta
            if ok.any():
                i = int(np.argmax(ok))
                lab_of[k], cost_of[k] = labs[i], costs[i]
                changed = True
        return changed

    repair()
    for _ in range(rounds):  # busca local
        improved = repair()
        for k in order:
            if k in fixed or not preferred.get(k):
                continue
            costs, labs = cand[k]
            better = np.flatnonzero(costs < cost_of[k] - 1e-9)
            if not len(better):
                continue
            ok = min_dist(labs[better], exclude={k}) >= min_delta
            if ok.any():  # cor melhor e livre
                i = better[int(np.argmax(ok))]
                lab_of[k], cost_of[k] = labs[i], costs[i]
                improved = True
                continue
            for i in better[:80]:  # cor melhor que conflita com uma única outra chave
                conflicts = [j for j, lab in lab_of.items() if j != k and np.linalg.norm(labs[i] - lab) < min_delta]
                if len(conflicts) != 1 or conflicts[0] in fixed:
                    continue
                j = conflicts[0]
                cj, lj = cand[j]
                others = np.asarray([lab for m, lab in lab_of.items() if m not in (j, k)] + [labs[i]])
                okj = np.linalg.norm(lj[:, None, :] - others[None, :, :], axis=2).min(axis=1) >= min_delta
                if not okj.any():
                    continue
                a = int(np.argmax(okj))
                if w[k] * (costs[i] - cost_of[k]) + w[j] * (cj[a] - cost_of[j]) < -1e-9:
                    lab_of[k], cost_of[k] = labs[i], costs[i]
                    lab_of[j], cost_of[j] = lj[a], cj[a]
                    improved = True
                    break
        if not improved:
            break
    return lab_of


def distinct_colors(preferred, weights=None, neutral=(), neutral_color="#8c8a85", min_delta=0.09,
                    secondary_penalty=0.1, rounds=30, neutral_dark="#6f6e68"):
    """Escolhe uma cor diferente para cada chave, o mais perto possível das cores preferidas.

    ``preferred`` é ``{chave: [cor principal, secundárias...]}`` (ex.: saída de ``logo_palette``).
    Cada chave pode usar as próprias cores ou variações próximas delas (claridade, saturação e
    um pouco de tom); usar uma cor secundária custa ``secondary_penalty`` a mais. Todas as cores
    ficam a pelo menos ``min_delta`` umas das outras (distância no espaço OKLab).

    Começa por uma escolha gulosa (chaves com mais ``weights`` primeiro) e depois faz uma busca
    local que minimiza o custo total ponderado por √peso: uma chave pode trocar para uma cor mais
    perto do logo, inclusive empurrando uma única outra chave para a melhor alternativa dela,
    se o total diminuir. Chaves sem cor preferida recebem a cor mais distante das demais; as de
    ``neutral`` (ex.: "sem partido") ficam cinza.

    Devolve ``{chave: (cor no tema claro, cor no tema escuro)}``. A cor do tema escuro parte da
    do tema claro (mesma cor do logo) e é escolhida do mesmo jeito na faixa de claridade do escuro.
    """
    weights = weights or {}
    light = {k: _lab_to_hex(v) for k, v in _assign_colors(
        preferred, weights, _LIGHT_L, neutral, neutral_color, min_delta, secondary_penalty, rounds).items()}
    dark = _assign_colors({k: [light[k]] for k in preferred}, weights, _DARK_L,
                          neutral, neutral_dark, min_delta, secondary_penalty, rounds)
    return {k: (neutral_color, neutral_dark) if k in neutral else (light[k], _lab_to_hex(dark[k])) for k in preferred}


def color_table(colors, images=None):
    """Tabela HTML (para o notebook) com cada cor nos temas claro e escuro e, se houver, a imagem."""
    from IPython.display import HTML

    def swatch(c, bg):
        return (f'<span style="display:inline-block;padding:3px;background:{bg};border-radius:6px">'
                f'<span style="display:block;width:36px;height:14px;border-radius:4px;background:{c}"></span></span>')

    rows = []
    for key, color in colors.items():
        light, dark = _theme_pair(color)
        img = (images or {}).get(key)
        logo = f'<img src="{html.escape(img)}" style="height:22px;max-width:90px" referrerpolicy="no-referrer">' if img else ""
        rows.append(f"<tr><td>{html.escape(str(key))}</td><td>{swatch(light, '#fcfcfb')}</td>"
                    f"<td>{swatch(dark, '#1a1a19')}</td><td><code>{light}</code> <code>{dark}</code></td><td>{logo}</td></tr>")
    return HTML('<table style="border-collapse:collapse"><tr><th>Chave</th><th>Claro</th><th>Escuro</th>'
                "<th>Cores</th><th>Imagem</th></tr>" + "".join(rows) + "</table>")


def graph_to_html(
    graphs,
    title: str = "Câmara dos Deputados",
    subtitle: str = "",
    label_attr: str = "name",
    color_attrs=None,
    size_attrs=None,
    info_attrs=None,
    sublabel_attrs=("Partido", "UF"),
    edge_attr: str = "similarity",
    edge_label: str = "Similaridade mínima",
    edge_flag_attr: str | None = "significativa",
    edge_flag_label: str = "Só signif.",
    edge_flag_hint: str = "Mostrar só as arestas significativas (as demais aparecem mais apagadas)",
    photo_attr: str | None = "Foto",
    link_attr: str | None = "Página",
    link_text: str = "Página na Câmara",
    logo_attr: str | None = "Partido URI",
    logos: dict | None = None,
    logo_for: str = "Partido",
    colors: dict | None = None,
    node_noun: str = "deputados",
    edge_noun: str = "conexões",
    timeline_label: str = "Tema",
    scale_label: str = "Escala",
    period_attrs=("inicio", "fim"),
    layout=None,
    layout_seconds: float = 4,
    transition_seconds: float = 3,
    play_seconds: float = 4,
    fa2_settings: dict | None = None,
    votacoes=None,
    votos=None,
    votacoes_attr: str = "ids_votacoes",
    seed: int = 42,
) -> str:
    """Gera a página HTML interativa para um grafo ou para linhas do tempo de grafos.

    graphs : igraph.Graph | dict
        Um grafo não direcionado, ``{linha: {quadro: Graph}}`` (ex.: ``{tema: {período: grafo}}``)
        ou ``{escala: {linha: {quadro: Graph}}}`` (ex.: ``{'Ano': {...}, 'Legislatura': {...}}``),
        com um seletor de escala na linha do tempo.
        Cada quadro é um grafo completo; os vértices são ligados entre quadros por ``label_attr``.
        Atributos de grafo (ex.: ``g['votações'] = 120``) aparecem como informação do quadro.
    period_attrs : (str, str)
        Atributos de grafo com a data de início e de fim (inclusive) de cada quadro (ex.:
        ``g['inicio'] = '2023-02-01'``, ``g['fim'] = '2024-01-31'``). Se existirem, ao trocar de
        tema ou de escala a página vai para o quadro que mais se sobrepõe ao atual no tempo.
    color_attrs : dict {atributo: rótulo}
        Atributos categóricos dos vértices que podem colorir os nós (o primeiro é o padrão).
    size_attrs : dict {atributo: rótulo}
        Atributos numéricos que podem definir o tamanho dos nós. ``"degree"`` é calculado.
    info_attrs : dict {atributo: rótulo}
        Campos exibidos no painel ao clicar em um nó (padrão: cores + tamanhos).
    edge_flag_attr, edge_flag_label, edge_flag_hint : str
        Atributo booleano opcional das arestas (ex.: ``significativa``). Se existir, as arestas com
        valor falso aparecem mais apagadas e um botão (``edge_flag_label``) permite escondê-las.
    edge_attr : str
        Atributo numérico das arestas mostrado na interface e usado no filtro. Arestas
        paralelas (mesmo par de vértices) são unidas mantendo o maior valor.
    photo_attr, link_attr : str
        Atributos com a URL da foto e da página externa de cada vértice.
    logos, logo_attr, logo_for : dict, str, str
        ``logos[vértice[logo_attr]]`` é a URL do logo exibido ao lado do campo ``logo_for``.
    colors : dict {atributo: {valor: cor}}
        Cores fixas para atributos de cor: ``{'Partido': {'PT': '#e0262a', ...}}`` ou pares
        ``(tema claro, tema escuro)`` como os de ``distinct_colors``. Nesses atributos todo valor tem
        a própria cor (sem o grupo "Outros"); para uma cor avulsa, a do tema escuro é calculada.
    layout : igraph.Layout | lista de (x, y) | None
        Posições iniciais (só para um único grafo). Se None, usa Fruchterman-Reingold
        ponderado e determinístico sobre a união de todos os quadros.
    layout_seconds : float
        Por quantos segundos o ForceAtlas2 roda ao abrir a página (0 = não roda).
    transition_seconds : float
        Por quantos segundos o ForceAtlas2 roda depois de trocar de quadro (0 = não roda).
    play_seconds : float
        Intervalo entre quadros no botão "▶ Animar".
    fa2_settings : dict
        Valores iniciais dos parâmetros do ForceAtlas2 (chaves em ``FA2_KEYS``), por exemplo
        ``{'gravity': 1, 'linLogMode': True}``. Na página, o botão "⚙ Parâmetros" permite
        ajustá-los com o layout rodando e copiar o dicionário resultante para cá.
    votacoes, votos : pandas.DataFrame
        Para a lista de votações da página (botão "⚖ Votações"). ``votacoes`` tem uma linha por
        votação, com a coluna ``id`` e, se houver, ``data``, ``titulo``, ``ementa``, ``descricao``,
        ``orgao``, ``aprovacao``, ``sim``, ``nao``, ``outros``, ``temas`` (lista) e ``url``.
        ``votos`` tem as colunas (id da votação, ``label_attr`` do deputado, voto), nessa ordem.
        As votações de cada quadro vêm do atributo de grafo ``votacoes_attr`` (lista de ids).
    """
    fa2 = {k: _clean(v) for k, v in (fa2_settings or {}).items()}
    unknown = sorted(set(fa2) - set(FA2_KEYS))
    if unknown:
        raise ValueError(f"fa2_settings: parâmetros desconhecidos {unknown}; válidos: {list(FA2_KEYS)}")

    timelines = _timelines(graphs)
    frames = [(label, g) for _, _, items in timelines for label, g in items]
    scales = list(dict.fromkeys(scale for scale, _, _ in timelines))
    single = len(frames) == 1
    real_vattrs = set().union(*(g.vs.attributes() for _, g in frames))
    vattrs = real_vattrs | {"degree"}

    color_attrs = _as_dict(color_attrs, DEFAULT_COLOR_ATTRS, vattrs)
    size_attrs = _as_dict(size_attrs, DEFAULT_SIZE_ATTRS, vattrs)
    info_attrs = _as_dict(info_attrs, {**color_attrs, **size_attrs}, vattrs)
    sublabel_attrs = [a for a in (sublabel_attrs or []) if a in vattrs]
    photo_attr = photo_attr if photo_attr in real_vattrs else None
    link_attr = link_attr if link_attr in real_vattrs else None
    logo_attr = logo_attr if (logos and logo_attr in real_vattrs) else None
    if label_attr not in real_vattrs and not single:
        raise ValueError(f"Os grafos precisam do atributo de vértice '{label_attr}' para ligar os quadros.")

    # --- Vértices: união dos quadros, identificados por `label_attr` ---
    index, names, frame_nodes = {}, [], []
    for label, g in frames:
        keys = ([str(_clean(v)) for v in g.vs[label_attr]] if label_attr in g.vs.attributes()
                else [str(i) for i in range(g.vcount())])
        if len(set(keys)) != len(keys):
            raise ValueError(f"Valores repetidos de '{label_attr}' no quadro '{label}'.")
        for k in keys:
            if k not in index:
                index[k] = len(names)
                names.append(k)
        frame_nodes.append(np.array([index[k] for k in keys], dtype=np.int64))
    n_nodes = len(names)
    if n_nodes >= _NONE:
        raise ValueError("Grafos com mais de 65.534 vértices não são suportados.")

    static = [{} for _ in range(n_nodes)]  # foto e link: último valor conhecido
    for (_, g), gi in zip(frames, frame_nodes):
        for attr in (photo_attr, link_attr):
            if attr and attr in g.vs.attributes():
                for i, v in zip(gi, g.vs[attr]):
                    v = _clean(v)
                    if v is not None:
                        static[i][attr] = v

    # Atributos que variam por quadro (partido, comunidade, métricas...)
    frame_attrs = [a for a in dict.fromkeys([*color_attrs, *size_attrs, *info_attrs, *sublabel_attrs, logo_attr])
                   if a and a in real_vattrs and a not in (photo_attr, link_attr)]
    kinds = {}
    for a in frame_attrs:
        values = [_clean(v) for _, g in frames if a in g.vs.attributes() for v in g.vs[a]]
        values = [v for v in values if v is not None]
        numeric = values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
        categorical = a in color_attrs or a in sublabel_attrs or a == logo_attr
        kinds[a] = "num" if numeric and not categorical else "cat"

    # --- Arestas: une paralelas (maior valor), remove laços ---
    frame_edges, lo, hi = [], math.inf, -math.inf
    has_flags = bool(edge_flag_attr) and any(edge_flag_attr in g.es.attributes() for _, g in frames)
    for (_, g), gi in zip(frames, frame_nodes):
        el = np.asarray(g.get_edgelist(), dtype=np.int64).reshape(-1, 2)
        if edge_attr in g.es.attributes():
            vals = _floats(g.es[edge_attr])
        elif "weight" in g.es.attributes():
            vals = _floats(g.es["weight"])
        else:
            vals = np.ones(g.ecount())
        s, t = gi[el[:, 0]], gi[el[:, 1]]
        a, b = np.minimum(s, t), np.maximum(s, t)
        ok = (a != b) & np.isfinite(vals)
        keys, inv = np.unique(a[ok] * n_nodes + b[ok], return_inverse=True)
        best = np.full(len(keys), -np.inf)
        np.maximum.at(best, inv, vals[ok])
        flags = None
        if has_flags:  # aresta sem o atributo conta como marcada; paralelas: basta uma marcada
            raw = g.es[edge_flag_attr] if edge_flag_attr in g.es.attributes() else [True] * g.ecount()
            marked = np.array([bool(_clean(v)) for v in raw], dtype=np.uint8)
            flags = np.zeros(len(keys), dtype=np.uint8)
            np.maximum.at(flags, inv, marked[ok])
        frame_edges.append((keys, best, flags))
        if len(best):
            lo, hi = min(lo, best.min()), max(hi, best.max())
    if not math.isfinite(lo):
        lo, hi = 0.0, 1.0
    span = (hi - lo) or 1.0

    # --- Layout inicial ("posições iniciais") ---
    if layout is not None:
        if not single:
            raise ValueError("`layout` só pode ser usado com um único grafo.")
        coords = [tuple(c[:2]) for c in (layout.coords if hasattr(layout, "coords") else layout)]
    else:
        keys = np.concatenate([k for k, _, _ in frame_edges])
        weights = np.concatenate([(v - lo) / span for _, v, _ in frame_edges])
        keys, inv = np.unique(keys, return_inverse=True)
        best = np.zeros(len(keys))
        np.maximum.at(best, inv, weights)
        if len(keys) > _MAX_LAYOUT_EDGES:
            top = np.argsort(-best, kind="stable")[:_MAX_LAYOUT_EDGES]
            keys, best = keys[top], best[top]
        union = frames[0][1].__class__(n=n_nodes, edges=np.column_stack([keys // n_nodes, keys % n_nodes]).tolist())
        if union.ecount():
            rng_state = random.getstate()  # o igraph usa o `random` do Python
            random.seed(seed)
            lay = union.layout_fruchterman_reingold(weights=(0.05 + 0.95 * best).tolist(), niter=1000 if single else 500)
            random.setstate(rng_state)
        else:
            lay = union.layout_circle()
        coords = lay.coords

    nodes = [{"l": names[i], "x": round(float(coords[i][0]), 3), "y": round(float(coords[i][1]), 3), "a": static[i]}
             for i in range(n_nodes)]

    # --- Quadros: arrays binários (uint16/float32) num único blob gzip ---
    buf = bytearray()

    def put(arr):
        raw = arr.tobytes()
        offset = len(buf)
        buf.extend(raw)
        buf.extend(b"\0" * (-len(raw) % 4))  # alinhamento para os typed arrays do JS
        return [offset, int(arr.size)]

    codes = {a: {} for a in frame_attrs if kinds[a] == "cat"}
    frames_meta = []
    for (label, g), gi, (keys, vals, flags) in zip(frames, frame_nodes, frame_edges):
        refs = {}
        for a in frame_attrs:
            column = g.vs[a] if a in g.vs.attributes() else [None] * g.vcount()
            if kinds[a] == "num":
                refs[a] = put(np.array([_number(v) for v in column], dtype="<f4"))
            else:
                arr = []
                for v in column:
                    v = _clean(v)
                    arr.append(_NONE if v is None else codes[a].setdefault(str(v), len(codes[a])))
                refs[a] = put(np.array(arr, dtype="<u2"))
        s, t = keys // n_nodes, keys % n_nodes  # já ordenado por (s, t)
        q = np.round((vals - lo) / span * _Q_MAX).astype(np.int64)
        attrs = g.attributes()
        info = {k: _clean(g[k]) for k in attrs if k not in ("name", votacoes_attr, *period_attrs)}
        start, end = (_iso_day(g[a]) if a in attrs else None for a in period_attrs)
        frames_meta.append({
            "label": label,
            **({"t0": start, "t1": end} if start and end else {}),
            "info": {k: v for k, v in info.items() if v is not None},
            "n": put(gi.astype("<u2")),
            "a": refs,
            "s": put(np.diff(s, prepend=0).astype("<u2")),  # origem codificada por diferença
            "t": put(t.astype("<u2")),
            "q": put(np.concatenate([q >> 8, q & 255]).astype("u1")),  # [bytes altos..., bytes baixos...]
            **({"f": put(flags)} if flags is not None else {}),
        })
    if any(len(c) >= _NONE for c in codes.values()):
        raise ValueError("Atributo categórico com valores distintos demais.")

    # --- Votações: informações (JSON comprimido) e matriz votação × vértice com o voto (uint8) ---
    votes_data = None
    if votacoes is not None and votos is not None:
        info_df = votacoes if "id" in votacoes.columns else votacoes.reset_index().rename(columns={votacoes.index.name or "index": "id"})
        info_df = info_df.assign(id=info_df["id"].astype(str))
        used = {str(v) for _, g in frames if votacoes_attr in g.attributes() for v in (g[votacoes_attr] or [])}
        if used:
            info_df = info_df[info_df["id"].isin(used)]
        if "data" in info_df.columns:
            info_df = info_df.sort_values("data", kind="stable")
        info_df = info_df.drop_duplicates("id").reset_index(drop=True)
        if len(info_df) >= _NONE:
            raise ValueError("Votações demais.")
        vidx = {v: i for i, v in enumerate(info_df["id"])}
        vv = votos.iloc[:, :3].set_axis(["id", "nome", "voto"], axis=1)
        vv = vv.assign(id=vv["id"].astype(str), nome=vv["nome"].astype(str))
        vv = vv[vv["id"].isin(vidx) & vv["nome"].isin(index)].drop_duplicates(["id", "nome"])
        code_of = {label: i + 1 for i, label in enumerate(_VOTE_LABELS)}
        matrix = np.zeros((len(vidx), n_nodes), dtype=np.uint8)
        matrix[vv["id"].map(vidx).to_numpy(), vv["nome"].map(index).to_numpy()] = (
            vv["voto"].map(code_of).fillna(len(_VOTE_LABELS)).to_numpy(dtype=np.uint8))
        for (_, g), meta in zip(frames, frames_meta):
            ids = g[votacoes_attr] if votacoes_attr in g.attributes() else list(vidx)
            meta["v"] = put(np.array(sorted({vidx[str(v)] for v in ids if str(v) in vidx}), dtype="<u2"))
        records = []
        for row in info_df.to_dict("records"):
            rec = {}
            for k, v in row.items():
                if k == "temas":
                    v = [str(t) for t in v] if isinstance(v, (list, tuple, set, np.ndarray)) else []
                else:
                    v = _clean(v)
                if v is not None and v != []:
                    rec[k] = v
            records.append(rec)
        meta_json = json.dumps(records, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        votes_data = {
            "m": put(matrix.ravel()),
            "labels": list(_VOTE_LABELS),
            "meta": base64.b64encode(gzip.compress(meta_json, compresslevel=9, mtime=0)).decode("ascii"),
        }

    frame_index = iter(range(len(frames)))
    data = {
        "libs": JS_LIBS,
        "config": {
            "title": title,
            "subtitle": subtitle,
            "colorAttrs": color_attrs,
            "sizeAttrs": size_attrs,
            "infoAttrs": info_attrs,
            "sublabelAttrs": sublabel_attrs,
            "attrKinds": kinds,
            "edgeLabel": edge_label,
            "edgeFlag": {"label": edge_flag_label, "hint": edge_flag_hint} if has_flags else None,
            "photoAttr": photo_attr,
            "linkAttr": link_attr,
            "linkText": link_text,
            "logoAttr": logo_attr,
            "logoFor": logo_for,
            "logos": {str(k): v for k, v in (logos or {}).items()} if logo_attr else {},
            "fixedColors": {attr: {str(k): _theme_pair(c) for k, c in mapping.items() if c}
                            for attr, mapping in (colors or {}).items() if attr in color_attrs},
            "nodeNoun": node_noun,
            "edgeNoun": edge_noun,
            "timelineLabel": timeline_label,
            "scaleLabel": scale_label,
            "scales": scales,
            "timelines": [{"name": name, "scale": scales.index(scale), "frames": [next(frame_index) for _ in items]}
                          for scale, name, items in timelines],
            "layoutSeconds": layout_seconds,
            "transitionSeconds": transition_seconds,
            "playSeconds": play_seconds,
            "fa2": fa2,
        },
        "nodes": nodes,
        "dicts": {a: list(c) for a, c in codes.items()},
        "valueRange": [float(lo), float(hi)],
        "qMax": _Q_MAX,
        "frames": frames_meta,
        "votes": votes_data,
        "blob": base64.b64encode(gzip.compress(bytes(buf), compresslevel=9, mtime=0)).decode("ascii"),
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("<", "\\u003c")  # seguro dentro de <script>
    return (_TEMPLATE
            .replace("__TITLE__", html.escape(title))
            .replace("__DATA__", payload))


def save_html(graphs, path="grafo.html", **kwargs) -> Path:
    """Salva a visualização em um arquivo HTML e devolve o caminho."""
    path = Path(path)
    path.write_text(graph_to_html(graphs, **kwargs), encoding="utf-8")
    return path


def show(graphs, path="grafo.html", height=820, **kwargs):
    """Salva o HTML (se ``path`` não for None) e exibe a visualização no notebook."""
    from IPython.display import HTML

    page = graph_to_html(graphs, **kwargs)
    if path:
        Path(path).write_text(page, encoding="utf-8")
        print(f"Visualização salva em {Path(path).resolve()} ({len(page) / 1e6:.1f} MB) — abra em qualquer navegador.")
    return HTML(
        f'<div><iframe srcdoc="{html.escape(page, quote=True)}" '
        f'style="width:100%;height:{int(height)}px;border:1px solid rgba(0,0,0,.1);border-radius:8px" '
        f'allow="fullscreen; clipboard-write"></iframe></div>'
    )


_TEMPLATE = r"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --raised: #ffffff;
  --text: #0b0b0b; --text-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,.10);
  --faded: #e6e5df; --edge: #898781; --shadow: rgba(11,11,11,.14); --accent: #2a78d6;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100;
  --s5: #e87ba4; --s6: #008300; --s7: #6250d6; --s8: #e34948; --other: #a3a29b;
  --font: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --good-text: #006300; --bad-text: #c62f2f;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --raised: #222221;
    --text: #f0efec; --text-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10);
    --faded: #2a2a28; --edge: #898781; --shadow: rgba(0,0,0,.5); --accent: #3987e5;
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
    --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767; --other: #6f6e68;
  --good-text: #0ca30c; --bad-text: #e66767;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --raised: #222221;
  --text: #f0efec; --text-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,.10);
  --faded: #2a2a28; --edge: #898781; --shadow: rgba(0,0,0,.5); --accent: #3987e5;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500;
  --s5: #d55181; --s6: #008300; --s7: #9085e9; --s8: #e66767; --other: #6f6e68;
  --good-text: #0ca30c; --bad-text: #e66767;
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body { background: var(--page); color: var(--text); font: 14px/1.4 var(--font); }
button, input, select { font: inherit; color: inherit; }
.app { display: flex; flex-direction: column; height: 100%; }

/* Barra superior: título + controles numa linha */
.bar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; padding: 10px 16px;
  background: var(--surface); border-bottom: 1px solid var(--border); }
.brand { min-width: 0; margin-right: auto; }
.brand h1 { margin: 0; font-size: 16px; font-weight: 650; }
.brand p { margin: 0; color: var(--text-2); font-size: 12.5px; }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
.field { display: flex; align-items: center; gap: 6px; color: var(--text-2); font-size: 12.5px; }
select, .btn, .search input { height: 32px; border: 1px solid var(--border); border-radius: 8px;
  background: var(--raised); padding: 0 10px; }
select { padding-right: 6px; max-width: 120px; text-overflow: ellipsis; }
.btn { cursor: pointer; display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.btn:hover, select:hover { border-color: var(--axis); }
.btn:disabled { opacity: .45; cursor: default; }
.btn[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); }
#layoutBtn { min-width: 96px; justify-content: center; }  /* ▶/❚❚ sem mudar a largura da barra */
.btn:focus-visible, select:focus-visible, input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.range { display: flex; align-items: center; gap: 6px; }
.range input { width: 80px; accent-color: var(--accent); }
.range output { min-width: 34px; font-variant-numeric: tabular-nums; color: var(--text); }

.search { position: relative; }
.search input { width: 125px; }
.suggest { position: absolute; z-index: 20; top: 36px; left: 0; right: 0; margin: 0; padding: 4px;
  list-style: none; background: var(--raised); border: 1px solid var(--border); border-radius: 8px;
  box-shadow: 0 8px 24px var(--shadow); max-height: 320px; overflow: auto; }
.suggest li { padding: 6px 8px; border-radius: 6px; cursor: pointer; display: flex; gap: 8px; align-items: center; }
.suggest li[aria-selected="true"], .suggest li:hover { background: var(--grid); }
.suggest small { color: var(--text-2); margin-left: auto; }

/* Área do grafo */
.stage { position: relative; flex: 1; min-height: 0; background: var(--surface); }
#graph { position: absolute; inset: 0; }
.overlay { position: absolute; z-index: 5; background: color-mix(in srgb, var(--raised) 94%, transparent);
  border: 1px solid var(--border); border-radius: 10px; box-shadow: 0 4px 16px var(--shadow); }

/* Coluna da esquerda: legenda + parâmetros do layout */
.side { position: absolute; z-index: 5; top: 12px; left: 12px; width: 270px; max-height: calc(100% - 24px);
  display: flex; flex-direction: column; gap: 8px; }
.side > .overlay { position: relative; flex: 0 1 auto; min-height: 0; overflow: auto; }
.side > .legend { flex-shrink: 0; max-height: 100%; }
.side.with-params > .legend { max-height: 40%; }  /* com o painel aberto, a legenda rola */
.legend { padding: 10px; }
.legend h2, .params h2, .compo h2, .votes h2 { margin: 0 0 2px; font-size: 12px; font-weight: 600; color: var(--text-2);
  text-transform: uppercase; letter-spacing: .04em; }

.params { padding: 10px 12px; flex-shrink: 2 !important; }
.params .head { display: flex; align-items: center; justify-content: space-between; }
.params .close, .compo .close, .votes .close { width: 24px; height: 24px; border: 0; background: none; border-radius: 6px; cursor: pointer;
  font-size: 16px; color: var(--text-2); }
.params .close:hover, .compo .close:hover, .votes .close:hover { background: var(--grid); }
.params .status { margin: 0 0 8px; font-size: 12px; color: var(--text-2); font-variant-numeric: tabular-nums; }
.params .btns { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 4px; }
.params .btn { height: 28px; padding: 0 8px; font-size: 12.5px; }
.params .hint { display: block; font-size: 11.5px; color: var(--text-2); line-height: 1.3; }
.params code { font: 11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: var(--muted); }
.params .lbl { font-weight: 550; }
.params .row { margin: 10px 0; }
.params .row-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; font-size: 13px; }
.params .row-head output { font-variant-numeric: tabular-nums; }
.params input[type="range"] { width: 100%; margin: 4px 0 2px; accent-color: var(--accent); }
.params .row.disabled { opacity: .45; }
.params .check { display: flex; gap: 8px; align-items: flex-start; margin: 8px 0; font-size: 13px; cursor: pointer; }
.params .check input { margin: 2px 0 0; accent-color: var(--accent); }
.params .code-head { display: flex; align-items: center; justify-content: space-between; gap: 8px;
  margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--grid); font-size: 12px; }
.params pre { margin: 6px 0 0; padding: 6px 8px; border-radius: 6px; background: var(--grid); white-space: pre-wrap;
  word-break: break-word; font: 11px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; user-select: all; }
.legend .hint { margin: 0 0 6px; font-size: 11.5px; color: var(--muted); }
.legend .actions { display: flex; gap: 10px; margin-bottom: 6px; font-size: 12px; }
.linklike { background: none; border: 0; padding: 0; color: var(--accent); cursor: pointer; }
.legend ul { list-style: none; margin: 0; padding: 0; }
.legend li button { width: 100%; display: flex; align-items: center; gap: 8px; padding: 4px 6px;
  border: 0; border-radius: 6px; background: none; cursor: pointer; text-align: left; }
.legend li button:hover { background: var(--grid); }
.legend li button[aria-pressed="false"] { opacity: .4; }
.legend li button[aria-pressed="false"] .sw { background: transparent !important; box-shadow: inset 0 0 0 1.5px var(--muted); }
.legend .name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.legend .count { color: var(--text-2); font-variant-numeric: tabular-nums; font-size: 12px; }
.legend details summary { cursor: pointer; padding: 4px 6px; color: var(--text-2); font-size: 12.5px; }
.sw { width: 10px; height: 10px; border-radius: 50%; flex: none; }

.zoom { right: 12px; bottom: 12px; display: flex; flex-direction: column; padding: 4px; gap: 2px; }
.zoom button { width: 32px; height: 32px; border: 0; border-radius: 6px; background: none; cursor: pointer; font-size: 16px; }
.zoom button:hover { background: var(--grid); }
.stage.right-open .zoom { right: 344px; }

.rside { position: absolute; z-index: 5; top: 12px; right: 12px; width: 320px; max-height: calc(100% - 24px);
  display: flex; flex-direction: column; gap: 8px; }
.rside > .overlay { position: relative; flex: 0 1 auto; min-height: 0; overflow: auto; }
.panel { padding: 14px; }

/* Composição: barras empilhadas grupo × categoria */
.compo { padding: 10px 12px; }
.compo .head { display: flex; align-items: center; justify-content: space-between; }
.compo .ctrls { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 6px 0; font-size: 12.5px; color: var(--text-2); }
.compo select { height: 28px; font-size: 12.5px; max-width: 120px; }
.compo .icon { height: 28px; width: 28px; padding: 0; justify-content: center; }
.seg-toggle { display: inline-flex; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.seg-toggle button { border: 0; background: var(--raised); height: 26px; padding: 0 9px; cursor: pointer; font-size: 12px; color: var(--text-2); }
.seg-toggle button[aria-pressed="true"] { background: var(--grid); color: var(--text); font-weight: 600; }
.compo .hint { margin: 2px 0 10px; font-size: 11.5px; color: var(--text-2); }
.crow { margin: 0 0 12px; }
.crow .clabel { display: flex; align-items: center; gap: 6px; width: 100%; border: 0; background: none; padding: 2px 0;
  cursor: pointer; font-size: 13px; text-align: left; }
.crow .clabel:hover .name, .crow .clabel.on .name { text-decoration: underline; }
.crow .clabel .n { margin-left: auto; color: var(--text-2); font-variant-numeric: tabular-nums; font-size: 12px; }
.cbar { display: flex; gap: 2px; height: 16px; margin: 3px 0; }
.cbar .seg { min-width: 3px; height: 100%; border: 0; padding: 0; cursor: pointer; }
.cbar .seg:first-child { border-radius: 4px 0 0 4px; }
.cbar .seg:last-child { border-radius: 0 4px 4px 0; }
.cbar .seg:only-child { border-radius: 4px; }
.cbar .seg:hover, .cbar .seg:focus-visible, .cbar .seg.on { outline: 2px solid var(--text); outline-offset: 1px; }
.crow .ctext { font-size: 11.5px; color: var(--text-2); line-height: 1.35; }
.compo table { border-collapse: collapse; width: 100%; font-size: 12px; }
.compo table { font-size: 11.5px; }
.compo th, .compo td { padding: 3px; border-bottom: 1px solid var(--grid); text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.compo td:first-child { max-width: 104px; overflow: hidden; text-overflow: ellipsis; }
.compo th:first-child, .compo td:first-child { text-align: left; }
.compo th { position: sticky; top: 0; background: var(--raised); color: var(--text-2); font-weight: 600; }
.compo td.cell { cursor: pointer; }
.compo td.cell:hover, .compo td.cell.on { background: var(--grid); }
.compo td small { color: var(--text-2); }
.compo tfoot td { font-weight: 600; border-bottom: 0; }
.compo .hl { margin: 4px 0 12px; padding: 8px; border-radius: 8px; background: var(--grid); font-size: 12.5px; }
.compo .hl .names { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; max-height: 92px; overflow: auto; }
.compo .hl .names button { border: 1px solid var(--border); background: var(--raised); border-radius: 999px; padding: 1px 8px;
  font-size: 11.5px; cursor: pointer; }
/* Votações: lista + votação selecionada */
.side > .votes { display: flex; flex-direction: column; overflow: hidden; padding: 10px 12px; flex-shrink: 2; min-height: 180px; }
.votes .head { display: flex; align-items: center; justify-content: space-between; }
.votes .count { font-size: 12px; color: var(--text-2); font-weight: 400; text-transform: none; letter-spacing: 0; }
.votes input[type="search"] { width: 100%; height: 30px; margin: 6px 0; border: 1px solid var(--border); border-radius: 8px;
  background: var(--raised); padding: 0 8px; }
.vsel { flex: none; max-height: 48%; overflow: auto; margin-bottom: 6px; padding: 8px; border-radius: 8px; background: var(--grid); font-size: 12.5px; }
.vsel .vhead { display: flex; gap: 6px; align-items: flex-start; }
.vsel .vhead strong { flex: 1; font-size: 13px; }
.vsel p { margin: 4px 0; }
.vsel .vementa { color: var(--text-2); }
.vtexts.clamp p { display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
.vcounts { display: flex; flex-wrap: wrap; gap: 3px 10px; margin: 6px 0 2px; }
.vcounts span { display: inline-flex; align-items: center; gap: 4px; font-variant-numeric: tabular-nums; }
.vofficial { font-size: 11.5px; color: var(--text-2); }
.chip { display: inline-block; margin: 2px 4px 0 0; padding: 0 7px; border: 1px solid var(--border); border-radius: 999px;
  background: var(--raised); font-size: 11px; }
.vactions { display: flex; flex-wrap: wrap; gap: 6px 12px; margin-top: 6px; }
.vactions a { color: var(--accent); }
.vlist { flex: 1 1 auto; min-height: 60px; overflow: auto; list-style: none; margin: 0; padding: 0; }
.vitem { display: block; width: 100%; padding: 6px 4px; border: 0; border-bottom: 1px solid var(--grid); background: none;
  text-align: left; cursor: pointer; }
.vitem:hover { background: var(--grid); }
.vitem.on { background: var(--grid); box-shadow: inset 3px 0 0 var(--accent); }
.vmeta { display: flex; gap: 6px; font-size: 11.5px; color: var(--text-2); font-variant-numeric: tabular-nums; }
.vres { margin-left: auto; white-space: nowrap; }
.vres.ok { color: var(--good-text); }
.vres.no { color: var(--bad-text); }
.vitem .vtitle { display: block; font-weight: 600; font-size: 12.5px; }
.vitem .vdesc { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
  font-size: 11.5px; color: var(--text-2); }
.vempty { padding: 8px 4px; font-size: 12px; color: var(--text-2); }
.tooltip { position: absolute; z-index: 40; pointer-events: none; max-width: 260px; padding: 6px 9px; font-size: 12px; line-height: 1.35;
  background: var(--raised); border: 1px solid var(--border); border-radius: 8px; box-shadow: 0 6px 18px var(--shadow); }
.tooltip strong { font-size: 13px; }
.tooltip span { display: block; color: var(--text-2); }
.panel .close { position: absolute; top: 8px; right: 8px; width: 28px; height: 28px; border: 0;
  background: none; border-radius: 6px; cursor: pointer; font-size: 18px; color: var(--text-2); }
.panel .close:hover { background: var(--grid); }
.panel header { display: flex; gap: 12px; align-items: center; margin-bottom: 10px; padding-right: 24px; }
.panel .photo { width: 64px; height: 84px; object-fit: cover; border-radius: 8px; background: var(--grid); flex: none; }
.panel h2 { margin: 0; font-size: 16px; line-height: 1.25; }
.panel .sub { color: var(--text-2); margin-top: 2px; display: flex; align-items: center; gap: 6px; }
.panel dl { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 4px 12px; margin: 0 0 10px; font-size: 13px; }
.panel dt { color: var(--text-2); }
.panel dd { margin: 0; max-width: 170px; text-align: right; font-variant-numeric: tabular-nums; display: flex;
  justify-content: flex-end; align-items: center; gap: 6px; overflow-wrap: anywhere; }
.panel .logo { height: 18px; max-width: 48px; object-fit: contain; }
.panel a { color: var(--accent); }
.panel h3 { font-size: 12px; font-weight: 600; color: var(--text-2); text-transform: uppercase;
  letter-spacing: .04em; margin: 14px 0 4px; }
.panel ol { list-style: none; margin: 0; padding: 0; }
.panel ol button { width: 100%; display: flex; gap: 8px; align-items: center; padding: 4px 6px; border: 0;
  border-radius: 6px; background: none; cursor: pointer; text-align: left; }
.panel ol button:hover { background: var(--grid); }
.panel ol .nm { display: flex; flex-direction: column; min-width: 0; line-height: 1.25; }
.panel ol li.weak { opacity: .55; }
.panel ol .v { margin-left: auto; color: var(--text-2); font-variant-numeric: tabular-nums; font-size: 12px; }
.panel ol small { color: var(--text-2); font-size: 11.5px; }

.table-view { position: absolute; inset: 0; z-index: 10; background: var(--surface); overflow: auto; padding: 0 16px 16px; }
.table-view table { border-collapse: collapse; width: 100%; font-size: 13px; }
.table-view th { position: sticky; top: 0; background: var(--surface); text-align: left; font-weight: 600;
  color: var(--text-2); padding: 10px 8px 6px; border-bottom: 1px solid var(--axis); cursor: pointer; white-space: nowrap; }
.table-view th[aria-sort="ascending"]::after { content: " ▲"; font-size: 10px; }
.table-view th[aria-sort="descending"]::after { content: " ▼"; font-size: 10px; }
.table-view td { padding: 5px 8px; border-bottom: 1px solid var(--grid); }
.table-view td.num { text-align: right; font-variant-numeric: tabular-nums; }
.table-view tbody tr { cursor: pointer; }
.table-view tbody tr:hover { background: var(--grid); }

/* Linha do tempo (embaixo) */
.timeline { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 16px; padding: 8px 16px 4px;
  background: var(--surface); border-top: 1px solid var(--border); }
.tl-buttons { display: flex; gap: 4px; }
#tlPlay { min-width: 104px; justify-content: center; }
#tlPrev, #tlNext { width: 32px; padding: 0; justify-content: center; font-size: 18px; }
.tl-track { flex: 1 1 260px; min-width: 180px; padding: 0 4px; }
.tl-track input { display: block; width: 100%; margin: 0; accent-color: var(--accent); }
.ticks { position: relative; height: 22px; }
.tick { position: absolute; top: 0; transform: translateX(-50%); background: none; border: 0; padding: 5px 3px 0;
  font-size: 11.5px; color: var(--text-2); cursor: pointer; white-space: nowrap; }
.tick::before { content: ""; position: absolute; top: 0; left: 50%; width: 1px; height: 4px; background: var(--axis); }
.tick.nolabel { width: 9px; height: 14px; padding: 0; }
.tick.nolabel span { display: none; }
.tick:hover { color: var(--text); }
.tick:hover::before { background: var(--text); }
.tick[aria-current="true"] { color: var(--text); font-weight: 650; }
.tick[aria-current="true"]::before { background: var(--accent); width: 3px; height: 6px; margin-left: -1px; }
.tl-info { display: flex; flex-direction: column; min-width: 150px; max-width: 420px; font-size: 12.5px; line-height: 1.3; padding-bottom: 4px; }
.tl-info strong { font-size: 14px; }
.tl-info span { color: var(--text-2); font-variant-numeric: tabular-nums; }

.empty-msg { position: absolute; inset: 0; display: grid; place-items: center; color: var(--text-2); font-size: 14px;
  pointer-events: none; z-index: 5; }
.loading { position: absolute; inset: 0; display: grid; place-items: center; color: var(--text-2); z-index: 30; background: var(--surface); }
[hidden] { display: none !important; }

@media (max-width: 720px) {
  .search input { width: 100%; }
  .search { flex: 1 1 100%; }
  .side { width: 220px; max-height: 45%; }
  .rside { top: auto; bottom: 12px; left: 12px; right: 12px; width: auto; max-height: 55%; }
  .zoom, .stage.right-open .zoom { bottom: auto; top: 12px; right: 12px; }
}
</style>
</head>
<body>
<div class="app">
  <header class="bar">
    <div class="brand"><h1 id="title"></h1><p id="subtitle"></p></div>
    <div class="controls">
      <div class="search" role="search">
        <input id="search" type="search" placeholder="Buscar…" autocomplete="off" aria-label="Buscar"
               aria-autocomplete="list" aria-controls="suggest">
        <ul id="suggest" class="suggest" role="listbox" hidden></ul>
      </div>
      <label class="field">Cor <select id="colorBy"></select></label>
      <label class="field">Tamanho <select id="sizeBy"></select></label>
      <label class="field range"><span id="edgeLabel"></span>
        <input id="threshold" type="range" min="0" max="1" step="0.001" value="0">
        <output id="thresholdOut"></output></label>
      <button class="btn" id="flagBtn" type="button" aria-pressed="false" hidden></button>
      <button class="btn" id="layoutBtn" type="button" aria-pressed="false">▶ Layout</button>
      <button class="btn" id="paramsBtn" type="button" aria-pressed="false" aria-controls="params"
              title="Parâmetros do layout (ForceAtlas2)" aria-label="Parâmetros do layout">⚙</button>
      <button class="btn" id="compoBtn" type="button" aria-pressed="false" aria-controls="compo"
              title="Distribuição de uma categoria (ex.: partidos) em cada grupo (ex.: comunidades)">▦ Composição</button>
      <button class="btn" id="votesBtn" type="button" aria-pressed="false" aria-controls="votes"
              title="Votações do período e tema: informações da proposição e voto de cada deputado">⚖ Votações</button>
      <button class="btn" id="tableBtn" type="button" aria-pressed="false">☰ Lista</button>
      <button class="btn" id="themeBtn" type="button" title="Tema: automático" aria-label="Tema: automático">◐</button>
      <button class="btn" id="fullBtn" type="button" title="Tela cheia" aria-label="Tela cheia">⛶</button>
    </div>
  </header>
  <div class="stage" id="stage">
    <div id="graph"></div>
    <div class="side">
      <aside class="overlay legend" id="legend" aria-label="Legenda"></aside>
      <section class="overlay params" id="params" hidden aria-label="Parâmetros do layout"></section>
      <section class="overlay votes" id="votes" hidden aria-label="Votações"></section>
    </div>
    <div class="rside">
      <aside class="overlay panel" id="panel" hidden aria-live="polite"></aside>
      <section class="overlay compo" id="compo" hidden aria-label="Composição dos grupos"></section>
    </div>
    <div class="overlay zoom">
      <button id="zoomIn" type="button" title="Aproximar" aria-label="Aproximar">+</button>
      <button id="zoomOut" type="button" title="Afastar" aria-label="Afastar">−</button>
      <button id="zoomReset" type="button" title="Centralizar" aria-label="Centralizar">⟲</button>
    </div>
    <div class="table-view" id="tableView" hidden></div>
    <div class="tooltip" id="tip" hidden role="tooltip"></div>
    <div class="empty-msg" id="emptyMsg" hidden></div>
    <div class="loading" id="loading">Carregando bibliotecas…</div>
  </div>
  <footer class="timeline" id="timeline" hidden aria-label="Linha do tempo">
    <div class="field" id="tlScaleField" hidden><span id="tlScaleName"></span>
      <div class="seg-toggle" id="tlScale" role="group"></div></div>
    <label class="field" id="tlField"><span id="tlName"></span> <select id="tlSelect"></select></label>
    <div class="tl-buttons">
      <button class="btn" id="tlPrev" type="button" title="Período anterior (←)" aria-label="Período anterior">‹</button>
      <button class="btn" id="tlPlay" type="button" aria-pressed="false">▶ Animar</button>
      <button class="btn" id="tlNext" type="button" title="Próximo período (→)" aria-label="Próximo período">›</button>
    </div>
    <div class="tl-track">
      <input id="tlRange" type="range" min="0" max="0" step="1" value="0" aria-label="Período">
      <div class="ticks" id="tlTicks"></div>
    </div>
    <div class="tl-info" aria-live="polite"><strong id="tlLabel"></strong><span id="tlMeta"></span></div>
  </footer>
</div>
<script type="application/json" id="graph-data">__DATA__</script>
<script type="module">
const DATA = JSON.parse(document.getElementById("graph-data").textContent);
const C = DATA.config;
const $ = (id) => document.getElementById(id);
const el = (tag, props = {}, ...kids) => {
  const n = Object.assign(document.createElement(tag), props);
  for (const k of kids) if (k != null) n.append(k);
  return n;
};
const nf = new Intl.NumberFormat("pt-BR");
const fmt = (v) => {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return "—";
    if (Number.isInteger(v) || Math.abs(v) >= 100) return nf.format(Math.round(v));
    return v.toLocaleString("pt-BR", { maximumFractionDigits: 3 });
  }
  return String(v);
};
const norm = (s) => String(s).normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
const MISSING = "Sem informação";
const SLOTS = 8;
const NONE = 65535;

// ---------- Tema (cores) ----------
let T = {};
const cssVar = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function readTheme() {
  T = {
    text: cssVar("--text"), text2: cssVar("--text-2"), raised: cssVar("--raised"), border: cssVar("--axis"),
    faded: cssVar("--faded"), edge: cssVar("--edge"), other: cssVar("--other"), shadow: cssVar("--shadow"),
    font: cssVar("--font"), slots: Array.from({ length: SLOTS }, (_, i) => cssVar(`--s${i + 1}`)),
    dark: matchMedia("(prefers-color-scheme: dark)").matches
      ? document.documentElement.dataset.theme !== "light"
      : document.documentElement.dataset.theme === "dark",
  };
}
// O WebGL do Sigma mistura cores com alfa pré-multiplicado (gl.ONE, gl.ONE_MINUS_SRC_ALPHA)
// e só aceita componentes inteiros: multiplicamos RGB pelo alfa aqui.
function rgba(hex, a) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.replace(/./g, "$&$&") : h, 16);
  const c = (shift) => Math.round(((n >> shift) & 255) * a);
  return `rgba(${c(16)},${c(8)},${c(0)},${a})`;
}

// ---------- Bibliotecas ----------
const CDN = "https://cdn.jsdelivr.net/npm/";
let Graph, Sigma, forceAtlas2, FA2Layout;
try {
  const V = DATA.libs;
  const mods = await Promise.all([
    import(`${CDN}graphology@${V.graphology}/+esm`),
    import(`${CDN}sigma@${V.sigma}/+esm`),
    import(`${CDN}graphology-layout-forceatlas2@${V.fa2}/+esm`),
    import(`${CDN}graphology-layout-forceatlas2@${V.fa2}/worker.js/+esm`),
  ]);
  [Graph, Sigma, forceAtlas2, FA2Layout] = mods.map((m) => m.default);
} catch (err) {
  $("loading").textContent = "Não foi possível carregar as bibliotecas do CDN (sem internet?). " + err;
  throw err;
}

// ---------- Dados: blob binário (gzip + base64) com os arrays de cada quadro ----------
async function inflate(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}
let BUF;
try {
  $("loading").textContent = "Descompactando dados…";
  BUF = await inflate(DATA.blob);
} catch (err) {
  $("loading").textContent = "Não foi possível ler os dados embutidos (navegador antigo?). " + err;
  throw err;
}
const NODES = DATA.nodes, FRAMES = DATA.frames, TIMELINES = C.timelines, KINDS = C.attrKinds, DICTS = DATA.dicts;
const [VLO, VHI] = DATA.valueRange;
const VSPAN = VHI - VLO || 1;
const QMAX = DATA.qMax || 65535;
const view = (Type, [offset, length]) => new Type(BUF, offset, length);
const catValue = (attr, code) => (code === NONE ? null : DICTS[attr][code]);
// Votações (opcional): informações de cada uma e matriz votação × deputado com o voto (1 byte)
const VOTES = DATA.votes ? {
  meta: JSON.parse(new TextDecoder().decode(await inflate(DATA.votes.meta))),
  matrix: view(Uint8Array, DATA.votes.m),
} : null;
const frameCache = new Map();
function frameData(fi) {
  if (frameCache.has(fi)) return frameCache.get(fi);
  const M = FRAMES[fi], attrs = {};
  for (const [attr, ref] of Object.entries(M.a)) attrs[attr] = view(KINDS[attr] === "num" ? Float32Array : Uint16Array, ref);
  const [qo, q2] = M.q;  // valor quantizado: bytes altos e depois bytes baixos
  const F = { nodes: view(Uint16Array, M.n), attrs, sd: view(Uint16Array, M.s), t: view(Uint16Array, M.t),
              qh: new Uint8Array(BUF, qo, q2 / 2), ql: new Uint8Array(BUF, qo + q2 / 2, q2 / 2),
              flag: M.f ? view(Uint8Array, M.f) : null };
  frameCache.set(fi, F);
  return F;
}

// Cores dos grupos: ordem pelo total em TODOS os quadros, fixada uma única vez
// (a cor acompanha o grupo ao longo do tempo e não muda quando se filtra).
const groups = {};
for (const attr of Object.keys(C.colorAttrs)) {
  const counts = new Map();
  const add = (key, k = 1) => counts.set(key, (counts.get(key) || 0) + k);
  FRAMES.forEach((M, fi) => {
    const arr = frameData(fi).attrs[attr];
    if (!arr) add(MISSING, M.n[1]);
    else for (const code of arr) add(KINDS[attr] === "num" ? String(code) : (catValue(attr, code) ?? MISSING));
  });
  const keys = [...counts.keys()].sort((a, b) => (a === MISSING) - (b === MISSING) || counts.get(b) - counts.get(a)
    || a.localeCompare(b, "pt-BR", { numeric: true }));
  let slot = 0;
  groups[attr] = { rank: new Map(keys.map((k, i) => [k, i])), slot: new Map() };
  for (const k of keys) groups[attr].slot.set(k, k !== MISSING && slot < SLOTS ? slot++ : -1);
}

// "Voto na votação selecionada" funciona como um atributo de cor virtual (legenda, composição...)
const VOTE = "__voto", NO_VOTE = "Não votou";
const VOTE_LABELS = [NO_VOTE, ...(DATA.votes ? DATA.votes.labels : [])];
if (VOTES) {
  C.fixedColors[VOTE] = { // Sim × Não em azul × laranja (distinguíveis também por daltônicos)
    Sim: ["#2a78d6", "#3987e5"], "Não": ["#eb6834", "#d95926"], "Abstenção": ["#eda100", "#c98500"],
    "Obstrução": ["#6250d6", "#9085e9"], "Artigo 17": ["#e87ba4", "#d55181"], Outro: ["#898781", "#898781"],
    [NO_VOTE]: ["#d3d2cb", "#45443f"],
  };
  groups[VOTE] = { rank: new Map([...VOTE_LABELS.slice(1), NO_VOTE].map((k, i) => [k, i])), slot: new Map() };
}
const attrLabel = (attr) => (attr === VOTE ? "Voto" : C.colorAttrs[attr]);

// ---------- Estado ----------
const state = {
  colorAttr: Object.keys(C.colorAttrs)[0] || null,
  sizeAttr: Object.keys(C.sizeAttrs)[0] || "uniform",
  threshold: VLO,
  hidden: new Set(),
  hovered: null,
  selected: null,
  focusNeighbors: new Set(),
  highlight: null, // critério de destaque da composição: {rows, rowKey, segs, segKeys}
  vote: null,      // índice da votação selecionada (cores pelo voto)
  onlyFlag: false, // só as arestas marcadas (ex.: significativas)
};
// Aresta passa nos filtros de valor mínimo e de marca (não olha os nós)
const edgeShown = (a) => a.value >= state.threshold && (!state.onlyFlag || a.sig);
const cur = { tl: 0, pos: 0 };
const curFrame = () => FRAMES[TIMELINES[cur.tl].frames[cur.pos]];
const multi = TIMELINES.length > 1 || TIMELINES[0].frames.length > 1;

// ---------- Grafo do quadro atual ----------
const graph = new Graph({ type: "undirected", multi: false, allowSelfLoops: false });
let sigma = null; // criado depois do primeiro quadro
// Mudanças estruturais (nós/arestas entrando e saindo) e sigma.setSetting agendam um reprocessamento
// completo, que descarta os índices de desenho até o próximo frame. Uma atualização em massa de
// atributos nesse intervalo tenta repintar itens sem índice e o Sigma lança "can't be repaint".
// Por isso, antes de recolorir/redimensionar, fazemos o reprocessamento pendente na hora.
const syncSigma = () => { if (sigma && sigma.needToProcess) sigma.refresh(); };
function attrValue(n, attr) {
  if (attr === VOTE) return state.vote == null ? null : (VOTE_LABELS[VOTES.matrix[state.vote * NODES.length + +n]] ?? "Outro");
  const a = graph.getNodeAttribute(n, "attrs");
  if (attr in a) return a[attr];
  if (attr === "degree") return graph.degree(n);
  return NODES[+n].a[attr] ?? null;
}
const fixedColors = () => state.colorAttr && C.fixedColors[state.colorAttr];
// Cor de um valor de qualquer atributo de cor (cores dadas no Python: [tema claro, tema escuro])
const groupColorOf = (attr, key) => {
  const fixed = attr && C.fixedColors[attr];
  if (fixed) return fixed[key] ? fixed[key][T.dark ? 1 : 0] : T.other;
  const s = attr ? groups[attr].slot.get(key) : 0;
  return s != null && s >= 0 ? T.slots[s] : T.other;
};
const groupColor = (key) => groupColorOf(state.colorAttr, key);
let highlightSet = null; // nós destacados pelo painel de composição (ou null)
const nodeVisible = (n) => !state.hidden.has(graph.getNodeAttribute(n, "g"));

// Troca o conteúdo do grafo pelo do quadro (linha `tl`, posição `pos`).
// Quem continua mantém a posição; quem entra nasce no centro dos vizinhos já posicionados.
function loadFrame(tl, pos, { home = false } = {}) {
  cur.tl = tl; cur.pos = pos;
  const F = frameData(TIMELINES[tl].frames[pos]);
  graph.clearEdges();
  const keep = new Set(Array.from(F.nodes, String));
  graph.nodes().forEach((n) => { if (!keep.has(n)) graph.dropNode(n); });
  const entering = [], attrNames = Object.keys(F.attrs);
  F.nodes.forEach((gi, k) => {
    const key = String(gi), attrs = {};
    for (const a of attrNames) {
      const v = F.attrs[a][k];
      attrs[a] = KINDS[a] === "num" ? (Number.isNaN(v) ? null : v) : catValue(a, v);
    }
    const sublabel = C.sublabelAttrs.map((a) => attrs[a] ?? NODES[gi].a[a]).filter((v) => v != null).join(" · ");
    if (graph.hasNode(key)) graph.mergeNodeAttributes(key, { attrs, sublabel });
    else {
      graph.addNode(key, { label: NODES[gi].l, x: NODES[gi].x, y: NODES[gi].y, attrs, sublabel, size: 4, color: "#999", g: "" });
      entering.push(key);
    }
  });
  let s = 0;
  for (let k = 0; k < F.t.length; k++) {
    s += F.sd[k];
    const w = ((F.qh[k] << 8) | F.ql[k]) / QMAX;
    graph.addEdge(String(s), String(F.t[k]), { value: VLO + w * VSPAN, weight: 0.05 + 0.95 * w, size: 0.4 + 0.8 * w,
      sig: F.flag ? F.flag[k] === 1 : true });
  }
  if (home) graph.updateEachNodeAttributes((n, a) => ({ ...a, x: NODES[+n].x, y: NODES[+n].y }), { attributes: ["x", "y"] });
  else placeNear(entering);
}
function placeNear(keys) {
  if (!keys.length) return;
  const pending = new Set(keys);
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, cx = 0, cy = 0, c = 0;
  graph.forEachNode((n, a) => {
    if (pending.has(n)) return;
    minX = Math.min(minX, a.x); maxX = Math.max(maxX, a.x); minY = Math.min(minY, a.y); maxY = Math.max(maxY, a.y);
    cx += a.x; cy += a.y; c++;
  });
  if (!c) return; // ninguém posicionado: ficam nas posições iniciais
  const ext = Math.max(maxX - minX, maxY - minY) || 1;
  const jitter = (k = 1) => (Math.random() - 0.5) * ext * 0.03 * k;
  for (let pass = 0; pass < 6 && pending.size; pass++) {
    const placed = [];
    for (const n of pending) {
      let sx = 0, sy = 0, k = 0;
      graph.forEachNeighbor(n, (m, ma) => { if (!pending.has(m)) { sx += ma.x; sy += ma.y; k++; } });
      if (k) placed.push([n, sx / k + jitter(), sy / k + jitter()]);
    }
    if (!placed.length) break;
    for (const [n, x, y] of placed) { graph.mergeNodeAttributes(n, { x, y }); pending.delete(n); }
  }
  // Sem vizinho posicionado (ex.: quadro sem arestas): espalhados num disco em volta do centro (filotaxia)
  const rest = [...pending], R = ext * 0.35, GOLDEN = Math.PI * (3 - Math.sqrt(5));
  rest.forEach((n, k) => {
    const r = R * Math.sqrt((k + 0.5) / rest.length), t = k * GOLDEN;
    graph.mergeNodeAttributes(n, { x: cx / c + r * Math.cos(t), y: cy / c + r * Math.sin(t) });
  });
}

function applyColors() {
  syncSigma();
  const attr = state.colorAttr;
  graph.updateEachNodeAttributes((n, a) => {
    const v = attr ? attrValue(n, attr) : null;
    const g = v == null ? MISSING : String(v);
    return { ...a, g, color: attr ? groupColor(g) : T.slots[0] };
  }, { attributes: ["g", "color"] });
  const intraA = T.dark ? 0.22 : 0.18, interA = T.dark ? 0.09 : 0.07;
  graph.updateEachEdgeAttributes((e, a, s, t, sa, ta) => {
    const f = a.sig ? 1 : 0.3;  // aresta não marcada (ex.: não significativa): mais apagada
    return { ...a, color: sa.g === ta.g ? rgba(sa.color, intraA * f) : rgba(T.edge, interA * f) };
  }, { attributes: ["color"] });
}

function applySizes() {
  syncSigma();
  const attr = state.sizeAttr;
  if (attr === "uniform") {
    graph.updateEachNodeAttributes((n, a) => ({ ...a, size: 5 }), { attributes: ["size"] });
    return;
  }
  let lo = Infinity, hi = -Infinity;
  graph.forEachNode((n) => {
    const v = attrValue(n, attr);
    if (typeof v === "number" && Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  });
  const span = hi - lo || 1;
  graph.updateEachNodeAttributes((n, a) => {
    const v = attrValue(n, attr);
    const t = typeof v === "number" && Number.isFinite(v) ? (v - lo) / span : 0;
    return { ...a, size: 2 + 8 * Math.sqrt(t) };
  }, { attributes: ["size"] });
}

function computeFocus() {
  const focus = state.hovered ?? state.selected;
  state.focusNeighbors = new Set();
  if (focus == null) return;
  graph.forEachEdge(focus, (e, a, s, t) => {
    const o = s === focus ? t : s;
    if (edgeShown(a) && nodeVisible(o)) state.focusNeighbors.add(o);
  });
}

// ---------- Sigma ----------
readTheme();
loadFrame(0, 0, { home: true });
applyColors();
applySizes();

function nodeReducer(node, data) {
  const res = { ...data };
  if (state.hidden.has(data.g)) { res.hidden = true; return res; }
  const focus = state.hovered ?? state.selected;
  if (focus != null) {
    if (node === focus) { res.highlighted = true; res.forceLabel = true; res.zIndex = 3; }
    else if (state.focusNeighbors.has(node)) { res.zIndex = 2; }
    else { res.color = T.faded; res.label = null; res.zIndex = 0; }
  } else if (highlightSet) {
    if (highlightSet.has(node)) { res.zIndex = 2; if (highlightSet.size <= 40) res.forceLabel = true; }
    else { res.color = T.faded; res.label = null; res.zIndex = 0; }
  } else {
    res.zIndex = 1;
  }
  if (node === state.selected) { res.highlighted = true; res.forceLabel = true; }
  return res;
}
function edgeReducer(edge, data) {
  const res = { ...data };
  if (!edgeShown(data)) { res.hidden = true; return res; }
  const [s, t] = graph.extremities(edge);
  if (!nodeVisible(s) || !nodeVisible(t)) { res.hidden = true; return res; }
  const focus = state.hovered ?? state.selected;
  if (focus != null) {
    if (s === focus || t === focus) {
      res.color = rgba(graph.getNodeAttribute(focus, "color"), 0.55);
      res.size = data.size + 0.8;
      res.zIndex = 1;
    } else res.hidden = true;
  } else if (highlightSet && !(highlightSet.has(s) && highlightSet.has(t))) {
    res.hidden = true;
  }
  return res;
}

function drawHover(ctx, data, settings) {
  const size = settings.labelSize, font = settings.labelFont;
  const name = data.label || "", sub = data.sublabel || "";
  ctx.font = `600 ${size}px ${font}`;
  const w1 = ctx.measureText(name).width;
  ctx.font = `400 ${size - 1}px ${font}`;
  const w2 = sub ? ctx.measureText(sub).width : 0;
  const padX = 8, padY = 6, lineH = size + 4;
  const boxW = Math.max(w1, w2) + 2 * padX, boxH = 2 * padY + lineH * (sub ? 2 : 1) - 4;
  const x = data.x + data.size + 6, y = data.y - boxH / 2;
  ctx.save();
  ctx.shadowColor = T.shadow; ctx.shadowBlur = 10;
  ctx.fillStyle = T.raised;
  ctx.beginPath(); ctx.arc(data.x, data.y, data.size + 3, 0, 2 * Math.PI); ctx.fill();
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, boxW, boxH, 6); else ctx.rect(x, y, boxW, boxH);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.strokeStyle = T.border; ctx.lineWidth = 1; ctx.stroke();
  ctx.textBaseline = "middle";
  ctx.fillStyle = T.text; ctx.font = `600 ${size}px ${font}`;
  ctx.fillText(name, x + padX, y + padY + (lineH - 4) / 2);
  if (sub) {
    ctx.fillStyle = T.text2; ctx.font = `400 ${size - 1}px ${font}`;
    ctx.fillText(sub, x + padX, y + padY + lineH + (lineH - 4) / 2);
  }
  ctx.restore();
}

const container = $("graph");
sigma = new Sigma(graph, container, {
  zIndex: true,
  renderEdgeLabels: false,
  labelFont: T.font,
  labelSize: 12,
  labelWeight: "500",
  labelColor: { color: T.text },
  labelDensity: 0.4,
  labelGridCellSize: 100,
  labelRenderedSizeThreshold: 7,
  minCameraRatio: 0.03,
  maxCameraRatio: 3,
  stagePadding: 40,
  nodeReducer,
  edgeReducer,
  defaultDrawNodeHover: drawHover,
});
const camera = sigma.getCamera();
$("loading").hidden = true;

const refresh = () => { updateRightSide(); sigma.refresh(); };

// ---------- Arrastar nós (como no novelo) ----------
// Com o layout parado, o nó fica onde for solto. Com o layout rodando, ele segue o mouse e os
// vizinhos acompanham; ao soltar, volta a obedecer às forças. No toque, arrastar move a câmera.
let drag = null;     // { node, x, y, moved }
let lastDragAt = 0;  // o clique que o navegador dispara ao soltar não deve selecionar nada
const justDragged = () => performance.now() - lastDragAt < 250;
// Durante o arraste o enquadramento fica congelado: sem isso, mover o nó muda o bounding box, que muda
// a conversão tela→grafo, que move o nó de novo (realimentação). Ele volta a se ajustar ao trocar de
// quadro, ao rodar o layout, em "Posições iniciais" e no botão ⟲.
const unfreezeView = () => sigma.setCustomBBox(null);

sigma.on("enterNode", ({ node }) => {
  if (drag) return;
  state.hovered = node; computeFocus(); container.style.cursor = "grab"; refresh();
});
sigma.on("leaveNode", () => {
  if (drag) return;
  state.hovered = null; computeFocus(); container.style.cursor = ""; refresh();
});
sigma.on("clickNode", ({ node }) => { if (!justDragged()) select(node); });
sigma.on("clickStage", () => { if (!justDragged()) select(null); });
sigma.on("downNode", ({ node, event, preventSigmaDefault }) => {
  if ("touches" in event.original) return;
  drag = { node, x: null, y: null, moved: false };
  state.hovered = node; computeFocus();
  if (!sigma.getCustomBBox()) sigma.setCustomBBox(sigma.getBBox());
  container.style.cursor = "grabbing";
  preventSigmaDefault();
});
sigma.getMouseCaptor().on("mousemovebody", (e) => {
  if (!drag) return;
  const p = sigma.viewportToGraph(e);
  if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) return;
  Object.assign(drag, { x: p.x, y: p.y, moved: true });
  graph.mergeNodeAttributes(drag.node, { x: p.x, y: p.y });
  e.preventSigmaDefault();
  e.original.preventDefault();
  e.original.stopPropagation();
});
function endDrag() {
  if (!drag) return;
  if (drag.moved) lastDragAt = performance.now();
  drag = null;
  // Com o layout rodando, o grafo continua mudando de escala: volta a enquadrar sozinho.
  // Parado, o enquadramento fica como está (sem pulo da câmera).
  if (layoutRunning()) unfreezeView();
  container.style.cursor = state.hovered ? "grab" : "";
}
sigma.getMouseCaptor().on("mouseup", endDrag);
window.addEventListener("mouseup", endDrag); // soltou fora do grafo

function select(node, { zoom = false } = {}) {
  state.selected = node;
  state.hovered = null;
  computeFocus();
  if (node == null) { $("panel").hidden = true; refresh(); return; }
  renderPanel(node);
  if (zoom) {
    const d = sigma.getNodeDisplayData(node);
    if (d) camera.animate({ x: d.x, y: d.y, ratio: Math.min(camera.ratio, 0.3) }, { duration: 600 });
  }
  refresh();
}

// ---------- Painel do nó selecionado ----------
function renderPanel(node) {
  const a = graph.getNodeAttributes(node);
  const get = (attr) => attrValue(node, attr);
  const panel = $("panel");
  panel.replaceChildren();
  panel.append(el("button", { className: "close", type: "button", title: "Fechar", ariaLabel: "Fechar",
    textContent: "×", onclick: () => select(null) }));
  const head = el("header");
  if (C.photoAttr && get(C.photoAttr)) {
    head.append(el("img", { className: "photo", src: get(C.photoAttr), alt: "", referrerPolicy: "no-referrer", loading: "lazy" }));
  }
  const sub = el("div", { className: "sub" },
    el("span", { className: "sw", style: `background:${a.color}` }),
    el("span", { textContent: a.sublabel || (state.colorAttr ? a.g : "") }));
  head.append(el("div", {}, el("h2", { textContent: a.label }), sub));
  panel.append(head);

  const dl = el("dl");
  if (state.vote != null) {
    const v = get(VOTE);
    dl.append(el("dt", { textContent: "Voto" }), el("dd", {}, el("span", { className: "sw", style: `background:${groupColorOf(VOTE, v)}` }), v));
  }
  for (const [attr, label] of Object.entries(C.infoAttrs)) {
    const dd = el("dd");
    const logo = C.logoAttr && C.logos[get(C.logoAttr)];
    if (attr === C.logoFor && logo) dd.append(el("img", { className: "logo", src: logo, alt: "", referrerPolicy: "no-referrer",
      onerror: (ev) => ev.target.remove() }));  // logo fora do ar: só o texto
    dd.append(fmt(get(attr)));
    dl.append(el("dt", { textContent: label }), dd);
  }
  panel.append(dl);
  if (C.linkAttr && get(C.linkAttr)) {
    panel.append(el("a", { href: get(C.linkAttr), target: "_blank", rel: "noopener", textContent: `${C.linkText} ↗` }));
  }

  const neigh = [];
  graph.forEachEdge(node, (e, ea, s, t) => {
    const o = s === node ? t : s;
    if (edgeShown(ea) && nodeVisible(o)) neigh.push([o, ea.value, ea.sig]);
  });
  neigh.sort((x, y) => y[1] - x[1]);
  panel.append(el("h3", { textContent: `Mais similares (${nf.format(neigh.length)} ${C.edgeNoun})` }));
  const ol = el("ol");
  for (const [o, v, sig] of neigh.slice(0, 15)) {
    const oa = graph.getNodeAttributes(o);
    ol.append(el("li", sig ? {} : { className: "weak", title: "Aresta não significativa" }, el("button", { type: "button", onclick: () => select(o, { zoom: true }) },
      el("span", { className: "sw", style: `background:${oa.color}` }),
      el("span", { className: "nm" }, el("span", { textContent: oa.label }), el("small", { textContent: oa.sublabel })),
      el("span", { className: "v", textContent: fmt(v) }))));
  }
  panel.append(ol);
  panel.hidden = false;
}

// ---------- Legenda (também é filtro) ----------
function renderLegend() {
  const box = $("legend");
  box.replaceChildren();
  if (!state.colorAttr) { box.hidden = true; return; }
  box.hidden = false;
  const info = groups[state.colorAttr];
  const counts = new Map();
  graph.forEachNode((n, a) => counts.set(a.g, (counts.get(a.g) || 0) + 1));
  const list = [...counts].map(([key, count]) => ({ key, count, slot: info.slot.get(key) ?? -1 }))
    .sort((a, b) => (info.rank.get(a.key) ?? 1e9) - (info.rank.get(b.key) ?? 1e9));
  // Rótulos numerados com o mesmo prefixo (ex.: "Comunidade 1, 2, 3"): em ordem numérica
  const numOf = (k) => /^(.*?)(\d+)$/.exec(k);
  const nums = list.filter((g) => g.key !== MISSING).map((g) => numOf(g.key));
  if (nums.length && nums.every((m) => m && m[1] === nums[0][1])) {
    const order = (g) => (g.key === MISSING ? Infinity : +numOf(g.key)[2]);
    list.sort((a, b) => order(a) - order(b));
  }
  box.append(el("h2", { textContent: attrLabel(state.colorAttr) }));
  if (state.colorAttr === VOTE) box.append(el("p", { className: "hint", textContent: voteTitle(state.vote) }));
  box.append(el("p", { className: "hint", textContent: "Clique para mostrar/ocultar" }));
  const setAll = (hide) => { state.hidden = new Set(hide ? list.map((g) => g.key) : []); afterFilter(); };
  box.append(el("div", { className: "actions" },
    el("button", { className: "linklike", type: "button", textContent: "Todos", onclick: () => setAll(false) }),
    el("button", { className: "linklike", type: "button", textContent: "Nenhum", onclick: () => setAll(true) })));
  const row = (g) => el("li", {}, el("button", {
    type: "button", ariaPressed: String(!state.hidden.has(g.key)),
    onclick: () => { state.hidden.has(g.key) ? state.hidden.delete(g.key) : state.hidden.add(g.key); afterFilter(); },
  }, el("span", { className: "sw", style: `background:${groupColor(g.key)}` }),
     el("span", { className: "name", textContent: g.key, title: g.key }),
     el("span", { className: "count", textContent: nf.format(g.count) })));
  // Com cores fixas, todo grupo tem a própria cor: lista completa, sem "Outros"
  const own = (g) => (fixedColors() ? fixedColors()[g.key] != null : g.slot >= 0);
  const main = list.filter(own), rest = list.filter((g) => !own(g));
  box.append(el("ul", {}, ...main.map(row)));
  if (rest.length) {
    const total = rest.reduce((s, g) => s + g.count, 0);
    const det = el("details", {}, el("summary", { textContent: `Outros: ${rest.length} grupos (${nf.format(total)})` }),
      el("ul", {}, ...rest.map(row)));
    det.open = rest.length <= 3;
    box.append(det);
  }
}

function afterFilter() {
  if (state.selected != null && !nodeVisible(state.selected)) state.selected = null;
  computeFocus();
  renderLegend();
  if (state.selected != null) renderPanel(state.selected); else $("panel").hidden = true;
  updateStats();
  refresh();
}

function updateStats() {
  $("emptyMsg").hidden = graph.order > 0;
  $("emptyMsg").textContent = `Sem ${C.nodeNoun} neste quadro.`;
  let nv = 0, ev = 0;
  graph.forEachNode((n) => { if (nodeVisible(n)) nv++; });
  graph.forEachEdge((e, a, s, t) => { if (edgeShown(a) && nodeVisible(s) && nodeVisible(t)) ev++; });
  const parts = [C.subtitle, TIMELINES[cur.tl].name, curFrame().label,
    `${nf.format(nv)} ${C.nodeNoun}`, `${nf.format(ev)} ${C.edgeNoun}`].filter(Boolean);
  $("subtitle").textContent = parts.join(" · ");
}

// ---------- Controles ----------
$("title").textContent = C.title;
document.title = C.title;
$("edgeLabel").textContent = C.edgeLabel;

const colorSel = $("colorBy");
function syncColorSelect() {
  const attrs = [...Object.keys(C.colorAttrs), ...(state.vote != null ? [VOTE] : [])];
  colorSel.replaceChildren(...attrs.map((a) => el("option", { value: a, textContent: attrLabel(a) })));
  colorSel.value = state.colorAttr;
  colorSel.closest("label").hidden = attrs.length < 2;
}
syncColorSelect();
colorSel.onchange = () => {
  state.colorAttr = colorSel.value; state.hidden = new Set();
  applyColors(); afterFilter();
};

const sizeSel = $("sizeBy");
for (const [attr, label] of Object.entries(C.sizeAttrs)) sizeSel.append(el("option", { value: attr, textContent: label }));
sizeSel.append(el("option", { value: "uniform", textContent: "Uniforme" }));
sizeSel.value = state.sizeAttr;
sizeSel.onchange = () => {
  state.sizeAttr = sizeSel.value; applySizes(); refresh();
  restartLayoutIfRunning(); // o worker lê os tamanhos só no início (usado por adjustSizes)
};

// Botão "Só significativas" (só existe se as arestas tiverem a marca)
if (C.edgeFlag) {
  const fb = $("flagBtn");
  fb.hidden = false;
  fb.textContent = C.edgeFlag.label;
  fb.title = C.edgeFlag.hint;
  fb.onclick = () => {
    state.onlyFlag = !state.onlyFlag;
    fb.setAttribute("aria-pressed", String(state.onlyFlag));
    afterFilter();
  };
}

const th = $("threshold");
if (!FRAMES.some((f) => f.t[1] > 0)) th.closest("label").hidden = true;
th.min = VLO; th.max = VHI; th.step = (VSPAN / 200) || 0.001; th.value = VLO;
const showThreshold = () => ($("thresholdOut").textContent = fmt(state.threshold));
let thTimer = 0;
th.oninput = () => {
  state.threshold = parseFloat(th.value); showThreshold();
  cancelAnimationFrame(thTimer);
  thTimer = requestAnimationFrame(() => afterFilter());
};
showThreshold();

$("zoomIn").onclick = () => camera.animatedZoom({ duration: 300 });
$("zoomOut").onclick = () => camera.animatedUnzoom({ duration: 300 });
$("zoomReset").onclick = () => { unfreezeView(); camera.animatedReset({ duration: 400 }); };

// Tela cheia (útil dentro do notebook)
$("fullBtn").onclick = () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.().catch(() => {});
};

// Tema: Auto → Claro → Escuro
const THEMES = [["auto", "◐", "Tema: automático"], ["light", "☀", "Tema: claro"], ["dark", "☾", "Tema: escuro"]];
let themeIdx = 0;
try { themeIdx = Math.max(0, THEMES.findIndex(([k]) => k === localStorage.getItem("graph-viz-theme"))); } catch {}
function setTheme() {
  const [key, icon, label] = THEMES[themeIdx];
  if (key === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = key;
  $("themeBtn").textContent = icon;
  $("themeBtn").title = label;
  $("themeBtn").setAttribute("aria-label", label);
  readTheme();
  sigma.setSetting("labelColor", { color: T.text });
}
function applyTheme() {
  setTheme();
  applyColors(); // cada tema tem seus tons (partidos, comunidades, votos)
  renderLegend();
  if (state.selected != null) renderPanel(state.selected);
  renderCompo();
  refresh();
}
$("themeBtn").onclick = () => {
  themeIdx = (themeIdx + 1) % THEMES.length;
  try { localStorage.setItem("graph-viz-theme", THEMES[themeIdx][0]); } catch {}
  applyTheme();
};
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (themeIdx === 0) applyTheme(); });

// ---------- Busca (nos nós do quadro atual) ----------
let searchIndex = [];
function rebuildSearch() {
  searchIndex = [];
  graph.forEachNode((n, a) => searchIndex.push({ n, key: norm(a.label), label: a.label, sub: a.sublabel }));
  $("search").placeholder = `Buscar ${C.nodeNoun}…`;
}
const input = $("search"), sug = $("suggest");
let matches = [], active = -1;
function renderSuggest() {
  sug.replaceChildren(...matches.map((m, i) => {
    const li = el("li", { role: "option", ariaSelected: String(i === active) },
      el("span", { className: "sw", style: `background:${graph.getNodeAttribute(m.n, "color")}` }),
      el("span", { textContent: m.label }), el("small", { textContent: m.sub }));
    li.onmousedown = (ev) => { ev.preventDefault(); pick(m); };
    return li;
  }));
  sug.hidden = matches.length === 0;
}
function pick(m) {
  input.value = m.label; sug.hidden = true; matches = [];
  if (!graph.hasNode(m.n)) return;
  if (!nodeVisible(m.n)) { state.hidden.delete(graph.getNodeAttribute(m.n, "g")); afterFilter(); }
  select(m.n, { zoom: true });
}
input.oninput = () => {
  const q = norm(input.value.trim());
  active = -1;
  if (!q) { matches = []; renderSuggest(); return; }
  matches = searchIndex
    .map((m) => ({ ...m, pos: m.key.indexOf(q) }))
    .filter((m) => m.pos >= 0)
    .sort((a, b) => (a.pos !== 0) - (b.pos !== 0) || a.key.localeCompare(b.key))
    .slice(0, 10);
  if (matches.length) active = 0;
  renderSuggest();
};
input.onkeydown = (ev) => {
  if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
    if (!matches.length) return;
    ev.preventDefault();
    active = (active + (ev.key === "ArrowDown" ? 1 : -1) + matches.length) % matches.length;
    renderSuggest();
  } else if (ev.key === "Enter" && matches[active]) { ev.preventDefault(); pick(matches[active]); }
  else if (ev.key === "Escape") { sug.hidden = true; }
};
input.onblur = () => setTimeout(() => (sug.hidden = true), 100);

// ---------- Layout ForceAtlas2 (Web Worker, com fallback na thread principal) ----------
const FA2_PARAMS = [
  { key: "gravity", label: "Gravidade", hint: "Puxa todos os nós para o centro.", min: 0.01, max: 20, log: true },
  { key: "scalingRatio", label: "Repulsão", hint: "Quanto os nós se empurram entre si.", min: 0.1, max: 200, log: true },
  { key: "slowDown", label: "Desaceleração", hint: "Maior = movimento mais lento e estável.", min: 0.5, max: 100, log: true },
  { key: "edgeWeightInfluence", label: "Influência da similaridade",
    hint: "0 = toda aresta puxa igual; maior = só as mais similares importam.", min: 0, max: 3, step: 0.05 },
  { key: "barnesHutTheta", label: "Precisão Barnes-Hut", hint: "Menor = repulsão mais exata e mais lenta.",
    min: 0.1, max: 2, step: 0.05, needs: "barnesHutOptimize" },
  { key: "strongGravityMode", label: "Gravidade forte", hint: "A gravidade cresce com a distância ao centro." },
  { key: "linLogMode", label: "Modo LinLog", hint: "Atração logarítmica: comunidades mais separadas e compactas." },
  { key: "outboundAttractionDistribution", label: "Distribuir atração", hint: "Divide a atração pelo grau: hubs vão para a periferia." },
  { key: "adjustSizes", label: "Evitar sobreposição", hint: "Considera o tamanho dos nós." },
  { key: "barnesHutOptimize", label: "Barnes-Hut", hint: "Aproxima a repulsão em O(n log n); desligado = exata, O(n²)." },
];
const fa2Defaults = {
  gravity: 0.5, scalingRatio: 10, slowDown: 5, edgeWeightInfluence: 1, barnesHutTheta: 0.5,
  strongGravityMode: true, linLogMode: false, outboundAttractionDistribution: false, adjustSizes: false,
  barnesHutOptimize: Math.max(...FRAMES.map((f) => f.n[1])) > 300,
  ...C.fa2,
};
const fa2Settings = { ...fa2Defaults };
let layout = null, rafId = 0, stopTimer = 0, useWorker = true, autoStop = false;
let stopAfter = 0; // segundos de layout pedidos; o relógio só começa na 1ª iteração aplicada
let iterations = 0, lastIterations = 0, iterRate = 0;
const layoutRunning = () => (layout ? layout.isRunning() : rafId !== 0);
// O ForceAtlas2 grava x/y sem "hints"; as atualizações desta página sempre passam hints.
graph.on("eachNodeAttributesUpdated", (e) => {
  if (e.hints || !layoutRunning()) return;
  iterations++;
  // Num computador lento o primeiro desenho pode ocupar a página por segundos: contar o tempo a
  // partir da 1ª iteração garante que "rodar 4 s" seja de fato 4 s de layout.
  if (stopAfter) { stopTimer = setTimeout(stopLayout, stopAfter * 1000); stopAfter = 0; }
});
const keepDragged = (node, attr) =>
  (drag && drag.x != null && node === drag.node ? { ...attr, x: drag.x, y: drag.y } : attr);

function startLayout(seconds = 0) {
  clearTimeout(stopTimer);
  if (!layoutRunning() && graph.size > 0) {
    if (!drag) unfreezeView(); // no meio de um arraste o enquadramento continua congelado
    if (useWorker) {
      // Um worker novo a cada início: respostas atrasadas de uma execução anterior não criam um 2º loop.
      // O outputReducer mantém o nó arrastado sob o mouse; como ele existe, o supervisor relê as
      // posições do grafo a cada iteração e os vizinhos reagem ao arraste.
      try {
        layout = new FA2Layout(graph, { settings: fa2Settings, getEdgeWeight: "weight", outputReducer: keepDragged });
        layout.start();
      } catch (err) { console.warn("Web Worker indisponível; layout na thread principal.", err); useWorker = false; layout = null; }
    }
    if (!useWorker) {
      const step = () => {
        forceAtlas2.assign(graph, { iterations: 1, settings: fa2Settings, getEdgeWeight: "weight" });
        if (drag && drag.x != null && graph.hasNode(drag.node)) graph.mergeNodeAttributes(drag.node, { x: drag.x, y: drag.y });
        rafId = requestAnimationFrame(step);
      };
      rafId = requestAnimationFrame(step);
    }
  }
  autoStop = seconds > 0;
  stopAfter = autoStop ? seconds : 0;
  updateLayoutUI();
}
function stopLayout() {
  clearTimeout(stopTimer);
  autoStop = false;
  stopAfter = 0;
  if (layout) { layout.kill(); layout = null; }
  cancelAnimationFrame(rafId); rafId = 0;
  updateLayoutUI();
}
function restartLayoutIfRunning() {
  if (!layoutRunning()) return;
  const keepAuto = autoStop;
  stopLayout();
  startLayout(keepAuto ? C.transitionSeconds : 0);
}

// Muda um parâmetro ao vivo: o supervisor reenvia `settings` ao worker a cada iteração.
function setFa2(key, value) {
  fa2Settings[key] = value;
  if (layout) layout.settings[key] = value;
  if (layoutRunning()) { clearTimeout(stopTimer); autoStop = false; stopAfter = 0; } // explorando: não para sozinho
  else startLayout();
  updateLayoutUI();
}
function resetFa2() {
  Object.assign(fa2Settings, fa2Defaults);
  if (layout) Object.assign(layout.settings, fa2Defaults);
  syncParams();
  updateLayoutUI();
}
function restorePositions() {
  const wasRunning = layoutRunning();
  stopLayout();
  graph.updateEachNodeAttributes((n, a) => ({ ...a, x: NODES[+n].x, y: NODES[+n].y }), { attributes: ["x", "y"] });
  iterations = lastIterations = iterRate = 0;
  unfreezeView();
  camera.animatedReset({ duration: 300 });
  if (wasRunning) startLayout();
  updateLayoutUI();
}

// ---------- Painel de parâmetros ----------
const STEPS = 1000; // sliders logarítmicos vão de 0 a STEPS
const fmtParam = (v) => v.toLocaleString("pt-BR", { maximumSignificantDigits: 3 });
const toSlider = (p, v) => (p.log ? (STEPS * Math.log(Math.max(v, p.min) / p.min)) / Math.log(p.max / p.min) : v);
const fromSlider = (p, t) => (p.log ? Number((p.min * Math.pow(p.max / p.min, t / STEPS)).toPrecision(2)) : Number(t));
const paramInputs = [];
const fa2Code = () => "fa2_settings={" + FA2_PARAMS.map((p) => {
  const v = fa2Settings[p.key];
  return `'${p.key}': ${typeof v === "boolean" ? (v ? "True" : "False") : String(v)}`;
}).join(", ") + "}";

function buildParams() {
  const box = $("params");
  const copyBtn = el("button", { className: "btn", type: "button", textContent: "Copiar" });
  const pre = el("pre", { id: "fa2Code" });
  copyBtn.onclick = () => copyCode(pre, copyBtn);
  box.append(
    el("div", { className: "head" }, el("h2", { textContent: "Layout · ForceAtlas2" }),
      el("button", { className: "close", type: "button", title: "Fechar", ariaLabel: "Fechar", textContent: "×",
        onclick: () => toggleParams(false) })),
    el("p", { className: "status", id: "fa2Status" }),
    el("div", { className: "btns" },
      el("button", { className: "btn", type: "button", id: "fa2Run", onclick: () => (layoutRunning() ? stopLayout() : startLayout()) }),
      el("button", { className: "btn", type: "button", textContent: "↺ Posições iniciais",
        title: "Volta ao layout calculado no Python", onclick: restorePositions }),
      el("button", { className: "btn", type: "button", textContent: "Padrões",
        title: "Volta aos parâmetros com que a página abriu", onclick: resetFa2 })),
    el("span", { className: "hint", textContent: "Mudar um parâmetro põe o layout para rodar. "
      + "Arraste um nó para reposicioná-lo; com o layout rodando, os vizinhos acompanham." }));
  for (const p of FA2_PARAMS) {
    const code = el("code", { textContent: p.key });
    if (typeof fa2Defaults[p.key] === "boolean") {
      const input = el("input", { type: "checkbox" });
      input.onchange = () => setFa2(p.key, input.checked);
      paramInputs.push({ p, input });
      box.append(el("label", { className: "check" }, input, el("span", {},
        el("span", { className: "lbl", textContent: p.label }), " ", code, el("span", { className: "hint", textContent: p.hint }))));
    } else {
      const out = el("output");
      const input = el("input", { type: "range", min: p.log ? 0 : p.min, max: p.log ? STEPS : p.max, step: p.log ? 1 : p.step });
      input.setAttribute("aria-label", p.label);
      input.oninput = () => { const v = fromSlider(p, input.value); out.textContent = fmtParam(v); setFa2(p.key, v); };
      const row = el("div", { className: "row" },
        el("div", { className: "row-head" }, el("span", {}, el("span", { className: "lbl", textContent: p.label }), " ", code), out),
        input, el("span", { className: "hint", textContent: p.hint }));
      paramInputs.push({ p, input, out, row });
      box.append(row);
    }
  }
  box.append(el("div", { className: "code-head" }, el("span", { textContent: "Para usar no graph_viz.show():" }), copyBtn), pre);
  syncParams();
}
// Põe os controles nos valores atuais (não é chamado durante o arraste, para o slider não "pular").
function syncParams() {
  for (const { p, input, out } of paramInputs) {
    const v = fa2Settings[p.key];
    if (input.type === "checkbox") input.checked = v;
    else { input.value = toSlider(p, v); out.textContent = fmtParam(v); }
  }
}
function updateLayoutUI() {
  const on = layoutRunning();
  $("layoutBtn").textContent = on ? "❚❚ Layout" : "▶ Layout";  // mesma largura nos dois estados
  $("layoutBtn").title = on ? "Pausar o layout" : "Rodar o layout";
  $("layoutBtn").setAttribute("aria-pressed", String(on));
  if (!$("fa2Run")) return;
  $("fa2Run").textContent = on ? "❚❚ Pausar" : "▶ Rodar";
  $("fa2Status").textContent = `${on ? "Rodando" : "Parado"} · ${nf.format(iterations)} iterações`
    + (on ? ` · ${nf.format(iterRate)}/s` : "");
  $("fa2Code").textContent = fa2Code();
  for (const { p, input, row } of paramInputs) {
    if (!p.needs) continue;
    const enabled = fa2Settings[p.needs];
    input.disabled = !enabled;
    row.classList.toggle("disabled", !enabled);
  }
}
async function copyCode(pre, btn) {
  let ok = false;
  try { await navigator.clipboard.writeText(fa2Code()); ok = true; } catch {}
  if (!ok) { // fallback: seleciona o texto (e tenta o comando antigo de cópia)
    const range = document.createRange(); range.selectNodeContents(pre);
    const sel = getSelection(); sel.removeAllRanges(); sel.addRange(range);
    try { ok = document.execCommand("copy"); } catch {}
  }
  btn.textContent = ok ? "Copiado ✓" : "Selecionado: Ctrl+C";
  setTimeout(() => (btn.textContent = "Copiar"), 1800);
}
function updateSideClass() { // com parâmetros ou votações abertos, a legenda encolhe e rola
  $("params").parentElement.classList.toggle("with-params", !$("params").hidden || !$("votes").hidden);
}
function toggleParams(show) {
  $("params").hidden = !show;
  updateSideClass();
  $("paramsBtn").setAttribute("aria-pressed", String(show));
  updateLayoutUI();
}
setInterval(() => {
  iterRate = iterations - lastIterations;
  lastIterations = iterations;
  if (!$("params").hidden) updateLayoutUI();
}, 1000);
$("layoutBtn").onclick = () => (layoutRunning() ? stopLayout() : startLayout());
$("paramsBtn").onclick = () => toggleParams($("params").hidden);
buildParams();

// ---------- Lista (tabela) ----------
const cols = [{ key: "__label", label: "Nome", num: false },
  ...Object.entries(C.infoAttrs).map(([key, label]) => ({ key, label, num: key in C.sizeAttrs }))];
let sortKey = "__label", sortDir = 1;
function renderTable() {
  const box = $("tableView");
  const val = (n, k) => (k === "__label" ? graph.getNodeAttribute(n, "label") : attrValue(n, k));
  const rows = graph.nodes().filter(nodeVisible).sort((a, b) => {
    const x = val(a, sortKey), y = val(b, sortKey);
    if (x == null) return 1; if (y == null) return -1;
    return sortDir * (typeof x === "number" && typeof y === "number" ? x - y
      : String(x).localeCompare(String(y), "pt-BR", { numeric: true }));
  });
  const thead = el("tr", {}, ...cols.map((c) => el("th", {
    scope: "col", textContent: c.label,
    ariaSort: c.key === sortKey ? (sortDir > 0 ? "ascending" : "descending") : "none",
    onclick: () => { sortDir = c.key === sortKey ? -sortDir : (c.num ? -1 : 1); sortKey = c.key; renderTable(); },
  })));
  const tbody = el("tbody", {}, ...rows.map((n) => el("tr", {
    onclick: () => { toggleTable(false); select(n, { zoom: true }); },
  }, ...cols.map((c) => el("td", { className: c.num ? "num" : "", textContent: fmt(val(n, c.key)) })))));
  box.replaceChildren(el("table", {}, el("thead", {}, thead), tbody));
}
function toggleTable(show) {
  $("tableView").hidden = !show;
  $("tableBtn").setAttribute("aria-pressed", String(show));
  if (show) renderTable();
}
$("tableBtn").onclick = () => toggleTable($("tableView").hidden);


// ---------- Composição: como uma categoria (ex.: partidos) se distribui em cada grupo (ex.: comunidades) ----------
// Barras empilhadas por grupo; clicar num trecho destaca esses nós no grafo. O destaque é um critério
// (ex.: "PL na Comunidade 3"), então continua valendo ao trocar de período ou de tema.
const CATS = Object.keys(C.colorAttrs);
const compo = { open: false, mode: "share", view: "bars" };
compo.rows = CATS.includes("Comunidade") ? "Comunidade" : CATS[0];
compo.segs = CATS.includes("Partido") && compo.rows !== "Partido" ? "Partido" : CATS.find((a) => a !== compo.rows);
const COMPO_DEFAULT = { rows: compo.rows, segs: compo.segs };
$("compoBtn").hidden = CATS.length < 2;
const FOLD = 0.025; // trechos com menos de 2,5% do grupo vão para "Outros" (detalhes no tooltip e na tabela)
const catOf = (n, attr) => { const v = attrValue(n, attr); return v == null ? MISSING : String(v); };
const pct = (x) => `${(x * 100).toLocaleString("pt-BR", { maximumFractionDigits: x < 0.1 ? 1 : 0 })}%`;
const rankOf = (attr, key) => groups[attr]?.rank.get(key) ?? 1e9;

function computeHighlight() {
  const h = state.highlight;
  if (!h) { highlightSet = null; return; }
  highlightSet = new Set();
  graph.forEachNode((n) => {
    if (catOf(n, h.rows) === h.rowKey && (!h.segKeys || h.segKeys.has(catOf(n, h.segs)))) highlightSet.add(n);
  });
}
const sameKeys = (a, b) => (!a && !b) || (a && b && a.size === b.size && [...a].every((k) => b.has(k)));
const isHighlighted = (rowKey, segKeys) => {
  const h = state.highlight;
  return !!h && h.rows === compo.rows && h.rowKey === rowKey && (segKeys ? h.segs === compo.segs : true) && sameKeys(h.segKeys, segKeys);
};
function setHighlight(h) {
  if (h && isHighlighted(h.rowKey, h.segKeys)) h = null; // clicar de novo desfaz
  state.highlight = h;
  computeHighlight();
  renderCompo();
  refresh();
}

function crosstab() {
  const rows = new Map(), segTot = new Map();
  graph.forEachNode((n) => {
    const r = catOf(n, compo.rows), k = catOf(n, compo.segs);
    if (!rows.has(r)) rows.set(r, { key: r, total: 0, segs: new Map() });
    const row = rows.get(r);
    row.total++;
    row.segs.set(k, (row.segs.get(k) || 0) + 1);
    segTot.set(k, (segTot.get(k) || 0) + 1);
  });
  const rowList = [...rows.values()].sort((a, b) => rankOf(compo.rows, a.key) - rankOf(compo.rows, b.key) || b.total - a.total);
  const segOrder = [...segTot.keys()].sort((a, b) => rankOf(compo.segs, a) - rankOf(compo.segs, b) || segTot.get(b) - segTot.get(a));
  return { rowList, segOrder, segTot };
}

function showTip(target, strong, lines) {
  const tip = $("tip"), stage = $("stage").getBoundingClientRect(), r = target.getBoundingClientRect();
  tip.replaceChildren(el("strong", { textContent: strong }), ...lines.map((t) => el("span", { textContent: t })));
  tip.hidden = false;
  const x = Math.min(Math.max(r.left + r.width / 2 - stage.left - tip.offsetWidth / 2, 8), stage.width - tip.offsetWidth - 8);
  const above = r.top - stage.top - tip.offsetHeight - 8;
  tip.style.left = `${x}px`;
  tip.style.top = `${above >= 8 ? above : r.bottom - stage.top + 8}px`;
}
const hideTip = () => { $("tip").hidden = true; };
function withTip(node, strong, lines) {
  node.onmouseenter = node.onfocus = () => showTip(node, strong, lines);
  node.onmouseleave = node.onblur = hideTip;
  return node;
}
function commonPrefix(keys) { // "Comunidade 1", "Comunidade 2" -> "Comunidade "
  if (keys.length < 2) return "";
  let p = keys[0];
  for (const k of keys) while (!k.startsWith(p)) p = p.slice(0, -1);
  return p.endsWith(" ") ? p : "";
}

function renderCompo() {
  const box = $("compo");
  hideTip();
  if (!compo.open) { box.hidden = true; updateRightSide(); return; }
  box.hidden = false;
  const { rowList, segOrder, segTot } = crosstab();
  const pick = (value, label, onchange) => {
    const sel = el("select");
    sel.setAttribute("aria-label", label);
    for (const a of [...CATS, ...(state.vote != null ? [VOTE] : [])]) sel.append(el("option", { value: a, textContent: attrLabel(a) }));
    sel.value = value;
    sel.onchange = () => onchange(sel.value);
    return sel;
  };
  const toggle = (options, value, onpick) => el("div", { className: "seg-toggle", role: "group" },
    ...options.map(([v, label]) => {
      const b = el("button", { type: "button", textContent: label, onclick: () => onpick(v) });
      b.setAttribute("aria-pressed", String(v === value));
      return b;
    }));
  box.replaceChildren(
    el("div", { className: "head" }, el("h2", { textContent: "Composição" }),
      el("button", { className: "close", type: "button", title: "Fechar", ariaLabel: "Fechar", textContent: "×",
        onclick: () => toggleCompo(false) })),
    el("div", { className: "ctrls" },
      pick(compo.rows, "Grupos", (v) => { if (v === compo.segs) compo.segs = compo.rows; compo.rows = v; renderCompo(); }),
      el("span", { textContent: "por" }),
      pick(compo.segs, "Categoria", (v) => { if (v === compo.rows) compo.rows = compo.segs; compo.segs = v; renderCompo(); }),
      el("button", { className: "btn icon", type: "button", title: "Inverter", ariaLabel: "Inverter", textContent: "⇄",
        onclick: () => { [compo.rows, compo.segs] = [compo.segs, compo.rows]; renderCompo(); } })),
    el("div", { className: "ctrls" },
      toggle([["bars", "Barras"], ["table", "Tabela"]], compo.view, (v) => { compo.view = v; renderCompo(); }),
      compo.view === "bars"
        ? toggle([["share", "% do grupo"], ["count", "Nº"]], compo.mode, (v) => { compo.mode = v; renderCompo(); }) : null),
    el("p", { className: "hint", textContent: "Clique num trecho para destacar esses " + C.nodeNoun + " no grafo." }));

  if (state.highlight) {
    const h = state.highlight, names = [...(highlightSet || [])].sort((a, b) => graph.degree(b) - graph.degree(a));
    const what = h.segKeys ? `${[...h.segKeys].join(", ")} em ${h.rowKey}` : h.rowKey;
    box.append(el("div", { className: "hl" },
      el("div", {}, el("strong", { textContent: `${nf.format(names.length)} ${C.nodeNoun}` }), ` · ${what} · `,
        el("button", { className: "linklike", type: "button", textContent: "limpar destaque", onclick: () => setHighlight(null) })),
      el("div", { className: "names" },
        ...names.slice(0, 60).map((n) => el("button", { type: "button", textContent: graph.getNodeAttribute(n, "label"),
          title: graph.getNodeAttribute(n, "sublabel"), onclick: () => select(n, { zoom: true }) })),
        names.length > 60 ? el("span", { textContent: `+${nf.format(names.length - 60)}` }) : null)));
  }

  if (compo.view === "bars") {
    const max = Math.max(1, ...rowList.map((r) => r.total));
    for (const row of rowList) {
      const label = el("button", { className: "clabel" + (isHighlighted(row.key, null) ? " on" : ""), type: "button",
        onclick: () => setHighlight({ rows: compo.rows, rowKey: row.key, segs: compo.segs, segKeys: null }) },
        el("span", { className: "sw", style: `background:${groupColorOf(compo.rows, row.key)}` }),
        el("span", { className: "name", textContent: row.key }),
        el("span", { className: "n", textContent: `${nf.format(row.total)} ${C.nodeNoun}` }));
      const segs = segOrder.filter((k) => row.segs.get(k)).map((k) => ({ keys: [k], n: row.segs.get(k) }));
      const main = segs.filter((g) => g.n / row.total >= FOLD), small = segs.filter((g) => g.n / row.total < FOLD);
      if (small.length === 1) main.push(small.pop());
      if (small.length) main.push({ keys: small.flatMap((g) => g.keys), n: small.reduce((t, g) => t + g.n, 0), other: true });
      const bar = el("div", { className: "cbar" });
      bar.style.width = compo.mode === "count" ? `${(100 * row.total) / max}%` : "100%";
      for (const g of main) {
        const name = g.other ? `Outros (${g.keys.length})` : g.keys[0];
        const keys = new Set(g.keys);
        const inCat = g.keys.reduce((t, k) => t + segTot.get(k), 0);
        const b = el("button", { type: "button", className: "seg" + (isHighlighted(row.key, keys) ? " on" : ""),
          onclick: () => setHighlight({ rows: compo.rows, rowKey: row.key, segs: compo.segs, segKeys: keys }) });
        b.style.flex = `${g.n} 1 0`;
        b.style.background = g.other ? T.other : groupColorOf(compo.segs, g.keys[0]);
        b.setAttribute("aria-label", `${name}: ${g.n} ${C.nodeNoun}, ${pct(g.n / row.total)} de ${row.key}`);
        bar.append(withTip(b, `${nf.format(g.n)} ${C.nodeNoun}`, [
          `${name} · ${pct(g.n / row.total)} de ${row.key}`,
          g.other ? g.keys.map((k) => `${k} ${row.segs.get(k)}`).join(", ") : `${pct(g.n / inCat)} dos ${C.nodeNoun} de ${name}`]));
      }
      const top = main.filter((g) => !g.other).sort((a, b) => b.n - a.n).slice(0, 4); // maiores primeiro
      const text = top.map((g) => `${g.keys[0]} ${pct(g.n / row.total)}`).join(" · ")
        + (segs.length > top.length ? ` · +${segs.length - top.length}` : "");
      box.append(el("div", { className: "crow" }, label, bar, el("div", { className: "ctext", textContent: text })));
    }
  } else {
    // Tabela: contagem e % do grupo em cada célula. As linhas são o atributo com mais valores
    // (normalmente os partidos), para a tabela caber na largura do painel.
    const byGroup = new Map(rowList.map((r) => [r.key, r]));
    const groupsAsCols = segOrder.length >= rowList.length;
    const colKeys = groupsAsCols ? rowList.map((r) => r.key) : segOrder;
    const lineKeys = groupsAsCols ? segOrder : rowList.map((r) => r.key);
    const colAttr = groupsAsCols ? compo.rows : compo.segs, lineAttr = groupsAsCols ? compo.segs : compo.rows;
    const prefix = commonPrefix(colKeys);
    const swatch = (attr, k, gap) => el("span", { className: "sw", style: `display:inline-block;margin-right:${gap}px;background:${groupColorOf(attr, k)}` });
    const head = el("tr", {}, el("th", { textContent: attrLabel(lineAttr) }),
      ...colKeys.map((k) => el("th", { title: k }, swatch(colAttr, k, 4), k.slice(prefix.length))),
      el("th", { textContent: "Total" }));
    const body = lineKeys.map((line) => el("tr", {},
      el("td", {}, swatch(lineAttr, line, 6), line),
      ...colKeys.map((col) => {
        const groupKey = groupsAsCols ? col : line, segKey = groupsAsCols ? line : col;
        const group = byGroup.get(groupKey), n = group.segs.get(segKey) || 0;
        const td = el("td", { className: "cell" + (isHighlighted(groupKey, new Set([segKey])) ? " on" : ""),
          title: n ? `${segKey} em ${groupKey}: ${n} (${pct(n / group.total)} de ${groupKey}, ${pct(n / segTot.get(segKey))} dos ${C.nodeNoun} de ${segKey})` : "" },
          n ? String(n) : "·", n ? el("small", { textContent: ` ${pct(n / group.total)}` }) : null);
        if (n) td.onclick = () => setHighlight({ rows: compo.rows, rowKey: groupKey, segs: compo.segs, segKeys: new Set([segKey]) });
        return td;
      }),
      el("td", { textContent: nf.format(groupsAsCols ? segTot.get(line) : byGroup.get(line).total) })));
    const foot = el("tr", {}, el("td", { textContent: "Total" }),
      ...colKeys.map((col) => el("td", { textContent: nf.format(groupsAsCols ? byGroup.get(col).total : segTot.get(col)) })),
      el("td", { textContent: nf.format(graph.order) }));
    box.append(el("table", {}, el("thead", {}, head), el("tbody", {}, ...body), el("tfoot", {}, foot)));
  }
  updateRightSide();
}
function toggleCompo(show) {
  compo.open = show;
  $("compoBtn").setAttribute("aria-pressed", String(show));
  renderCompo();
}
function updateRightSide() {
  $("stage").classList.toggle("right-open", !$("panel").hidden || !$("compo").hidden);
}
$("compoBtn").onclick = () => toggleCompo(!compo.open);

// ---------- Votações: lista do quadro atual, informações da proposição e voto de cada deputado ----------
const votesUI = { open: false, query: "", expanded: false };
let prevColorAttr = null;
$("votesBtn").hidden = !VOTES;
const VOTE_TEXT = VOTES ? VOTES.meta.map((m) => norm([m.titulo, m.ementa, m.descricao, ...(m.temas || [])].filter(Boolean).join(" "))) : [];
const frameVotes = () => (VOTES && curFrame().v ? Array.from(view(Uint16Array, curFrame().v)) : []);
const fmtDate = (iso) => { if (!iso) return ""; const [y, m, d] = String(iso).slice(0, 10).split("-"); return `${d}/${m}/${y}`; };
const voteTitle = (v) => { const m = VOTES.meta[v]; return [m.titulo || m.id, fmtDate(m.data)].filter(Boolean).join(" · "); };
const resultOf = (m) => (m.aprovacao == null ? null : m.aprovacao ? ["ok", "✓ Aprovada"] : ["no", "✗ Rejeitada"]);

function selectVote(v) {
  if (state.vote === v) return clearVote(); // clicar de novo desfaz
  if (state.colorAttr !== VOTE) prevColorAttr = state.colorAttr;
  state.vote = v;
  votesUI.expanded = false;
  state.colorAttr = VOTE;
  state.hidden = new Set();
  refreshVoteViews();
}
function dropVote() { // desfaz a seleção (sem redesenhar)
  state.vote = null;
  if (state.colorAttr === VOTE) { state.colorAttr = prevColorAttr ?? CATS[0] ?? null; state.hidden = new Set(); }
  if (state.highlight && (state.highlight.rows === VOTE || state.highlight.segs === VOTE)) state.highlight = null;
  if (compo.rows === VOTE || compo.segs === VOTE) Object.assign(compo, COMPO_DEFAULT);
}
function clearVote() { dropVote(); refreshVoteViews(); }
function refreshVoteViews() {
  syncColorSelect();
  applyColors();
  computeHighlight();
  afterFilter();
  renderVotes();
  renderCompo();
}
function showPartyVotes() { // composição: uma barra por partido, dividida pelo voto
  compo.rows = CATS.includes("Partido") ? "Partido" : CATS[0];
  compo.segs = VOTE;
  compo.view = "bars";
  compo.mode = "share";
  toggleCompo(true);
}

// O esqueleto do painel é criado uma vez (a busca não perde o foco); detalhes e lista são refeitos.
const votesBox = $("votes");
const votesCount = el("span", { className: "count" });
const votesSearch = el("input", { type: "search", placeholder: "Buscar proposição, ementa, tema…", autocomplete: "off" });
votesSearch.setAttribute("aria-label", "Buscar votação");
const votesSel = el("div", { className: "vsel" });
const votesList = el("ol", { className: "vlist" });
votesBox.append(
  el("div", { className: "head" }, el("h2", {}, "Votações ", votesCount),
    el("button", { className: "close", type: "button", title: "Fechar", ariaLabel: "Fechar", textContent: "×", onclick: () => toggleVotes(false) })),
  votesSearch, votesSel, votesList);
let searchTimer = 0;
votesSearch.oninput = () => {
  votesUI.query = norm(votesSearch.value.trim());
  clearTimeout(searchTimer); // espera a digitação parar (não depende do próximo quadro desenhado)
  searchTimer = setTimeout(renderVoteList, 120);
};

function renderVotes() {
  if (!VOTES || !votesUI.open) { votesBox.hidden = true; updateSideClass(); return; }
  votesBox.hidden = false;
  updateSideClass();
  renderVoteDetails();
  renderVoteList();
}
function renderVoteDetails() {
  votesSel.replaceChildren();
  votesSel.hidden = state.vote == null;
  if (state.vote == null) return;
  const m = VOTES.meta[state.vote], res = resultOf(m);
  const long = (m.descricao || "").length + (m.ementa || "").length > 240;
  const tally = new Map();
  graph.forEachNode((n) => { const v = attrValue(n, VOTE); tally.set(v, (tally.get(v) || 0) + 1); });
  const order = [...VOTE_LABELS.slice(1), NO_VOTE];
  votesSel.append(...[
    el("div", { className: "vhead" }, el("strong", { textContent: m.titulo || m.id }),
      el("button", { className: "close", type: "button", title: "Limpar seleção", ariaLabel: "Limpar seleção", textContent: "×", onclick: clearVote })),
    el("div", { className: "vmeta" }, el("span", { textContent: [fmtDate(m.data), m.orgao].filter(Boolean).join(" · ") }),
      res ? el("span", { className: `vres ${res[0]}`, textContent: res[1] }) : null),
    el("div", { className: "vtexts" + (long && !votesUI.expanded ? " clamp" : "") },
      m.descricao ? el("p", { textContent: m.descricao }) : null,
      m.ementa ? el("p", { className: "vementa", textContent: m.ementa }) : null),
    long ? el("button", { className: "linklike", type: "button", textContent: votesUI.expanded ? "ver menos" : "ver ementa completa",
      onclick: () => { votesUI.expanded = !votesUI.expanded; renderVoteDetails(); } }) : null,
    el("div", { className: "vcounts", title: `Votos dos ${C.nodeNoun} deste grafo` },
      ...order.filter((k) => tally.get(k)).map((k) => el("span", {},
        el("span", { className: "sw", style: `background:${groupColorOf(VOTE, k)}` }), `${k} ${nf.format(tally.get(k))}`))),
    m.sim != null ? el("div", { className: "vofficial", textContent:
      `Placar oficial: Sim ${nf.format(m.sim)} · Não ${nf.format(m.nao ?? 0)} · Outros ${nf.format(m.outros ?? 0)}` }) : null,
    m.temas && m.temas.length ? el("div", {}, ...m.temas.map((t) => el("span", { className: "chip", textContent: t }))) : null,
    el("div", { className: "vactions" },
      m.url ? el("a", { href: m.url, target: "_blank", rel: "noopener", textContent: "Proposição na Câmara ↗" }) : null,
      CATS.length ? el("button", { className: "linklike", type: "button", textContent: "Como votou cada partido", onclick: showPartyVotes }) : null),
  ].filter((x) => x != null));  // append(null) escreveria o texto "null"
}
function renderVoteList() {
  const ids = frameVotes().sort((a, b) => String(VOTES.meta[b].data || "").localeCompare(String(VOTES.meta[a].data || "")));
  const shown = votesUI.query ? ids.filter((v) => VOTE_TEXT[v].includes(votesUI.query)) : ids;
  votesCount.textContent = votesUI.query ? `(${nf.format(shown.length)} de ${nf.format(ids.length)})` : `(${nf.format(ids.length)})`;
  if (!shown.length) {
    votesList.replaceChildren(el("li", { className: "vempty", textContent: ids.length ? "Nenhuma votação encontrada." : "Sem votações neste quadro." }));
    return;
  }
  votesList.replaceChildren(...shown.map((v) => {
    const m = VOTES.meta[v], res = resultOf(m);
    const b = el("button", { type: "button", className: "vitem" + (v === state.vote ? " on" : ""), onclick: () => selectVote(v) },
      el("span", { className: "vmeta" }, el("span", { textContent: [fmtDate(m.data), m.orgao].filter(Boolean).join(" · ") }),
        res ? el("span", { className: `vres ${res[0]}`, textContent: res[1] }) : null),
      el("span", { className: "vtitle", textContent: m.titulo || m.id }),
      m.descricao ? el("span", { className: "vdesc", textContent: m.descricao }) : null);
    b.setAttribute("aria-pressed", String(v === state.vote));
    return el("li", {}, b);
  }));
}
function toggleVotes(show) {
  votesUI.open = show;
  $("votesBtn").setAttribute("aria-pressed", String(show));
  renderVotes();
  if (show) votesSearch.focus({ preventScroll: true });
}
$("votesBtn").onclick = () => toggleVotes(!votesUI.open);

// ---------- Linha do tempo ----------
// Ao trocar de quadro: o grafo muda e o ForceAtlas2 continua das posições atuais por
// `transitionSeconds` (ou sem parar, se o layout já estava rodando de forma contínua).
function afterFrame() {
  state.hovered = null;
  if (state.selected != null && !graph.hasNode(state.selected)) state.selected = null;
  if (state.vote != null && !frameVotes().includes(state.vote)) { dropVote(); syncColorSelect(); }
  applyColors();
  applySizes();
  computeFocus();
  renderLegend();
  if (state.selected != null) renderPanel(state.selected); else $("panel").hidden = true;
  updateStats();
  rebuildSearch();
  updateTimelineUI();
  if (!$("tableView").hidden) renderTable();
  computeHighlight();
  renderCompo();
  renderVotes();
  refresh();
}
function go(tl, pos) {
  const frames = TIMELINES[tl].frames;
  pos = Math.max(0, Math.min(frames.length - 1, pos));
  if (tl === cur.tl && pos === cur.pos) return;
  const continuous = layoutRunning() && !autoStop;
  endDrag();
  stopLayout();
  unfreezeView();
  loadFrame(tl, pos);
  lastPos.set(tl, pos);
  afterFrame();
  if (continuous) startLayout();
  else if (C.transitionSeconds > 0) startLayout(C.transitionSeconds);
}

// Datas dos quadros (atributos de grafo `inicio`/`fim`, fim inclusive)
const DAY = 864e5, MONTHS = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"];
const dayOf = (iso) => Date.UTC(+iso.slice(0, 4), +iso.slice(5, 7) - 1, +iso.slice(8, 10));
function spanText(M) {
  if (!M.t0 || !M.t1) return "";
  const a = new Date(dayOf(M.t0)), b = new Date(dayOf(M.t1)), next = new Date(dayOf(M.t1) + DAY);
  const whole = a.getUTCDate() === 1 && next.getUTCDate() === 1;  // meses inteiros: "fev/2023–jan/2024"
  const f = (d) => whole ? `${MONTHS[d.getUTCMonth()]}/${d.getUTCFullYear()}`
    : `${String(d.getUTCDate()).padStart(2, "0")}/${String(d.getUTCMonth() + 1).padStart(2, "0")}/${d.getUTCFullYear()}`;
  return `${f(a)}–${f(b)}`;
}
const frameInfo = (M) => [spanText(M), ...Object.entries(M.info).map(([k, v]) =>
  typeof v === "number" && Number.isInteger(v) ? `${nf.format(v)} ${k}` : `${k} ${fmt(v)}`)].filter(Boolean).join(" · ");

// Intervalo [início, fim) de cada quadro; sem datas, a ordem em que os rótulos dos períodos aparecem
const PERIOD_ORDER = new Map();
for (const t of TIMELINES) for (const fi of t.frames) {
  if (!PERIOD_ORDER.has(FRAMES[fi].label)) PERIOD_ORDER.set(FRAMES[fi].label, PERIOD_ORDER.size);
}
function spanOf(fi) {
  const M = FRAMES[fi];
  if (M.t0 && M.t1) return [dayOf(M.t0), dayOf(M.t1) + DAY];
  const k = PERIOD_ORDER.get(M.label);
  return [k, k + 1];
}
// Ao trocar de tema ou de escala: o quadro da linha `tl` que mais se sobrepõe no tempo ao atual
// (empate: o último visitado nessa linha, senão o mais antigo; sem sobreposição: o mais próximo).
const lastPos = new Map([[0, 0]]);
function matchingPos(tl) {
  const [a0, a1] = spanOf(TIMELINES[cur.tl].frames[cur.pos]), prefer = lastPos.get(tl);
  let best = 0, bestScore = -Infinity;
  TIMELINES[tl].frames.forEach((fi, i) => {
    const [b0, b1] = spanOf(fi);
    const overlap = Math.min(a1, b1) - Math.max(a0, b0);
    // fração do período mais curto (anos de 365 e 366 dias contam igual); sem sobreposição: -(distância)
    const score = overlap > 0 ? Math.round(1000 * overlap / Math.min(a1 - a0, b1 - b0)) : overlap - 1;
    if (score > bestScore || (score === bestScore && i === prefer)) { best = i; bestScore = score; }
  });
  return best;
}

const tlSel = $("tlSelect"), tlRange = $("tlRange");
const SCALES = C.scales && C.scales.length ? C.scales : [""];
const scaleOf = (tl) => TIMELINES[tl].scale || 0;
const linesOf = (s) => TIMELINES.flatMap((t, i) => scaleOf(i) === s ? [i] : []);
$("tlName").textContent = C.timelineLabel;
$("tlScaleName").textContent = C.scaleLabel;
$("tlScale").setAttribute("aria-label", C.scaleLabel);
$("tlScaleField").hidden = SCALES.length < 2;
$("tlScale").replaceChildren(...SCALES.map((s, i) =>
  el("button", { type: "button", textContent: s || "—", onclick: () => switchScale(i) })));
$("timeline").hidden = !multi;
let selectScale = -1;  // escala cujas linhas estão no seletor de tema
function fillLineSelect() {
  const s = scaleOf(cur.tl);
  if (s === selectScale) return;
  selectScale = s;
  const lines = linesOf(s);
  tlSel.replaceChildren(...lines.map((i) => el("option", { value: i, textContent: TIMELINES[i].name || "—" })));
  $("tlField").hidden = lines.length < 2;
}
function switchScale(s) {
  if (s === scaleOf(cur.tl)) return;
  stopPlay();
  const lines = linesOf(s), name = TIMELINES[cur.tl].name;
  const tl = lines.find((i) => TIMELINES[i].name === name) ?? lines[0];
  go(tl, matchingPos(tl));
}
tlSel.onchange = () => {
  stopPlay();
  go(+tlSel.value, matchingPos(+tlSel.value));
};
let rangeTimer = 0;
tlRange.oninput = () => {
  stopPlay();
  cancelAnimationFrame(rangeTimer);
  rangeTimer = requestAnimationFrame(() => go(cur.tl, +tlRange.value));
};
$("tlPrev").onclick = () => { stopPlay(); go(cur.tl, cur.pos - 1); };
$("tlNext").onclick = () => { stopPlay(); go(cur.tl, cur.pos + 1); };

function updateTimelineUI() {
  if (!multi) return;
  const frames = TIMELINES[cur.tl].frames, n = frames.length;
  fillLineSelect();
  tlSel.value = String(cur.tl);
  [...$("tlScale").children].forEach((b, i) => b.setAttribute("aria-pressed", String(i === scaleOf(cur.tl))));
  tlRange.max = String(Math.max(0, n - 1));
  tlRange.value = String(cur.pos);
  tlRange.disabled = n < 2;
  $("tlPrev").disabled = cur.pos <= 0;
  $("tlNext").disabled = cur.pos >= n - 1;
  $("tlPlay").disabled = n < 2;
  renderTicks();
  $("tlLabel").textContent = curFrame().label || TIMELINES[cur.tl].name;
  $("tlMeta").textContent = frameInfo(curFrame());
}
// Marcas sob a barra: uma para cada quadro; rótulo de `every` em `every` quadros a partir do primeiro,
// conforme a largura disponível (o quadro atual tem a marca destacada e aparece por extenso ao lado).
function renderTicks() {
  const frames = TIMELINES[cur.tl].frames, n = frames.length;
  const width = $("tlTicks").clientWidth || 600;
  const longest = Math.max(...frames.map((fi) => FRAMES[fi].label.length));
  const every = Math.max(1, Math.ceil((n - 1) * (longest * 7 + 14) / Math.max(1, width - 16)));
  $("tlTicks").replaceChildren(...frames.map((fi, i) => {
    const label = FRAMES[fi].label;
    const b = el("button", { type: "button", className: "tick" + (i % every ? " nolabel" : ""), title: label,
      onclick: () => { stopPlay(); go(cur.tl, i); } }, el("span", { textContent: label }));
    b.setAttribute("aria-label", label);
    b.style.left = n > 1 ? `calc(8px + (100% - 16px) * ${i / (n - 1)})` : "50%";
    b.setAttribute("aria-current", String(i === cur.pos));
    return b;
  }));
}
if (multi && window.ResizeObserver) {
  let tickWidth = 0;
  new ResizeObserver(([entry]) => {
    const w = Math.round(entry.contentRect.width);
    if (w !== tickWidth) { tickWidth = w; renderTicks(); }
  }).observe($("tlTicks"));
}

let playTimer = 0;
function stopPlay() {
  clearInterval(playTimer); playTimer = 0;
  $("tlPlay").textContent = "▶ Animar";
  $("tlPlay").setAttribute("aria-pressed", "false");
}
$("tlPlay").onclick = () => {
  if (playTimer) return stopPlay();
  const n = TIMELINES[cur.tl].frames.length;
  go(cur.tl, cur.pos >= n - 1 ? 0 : cur.pos + 1);
  playTimer = setInterval(() => {
    if (cur.pos >= TIMELINES[cur.tl].frames.length - 1) return stopPlay();
    go(cur.tl, cur.pos + 1);
  }, Math.max(0.5, C.playSeconds) * 1000);
  $("tlPlay").textContent = "❚❚ Pausar";
  $("tlPlay").setAttribute("aria-pressed", "true");
};

document.addEventListener("keydown", (ev) => {
  const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(ev.target.tagName);
  if (ev.key === "Escape" && ev.target !== input) {
    if (!$("tableView").hidden) toggleTable(false);
    else if (state.selected != null) select(null);
    else if (state.highlight) setHighlight(null);
    else if (state.vote != null) clearVote();
  } else if (multi && !typing && (ev.key === "ArrowLeft" || ev.key === "ArrowRight")) {
    ev.preventDefault(); stopPlay(); go(cur.tl, cur.pos + (ev.key === "ArrowRight" ? 1 : -1));
  }
});

// ---------- Início ----------
setTheme();   // tema salvo (claro/escuro) antes do primeiro desenho
afterFrame(); // cores, tamanhos, legenda, busca e linha do tempo do primeiro quadro
if (C.layoutSeconds > 0) startLayout(C.layoutSeconds);
window.__graphViz = {
  graph, sigma, state, select, startLayout, stopLayout, setFa2, fa2Settings, go, cur, TIMELINES, FRAMES,
  compo, setHighlight, toggleCompo, selectVote, clearVote, toggleVotes, VOTES, frameVotes, switchScale, matchingPos,
  get layout() { return layout; }, get iterations() { return iterations; }, get drag() { return drag; },
  get highlightSet() { return highlightSet; },
};
</script>
</body>
</html>
"""
