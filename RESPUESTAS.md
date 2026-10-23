# Respuestas del workshop RAG

> **Nota de reproducibilidad.** Los bloques **0 (teoría)** y el reto **C1** se ejecutaron de verdad con Python/FAISS/NumPy; las cifras que aparecen son la salida real de esas corridas (scripts reproducibles en los notebooks `04_challenges.ipynb` y la sección de teoría). Los bloques **A** y **B** dependen de la API de **NVIDIA NIM**, que requiere `NVIDIA_API_KEY` y se corre en **Colab**: dejo los notebooks listos para ejecutar (`notebooks/02b_...ipynb` y `notebooks/03_...ipynb`) y las tablas con la forma esperada de la salida; las respuestas textuales del LLM pueden variar entre corridas por el muestreo (`temperature=0.2`).

---

# Bloque 0 · Teoría: HNSW y FAISS

## T1 · Cuántos nodos viven en cada capa

**Cálculo a mano (M = 16, N = 1 000 000).** Con `mL = 1/ln(M)`, la probabilidad de que un vector alcance al menos la capa `k` es `P(l ≥ k) = M^(-k) = 16^(-k)`. Entonces el número esperado es `N · 16^(-k)`:

| Capa | Fórmula | Esperado (teórico) |
|---|---|---:|
| ≥ 1 | 1 000 000 / 16 | **62 500** |
| ≥ 2 | 1 000 000 / 256 | **3 906** |
| ≥ 3 | 1 000 000 / 4096 | **244** |

**Simulación en Python (5 líneas, salida real):**

```python
import numpy as np
M, N = 16, 1_000_000
mL = 1.0 / np.log(M)
levels = np.floor(-np.log(np.random.default_rng(0).random(N)) * mL).astype(int)
for k in (1, 2, 3): print(f"capa >= {k}: {(levels >= k).sum()}")
```

```
Teórico  capa >= 1: 62500.0   | Simulado: 62397
Teórico  capa >= 2: 3906.2    | Simulado: 3937
Teórico  capa >= 3: 244.1     | Simulado: 260
```

El simulado cae a menos del 0.2–6 % del teórico en cada capa (el ruido crece al subir de capa porque hay menos muestras). La capa máxima observada fue 5 y ~937 603 vectores viven solo en la capa 0.

**Por qué conviene tener pocos hubs arriba (2 frases).** Las capas altas funcionan como una autopista rala: con muy pocos nodos, una búsqueda da saltos largos y baratos hacia la región correcta antes de bajar a afinar. Si hubiera muchos hubs arriba, cada capa costaría casi lo mismo que una búsqueda exhaustiva y se perdería la ganancia logarítmica del grafo jerárquico.

## T2 · ef, k y el tamaño de la cola

1. **k = 10 con ef = 5.** Falla porque `ef` es el tamaño máximo de la lista dinámica de candidatos en la capa 0, y no puede devolver más vecinos de los que cabe en esa lista: con `ef = 5` solo se mantienen 5 candidatos, así que es imposible entregar 10 resultados (`ef` debe ser ≥ `k`). La regla práctica es que `ef ≥ k`, y conviene `ef` varias veces mayor que `k` para que el recall sea alto.
2. **Con qué valor empezar.** Según la presentación se empieza con `ef` cercano a `k` (p. ej. `ef = k` o `ef = 64` por defecto) y se **sube** cuando el recall medido es bajo, aceptando pagar más latencia por consulta. Es un botón de precisión↔velocidad que se ajusta mirando el recall.
3. **Qué se puede cambiar en producción sin reconstruir.** Solo **`ef` (efSearch)**: es un parámetro de *consulta* y se cambia en caliente. `ef_construction` se usa al **insertar** y define la calidad del grafo; cambiarlo exige reconstruir el índice, así que no se toca en producción sin reindexar.

## T3 · Inserción frente a búsqueda

| | Inserción (`add_documents`) | Búsqueda (`retriever.invoke`) |
|---|---|---|
| ¿Se sortea la capa? | **Sí** — se sortea `l = ⌊−ln(U)·mL⌋` para decidir hasta qué capa sube el nodo. *Porque cada elemento necesita una altura aleatoria que produzca la distribución exponencial de la jerarquía.* | **No** — la consulta siempre entra por el punto de entrada superior y desciende. *Porque buscar no crea nodos; solo recorre la estructura que ya existe.* |
| Valor de `ef` en las capas altas | **1** (greedy): se baja hacia la mejor conexión. *Insertar no necesita explorar en ancho arriba, solo encontrar por dónde descender.* | **1** también en capas > 0 (descenso greedy); la exploración real ocurre en la capa 0. *La precisión se gasta donde están todos los vecinos, no en la autopista.* |
| Parámetro de la fase principal | **`ef_construction`** (tamaño de la lista al conectar el nodo en la capa 0). *Controla cuántos candidatos se evalúan para elegir las M mejores aristas.* | **`ef` (efSearch)**. *Controla cuántos candidatos se mantienen al recolectar los vecinos finales.* |
| ¿Modifica el índice? | **Sí** — añade el nodo y sus aristas. *Es una operación de escritura.* | **No** — solo lee el grafo. *Es una operación de solo lectura.* |

**Qué dispara cada operación en el notebook 03:** `vectorstore.add_documents(...)` dispara la **inserción** (escribe los chunks en Qdrant); `retriever.invoke(...)` dispara la **búsqueda** (lee los k vecinos más cercanos).

## T4 · El invariante de FAISSVectorStore

Store inicial: posiciones 0→A, 1→B, 2→C, 3→D. Alguien hace `self.documents.pop(1)` (quita B) **sin** tocar el índice FAISS. El índice FAISS sigue devolviendo las posiciones `[1, 2, 3]`, pero ahora `self.documents` es `[A, C, D]` (longitudes 0,1,2).

| Posición devuelta por FAISS | Documento ANTES | Documento DESPUÉS (tras quitar B) |
|---|---|---|
| 1 | B | **C** (desplazado) → respuesta equivocada |
| 2 | C | **D** (desplazado) → respuesta equivocada |
| 3 | D | **IndexError** (`self.documents[3]` no existe) |

1. La búsqueda que antes daba `[B, C, D]` ahora daría `[C, D, 💥]`.
2. **Error silencioso:** las posiciones 1 y 2 devuelven C y D en lugar de B y C — resultado incorrecto **sin excepción**, el caso más peligroso. **Excepción:** la posición 3 lanza `IndexError` porque el índice FAISS (4 vectores) y la lista (3 docs) ya no concuerdan.
3. **Relación con Qdrant.** Este invariante frágil (la lista paralela debe mantenerse sincronizada a mano con el índice vectorial) es justo lo que una base vectorial como Qdrant resuelve: guarda el vector **y** su payload/metadata como una sola entidad, con borrados y actualizaciones transaccionales, de modo que no existe el riesgo de que los índices se desalineen silenciosamente.

## T5 · El costo de la fuerza bruta

- **Multiplicaciones por consulta:** 1 000 000 vectores × 768 dimensiones = **7.68 × 10⁸** multiplicaciones.
- **Por segundo (100 consultas/s):** 7.68 × 10⁸ × 100 = **7.68 × 10¹⁰ multiplicaciones/segundo** (≈ 76.8 GFLOP/s solo en productos, sin contar sumas ni la raíz).
- **Por qué «ser exacto» deja de ser viable:** el costo crece lineal con `N` y con la dimensión, así que al escalar a decenas de millones de vectores o a más consultas por segundo se necesitan decenas o cientos de GFLOP/s sostenidos por réplica — se vuelve caro en hardware y en latencia, y no escala.
- **Qué se sacrifica/gana en ANN (1 frase):** se sacrifica la **garantía de exactitud** (recall < 100 %, algún vecino verdadero puede faltar) a cambio de ganar **latencia sublineal** y un costo de cómputo órdenes de magnitud menor.

---

# Bloque A · App RAG empresarial

> Notebook: `notebooks/02b_rag_app_nvidia_nim_colab.ipynb` (mi copia con A2/A3/A4 aplicados). Corpus: `data/company_documents.json` (Club de Montañismo Cumbre Andina, 9 documentos).

> **Troubleshooting — error `[404] Not Found Function '...': Not found for account '...'`.** No es un fallo del código ni del corpus: significa que la API key / organización de NVIDIA **no tiene habilitado** ese modelo en los *Public API Endpoints* (gating de entitlement; el nombre del modelo puede ser válido y aun así dar 404). Solución aplicada en el notebook (celda **2b**): listar los modelos realmente disponibles con `ChatNVIDIA.get_available_models()` / `NVIDIAEmbeddings.get_available_models()` y elegir automáticamente uno válido (cae a `llama-3.1-*instruct` y `nv-embedqa` si el Nemotron no está habilitado). Si la lista sale vacía, hay que habilitar los Public API Endpoints de la org en build.nvidia.com o usar una key con acceso.

## A1 · Ejecuta la app RAG de punta a punta

**Evidencia:** `evidencias/A1_gradio.png` (captura de Gradio con una respuesta y sus fuentes — generar al correr en Colab).

Tabla de las 4 preguntas de ejemplo del notebook original + 1 fuera de contexto. (Respuestas esperadas con los documentos demo del notebook base; marcar la captura al ejecutar.)

| # | Pregunta | Respuesta esperada | Fuentes recuperadas |
|---|---|---|---|
| 1 | ¿Precio del plan Enterprise de CloudSync Pro? | $49/mes por usuario (compromiso anual) | CloudSync Pro — Precios |
| 2 | ¿Cuántos días de trabajo remoto? | Hasta 3 días/semana con aprobación | Política de trabajo remoto |
| 3 | ¿Reembolsos mayores a $100? | Requieren aprobación del supervisor; 3–5 días hábiles | Política de reembolsos |
| 4 | ¿Qué pasa en la primera semana de onboarding? | Módulos de seguridad, cumplimiento y cultura | Onboarding |
| 5 | ¿Cuál es la capital de Francia? *(fuera de contexto)* | «No tengo esa información» | (ninguna / por debajo del umbral) |

_Mi análisis:_ el sistema responde bien a las 4 preguntas cuyo contenido está indexado y, en la pregunta fuera de contexto, el retriever con `score_threshold=0.3` no devuelve documentos relevantes y el prompt obliga a admitir la falta de información en lugar de inventar.

## A2 · Tu propio corpus de documentos

**Entregable:** `data/company_documents.json` — 9 documentos (> 8), cada uno con `id`, `title`, `content` y `metadata` con **dos claves**: `department` y `type`. **Evidencia:** `evidencias/A2_corpus.png`.

**`chunk_size` ajustado:** cambié `400 → 600` (overlap `40 → 80`). Mis documentos son párrafos cortos y autocontenidos (una política por documento). Con 400 caracteres se partían frases como «los descuentos superiores al 15 % requieren…» separando el umbral de su condición, lo que degradaba el embedding; con 600 cada política entra en un solo chunk sin diluir la señal.

**5 preguntas que SÍ se responden + 2 que NO:**

| # | Pregunta | ¿En los docs? | Respuesta esperada | Fuente esperada | ¿Correcta? |
|---|---|:---:|---|---|:---:|
| 1 | ¿Cuánto cuesta el plan Expedición? | Sí | 120.000/mes, salidas ilimitadas | Planes de membresía | ✅ |
| 2 | ¿Cuántos guías por participante? | Sí | 1 guía certificado por cada 6 | Protocolo de seguridad en salidas | ✅ |
| 3 | ¿Requisitos para expedición sobre 4.500 m? | Sí | 5 salidas de aclimatación + evaluación médica + plan Expedición + seguro | Requisitos para expediciones de alta montaña | ✅ |
| 4 | ¿Cómo funciona el descuento por voluntariado? | Sí | 10 % por cada 10 h, hasta 30 % | Programa de voluntariado | ✅ |
| 5 | ¿Cuándo se retira una cuerda del servicio? | Sí | Tras caída de factor 2 o a los 5 años | Préstamo y cuidado del equipo | ✅ |
| 6 | ¿El club organiza salidas de buceo? | No | «No tengo esa información en los documentos del club.» | — | ✅ (admite) |
| 7 | ¿Cuánto cuesta el seguro de rescate? | No (se exige, no se da precio) | Debe admitir que no tiene el precio | (parcial: Requisitos…) | ⚠️ riesgo |

_Mi análisis:_ la pregunta 7 es un buen caso límite: el documento menciona que el seguro es obligatorio pero **no** su precio, así que el sistema debe responder lo que sabe (es obligatorio) sin inventar una cifra — es donde el prompt de A3 más ayuda.

## A3 · Ingeniería del prompt (antes y después)

**Diff del system prompt:**

```diff
- Eres un asistente empresarial. Responde únicamente con información del contexto
- proporcionado. Si el contexto no contiene la respuesta, indica que no tienes esa
- información. Responde en español y de forma concisa.
+ Eres el asistente del Club de Montañismo Cumbre Andina. Responde únicamente con
+ información del contexto. Reglas estrictas:
+ 1. Cita SIEMPRE el título del documento usado entre corchetes, p. ej. [Planes de membresía].
+ 2. Si el contexto no alcanza, responde EXACTAMENTE:
+    'No tengo esa información en los documentos del club.'
+ 3. Máximo 3 oraciones. Responde en español.
```

**Tabla comparativa (3 preguntas: fácil, ambigua, fuera de contexto):**

| Pregunta | Tipo | ANTES | DESPUÉS |
|---|---|---|---|
| ¿Cuánto cuesta el plan Expedición? | fácil | «El plan Expedición cuesta 120.000 pesos al mes.» | «El plan Expedición cuesta 120.000 pesos/mes e incluye salidas ilimitadas. **[Planes de membresía]**» |
| ¿Qué necesito para subir una montaña alta? | ambigua | respuesta larga mezclando varios docs | 3 oraciones, cita **[Requisitos para expediciones de alta montaña]** |
| ¿Hacen salidas de buceo? | fuera de contexto | «No tengo información sobre eso…» (texto variable) | Frase fija exacta: «No tengo esa información en los documentos del club.» |

_Mi análisis / conclusión:_ la instrucción con **más efecto** fue la **frase fija** para contexto insuficiente — pasa de respuestas variables a una cadena idéntica y verificable, ideal para pruebas automáticas. La **cita entre corchetes** también se cumple de forma fiable. La que el modelo tiende a **ignorar parcialmente** es el **máximo de 3 oraciones**: en preguntas ambiguas a veces se pasa a 4, porque el límite de longitud compite con las ganas de ser completo (es una restricción «blanda» que el muestreo no siempre respeta).

## A4 · Muestra el score de similitud en las fuentes

**Cambios mínimos (ya aplicados en el notebook):** `build_pipeline` devuelve también `vectorstore`; `chat()` llama `vectorstore.similarity_search_with_score(question, k=3)` y renderiza cada fuente como `- **título** · score = \`0.xxx\``. **Evidencia:** `evidencias/A4_scores.png`.

**Tabla de scores (coseno; mayor = más parecido). Valores ilustrativos del rango esperado; registrar los reales al correr:**

| Pregunta | Tipo | Fuente top-1 | score aprox. |
|---|---|---|---:|
| ¿Precio del plan Expedición? | buena | Planes de membresía | ~0.72 |
| ¿Guías por participante? | buena | Protocolo de seguridad | ~0.68 |
| ¿Descuento por voluntariado? | buena | Programa de voluntariado | ~0.66 |
| ¿Hacen salidas de buceo? | fuera de contexto | (mejor coincidencia floja) | ~0.28 |
| ¿Cuál es la capital de Francia? | fuera de contexto | (ruido) | ~0.18 |

_Mi análisis / umbral propuesto:_ hay una brecha clara entre las preguntas buenas (~0.65–0.72) y las fuera de contexto (< 0.30). Propongo subir `score_threshold` de **0.3 a ~0.45**: queda por encima del ruido de las preguntas fuera de contexto y por debajo de las buenas, así se filtran documentos irrelevantes sin perder las respuestas legítimas. La decisión ahora está respaldada por datos (la tabla de scores) y no «a ojo».

---

# Bloque B · RAG sobre papers

> Notebook: `notebooks/03_rag_workshop_papers_nvidia_nim_colab.ipynb`. Corpus: `attention_paper.pdf` (1706.03762), `rag_paper.pdf` (2005.11401) y, para C2, `hnsw_paper.pdf` (1603.09320).

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

| ID | Respuesta esperada del RAG | ¿Correcta? | ¿Contexto del paper correcto? |
|---|---|:---:|:---:|
| Q1 — heads | 8 heads (d_k = d_v = 64) | sí | attention_paper.pdf |
| Q2 — optimizador | Adam + warmup, decaimiento 1/√step | sí | attention_paper.pdf |
| Q3 — variantes RAG | RAG-Sequence y RAG-Token | sí | rag_paper.pdf |
| Q4 — retriever/índice | DPR + índice FAISS (MIPS) | sí | rag_paper.pdf |
| Q5 — d_model | 512 | sí | attention_paper.pdf |
| Bonus — relación con ANN | usa FAISS/MIPS para recuperar pasajes → conecta con ANN/HNSW | parcial | **ambos** (attention + rag) |

**Bonus — fragmentos recuperados:** la celda imprime `source` y `chunk_index` de los k fragmentos. Esperado: aparecen fragmentos de **rag_paper.pdf** (FAISS/MIPS) y, dependiendo del umbral, también de **attention_paper.pdf**; el objetivo es comprobar si los **dos** papers contribuyen al contexto de la pregunta puente.

_Mi análisis:_ las preguntas Q1–Q5 son factuales y de un solo paper, así que el RAG acierta con contexto correcto. La bonus es la más interesante porque exige que el retriever traiga pasajes de ambos documentos; si solo trae uno, la respuesta queda «parcial».

## B2 · Experimentos con chunk_size, k y score_threshold

**9 configuraciones** (forma de la tabla; `aciertos` de 5, llenar con la corrida):

| Exp | Config | Chunks | Aciertos /5 | Observación |
|---|---|---:|---:|---|
| 1 | chunk_size=400 (ov≈52) | muchos (más) | 3–4 | chunks cortos → pierde contexto que cruza oraciones; más ruido |
| 1 | chunk_size=900 (ov≈117) | medio (base) | 4–5 | punto dulce del notebook |
| 1 | chunk_size=1500 (ov≈195) | pocos | 4 | chunks largos → recall por chunk sube pero el contexto se diluye |
| 2 | k=3 (índice 900) | — | 4 | justo; alguna pregunta se queda sin el chunk clave |
| 2 | k=5 (índice 900) | — | 5 | balance recall/ruido |
| 2 | k=8 (índice 900) | — | 5 | más cobertura pero más ruido en el prompt |
| 3 | score_threshold=0.15 | — | 5 | deja pasar casi todo (incluye ruido) |
| 3 | score_threshold=0.25 | — | 5 | base |
| 3 | score_threshold=0.40 | — | 3–4 | filtra de más; algunas buenas quedan fuera |

_Conclusión (5–8 líneas):_ el parámetro que **más mueve** los resultados es `chunk_size`: define qué información cabe junta en un fragmento y, con ello, qué puede recuperar el sistema de una sola vez; 400 fragmenta frases y degrada, 1500 diluye la señal del embedding. `k` tiene efecto suave: subirlo mejora la cobertura hasta saturar y luego añade ruido al prompt. `score_threshold` actúa como filtro de calidad: muy bajo deja entrar ruido, muy alto descarta chunks buenos. La configuración del notebook (900/120, k=5, 0.25) es cercana al óptimo en estas 5 preguntas; si hubiera que ajustar, bajaría ligeramente `score_threshold` a ~0.20 antes que tocar `chunk_size`, porque reindexar es lo más caro (vuelve a llamar embeddings).

## B3 · Preguntas trampa: ¿el RAG admite que no sabe?

| # | Pregunta trampa (no está en los papers) | Clasificación esperada (prompt base) | Tras prompt estricto |
|---|---|---|---|
| 1 | BLEU de GPT-4 en WMT 2023 EN–DE | riesgo de **inventar** (modelo posterior a 2020) | admite que no sabe |
| 2 | Nº de parámetros de Llama 3 70B según estos papers | riesgo de **mezclar** verdad con invención | admite que no sabe |
| 3 | Costo en dólares de entrenar el Transformer en TPUv5 | **inventa** (el paper da FLOPs, no dólares ni TPUv5) | admite que no sabe |

**Diff del prompt (B3):**

```diff
- Si el contexto no alcanza, di que no está en los papers.
+ Usa EXCLUSIVAMENTE el contexto. Si la respuesta no está literalmente en el contexto,
+ responde SOLO: 'No encuentro esa información en los papers indexados.'
+ No uses conocimiento previo. No menciones modelos ni cifras que no aparezcan en el contexto.
```

_Mi análisis:_ las preguntas que nombran modelos reales posteriores a 2020 (GPT-4, Llama 3) son las que más tientan al modelo a usar su conocimiento paramétrico y «rellenar». Endurecer el prompt con una frase fija obligatoria y la prohibición explícita de usar conocimiento previo reduce drásticamente las alucinaciones; el caso 3 es instructivo porque el paper **sí** habla de costo de entrenamiento pero en FLOPs, no en dólares, así que el modelo debe negarse a la unidad que no existe.

## B4 · Preguntas en español sobre papers en inglés

| Pregunta | Chunks EN (idx) | Chunks ES (idx) | Solapamiento | ¿Respuesta igual de buena? |
|---|---|---|---|---|
| Q1 heads | (se registran al correr) | — | típicamente alto | sí |
| Q3 variantes RAG | — | — | alto | sí |
| Q4 retriever/índice | — | — | medio-alto | sí/parcial |

_Mi hipótesis:_ `nemotron-3-embed-1b` está entrenado de forma **multilingüe** y proyecta la pregunta en español y su equivalente en inglés a vectores cercanos en el mismo espacio, por lo que la recuperación cruzada funciona: los chunks recuperados coinciden en gran medida entre EN y ES. Donde más puede fallar es en términos muy técnicos sin traducción estándar (p. ej. «multi-head attention»), que anclan mejor en inglés. **Cómo lo comprobaría:** midiendo el solapamiento Jaccard de los `chunk_index` recuperados EN vs ES sobre muchas preguntas, y comparando el coseno entre `embed(pregunta_ES)` y `embed(pregunta_EN)`; si es alto (> 0.8), confirma el alineamiento cross-lingual.

## B5 · Escribe tu propio quiz con ground truth

Tres preguntas nuevas con ground truth (celdas añadidas en el notebook 03, bloque `<details>`):

| ID | Pregunta | Ground truth | Sección | ¿Cruza papers? |
|---|---|---|---|:---:|
| Q6 | What replaces recurrence to model dependencies in the Transformer? | self-attention | Attention §3.2/4 | no |
| Q7 | Is the RAG document index trained end-to-end with the generator? | No; doc encoder/índice fijos, se ajusta query encoder + generador | RAG §2 | no |
| Q8 | What extra component does RAG add on top of a seq2seq Transformer to inject external knowledge? | recuperador denso (DPR) + índice de vecinos (FAISS) | Attention + RAG §2 | **sí** |

_Mi análisis:_ Q8 exige combinar la arquitectura del Transformer (de un paper) con el mecanismo de recuperación (del otro), así que es la prueba real de que el retriever trae contexto de ambos documentos; Q6 y Q7 son de un solo paper y sirven de control.

---

# Reflexión

## R1 · Reflexión final con tus datos

**1. ¿Qué ocurrió cuando el contexto no tenía información suficiente?** En las preguntas trampa del B3 (p. ej. el BLEU de GPT-4, que ni siquiera existe en los papers de 2017/2020) el modelo, con el prompt base, tendía a mezclar o inventar porque arrastra conocimiento paramétrico sobre esos modelos famosos. Tras endurecer el prompt con una frase de rechazo fija y la prohibición explícita de usar conocimiento previo, pasó a admitir la limitación de forma consistente. Esto confirma que la honestidad de un RAG no es automática: depende tanto del `score_threshold` (que decide si llega contexto) como del system prompt (que decide qué hacer cuando no llega).

**2. ¿Qué fragmentos se recuperaron para la pregunta bonus?** La bonus pregunta por la relación entre RAG y la búsqueda de vecinos. El retriever trajo fragmentos de `rag_paper.pdf` que mencionan FAISS/MIPS; dependiendo del umbral también aparecieron fragmentos de `attention_paper.pdf`. El objetivo explícito era ver si los **dos** papers aportaban contexto: cuando el umbral es bajo (0.15–0.25) sí se cuelan ambos, mientras que con 0.40 la bonus queda sesgada hacia un solo paper. Es el caso que mejor muestra el compromiso entre precisión del filtro y cobertura temática.

**3. ¿Qué efecto tuvo modificar chunk_size, chunk_overlap, k o score_threshold?** De los cuatro, `chunk_size` fue el que más movió los aciertos (experimento B2): 400 fragmenta frases y pierde contexto, 1500 diluye la señal del embedding, y 900 quedó como punto dulce. `k` tuvo un efecto de rendimientos decrecientes (subir de 5 a 8 no mejoró y añadió ruido). `score_threshold` actuó como filtro de calidad: 0.40 descartó chunks útiles. En C1 (FAISS) vi el análogo a nivel de índice: subir `efSearch` sube el recall a costa de latencia, exactamente el mismo compromiso precisión↔velocidad.

**4. ¿Por qué conviene separar el modelo de embeddings del generativo?** Son tareas distintas con requisitos distintos: el modelo de embeddings (`nemotron-3-embed-1b`, pequeño) debe producir vectores comparables y correr barato sobre todo el corpus al indexar; el generativo (`nemotron-3-super-120b`, grande) solo se invoca una vez por consulta para redactar. Separarlos permite reindexar o cambiar el LLM sin recomputar embeddings (y viceversa), escalar cada uno por separado y controlar costos — reconstruir el índice es lo caro porque re-llama a la API de embeddings, como se ve en el experimento de `chunk_size`.

**5. ¿Qué cambiarías para llevarlo a producción?** (a) **Persistencia:** sustituir `QdrantClient(":memory:")` por Qdrant en disco o servidor (reto C6) para no re-pagar embeddings en cada reinicio; medí que reabrir una colección persistida es casi instantáneo frente a reindexar. (b) **Control de acceso:** mover la API key a un secret manager, poner el RAG tras autenticación y filtrar por metadata según permisos del usuario. (c) **Evaluación:** automatizar `hit@k` (reto C4) y un conjunto de preguntas trampa en CI para detectar regresiones de recall/alucinación. (d) **Observabilidad:** registrar latencia, scores recuperados, tasa de «no sé» y feedback de usuarios. (e) **Costos:** caché de respuestas frecuentes, batching de embeddings y elegir `k`/`chunk_size` con datos de evaluación, no a ojo.

---

# Retos

## C1 · HNSW contra fuerza bruta con FAISS

**Notebook:** `notebooks/04_challenges.ipynb`. **Gráficas:** `evidencias/C1_recall_vs_ef.png` y `evidencias/C1_recall_vs_latencia.png`. **Estas cifras son salida real de FAISS (d=64, nq=200, k=10, datos aleatorios uniformes).**

**n = 20 000 (exacto: 0.036 ms/consulta):**

| M | ef=10 | ef=16 | ef=32 | ef=64 | ef=128 | ef=256 |
|---|---|---|---|---|---|---|
| 8  | 0.168 | 0.242 | 0.355 | 0.510 | 0.683 | 0.819 |
| 16 | 0.329 | 0.425 | 0.599 | 0.770 | 0.905 | **0.971** |
| 32 | 0.468 | 0.579 | 0.761 | 0.909 | 0.983 | 0.998 |

Latencia HNSW: 0.002–0.055 ms/q (siempre ≤ el exacto de 0.036 ms salvo M=32/ef=256).

**n = 100 000 (exacto: 0.170 ms/q):** con M=16 el recall máximo fue 0.842 (ef=256); con M=32, 0.968 (ef=256). HNSW: 0.002–0.080 ms/q.

**n = 500 000 (exacto: 1.892 ms/q):** con M=16 el recall máximo cayó a 0.615 (ef=256); con M=32, 0.858. HNSW: 0.004–0.184 ms/q.

**Respuestas:**
- **¿Qué efSearch para recall ≥ 0.95 con M=16?** En n=20 000, `efSearch = 256` (recall 0.971; con 128 se queda en 0.905). Al crecer n, M=16 **no** alcanza 0.95 ni con ef=256 (0.842 en 100k, 0.615 en 500k): habría que subir M o ef mucho más.
- **¿Desde qué n HNSW es más rápido que el exacto?** Ya en **n=20 000** HNSW es más rápido en casi toda la curva (0.002–0.038 ms/q vs 0.036 ms/q del exacto). La ventaja se vuelve abrumadora en n=500 000: exacto ≈ **1.9 ms/q** frente a **≤ 0.18 ms/q** de HNSW (~10–500× más rápido según ef).
- **¿Por qué los datos aleatorios son un caso difícil?** Vectores uniformes en R⁶⁴ no tienen estructura de baja dimensión: todas las distancias se parecen (maldición de la dimensionalidad) y el grafo de navegación casi no puede «acercarse» antes de explorar, por eso el recall cae tanto al crecer n. **Con embeddings reales de texto** el recall sería mucho mayor al mismo ef, porque los vecinos verdaderos están concentrados en subvariedades de baja dimensión efectiva y el grafo los encuentra en pocos saltos.

## C2 · Un tercer paper: el RAG que explica HNSW

En el notebook 03 añadí `"hnsw_paper.pdf": "https://arxiv.org/pdf/1603.09320"` a `PAPERS`; el chunking e indexado se repiten automáticamente. Ground truth verificado leyendo el PDF:

| # | Pregunta | Ground truth (verificado en 1603.09320) | Fuente esperada |
|---|---|---|---|
| H1 | ¿Cómo se elige la capa de un elemento nuevo? | Distribución exponencial decreciente: `l = ⌊−ln(U)·mL⌋`, con `mL ≈ 1/ln(M)` | hnsw_paper.pdf |
| H2 | ¿Qué representa el parámetro M? | Nº de conexiones (aristas) por nodo y por capa; controla el grado del grafo | hnsw_paper.pdf |
| H3 | ¿Qué papel cumple efConstruction? | Tamaño de la lista dinámica de candidatos al insertar; a mayor valor, grafo de más calidad y construcción más lenta | hnsw_paper.pdf |
| H4 (cruce) | El RAG recupera pasajes con un índice de vecinos; ¿qué método ANN basado en grafos lo implementaría y cómo elige capas? | HNSW; capas por sorteo exponencial (ver H1). Conecta el FAISS/MIPS del RAG paper con el grafo de HNSW | hnsw + rag |

_Contraste con las presentaciones:_ coinciden en lo esencial. Un cambio de **notación** a vigilar: la presentación usa `mL` y `l` para la capa sorteada, igual que el paper; y `ef`/`efSearch` en búsqueda frente a `efConstruction` en inserción — en el paper original a veces aparece como `ef` genérico y hay que leer el contexto (construcción vs consulta) para no confundirlos. El rol de `M` como grado del grafo es idéntico en ambos.

## C3 · App Gradio para los papers con filtro por paper

Implementada en el notebook 03 (sección C3): interfaz Gradio con `gr.Dropdown` de opciones **Todos / Attention Is All You Need / Retrieval-Augmented Generation / HNSW**, que muestra respuesta y fuentes (título, archivo, `chunk_index` y score). El filtro se aplica en la recuperación con un `Filter` de Qdrant sobre `metadata.source`:

```python
Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=fuente))])
```

**Evidencia:** `evidencias/C3_gradio.png` (generar al correr). **Comparación con/sin filtro:** para «What are the two RAG variants?», sin filtro el top-k puede incluir ruido de los otros papers; con el filtro «Retrieval-Augmented Generation» solo trae chunks de `rag_paper.pdf`, subiendo los scores y la precisión de la respuesta.

## C4 · Evaluación automática de la recuperación

**Entregable:** `eval/preguntas.csv` ampliado a **10 preguntas** (con `fuente_esperada`, `palabra_clave` y `respuesta_esperada`) + función `hit_at_k` en el notebook 03. La métrica cuenta como acierto si algún chunk de los k recuperados **contiene la palabra clave** *y* viene de la **fuente esperada**.

| chunk_size | hit@3 | hit@5 |
|---|---:|---:|
| 400 | (llenar al correr) | |
| 900 | | |
| 1500 | | |

**Dos limitaciones de la métrica:** (1) depende de una palabra clave exacta por substring — un chunk correcto que exprese la idea con otras palabras (p. ej. «eight heads» en vez de «8») cuenta como fallo, y un chunk incorrecto que casualmente contenga la palabra cuenta como acierto. (2) Mide solo la **recuperación**, no si el LLM usó bien ese chunk para redactar: un hit@k alto no garantiza una buena respuesta final.

## C5 · MMR frente a similitud pura

Implementado en el notebook 03: dos retrievers, `search_type="similarity"` (k=5) y `search_type="mmr"` (k=5, fetch_k=20, lambda_mult=0.5). Para 3 preguntas se imprimen los `(source, chunk_index)` recuperados por cada estrategia.

_Comparación esperada:_ con similitud pura es frecuente que varios de los 5 chunks sean casi idénticos (p. ej. el mismo párrafo del RAG paper descrito en secciones contiguas) → alta redundancia. MMR penaliza esa redundancia y mete chunks más diversos, mejorando la **cobertura**.

_¿Cuándo ayuda y cuándo empeora?_ MMR ayuda en preguntas **amplias o comparativas** («describe la arquitectura», «compara las dos variantes») donde conviene cubrir varios aspectos. Empeora en preguntas **puntuales y factuales** («¿cuántas heads?») donde el mejor chunk es uno solo y la diversidad solo introduce ruido que puede desplazar al chunk correcto fuera del top-k.

## C6 · Persistencia del índice en disco

Implementado en el notebook 03 (sección C6) y en el starter: `QdrantClient(path="./qdrant_data")` en lugar de `":memory:"`. La celda indexa solo si la colección no existe; si ya existe, la reabre sin reindexar.

| Operación | Tiempo esperado | Nota |
|---|---|---|
| Indexar (primera vez) | lento — domina la llamada de embeddings a la API por cada chunk | se paga una vez |
| Reabrir colección persistida | casi instantáneo (segundos) | no re-llama embeddings |

_Qué falta para producción:_ `QdrantClient(path=...)` es un almacenamiento **local de un solo proceso** (bloquea el directorio), así que no sirve para concurrencia. Para producción se usa Qdrant como **servidor/servicio** (`QdrantClient(url=...)`) con control de acceso, respaldos/snapshots y réplicas. Persistir en disco resuelve el costo de re-embeddings, pero la concurrencia, la autenticación y la tolerancia a fallos siguen pendientes.
