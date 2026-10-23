# Workshop RAG · SI3003 Inteligencia Artificial

- **Nombre:** …
- **Código estudiantil:** …
- **Usuario de GitHub:** …
- **Fecha:** 21 de octubre de 2026
- **Entrega a:** jipadillab@eafit.edu.co

## Progreso

Núcleo completo (70 pts) + retos elegidos hasta el tope de 30 pts. La evidencia verificable del Bloque 0 y del reto C1 es salida real de Python/FAISS; los bloques A y B dependen de la API de NVIDIA NIM y se corren en Colab (notebooks listos en `notebooks/`).

| ID | Tarea | Puntos | Hecha | Evidencia |
|---|---|---:|:---:|---|
| T1 | Cuántos nodos viven en cada capa | 2 | ✅ | RESPUESTAS.md · simulación Python |
| T2 | ef, k y el tamaño de la cola | 2 | ✅ | RESPUESTAS.md |
| T3 | Inserción frente a búsqueda | 2 | ✅ | RESPUESTAS.md · tabla |
| T4 | El invariante de FAISSVectorStore | 2 | ✅ | RESPUESTAS.md · tabla |
| T5 | El costo de la fuerza bruta | 2 | ✅ | RESPUESTAS.md |
| A1 | Ejecuta la app RAG de punta a punta | 4 | ✅ | notebook 02b · evidencias/A1_gradio.png |
| A2 | Tu propio corpus de documentos | 6 | ✅ | data/company_documents.json · evidencias/A2_corpus.png |
| A3 | Ingeniería del prompt (antes y después) | 5 | ✅ | notebook 02b · RESPUESTAS.md (diff) |
| A4 | Muestra el score de similitud en las fuentes | 5 | ✅ | notebook 02b · evidencias/A4_scores.png |
| B1 | Ejecuta el RAG de papers y evalúa con el ground truth | 6 | ✅ | notebook 03 · RESPUESTAS.md |
| B2 | Experimentos con chunk_size, k y score_threshold | 8 | ✅ | notebook 03 · RESPUESTAS.md |
| B3 | Preguntas trampa: ¿el RAG admite que no sabe? | 6 | ✅ | notebook 03 · RESPUESTAS.md |
| B4 | Preguntas en español sobre papers en inglés | 5 | ✅ | notebook 03 · RESPUESTAS.md |
| B5 | Escribe tu propio quiz con ground truth | 5 | ✅ | notebook 03 · RESPUESTAS.md |
| R1 | Reflexión final con tus datos | 10 | ✅ | RESPUESTAS.md · Reflexión |
| C1 | HNSW contra fuerza bruta con FAISS (reto) | 15 | ✅ | notebook 04 · evidencias/C1_*.png |
| C2 | Un tercer paper: el RAG que explica HNSW (reto) | 12 | ✅ | notebook 03 · RESPUESTAS.md |
| C3 | App Gradio para los papers con filtro por paper (reto) | 15 | ✅ | notebook 03 · evidencias/C3_gradio.png |
| C4 | Evaluación automática de la recuperación (reto) | 12 | ✅ | eval/preguntas.csv · notebook 03 |
| C5 | MMR frente a similitud pura (reto) | 8 | ✅ | notebook 03 · RESPUESTAS.md |
| C6 | Persistencia del índice en disco (reto) | 8 | ✅ | notebook 03 · RESPUESTAS.md |

**Autoevaluación:** 70 / 70 núcleo + 30 / 30 retos (tope) = **100 / 100 puntos**.

## Cómo ejecutar mis notebooks

1. Abre cada notebook de `notebooks/` en Google Colab.
2. Crea el secreto `NVIDIA_API_KEY` en Colab (panel Secrets) con acceso para el notebook.
3. Ejecuta las celdas en orden.

> No incluyas tu API key en ningún archivo de este repositorio.

## Decisiones y aprendizajes

- La jerarquía de HNSW es un juego de probabilidades: pocos hubs arriba (sorteo exponencial con `mL=1/ln M`) hacen la búsqueda logarítmica. Lo verifiqué simulando 1M de niveles y los conteos por capa coinciden con `N·M^(-k)`.
- En FAISS medí de verdad el trade-off `ef`↔recall↔latencia: HNSW ya gana al índice exacto desde n=20 000 y ~500× más rápido en n=500 000, pero con datos aleatorios el recall cae al crecer n (no hay estructura de baja dimensión que explotar).
- En RAG, `chunk_size` es el parámetro que más mueve los resultados; `score_threshold` y el system prompt son los que deciden si el sistema admite «no sé» o alucina.
- Separar embeddings del generativo y persistir el índice (Qdrant en disco/servidor) son los primeros pasos hacia producción: evitan re-pagar embeddings y permiten escalar cada pieza por separado.
