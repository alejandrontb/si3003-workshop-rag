# Respuestas del workshop RAG
Escribe aquí tus respuestas con tus propias palabras. Pega tablas, resultados y enlaces a las capturas de `evidencias/`.

---

# Bloque 0 · Teoría

## T1 · Cuántos nodos viven en cada capa

_Evidencia:_ … 
<img width="459" height="152" alt="image" src="https://github.com/user-attachments/assets/fe623c00-cd28-4693-8d6a-3649238f8007" />



_Mi respuesta / análisis:_
Datos:

M = 16
N = 1.000.000 vectores
Probabilidad de llegar al menos a la capa k: P(L >= k) = M^(-k)

La fórmula para calcular el número esperado de nodos es:

E_k = N / M^k

Capa >= 1:

E_1 = 1.000.000 / 16 = 62.500 nodos

Capa >= 2:

E_2 = 1.000.000 / 16^2
E_2 = 1.000.000 / 256 = 3.906,25 nodos

Capa >= 3:

E_3 = 1.000.000 / 16^3
E_3 = 1.000.000 / 4.096 = 244,14 nodos

SIMULACIÓN A MANO

Cada vector tiene una probabilidad de 1/16 de subir a la siguiente capa.

Capa >= 1: de 1.000.000 de vectores, esperamos que 1.000.000 / 16 = 62.500 lleguen a esta capa.
Capa >= 2: de los 62.500 vectores anteriores, esperamos que 62.500 / 16 = 3.906,25 lleguen a esta capa.
Capa >= 3: de los 3.906,25 vectores anteriores, esperamos que 3.906,25 / 16 = 244,14 lleguen a esta capa.

Resultados teóricos:

Capa >= 1: 62.500 nodos
Capa >= 2: 3.906,25 nodos
Capa >= 3: 244,14 nodos

Estas cifras representan aproximaciones teóricas; al ejecutar una simulación en Python, los valores pueden variar ligeramente por el componente aleatorio.

## T2 · ef, k y el tamaño de la cola

_Evidencia:_ …

_Mi respuesta / análisis:_


## T3 · Inserción frente a búsqueda

_Evidencia:_ …

_Mi respuesta / análisis:_


## T4 · El invariante de FAISSVectorStore

_Evidencia:_ …

_Mi respuesta / análisis:_


## T5 · El costo de la fuerza bruta

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Bloque A · App RAG empresarial

## A1 · Ejecuta la app RAG de punta a punta

_Evidencia:_ …

_Mi respuesta / análisis:_


## A2 · Tu propio corpus de documentos

_Evidencia:_ …
<img width="1441" height="413" alt="image" src="https://github.com/user-attachments/assets/a15fa0ee-7db1-498c-87ff-218141267d57" />


_Mi respuesta / análisis:_

user: ¿Cuántos días pueden trabajar remotamente los empleados?

assistant: Los empleados pueden trabajar remotamente hasta **3 días por semana**, previa aprobación del jefe directo.

user: Qué ocurre durante la primera semana de onboarding?

assistant: Durante la primera semana de onboarding, los empleados completan módulos obligatorios de seguridad, cumplimiento y cultura de la empresa.

user: ¿Cuál es el precio del plan Enterprise de CloudSync Pro?

assistant: El plan Enterprise de CloudSync Pro cuesta **$49/mes por usuario** con compromiso anual.

user: ¿Qué pasa con los reembolsos mayores a $100?

assistant: Los reembolsos mayores a $100 requieren aprobación del supervisor. Una vez aprobados, se procesan en 3 a 5 días hábiles al método de pago original.

user: Cuando fue la tercera guerra mundial?

assistant: No tengo esa información.
## A3 · Ingeniería del prompt (antes y después)

_Evidencia:_ …

_Mi respuesta / análisis:_


## A4 · Muestra el score de similitud en las fuentes

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Bloque B · RAG sobre papers

## B1 · Ejecuta el RAG de papers y evalúa con el ground truth

_Evidencia:_ …

_Mi respuesta / análisis:_


## B2 · Experimentos con chunk_size, k y score_threshold

_Evidencia:_ …

_Mi respuesta / análisis:_


## B3 · Preguntas trampa: ¿el RAG admite que no sabe?

_Evidencia:_ …

_Mi respuesta / análisis:_


## B4 · Preguntas en español sobre papers en inglés

_Evidencia:_ …

_Mi respuesta / análisis:_


## B5 · Escribe tu propio quiz con ground truth

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Reflexión

## R1 · Reflexión final con tus datos

_Evidencia:_ …

_Mi respuesta / análisis:_


---

# Retos

## C1 · HNSW contra fuerza bruta con FAISS

_Evidencia:_ …

_Mi respuesta / análisis:_


## C2 · Un tercer paper: el RAG que explica HNSW

_Evidencia:_ …

_Mi respuesta / análisis:_


## C3 · App Gradio para los papers con filtro por paper

_Evidencia:_ …

_Mi respuesta / análisis:_


## C4 · Evaluación automática de la recuperación

_Evidencia:_ …

_Mi respuesta / análisis:_


## C5 · MMR frente a similitud pura

_Evidencia:_ …

_Mi respuesta / análisis:_


## C6 · Persistencia del índice en disco

_Evidencia:_ …

_Mi respuesta / análisis:_

