# Respuestas del workshop RAG
Escribe aquí tus respuestas con tus propias palabras. Pega tablas, resultados y enlaces a las capturas de `evidencias/`.

---

# Bloque 0 · Teoría

## T1 · Cuántos nodos viven en cada capa

_Evidencia:_ … 
<img width="459" height="152" alt="image" src="https://github.com/user-attachments/assets/fe623c00-cd28-4693-8d6a-3649238f8007" />



T1. ¿CUÁNTOS NODOS VIVEN EN CADA CAPA?

CÁLCULO A MANO

Datos:

M = 16
N = 1.000.000 vectores
Probabilidad de llegar al menos a la capa k: P(L >= k) = M^(-k)

La fórmula para calcular el número esperado de nodos es:

E_k = N / M^k

Capa >= 1:

E_1 = 1.000.000 / 16 = 62.500 nodos

Capa >= 2:

E_2 = 1.000.000 / 16^2 E_2 = 1.000.000 / 256 = 3.906,25 nodos

Capa >= 3:

E_3 = 1.000.000 / 16^3 E_3 = 1.000.000 / 4.096 = 244,14 nodos

SIMULACIÓN A MANO

Cada vector tiene una probabilidad de 1/16 de subir a la siguiente capa.

Capa >= 1: de 1.000.000 de vectores, esperamos que 1.000.000 / 16 = 62.500 lleguen a esta capa.
Capa >= 2: de los 62.500 vectores anteriores, esperamos que 62.500 / 16 = 3.906,25 lleguen a esta capa.
Capa >= 3: de los 3.906,25 vectores anteriores, esperamos que 3.906,25 / 16 = 244,14 lleguen a esta capa.

Resultados teóricos:

Capa >= 1: 62.500 nodos Capa >= 2: 3.906,25 nodos Capa >= 3: 244,14 nodos

Estos valores son estimaciones matemáticas. Una simulación real en Python puede producir resultados ligeramente diferentes debido al azar.


_Mi respuesta / análisis: ¿POR QUÉ MEJORA LA VELOCIDAD DE BÚSQUEDA?

Tener pocos nodos en las capas altas permite realizar saltos largos y recorrer rápidamente grandes partes del índice. Después, las capas inferiores permiten encontrar el resultado con mayor precisión, evitando revisar todos los nodos.


## T2 · ef, k y el tamaño de la cola
PREGUNTAS:
Si pides k = 10 vecinos y configuras ef = 5, ¿qué falla y por qué?
R/: El parámetro ef siempre debe igualar o superar a k. Al solicitar 10 resultados (k=10) pero limitar la evaluación a 5 candidatos (ef=5), el sistema carece de suficientes opciones para devolver la cantidad pedida.

Según la presentación, ¿con qué valor conviene empezar y cuándo se sube?
R/: Se recomienda iniciar estableciendo ef = 2k. Este valor debe incrementarse si la métrica de exhaustividad (recall) no alcanza el 95%.

¿Cuál de los dos parámetros, ef o ef_construction, puedes cambiar en producción sin reconstruir el índice? Justifica.
R/: Solamente ef, dado que ajustarlo en tiempo de ejecución permite incrementar el recall a costa de una mayor latencia, sin necesidad de reindexar los datos.

## T3 · Inserción frente a búsqueda

_Evidencia:_ 



| Característica | Inserción | Búsqueda |
|---|---|---|
| ¿Se sortea la capa? | Sí, se asigna una altura aleatoria al nuevo nodo para mantener la estructura jerárquica del índice. | No, se recorren las capas existentes porque los nodos ya tienen una altura asignada. |
| Valor de ef en las capas altas | Se utiliza `ef_construction` para controlar los candidatos considerados al conectar el nuevo nodo. | Se utiliza un `ef` pequeño para avanzar rápidamente por las capas superiores hacia la zona del resultado. |
| Parámetro de la fase principal | `ef_construction`, que controla la búsqueda de candidatos durante la construcción del índice. | `ef`, que controla cuántos candidatos se consideran durante la búsqueda. |
| ¿Modifica el índice? | Sí, añade el nodo y crea o actualiza sus conexiones. | No, solo recorre las conexiones existentes para encontrar los vecinos más cercanos. |



_Mi respuesta / análisis:_
R/: Durante la fase de inserción se agregan documentos al índice, determinando la altura de los nodos y estableciendo los enlaces necesarios para su futura localización. Por su parte, la fase de búsqueda simplemente navega a través de esta red de enlaces preexistente para encontrar la información pertinente, manteniendo intacta la estructura del índice.


## T4 · El invariante de FAISSVectorStore


## 1. Tabla de posiciones antes y después de borrar B

Antes de eliminar B, la lista de documentos y el índice FAISS están sincronizados. Si se elimina B únicamente de `self.documents`, las posiciones dejan de corresponder.

| Posición | Documento antes de borrar | Documento después de borrar | Vector FAISS |
|---|---|---|---|
| 0 | A | A | A |
| 1 | B | C | B |
| 2 | C | D | C |
| 3 | D | No existe | D |

Después de borrar B, la lista contiene `[A, C, D]`, pero FAISS conserva los vectores `[A, B, C, D]`.

Si una búsqueda devuelve los índices `[1, 2, 3]`, sucede lo siguiente:

- Índice 1: devuelve C, aunque el vector corresponde a B.
- Índice 2: devuelve D, aunque el vector corresponde a C.
- Índice 3: provoca un error porque la lista solo tiene tres elementos y las posiciones válidas son 0, 1 y 2.

## 2. ¿Qué resultado sería un error silencioso y cuál lanzaría una excepción?

Los índices 1 y 2 producen errores silenciosos, porque devuelven documentos equivocados sin avisar. El índice 3 lanza una excepción `IndexError`, porque intenta acceder a una posición que ya no existe en la lista.

## 3. Relación con las bases vectoriales como Qdrant

Las bases de datos vectoriales especializadas, como Qdrant, utilizan identificadores para relacionar los vectores con sus documentos, en lugar de depender únicamente de las posiciones de una lista de Python. Esto permite gestionar inserciones y eliminaciones de forma más segura y evita que una búsqueda devuelva documentos incorrectos por una desincronización.



## T5 · El costo de la fuerza bruta

Con ~768 millones de multiplicaciones por consulta y 100 consultas por segundo, ¿cuántas multiplicaciones por segundo necesitas?
R/: Contamos con 768 millones de vectores de 768 dimensiones procesando 100 consultas cada segundo.

La operación es:

768.000.000 × 768 × 100 = 58.982.400.000.000

Escribe la cifra en notación científica y explica por qué «ser exacto» deja de ser viable.
R/: 5,9 × 10¹³ multiplicaciones por segundo.
Realizar búsquedas exhaustivas obliga a cotejar la consulta contra la totalidad de los vectores guardados. Este proceso exige un volumen gigantesco de cómputo. Al manejar millones de registros y 100 peticiones por segundo, los requerimientos de memoria, energía y procesamiento se disparan, elevando significativamente los tiempos de respuesta y degradando el desempeño general.

Una frase: ¿qué se sacrifica en ANN (Approximate Nearest Neighbor) y qué se gana?
R/: El enfoque ANN (Approximate Nearest Neighbor) renuncia a la certeza absoluta de hallar el vecino más cercano exacto, a cambio de obtener una notable velocidad y eficiencia, lo que viabiliza atender un mayor volumen de consultas usando una fracción de los recursos.
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

