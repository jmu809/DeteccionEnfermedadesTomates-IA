# Guion de exposición · 10 minutos

> **Duración objetivo:** ~10 min · 15 slides · ~38-40 segundos por slide de media
> **Consejo general:** no leas el slide, *cuéntalo*. Cada bloque te da la idea principal + lo que no aparece en la slide.

---

## Slide 1 · Portada (≈30 s)

**Saludo y presentación.**

> "Buenos días. Vamos a presentar nuestro proyecto de Inteligencia Artificial: un sistema de clasificación automática de enfermedades en hojas de tomate, usando Redes Neuronales Convolucionales y Transfer Learning sobre ResNet50."
>
> "Como adelanto, los tres números clave del trabajo: hemos entrenado sobre casi **247 000 imágenes** de **10 clases distintas**, y hemos conseguido un **96% de accuracy** en test con un **F1-score macro de 0,933**."

**Truco:** menciona los números pero no te detengas en ellos, ya vendrán al final.

---

## Slide 2 · El problema (≈40 s)

**Idea clave:** justificar *por qué* este problema importa y *qué* tipo de problema es técnicamente.

> "El tomate es uno de los cultivos más importantes del mundo (180 millones de toneladas al año) y especialmente clave para Almería. El problema es que las enfermedades foliares pueden tirar abajo entre el 20% y el 80% de una cosecha si no se detectan a tiempo, y el diagnóstico tradicional depende de un agrónomo que vaya al campo: es lento, caro y poco escalable."
>
> "Técnicamente, lo enfocamos como una **clasificación multiclase supervisada**: el sistema recibe una foto de una hoja y tiene que decir a cuál de estas 10 categorías pertenece, **9 enfermedades más planta sana**."

**Remate (importante):** "Distinguir bien entre clases no es un detalle: un fungicida no sirve contra una infección bacteriana, ni un insecticida contra un virus. Una clasificación errónea puede llevar a perder la planta."

---

## Slide 3 · Objetivos (≈25 s)

**Idea clave:** ir rápido. Es la slide de "carta de navegación", no te detengas demasiado.

> "El objetivo general era desarrollar un modelo de clasificación basado en CNN con Transfer Learning. Y desglosado en cuatro objetivos específicos: implementar un **pipeline completo** end-to-end, aplicar **Transfer Learning** con ResNet50 en dos fases, **gestionar el desbalance** de clases —y aquí spoiler: el dataset está muy desbalanceado, hay clases con 14 veces más imágenes que otras— y por último **evaluar rigurosamente** con métricas multiclase apropiadas."

---

## Slide 4 · Stack tecnológico (≈30 s)

**Idea clave:** no leer la tabla. Mencionar lo destacable.

> "El stack está construido sobre **Python 3.9 y PyTorch 2.8 con CUDA 12.8**. Usamos torchvision para el ResNet50 preentrenado, scikit-learn para las métricas y matplotlib y seaborn para las visualizaciones."
>
> "Donde sí merece la pena pararse es en el hardware: hemos entrenado sobre una **RTX 5080 con 16 GB de VRAM**, arquitectura Blackwell. Y dos decisiones importantes: usamos **PyTorch 2.8 específicamente porque Blackwell es muy nuevo** y necesita CUDA 12.8, y activamos **Mixed Precision en FP16**, que recorta a la mitad el consumo de VRAM y aproximadamente duplica la velocidad."

---

## Slide 5 · Dataset (≈45 s)

**Idea clave:** introducir el dataset Y revelar el problema del desbalance, que justificará decisiones más adelante.

> "El dataset es **Plant Village Augmented**, disponible en Kaggle. Es una versión aumentada del benchmark original, donde a cada imagen le han aplicado 17 tipos de aumentación distintos: ajustes de brillo, ruido, rotaciones, filtros Sobel, Laplaciano… Esto nos da casi **247 000 imágenes** procesadas a 224×224 píxeles."
>
> *(Señala el gráfico de la derecha)*
>
> "Pero el análisis exploratorio reveló un problema crítico que ves aquí: **la distribución está muy desbalanceada**. La clase Yellow Leaf Curl Virus tiene casi 73 000 imágenes —el 29,5% del total— mientras que Tomato Mosaic Virus tiene solo 5 083. Una **ratio de 14 a 1**. Esto va a marcar decisiones de diseño que veremos enseguida."

---

## Slide 6 · Preprocesamiento y data augmentation (≈45 s)

**Idea clave:** explicar la división en 3 conjuntos y *por qué* se aplican transformaciones distintas a train vs val/test.

> "Lo primero, dividimos el dataset de forma **estratificada** en 70% train, 15% validación y 15% test. La estratificación garantiza que cada conjunto mantenga la misma proporción de clases."
>
> "En cuanto a las transformaciones, aplicamos cosas distintas según el conjunto. **En entrenamiento** hacemos data augmentation: resize, recorte aleatorio, volteo horizontal, rotación de hasta 15 grados y variación de color. La idea es exponer al modelo a más variabilidad para que generalice mejor."
>
> "**En validación y test no se aplica nada aleatorio**. Solo redimensionar, convertir a tensor y normalizar. Porque si aplicáramos transformaciones aleatorias, la misma imagen daría resultados distintos en evaluaciones sucesivas y las métricas no serían fiables."
>
> "Un detalle técnico importante: la **normalización usa la media y desviación de ImageNet**, no del dataset propio. Es obligatorio cuando usas Transfer Learning, porque los pesos preentrenados esperan esa distribución de entrada."

---

## Slide 7 · Arquitectura del modelo (≈40 s)

**Idea clave:** explicar el flujo del modelo y las tres decisiones de diseño.

> "La arquitectura es la siguiente: entra una imagen 224×224 RGB, pasa por **ResNet50 preentrenado en ImageNet**, que actúa como extractor de características y nos devuelve un vector de 2048 dimensiones. Después le añadimos nuestra propia cabeza: una capa de Dropout, una capa lineal de 2048 a 10 neuronas y un Softmax que da la probabilidad para cada clase."
>
> *(Señala las tarjetas de abajo)*
>
> "Tres decisiones importantes: El **Dropout al 30%** desactiva neuronas aleatoriamente para que el modelo no dependa de ninguna característica concreta. La **cabeza nueva** son apenas 20 000 parámetros añadidos sobre los 23,5 millones del backbone. Y muy importante: la función de pérdida es **Cross-Entropy con pesos por clase**, inversamente proporcionales a la frecuencia, para penalizar más los errores en clases minoritarias."

---

## Slide 8 · Estrategia de entrenamiento (≈70 s — slide larga)

**Idea clave:** las dos fases y *por qué* es necesaria cada una. Esta es de las slides más técnicas.

> "Entrenamos durante 20 épocas en total, divididas en dos fases."
>
> "**Fase 1: Feature Extraction**, cinco épocas. Aquí el backbone de ResNet50 está **congelado**: solo entrenamos los 20 000 parámetros de la cabeza nueva. ¿Por qué? Porque cuando conectas una cabeza nueva a un backbone preentrenado, los primeros gradientes son enormes y caóticos. Si el backbone estuviera abierto en ese momento, esos gradientes destruirían los pesos que ResNet50 tardó **semanas** en aprender sobre ImageNet. Es lo que se llama *catastrophic forgetting*. Al final de la Fase 1 estamos en torno al 70% de accuracy."
>
> "**Fase 2: Fine-tuning completo**, 15 épocas. Aquí desbloqueamos todos los parámetros y dejamos que toda la red se adapte. Pero con un truco: **learning rates discriminativos**. El backbone se actualiza con un LR muy bajo, **diez veces más lento que la cabeza**, porque el backbone ya tiene conocimiento valioso de ImageNet y solo queremos afinarlo, no reescribirlo. La cabeza sí puede permitirse cambios más grandes."
>
> "Como optimizador usamos **AdamW** —Adam con weight decay desacoplado—, y un **scheduler CosineAnnealingLR** que reduce el LR suavemente siguiendo una curva coseno hasta cerca de cero."

---

## Slide 9 · Gestión del desbalance (≈30 s)

**Idea clave:** rescatar el problema introducido en slide 5 y explicar cómo lo combatimos.

> "Como vimos antes, había un desbalance de **14 a 1** entre la clase más numerosa y la menos numerosa. Lo hemos combatido con dos mecanismos complementarios:"
>
> "El primero, durante el muestreo, un **WeightedRandomSampler**. El DataLoader sobremuestrea las clases minoritarias y submuestrea las mayoritarias, de forma que cada batch tiene una distribución prácticamente uniforme."
>
> "El segundo, en la función de pérdida, **class weights**: penalizaciones mayores cuando se equivoca en clases minoritarias. Spoiler: aun así, la clase más pequeña va a seguir siendo la más débil."

---

## Slide 10 · Curvas de entrenamiento (≈70 s — slide clave)

**Idea clave:** ESTA ES LA SLIDE ESTRELLA. El "momento ajá" del proyecto.

> "Tenéis las curvas reales del entrenamiento: a la izquierda la **loss**, en el centro la **accuracy** y a la derecha el **F1 macro de validación**. La línea discontinua vertical marca el inicio del fine-tuning, en la época 6."
>
> *(Señala el primer tramo de la accuracy)*
>
> "Durante la **Fase 1**, las primeras cinco épocas, subimos del 63% al 70%. Apenas **7 puntos en 5 épocas**: avance lento, porque solo estamos entrenando la cabeza, el backbone no contribuye."
>
> *(Señala la línea discontinua)*
>
> "Y aquí pasa **lo más interesante del proyecto**. En la transición entre Fase 1 y Fase 2, al desbloquear el backbone, **en una sola época saltamos del 70% al 87%**. **17 puntos en una época**, más del doble de lo que la Fase 1 entera consiguió en cinco. Lo vemos en los tres paneles: la loss se desploma, la accuracy se dispara y el F1 macro también. Esto confirma que el verdadero poder del Transfer Learning está en el fine-tuning."
>
> *(Señala la convergencia final en la accuracy)*
>
> "Y a partir de ahí, la Fase 2 converge suavemente hasta el 96%. Pero fijaos en una cosa importante: las curvas de train y validación están **prácticamente solapadas todo el tiempo**. Eso significa que el modelo **no está sobreajustando**, está generalizando bien."

---

## Slide 11 · Métricas finales (≈25 s)

**Idea clave:** vendamos los resultados. Habla con seguridad.

> "Estos son los resultados sobre el conjunto de test, que el modelo nunca había visto: **96% de accuracy y un F1-score macro de 0,933**, con **9 de las 10 clases superando F1 = 0,90**."
>
> "Para contextualizar: una clasificación aleatoria entre 10 clases daría un 10%. Un clasificador que siempre predijera la clase mayoritaria daría 29,5%. Estamos muy por encima de ambas referencias, y en línea con el estado del arte publicado."

---

## Slide 12 · F1-Score por clase (≈40 s)

**Idea clave:** mostrar que el rendimiento es heterogéneo y que la causa principal es el desbalance.

> "Aquí tenéis el F1 desglosado clase por clase. La línea verde discontinua marca el umbral de 0,90 y la naranja el 0,70."
>
> "**Las dos clases con mejor F1, 0,974**, son Yellow Curl y Healthy. Son las más numerosas y las visualmente más distintivas: hojas enrolladas y amarillas en un caso, verde uniforme sin manchas en el otro."
>
> "**Ocho clases más están por encima de 0,90**. Y solo **Tomato Mosaic Virus se queda en 0,878**, en naranja. Es la más minoritaria del dataset. Y un detalle interesante: tiene **recall muy alto pero precisión más baja**, en torno al 0,78. Es decir, el modelo *sobreclasifica* esta clase: dice 'mosaic virus' demasiado a menudo. Es una consecuencia directa del WeightedRandomSampler — al sobrerrepresentarla durante el entrenamiento, el modelo aprende a sospechar de ella demasiado fácilmente."

---

## Slide 13 · Matriz de confusión (≈55 s — slide nueva)

**Idea clave:** la matriz confirma que los errores tienen sentido. Tómate tu tiempo en señalar.

> "Esta es la matriz de confusión. A la **izquierda en valores absolutos**, a la **derecha normalizada**. La diagonal son los aciertos, lo de fuera de la diagonal son los errores."
>
> *(Señala la diagonal de la matriz normalizada)*
>
> "Lo primero que veis es que **toda la diagonal está por encima de 0,91**. Todas las clases se identifican correctamente más del 91% de las veces. **Mosaic Virus llega al 1,00** de recall: cuando una imagen es realmente Mosaic Virus, el modelo siempre la detecta."
>
> *(Señala fuera de la diagonal)*
>
> "Pero lo más interesante son los errores. **No son aleatorios**. Mirad por ejemplo: **Target Spot se confunde con Spider Mites el 7% de las veces** —es el error más concentrado—. **Bacterial Spot tiene errores repartidos con Septoria y con Yellow Curl**, porque las tres producen manchas pequeñas. Y **Yellow Curl, al ser la clase mayoritaria, absorbe pequeños errores de muchas otras clases**, pero en proporción solo es el 5%."
>
> *(Remate)*
>
> "Lo importante: **el modelo nunca confunde Healthy con Late Blight o con Mosaic Virus**. Es decir, no falla de forma absurda. Falla solo donde un agrónomo humano también dudaría sin ayuda de laboratorio. Los errores son **semánticamente coherentes**."

---

## Slide 14 · Análisis de errores (≈45 s)

**Idea clave:** matizar el 96%. Al inspeccionar los errores manualmente descubrimos algo importante.

> "Al inspeccionar visualmente los ejemplos mal clasificados, encontramos **dos tipos de fallo muy distintos**."
>
> "**Categoría 1: aumentaciones extremas.** El dataset incluye filtros como Sobel, Laplaciano y High Pass, que transforman la imagen en un mapa de bordes en escala de grises. **Se pierde toda la información de color**, que es justo lo más importante para distinguir enfermedades foliares. Una parte significativa de los errores cae aquí."
>
> "Esto tiene una implicación importante: **el 96% medido es una estimación conservadora**. En un caso de uso real, donde un agricultor saca una foto con el móvil, este tipo de imágenes no existen. **El rendimiento práctico sería superior al 96%.**"
>
> "**Categoría 2:** los errores 'legítimos' entre clases visualmente similares: los pares Bacterial/Septoria y Early Blight/Target Spot, que ya hemos visto en la matriz de confusión."

---

## Slide 15 · Conclusiones y trabajo futuro (≈40 s)

**Idea clave:** cerrar con cuatro mensajes principales y dejar abiertas las líneas siguientes.

> "Cuatro conclusiones principales:"
>
> "Uno, **viabilidad técnica confirmada**: 96% de accuracy y F1 macro 0,933, competitivo con el estado del arte."
>
> "Dos, **el Transfer Learning ha sido decisivo**: 20 épocas en una GPU de consumo. Sin preentrenamiento, esto habría requerido millones de imágenes y semanas de cómputo."
>
> "Tres, **la estrategia de dos fases es esencial**: el salto de 17 puntos al iniciar la Fase 2 lo demuestra."
>
> "Y cuatro, **el desbalance es el principal limitante**: la única clase que no llega a F1 = 0,90 es precisamente la más minoritaria. Recopilar más datos de Tomato Mosaic Virus sería la intervención más efectiva."
>
> "De cara al futuro, vemos cuatro líneas: ampliar el dataset, limpiar el test set quitando los filtros extremos, pasar a **detección de objetos** con YOLO para localizar la zona enferma, y aplicar **técnicas de XAI como Grad-CAM** para visualizar en qué partes de la imagen se fija el modelo."
>
> "**Muchas gracias. ¿Alguna pregunta?**"

---

# Posibles preguntas del tribunal (prepara estas)

**P: ¿Por qué ResNet50 y no algo más moderno como EfficientNet o ViT?**
> "ResNet50 está muy estudiado, tiene pesos preentrenados muy estables y para 250 000 imágenes es más que suficiente. Modelos más grandes podrían sobreajustar o requerir más datos para justificar su capacidad. Como trabajo futuro sería interesante comparar."

**P: ¿Cómo aseguráis que no hay fuga de datos entre train, val y test?**
> "La división es estratificada y aleatoria pero con semilla fija. Como cada imagen aumentada se considera independiente, comprobamos que no haya repetidos. Al hacer la división antes del entrenamiento, garantizamos que el modelo no ve test durante el ajuste."

**P: ¿Por qué accuracy y F1 macro y no otras métricas?**
> "Accuracy da una visión global pero engaña con desbalance. F1 macro promedia el F1 de cada clase tratándolas por igual, así que captura el rendimiento real incluso en las minoritarias. La matriz de confusión la usamos para análisis cualitativo."

**P: En la matriz de confusión, ¿por qué Target Spot se confunde con Spider Mites? Son cosas distintas (un hongo vs un ácaro).**
> "Buena observación. Visualmente ambos producen un punteado distribuido por la hoja en sus fases iniciales: Target Spot da pequeños puntos antes de formar los anillos concéntricos característicos, y Spider Mites también deja un punteado fino por la succión del ácaro. En imágenes de baja calidad o fases tempranas, la distinción es sutil. Por eso vemos ese 7%."

**P: ¿Qué pasaría con una hoja sana de otra planta que no sea tomate?**
> "El modelo solo ha visto hojas de tomate. Ante una hoja de otra especie, probablemente la clasificaría con alta confianza en alguna de las 10 clases conocidas, porque no tiene una clase 'desconocido'. Una mejora sería añadir detección de out-of-distribution."

**P: ¿Cuánto tardó el entrenamiento?**
> "20 épocas sobre 173 000 imágenes en una RTX 5080 con Mixed Precision FP16. En total, en torno a [poned aquí el tiempo real que tardó vuestro entrenamiento]."

**P: ¿Por qué Dropout en la cabeza pero no en el backbone?**
> "ResNet50 ya tiene su propia regularización (BatchNorm) y los pesos preentrenados ya son robustos. El Dropout lo aplicamos solo en la cabeza nueva, que es lo que más riesgo tiene de sobreajustar porque se entrena desde cero."

---

# Tips finales para el día

- **Cronometra una vez en casa.** Algunas slides irán más rápido y otras más lento de lo previsto.
- **Si vas mal de tiempo:** acorta slide 6 (preprocesamiento), slide 9 (desbalance) y slide 14 (análisis de errores).
- **Si te sobra tiempo:** alarga la slide 10 (curvas) y la slide 13 (matriz de confusión). Son las que más impresionan.
- **Si te quedas en blanco:** vuelve siempre al título del slide. Cada título es ya el mensaje principal.
- **No leas la slide.** El tribunal lee mucho más rápido que tú hablas. Cuenta la historia.
- **Señala los gráficos cuando los menciones.** Slides 5, 10, 12 y 13 lo piden.
