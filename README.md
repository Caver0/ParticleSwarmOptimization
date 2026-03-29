# ParticleSwarmOptimization

Particle Swarm Optimization (PSO) Lab

Este proyecto implementa el algoritmo de optimización por enjambre de partículas (Particle Swarm Optimization, PSO) con una arquitectura modular orientada a la extensibilidad, la reproducibilidad y la comparación de distintas estrategias de evaluación.

---

## Estructura del proyecto

```text
PSO/
├── src/
│   └── pso_lab/
│       ├── core/         # Núcleo del algoritmo PSO
│       ├── objectives/   # Funciones objetivo
│       ├── parallel/     # Estrategias de evaluación (V0, V1, etc.)
│       ├── experiments/  # Ejecución de experimentos
│       ├── io/           # Entrada y salida de datos
│       └── viz/          # Visualización de resultados
│
├── tests/                # Tests unitarios y de integración
├── configs/              # Archivos de configuración
├── docs/                 # Documentación del diseño
├── results/              # Resultados generados
│
├── run_pso.py            # Ejecución de una instancia de PSO
├── run_benchmarks.py     # Comparación entre variantes
├── run_grid_search.py    # Búsqueda de hiperparámetros
├── analyze_results.py    # Análisis de resultados
├── make_viz.py           # Generación de visualizaciones
│
├── README.md
└── pyproject.toml
```

---

## Descripción de los módulos

### core/

Contiene la implementación del algoritmo PSO:

- Inicialización de partículas
- Actualización de velocidades y posiciones
- Gestión de mejores soluciones locales y globales
- Control de parámetros y topologías

Este módulo constituye el núcleo del sistema y no debe depender de componentes externos como visualización o almacenamiento.

---

### objectives/

Define las funciones objetivo utilizadas en los experimentos (por ejemplo, Sphere, Rosenbrock, Rastrigin o Ackley).

Se mantienen separadas del núcleo para permitir reutilizar el algoritmo con distintos problemas.

---

### parallel/

Implementa diferentes estrategias de evaluación del fitness:

- Evaluación secuencial (V0)
- Evaluación paralela mediante hilos (V1)

El objetivo es comparar rendimiento sin modificar la lógica principal del algoritmo.

---

### experiments/

Gestiona la ejecución de experimentos:

- Configuración y lanzamiento de ejecuciones
- Comparación entre variantes
- Benchmarks y estudios de rendimiento

---

### io/

Responsable de la gestión de entrada y salida:

- Lectura de configuraciones
- Almacenamiento de resultados en formatos estructurados (JSON, CSV, NPZ)

---

### viz/

Incluye herramientas para la visualización:

- Gráficas de convergencia
- Representación de trayectorias
- Generación de animaciones

---

### tests/

Contiene pruebas unitarias y de integración para validar:

- Reproducibilidad
- Correctitud del algoritmo
- Consistencia entre distintas variantes

---

### configs/

Archivos de configuración que definen parámetros del algoritmo y de los experimentos.

---

### results/

Directorio destinado exclusivamente a resultados generados durante la ejecución:

- Históricos
- Resúmenes
- Archivos de salida
