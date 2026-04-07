# ParticleSwarmOptimization

Proyecto de PSO para comparar distintas formas de evaluar el fitness sin tocar el nucleo del algoritmo. La idea del proyecto es sencilla: poder lanzar ejecuciutables reproducibles, guardar resultados en JSON y luego convertirlos en tablas y graficas que ayuden a la comprensión del programa.

## Estructura del proyecto

```text
ParticleSwarmOptimization/
├── src/
│   └── pso_lab/
│       ├── core/
│       │   ├── config.py          # Parametros de un PSO
│       │   ├── models.py          # Modelos auxiliares y metricas temporales
│       │   ├── boundaries.py      # Gestion de limites del espacio de busqueda
│       │   └── optimizer.py       # Implementacion principal del algoritmo
│       ├── objectives/
│       │   ├── base.py            # Interfaz comun para funciones objetivo
│       │   ├── benchmarks.py      # Sphere, Rosenbrock, Rastrigin y Ackley
│       │   └── __init__.py        # Fabrica build_objective(...)
│       ├── parallel/
│       │   └── evaluators.py      # Evaluacion sequential, threading y multiprocessing
│       ├── experiments/
│       │   ├── runner.py          # Realiza una ejecución completa con el modo elegido
│       │   ├── summary.py         # Resume varias ejecuciones en medias, minimos y maximos
│       │   └── pyswarm_runner.py  # Adaptador para comparar contra pyswarm
│       ├── io/
│       │   ├── logging_utils.py   # Configuracion de logs
│       │   └── results.py         # Guardado de resultados y resumenes en JSON
│       ├── viz/
│       │   └── plots.py           # Graficas de convergencia, tiempos y trayectorias
│       └── cli.py                 # Argumentos compartidos por los scripts
├── tests/                         # Tests del nucleo, evaluadores, plots y persistencia
├── results/                       # Resultados crudos de benchmarks y comparativas
├── reports/
│   └── plots/                     # Figuras generadas a partir de los resultados
├── logs/                          # Logs de ejecucion con marca de tiempo
├── run_pso.py                     # Ejecución individual de PSO
├── run_benchmarks.py              # Comparativa entre modos de evaluacion
├── run_grid_search.py             # Barrido de hiperparametros w, c1 y c2
├── run_best_configs_comparison.py # Reejecuta las mejores configuraciones encontradas
├── run_pyswarm_baseline.py        # Comparacion entre esta implementacion y pyswarm
├── analyze_results.py             # Lee results/ y genera tablas y graficas
├── make_viz.py                    # Genera visualizaciones del movimiento de particulas
├── _repo_bootstrap.py             # Hace visible src/ al ejecutar desde la raiz
└── README.md
```

## Que hace cada parte

`src/pso_lab/core` es el corazon del proyecto. Aqui se encuentra el optimizador, la configuracion de cada ejecución y la logica de los limites. 

`src/pso_lab/objectives` reune las funciones objetivo de benchmark. Esta separado del nucleo para que el algoritmo pueda reutilizarse con otros problemas sin mezclar responsabilidades.

`src/pso_lab/parallel` contiene las estrategias de evaluacion del fitness. En este repo se comparan tres variantes: `sequential`, `threading` y `multiprocessing`.

`src/pso_lab/experiments` conecta las piezas anteriores para lanzar experimentos completos, resumir varias semillas y, cuando hace falta, comparar contra `pyswarm`.

`src/pso_lab/io` se encarga de guardar resultados y preparar logs legibles. Es la parte que evita que cada script acabe escribiendo JSON a mano.

`src/pso_lab/viz` transforma los resultados en graficas utiles: convergencia, tiempo frente a calidad y trayectorias de particulas.

`tests/` cubre lo importante: correctitud del optimizador, consistencia de evaluadores, salida de resultados y generacion de plots.

## Scripts principales

`run_pso.py` sirve para una ejecución individual y es la forma mas rapida de comprobar que todo esta bien.

`run_benchmarks.py` compara los modos de evaluacion con varias dimensiones, objetivos y semillas. Es el script mas directo para medir tiempos y calidad.

`run_grid_search.py` busqueda los mejores hiperparametros. Es util cuando no interesa tanto una ejecución concreta como encontrar combinaciones razonables de `w`, `c1` y `c2`.

`run_best_configs_comparison.py` toma las mejores configuraciones ya seleccionadas y las vuelve a ejecutar para compararlas de forma mas limpia entre modos.

`run_pyswarm_baseline.py` Compara los resultados obtenidos por el PSO desarrollado durante el proyecto y la librería ya definda `pyswarm`

`analyze_results.py` y `make_viz.py` son el cierre natural del flujo: uno resume y grafica, el otro ayuda a ver el movimiento del enjambre.

## Nota de uso

Los scripts de la raiz se pueden lanzar directamente desde VS Code con el boton de ejecutar. Cada uno lleva un bloque `if __name__ == "__main__"` con argumentos editables para no depender de escribir la orden completa en terminal.

## Lectura rapida de las comparativas

Con los benchmarks que hay ahora mismo, la conclusion mas clara es que `sequential` sale mejor parado en tiempo casi siempre. En los resumenes de `best_config_comparison`, sus tiempos medios se mueven alrededor de `0.006 - 0.016 s`, mientras que `threading` suele quedar en `0.026 - 0.05 s` y `multiprocessing` ronda `0.23 - 0.31 s`. Para funciones objetivo ligeras y escritas en Python puro, el coste extra de paralelizar pesa mas que la ganancia.

Eso no significa que `threading` o `multiprocessing` no tengan sentido. `threading` puede ser interesante si la evaluacion del fitness bloquea, hace entrada/salida o delega trabajo a librerias que liberan el GIL. `multiprocessing` tiene mas recorrido cuando cada evaluacion es realmente cara y el overhead de crear procesos deja de ser el cuello de botella.

Frente a `pyswarm`, no hay un ganador absoluto. En estas pruebas ligeras `pyswarm` suele ser mas rapido en tiempo bruto, pero la implementacion propia ofrece mas control, mas trazabilidad y mejores herramientas para inspeccionar la ejecución.

Las tres variantes producen resultados numéricamente equivalentes, lo que valida la correctitud de la abstracción de evaluador. Sin embargo, en la configuración experimental utilizada, la versión `sequential` resulta más rápida que las versiones paralelas. Esto se explica por el bajo coste computacional de cada evaluación y por el overhead asociado a la gestión de hilos y procesos. En particular, `threading` se ve limitado por el GIL de Python en cargas CPU-bound, mientras que `multiprocessing` incurre en costes adicionales de serialización y comunicación entre procesos.

### Repositorio

Codigo fuente y seguimiento del proyecto:
[GitHub - NOMBRE-DEL-REPO](https://github.com/Caver0/ParticleSwarmOptimization.git)