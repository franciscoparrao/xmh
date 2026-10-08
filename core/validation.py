"""Validacion de la instrumentacion de un algoritmo.

El modo de falla de este framework no es ruidoso. Si un operador queda
registrado pero el bucle de evolucion no lo aplica, o si neutralizarlo no tiene
efecto, su valor de Shapley sale exactamente cero: indistinguible de un jugador
nulo legitimo. Ha ocurrido dos veces en este proyecto, con la aceptacion voraz
de DE en linea en vez de enrutada por el operador, y con PSO registrando
``velocity_update`` mientras la fabrica de neutrales solo conocia ``velocity``.

El chequeo ``v(emptyset) = 0`` que la Proposicion 1 evalua empiricamente **no
detecta ninguno de los dos**. En la coalicion vacia todos los operadores estan
neutralizados, el individuo de prueba coincide con el objetivo, nada se acepta y
la mejora es nula tanto si la seleccion estaba bien enrutada como si no. Es
condicion necesaria, no suficiente. Verificado inyectando el defecto: con la
seleccion de DE sin enrutar, ``v(emptyset) = 0`` sigue valiendo y el Shapley de
la seleccion sale +0.0000.

La condicion que si lo detecta es la que el metodo necesita de todos modos:
neutralizar un operador tiene que cambiar el resultado de la corrida. Un
operador cuya neutralizacion no cambia nada recibe cero por construccion, y eso
es indistinguible de un jugador nulo hasta que alguien lo mira.

Trampa de la API, encontrada al escribir esto: ``disable_operator`` por si solo
**no sobrevive a** ``run()``, porque ``run()`` vuelve a llamar a
``_initialize_operators`` y reconstruye el registro. La evaluacion de coaliciones
parchea ``_initialize_operators`` para re-aplicar la desactivacion despues de
cada reinicializacion; esta funcion usa el mismo parcheo. Quien llame a
``disable_operator`` y luego a ``run()`` obtiene el algoritmo completo, en
silencio.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np


def _corre(algorithm_class, kwargs, objective_func, desactivar: List[str],
           seeds: List[int]) -> List[float]:
    """Corre el algoritmo con `desactivar` neutralizado, con el mismo parcheo
    que usa la evaluacion de coaliciones."""
    out = []
    for s in seeds:
        alg = algorithm_class(**{**kwargs, "seed": s})
        if desactivar:
            original = alg._initialize_operators

            def parcheado(_self=alg, _orig=original, _ops=list(desactivar)):
                _orig()
                for op in _ops:
                    _self.disable_operator(op)

            alg._initialize_operators = parcheado
        out.append(alg.run(objective_func, verbose=False)["best_fitness"])
    return out


def validate_instrumentation(
    algorithm_class,
    algorithm_kwargs: Dict[str, Any],
    objective_func: Callable,
    n_runs: int = 5,
    base_seed: int = 0,
    raise_on_failure: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Comprueba que neutralizar cada operador cambie el resultado.

    Para cada operador registrado corre ``n_runs`` semillas con la coalicion
    completa y las mismas semillas con ese operador neutralizado. Si las dos
    series coinciden exactamente, neutralizarlo no hizo nada: su Shapley sera
    cero por construccion y no por medicion.

    Devuelve, por operador, ``max_abs_diff``, ``runs_differing`` y ``ok``. Un
    ``ok=False`` no prueba que haya un bug, prueba que el valor de ese operador
    no es interpretable sin revisar su registro a mano.
    """
    sonda = algorithm_class(**{**algorithm_kwargs, "seed": base_seed})
    sonda._initialize_operators()
    nombres = sonda.get_operators()
    seeds = [base_seed + i for i in range(n_runs)]

    completa = _corre(algorithm_class, algorithm_kwargs, objective_func, [], seeds)
    informe: Dict[str, Dict[str, Any]] = {}
    for nombre in nombres:
        sin = _corre(algorithm_class, algorithm_kwargs, objective_func, [nombre], seeds)
        difs = [abs(a - b) for a, b in zip(completa, sin)]
        distintas = sum(d > 0 for d in difs)
        ok = distintas > 0
        informe[nombre] = {"max_abs_diff": float(np.max(difs)),
                           "runs_differing": distintas, "n_runs": n_runs, "ok": ok}
        if not ok and raise_on_failure:
            raise RuntimeError(
                f"instrumentacion sospechosa en '{nombre}': neutralizarlo no cambio "
                f"el resultado en ninguna de las {n_runs} corridas. Su valor de Shapley "
                f"seria cero por construccion. Revisar que el bucle de evolucion lo "
                f"aplique a traves del operador registrado y que exista un neutral para "
                f"su tipo."
            )
    return informe
