import math
import unittest

from metodos import (
    avaliar_modelo_exponencial,
    avaliar_polinomio,
    construir_coeficientes_polinomio,
    ajustar_minimos_quadrados_exponencial,
)


class TestLagrangeInterpolacao(unittest.TestCase):
    def test_coeficientes_lagrange_avaliam_nos_pontos(self) -> None:
        pontos = [(10, 0.5), (50, 2.1), (100, 4.8), (500, 22), (1000, 48.5)]
        coeficientes = construir_coeficientes_polinomio(pontos, "lagrange")

        for x, y in pontos:
            with self.subTest(x=x):
                self.assertAlmostEqual(avaliar_polinomio(coeficientes, x), y, places=8)

    def test_ajuste_exponencial_aproxima_os_dados(self) -> None:
        pontos = [(0, 1.0), (1, 2.7), (2, 7.4), (3, 20.1)]
        a, b = ajustar_minimos_quadrados_exponencial(pontos)

        for x, y in pontos:
            with self.subTest(x=x):
                self.assertAlmostEqual(avaliar_modelo_exponencial((a, b), x), y, places=1)


if __name__ == "__main__":
    unittest.main()
