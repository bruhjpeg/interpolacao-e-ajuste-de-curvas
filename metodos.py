from __future__ import annotations

from typing import List, Tuple

# Tipo usado para armazenar um ponto (x, y).
Ponto = Tuple[float, float]


def _multiplicar_por_fator_linear(pol: List[float], raiz: float) -> List[float]:
    """Multiplica um polinômio por (x - raiz)."""
    resultado = [0.0] * (len(pol) + 1)
    for i, coef in enumerate(pol):
        resultado[i] += -raiz * coef
        resultado[i + 1] += coef
    return resultado


def construir_coeficientes_polinomio(pontos: List[Ponto], metodo: str, grau: int | None = None) -> List[float]:
    """Retorna os coeficientes do polinômio para o método escolhido."""
    if metodo == "lagrange":
        return _coeficientes_lagrange(pontos)
    if metodo == "newton":
        return _coeficientes_newton(pontos)
    if metodo == "minimos_quadrados":
        if grau is None:
            raise ValueError("O grau é obrigatório para mínimos quadrados.")
        return ajustar_minimos_quadrados(pontos, grau)
    raise ValueError("Método inválido.")


def _coeficientes_lagrange(pontos: List[Ponto]) -> List[float]:
    """Constrói os coeficientes do polinômio interpolador de Lagrange."""
    grau = len(pontos) - 1
    resultado = [0.0] * (grau + 1)

    for i, (xi, yi) in enumerate(pontos):
        base = [1.0]
        denominador = 1.0
        for j, (xj, _) in enumerate(pontos):
            if i == j:
                continue
            base = _multiplicar_por_fator_linear(base, xj)
            denominador *= xi - xj

        for poder, coef in enumerate([c / denominador for c in base]):
            resultado[poder] += yi * coef

    return [round(c, 12) for c in resultado if abs(c) > 1e-12]


def _coeficientes_newton(pontos: List[Ponto]) -> List[float]:
    """Constrói os coeficientes do polinômio interpolador de Newton."""
    diferencas = [[0.0] * len(pontos) for _ in range(len(pontos))]
    for i, (_, yi) in enumerate(pontos):
        diferencas[i][0] = yi

    for j in range(1, len(pontos)):
        for i in range(len(pontos) - j):
            numerador = diferencas[i + 1][j - 1] - diferencas[i][j - 1]
            denominador = pontos[i + j][0] - pontos[i][0]
            diferencas[i][j] = numerador / denominador

    polinomio = [diferencas[0][0]]
    for i in range(1, len(pontos)):
        polinomio = _multiplicar_por_fator_linear(polinomio, pontos[i - 1][0])
        polinomio[0] += diferencas[0][i]

    return [round(c, 12) for c in polinomio]


def avaliar_polinomio(coeficientes: List[float], x: float) -> float:
    """Avalia um polinômio cujos coeficientes estão em ordem crescente de grau."""
    valor = 0.0
    for coef in reversed(coeficientes):
        valor = valor * x + coef
    return valor


def formatar_polinomio(coeficientes: List[float]) -> str:
    """Converte os coeficientes em uma string legível para o polinômio."""
    termos = []
    for grau, coef in enumerate(coeficientes):
        if abs(coef) < 1e-12:
            continue
        if grau == 0:
            termo = f"{coef:.6g}"
        elif grau == 1:
            termo = f"{coef:.6g}x"
        else:
            termo = f"{coef:.6g}x^{grau}"
        termos.append(termo)
    return "0" if not termos else " + ".join(termos)


def plotar_polinomio(pontos: List[Ponto], coeficientes: List[float], caminho_salvar: str | None = None) -> None:
    """Plota os pontos e o polinômio correspondente."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valores_x = [x for x, _ in pontos]
    valores_y = [y for _, y in pontos]
    xmin, xmax = min(valores_x) - 1, max(valores_x) + 1
    eixos_x = [xmin + (xmax - xmin) * i / 200 for i in range(201)]
    eixos_y = [avaliar_polinomio(coeficientes, x) for x in eixos_x]

    plt.figure(figsize=(7, 4))
    plt.plot(eixos_x, eixos_y, label="Polinômio", color="tab:blue")
    plt.scatter(valores_x, valores_y, color="tab:red", label="Pontos")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Interpolação / Ajuste de Curvas")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if caminho_salvar:
        plt.savefig(caminho_salvar)
    plt.close()


def interpolacao_lagrange(pontos: List[Ponto], x: float) -> float:
    """Calcula o valor interpolado pelo método de Lagrange."""
    if not pontos:
        raise ValueError("A lista de pontos não pode estar vazia.")

    xs = [xi for xi, _ in pontos]
    if len(set(xs)) != len(xs):
        raise ValueError("Os pontos precisam ter valores de x diferentes para usar Lagrange.")

    valor = 0.0
    for i, (xi, yi) in enumerate(pontos):
        base = 1.0
        for j, (xj, _) in enumerate(pontos):
            if i != j:
                base *= (x - xj) / (xi - xj)
        valor += yi * base
    return valor


def interpolacao_newton(pontos: List[Ponto], x: float) -> float:
    """Calcula o valor interpolado pelo método de Newton."""
    if not pontos:
        raise ValueError("A lista de pontos não pode estar vazia.")

    diferencas = [[0.0] * len(pontos) for _ in range(len(pontos))]
    for i, (_, yi) in enumerate(pontos):
        diferencas[i][0] = yi

    for j in range(1, len(pontos)):
        for i in range(len(pontos) - j):
            numerador = diferencas[i + 1][j - 1] - diferencas[i][j - 1]
            denominador = pontos[i + j][0] - pontos[i][0]
            diferencas[i][j] = numerador / denominador

    valor = 0.0
    for i in range(len(pontos)):
        termo = 1.0
        for j in range(i):
            termo *= x - pontos[j][0]
        valor += diferencas[0][i] * termo
    return valor


def ajustar_minimos_quadrados(pontos: List[Ponto], grau: int) -> List[float]:
    """Ajusta um polinômio aos pontos por mínimos quadrados."""
    if not pontos:
        raise ValueError("A lista de pontos não pode estar vazia.")
    if grau < 0:
        raise ValueError("O grau do polinômio deve ser não negativo.")
    if len(pontos) < grau + 1:
        raise ValueError("São necessários pelo menos grau+1 pontos para o ajuste.")

    m = grau + 1
    matriz = [[0.0] * m for _ in range(m)]
    vetor = [0.0] * m

    for i in range(m):
        for j in range(m):
            matriz[i][j] = sum((x ** i) * (x ** j) for x, _ in pontos)
        vetor[i] = sum((x ** i) * y for x, y in pontos)

    for i in range(m):
        pivô = max(range(i, m), key=lambda r: abs(matriz[r][i]))
        if abs(matriz[pivô][i]) < 1e-12:
            raise ValueError("Não foi possível resolver o sistema de mínimos quadrados.")
        matriz[i], matriz[pivô] = matriz[pivô], matriz[i]
        vetor[i], vetor[pivô] = vetor[pivô], vetor[i]
        for j in range(i + 1, m):
            fator = matriz[j][i] / matriz[i][i]
            for k in range(i, m):
                matriz[j][k] -= fator * matriz[i][k]
            vetor[j] -= fator * vetor[i]

    coeficientes = [0.0] * m
    for i in range(m - 1, -1, -1):
        soma = vetor[i]
        for j in range(i + 1, m):
            soma -= matriz[i][j] * coeficientes[j]
        coeficientes[i] = soma / matriz[i][i]

    return [round(c, 12) for c in coeficientes]


def _obter_coeficientes(pontos: List[Ponto], opcao: str, grau: int) -> List[float]:
    """Escolhe os coeficientes com base no método selecionado."""
    if opcao == "1":
        return construir_coeficientes_polinomio(pontos, "lagrange")
    if opcao == "2":
        return construir_coeficientes_polinomio(pontos, "newton")
    return ajuste_minimos_quadrados(pontos, grau)


def executar_menu_interativo() -> None:
    """Exibe um menu interativo no terminal para escolher o método e avaliar o polinômio."""
    print("=" * 60)
    print("INTERPOLAÇÃO E AJUSTE DE CURVAS")
    print("=" * 60)

    while True:
        print("\nEscolha um método:")
        print("1 - Interpolação de Lagrange")
        print("2 - Interpolação de Newton")
        print("3 - Ajuste por Mínimos Quadrados")
        print("0 - Sair")
        opcao = input("Opção: ").strip()

        if opcao == "0":
            print("Encerrando...")
            break
        if opcao not in {"1", "2", "3"}:
            print("Opção inválida.")
            continue

        n = int(input("\nQuantidade de pontos: ").strip())
        pontos: List[Ponto] = []
        valores = input(f"Digite os pontos no formato x0:y0:x1:y1:...:x{n - 1}:y{n - 1}: ").split(":")
        if len(valores) != 2 * n:
            print("Quantidade de valores incompatível com o número de pontos informado.")
            continue

        for i in range(0, 2 * n, 2):
            pontos.append((float(valores[i]), float(valores[i + 1])))

        print("\nDeseja:")
        print("1 - Encontrar P(x)")
        print("2 - Apenas mostrar o polinômio")
        print("3 - Plotar o gráfico")
        acao = input("Opção: ").strip()

        if acao == "1":
            valor_x = float(input("Digite x: ").strip())
            grau = int(input("Grau do polinômio (para mínimos quadrados): ").strip()) if opcao == "3" else 0
            coeficientes = _obter_coeficientes(pontos, opcao, grau)
            valor = avaliar_polinomio(coeficientes, valor_x) if opcao != "1" and opcao != "2" else (
                interpolacao_lagrange(pontos, valor_x) if opcao == "1" else interpolacao_newton(pontos, valor_x)
            )
            print(f"\nPolinômio: P(x) = {formatar_polinomio(coeficientes)}")
            print(f"P({valor_x}) = {valor}")
        elif acao == "2":
            grau = int(input("Grau do polinômio (para mínimos quadrados): ").strip()) if opcao == "3" else 0
            coeficientes = _obter_coeficientes(pontos, opcao, grau)
            print(f"\nPolinômio: P(x) = {formatar_polinomio(coeficientes)}")
        elif acao == "3":
            grau = int(input("Grau do polinômio (para mínimos quadrados): ").strip()) if opcao == "3" else 0
            coeficientes = _obter_coeficientes(pontos, opcao, grau)
            caminho = input("Nome do arquivo para salvar o gráfico (opcional, pressione Enter para só mostrar): ").strip()
            plotar_polinomio(pontos, coeficientes, caminho_salvar=caminho or None)
        else:
            print("Opção inválida.")


if __name__ == "__main__":
    executar_menu_interativo()
