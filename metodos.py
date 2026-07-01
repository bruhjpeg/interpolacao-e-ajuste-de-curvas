from __future__ import annotations

import math
from typing import List, Tuple

# Este módulo reúne as principais ferramentas do projeto.
# Ele implementa três abordagens para trabalhar com dados discretos:
# 1) interpolação de Lagrange,
# 2) interpolação de Newton,
# 3) ajuste por mínimos quadrados.
#
# A ideia central é transformar um conjunto de pontos (x, y) em uma função
# que aproxime ou passe exatamente pelos valores fornecidos, dependendo do método.

# Tipo usado para representar um ponto do plano cartesiano.
Ponto = Tuple[float, float]


def _multiplicar_por_fator_linear(pol: List[float], raiz: float) -> List[float]:
    """Multiplica um polinômio por (x - raiz).

    Esta função é usada como peça auxiliar para montar polinômios a partir
    de fatores lineares. Em termos práticos, ela transforma um polinômio
    representado por coeficientes em um novo polinômio que já contém o fator
    (x - raiz).
    """
    resultado = [0.0] * (len(pol) + 1)
    for i, coef in enumerate(pol):
        # O coeficiente atual é deslocado para a direita em uma posição,
        # porque multiplicar por (x - raiz) aumenta o grau em 1.
        resultado[i] += -raiz * coef
        resultado[i + 1] += coef
    return resultado


def construir_coeficientes_polinomio(pontos: List[Ponto], metodo: str, grau: int | None = None) -> List[float]:
    """Retorna os coeficientes do polinômio correspondente ao método escolhido.

    Os coeficientes são devolvidos em ordem crescente de grau, que é a forma
    usada pelas funções de avaliação e formatação do projeto.
    """
    if metodo == "lagrange":
        return _coeficientes_lagrange(pontos)
    if metodo == "newton":
        return _coeficientes_newton(pontos)
    if metodo == "minimos_quadrados":
        if grau is None:
            raise ValueError("O grau é obrigatório para mínimos quadrados.")
        return ajustar_minimos_quadrados(pontos, grau)
    raise ValueError("Método inválido.")


def _normalizar_coeficientes(coeficientes: List[float]) -> List[float]:
    """Remove termos nulos no final sem arredondar demais os coeficientes."""
    coeficientes = [float(coef) for coef in coeficientes]
    while len(coeficientes) > 1 and abs(coeficientes[-1]) < 1e-15:
        coeficientes.pop()
    return coeficientes


def _coeficientes_lagrange(pontos: List[Ponto]) -> List[float]:
    # 1. Preparar o polinômio final
    grau = len(pontos) - 1
    resultado = [0.0] * (grau + 1)

    # 2. Construir uma base de Lagrange para cada ponto
    for i, (xi, yi) in enumerate(pontos):
        base = [1.0]
        denominador = 1.0

        # 3. Calcular a base L_i(x)
        for j, (xj, _) in enumerate(pontos):
            if i == j:
                continue

            # Monta o numerador: (x - xj)
            base = _multiplicar_por_fator_linear(base, xj)

            # Calcula o denominador: (xi - xj)
            denominador *= xi - xj

        # 4. Multiplicar a base pelo valor yi
        for poder, coef in enumerate([c / denominador for c in base]):
            resultado[poder] += yi * coef

    # 5. Retornar o polinômio interpolador
    return _normalizar_coeficientes(resultado)


def _coeficientes_newton(pontos: List[Ponto]) -> List[float]:
    """Constrói os coeficientes do polinômio interpolador de Newton.

    A interpolação de Newton usa diferenças divididas para montar um polinômio
    na forma:
    P(x) = a0 + a1(x - x0) + a2(x - x0)(x - x1) + ...

    Isso torna a construção incremental e bastante útil quando se quer adicionar
    novos pontos sem reconstruir tudo do zero.
    """
    diferencas = [[0.0] * len(pontos) for _ in range(len(pontos))]
    for i, (_, yi) in enumerate(pontos):
        diferencas[i][0] = yi

    # Cálculo das diferenças divididas.
    for j in range(1, len(pontos)):
        for i in range(len(pontos) - j):
            numerador = diferencas[i + 1][j - 1] - diferencas[i][j - 1]
            denominador = pontos[i + j][0] - pontos[i][0]
            diferencas[i][j] = numerador / denominador

    # Agora o polinômio é montado a partir das diferenças divididas.
    polinomio = [diferencas[0][0]]
    for i in range(1, len(pontos)):
        polinomio = _multiplicar_por_fator_linear(polinomio, pontos[i - 1][0])
        polinomio[0] += diferencas[0][i]

    return [round(c, 12) for c in polinomio]


def avaliar_polinomio(coeficientes: List[float], x: float) -> float:
    """Avalia um polinômio cujos coeficientes estão em ordem crescente de grau.

    A implementação usa o método de Horner, que é eficiente e numericamente estável.
    """
    valor = 0.0
    for coef in reversed(coeficientes):
        valor = valor * x + coef
    return valor


def formatar_polinomio(coeficientes: List[float]) -> str:
    """Converte os coeficientes em uma string legível para o polinômio.

    A função transforma a lista de coeficientes em uma representação parecida
    com "3x^2 - 2x + 1", facilitando a leitura do resultado.
    """
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


def plotar_polinomio(
    pontos: List[Ponto],
    coeficientes: List[float],
    caminho_salvar: str | None = None,
    titulo: str = "Interpolação / Ajuste de Curvas",
    destacar_x: float | None = None,
) -> None:
    """Plota os pontos e o polinômio correspondente com destaque opcional para um x específico."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valores_x = [x for x, _ in pontos]
    valores_y = [y for _, y in pontos]

    if len(valores_x) >= 2:
        margem_x = 0.08 * (max(valores_x) - min(valores_x) or 1)
        xmin, xmax = min(valores_x) - margem_x, max(valores_x) + margem_x
    else:
        xmin, xmax = valores_x[0] - 1, valores_x[0] + 1

    if len(valores_y) >= 2:
        margem_y = 0.08 * (max(valores_y) - min(valores_y) or 1)
        ymin, ymax = min(valores_y) - margem_y, max(valores_y) + margem_y
    else:
        ymin, ymax = valores_y[0] - 1, valores_y[0] + 1

    eixos_x = [xmin + (xmax - xmin) * i / 500 for i in range(501)]
    eixos_y = [avaliar_polinomio(coeficientes, x) for x in eixos_x]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(eixos_x, eixos_y, label="Polinômio", color="tab:blue", linewidth=2.2)
    ax.scatter(valores_x, valores_y, color="tab:red", s=70, label="Pontos", zorder=3)

    for x, y in zip(valores_x, valores_y):
        ax.annotate(f"({x:.3g}, {y:.3g})", (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8)

    if destacar_x is not None:
        valor_destacado = avaliar_polinomio(coeficientes, destacar_x)
        ax.axvline(destacar_x, color="tab:orange", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.scatter([destacar_x], [valor_destacado], color="tab:orange", s=90, zorder=4)
        ax.annotate(
            f"x = {destacar_x:.0f}\ny ≈ {valor_destacado:.3f}",
            (destacar_x, valor_destacado),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color="tab:orange",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(titulo)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if caminho_salvar:
        fig.savefig(caminho_salvar, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plotar_modelo_exponencial(
    pontos: List[Ponto],
    parametros: Tuple[float, float],
    caminho_salvar: str | None = None,
    titulo: str = "Ajuste Exponencial",
    destacar_x: float | None = None,
) -> None:
    """Plota os pontos e o modelo exponencial correspondente."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valores_x = [x for x, _ in pontos]
    valores_y = [y for _, y in pontos]

    if len(valores_x) >= 2:
        margem_x = 0.08 * (max(valores_x) - min(valores_x) or 1)
        xmin, xmax = min(valores_x) - margem_x, max(valores_x) + margem_x
    else:
        xmin, xmax = valores_x[0] - 1, valores_x[0] + 1

    if len(valores_y) >= 2:
        margem_y = 0.08 * (max(valores_y) - min(valores_y) or 1)
        ymin, ymax = min(valores_y) - margem_y, max(valores_y) + margem_y
    else:
        ymin, ymax = valores_y[0] - 1, valores_y[0] + 1

    eixos_x = [xmin + (xmax - xmin) * i / 500 for i in range(501)]
    eixos_y = [avaliar_modelo_exponencial(parametros, x) for x in eixos_x]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(eixos_x, eixos_y, label="Modelo exponencial", color="tab:green", linewidth=2.2)
    ax.scatter(valores_x, valores_y, color="tab:red", s=70, label="Pontos", zorder=3)

    if destacar_x is not None:
        valor_destacado = avaliar_modelo_exponencial(parametros, destacar_x)
        ax.axvline(destacar_x, color="tab:orange", linestyle="--", linewidth=1.2, alpha=0.8)
        ax.scatter([destacar_x], [valor_destacado], color="tab:orange", s=90, zorder=4)
        ax.annotate(
            f"x = {destacar_x:.3g}\ny ≈ {valor_destacado:.3f}",
            (destacar_x, valor_destacado),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color="tab:orange",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(titulo)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if caminho_salvar:
        fig.savefig(caminho_salvar, dpi=200, bbox_inches="tight")
    plt.close(fig)


def interpolacao_lagrange(pontos: List[Ponto], x: float) -> float:
    """Calcula o valor interpolado pelo método de Lagrange.

    Esse método constrói um polinômio que passa exatamente pelos pontos informados.
    O valor em x é obtido somando as contribuições de cada ponto por meio das
    funções basis de Lagrange.
    """
    if not pontos:
        raise ValueError("A lista de pontos não pode estar vazia.")

    xs = [xi for xi, _ in pontos]
    if len(set(xs)) != len(xs):
        raise ValueError("Os pontos precisam ter valores de x diferentes para usar Lagrange.")

    valor = 0.0
    for i, (xi, yi) in enumerate(pontos):
        # Cada termo representa a contribuição do ponto i na interpolação.
        base = 1.0
        for j, (xj, _) in enumerate(pontos):
            if i != j:
                base *= (x - xj) / (xi - xj)
        valor += yi * base
    return valor


def interpolacao_newton(pontos: List[Ponto], x: float) -> float:
    """Calcula o valor interpolado pelo método de Newton.

    O método de Newton usa diferenças divididas para montar um polinômio em uma
    forma conveniente para avaliação. É particularmente útil quando a lista de
    pontos aumenta aos poucos.
    """
    if not pontos:
        raise ValueError("A lista de pontos não pode estar vazia.")

    diferencas = [[0.0] * len(pontos) for _ in range(len(pontos))]
    for i, (_, yi) in enumerate(pontos):
        diferencas[i][0] = yi

    # Computa as diferenças divididas em tabela.
    for j in range(1, len(pontos)):
        for i in range(len(pontos) - j):
            numerador = diferencas[i + 1][j - 1] - diferencas[i][j - 1]
            denominador = pontos[i + j][0] - pontos[i][0]
            diferencas[i][j] = numerador / denominador

    valor = 0.0
    for i in range(len(pontos)):
        # Cada termo acrescenta um fator do tipo (x - x0)(x - x1)...
        termo = 1.0
        for j in range(i):
            termo *= x - pontos[j][0]
        valor += diferencas[0][i] * termo
    return valor


def ajustar_minimos_quadrados(pontos: List[Ponto], grau: int) -> List[float]:
    """Ajusta um polinômio aos pontos por mínimos quadrados.

    Diferente da interpolação, este método não exige que o polinômio passe
    exatamente por todos os pontos. Em vez disso, ele busca um polinômio de grau
    escolhido que minimize o erro quadrático total entre o modelo e os dados.
    """
    if not pontos:
        raise ValueError("A lista de pontos não pode estar vazia.")
    if grau < 0:
        raise ValueError("O grau do polinômio deve ser não negativo.")
    if len(pontos) < grau + 1:
        raise ValueError("São necessários pelo menos grau+1 pontos para o ajuste.")

    m = grau + 1
    matriz = [[0.0] * m for _ in range(m)]
    vetor = [0.0] * m

    # Monta o sistema normal associado ao ajuste por mínimos quadrados.
    for i in range(m):
        for j in range(m):
            matriz[i][j] = sum((x ** i) * (x ** j) for x, _ in pontos)
        vetor[i] = sum((x ** i) * y for x, y in pontos)

    # Resolve o sistema linear usando eliminação de Gauss com pivoteamento.
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


def ajustar_minimos_quadrados_exponencial(pontos: List[Ponto]) -> Tuple[float, float]:
    """Ajusta um modelo exponencial do tipo y = a * exp(b * x) por mínimos quadrados.

    A estratégia é linearizar o modelo aplicando o logaritmo natural em ambos os lados:
    ln(y) = ln(a) + b x.
    Em seguida, resolve-se um ajuste linear em z = ln(y).
    """
    if not pontos:
        raise ValueError("A lista de pontos não pode estar vazia.")
    if any(y <= 0 for _, y in pontos):
        raise ValueError("O ajuste exponencial exige valores positivos de y.")

    xs = [x for x, _ in pontos]
    zs = [__import__("math").log(y) for _, y in pontos]

    # Ajuste linear z = c0 + c1 * x
    m = 2
    matriz = [[0.0] * m for _ in range(m)]
    vetor = [0.0] * m

    for i in range(m):
        for j in range(m):
            matriz[i][j] = sum((x ** i) * (x ** j) for x in xs)
        vetor[i] = sum((x ** i) * z for x, z in zip(xs, zs))

    for i in range(m):
        pivô = max(range(i, m), key=lambda r: abs(matriz[r][i]))
        if abs(matriz[pivô][i]) < 1e-12:
            raise ValueError("Não foi possível resolver o ajuste exponencial.")
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

    c0, c1 = coeficientes
    a = math.exp(c0)
    b = c1
    return a, b


def avaliar_modelo_exponencial(parametros: Tuple[float, float], x: float) -> float:
    """Avalia o modelo exponencial y = a * exp(b * x)."""
    a, b = parametros
    return a * math.exp(b * x)


def _obter_coeficientes(pontos: List[Ponto], opcao: str, grau: int) -> List[float]:
    """Escolhe os coeficientes com base no método selecionado.

    Esta função atua como um adaptador entre a interface do menu e os métodos
    numéricos implementados abaixo.
    """
    if opcao == "1":
        return construir_coeficientes_polinomio(pontos, "lagrange")
    if opcao == "2":
        return construir_coeficientes_polinomio(pontos, "newton")
    return ajuste_minimos_quadrados(pontos, grau)


def executar_menu_interativo() -> None:
    """Exibe um menu interativo no terminal para escolher o método e avaliar o polinômio.

    O usuário pode informar os pontos, escolher o método desejado e então:
    - avaliar o valor do polinômio em um ponto x;
    - apenas mostrar o polinômio encontrado;
    - ou gerar um gráfico com os pontos e a curva.
    """
    print("=" * 60)
    print("INTERPOLAÇÃO E AJUSTE DE CURVAS")
    print("=" * 60)

    while True:
        print("\nEscolha um método:")
        print("1 - Interpolação de Lagrange")
        print("2 - Interpolação de Newton")
        print("3 - Ajuste por Mínimos Quadrados")
        print("4 - Ajuste Exponencial")
        print("0 - Sair")
        opcao = input("Opção: ").strip()

        if opcao == "0":
            print("Encerrando...")
            break
        if opcao not in {"1", "2", "3", "4"}:
            print("Opção inválida.")
            continue

        # Leitura da quantidade de pontos e dos próprios pontos.
        n = int(input("\nQuantidade de pontos: ").strip())
        pontos: List[Ponto] = []
        valores = input(f"Digite os pontos no formato x0:y0:x1:y1:...:x{n - 1}:y{n - 1}: ").split(":")
        if len(valores) != 2 * n:
            print("Quantidade de valores incompatível com o número de pontos informado.")
            continue

        for i in range(0, 2 * n, 2):
            pontos.append((float(valores[i]), float(valores[i + 1])))

        print("\nDeseja:")
        print("1 - Encontrar valor")
        print("2 - Mostrar modelo")
        print("3 - Plotar o gráfico")
        acao = input("Opção: ").strip()

        if opcao == "4":
            parametros = ajustar_minimos_quadrados_exponencial(pontos)
            if acao == "1":
                valor_x = float(input("Digite x: ").strip())
                valor = avaliar_modelo_exponencial(parametros, valor_x)
                print(f"\nModelo exponencial: y = {parametros[0]:.6g} * exp({parametros[1]:.6g}x)")
                print(f"f({valor_x}) = {valor}")
            elif acao == "2":
                print(f"\nModelo exponencial: y = {parametros[0]:.6g} * exp({parametros[1]:.6g}x)")
            elif acao == "3":
                caminho = input("Nome do arquivo para salvar o gráfico (opcional, pressione Enter para só mostrar): ").strip()
                plotar_modelo_exponencial(pontos, parametros, caminho_salvar=caminho or None)
            else:
                print("Opção inválida.")
            continue

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
