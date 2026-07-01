from metodos import (
    ajustar_minimos_quadrados,
    ajustar_minimos_quadrados_exponencial,
    avaliar_modelo_exponencial,
    construir_coeficientes_polinomio,
    interpolacao_lagrange,
    interpolacao_newton,
    plotar_modelo_exponencial,
    plotar_polinomio,
)


def converter_numero(texto: str) -> float:
    """Converte um número aceitando ponto ou vírgula como separador decimal.

    Esta função foi criada para deixar a entrada do usuário mais amigável,
    já que muitas pessoas escrevem 2,5 em vez de 2.5.
    """
    texto = texto.strip()
    try:
        return float(texto.replace(",", "."))
    except ValueError as erro:
        raise ValueError("Use um número válido, por exemplo 2.5 ou 2,5.") from erro


def ler_pontos() -> list[tuple[float, float]]:
    """Pede ao usuário os pontos no formato x:y.

    A função lê uma quantidade informada de pontos e transforma as entradas
    do usuário em uma lista de tuplas (x, y), que é o formato esperado pelas
    funções numéricas do projeto.
    """
    try:
        quantidade = int(input("Quantos pontos você quer informar? "))
    except ValueError as erro:
        raise ValueError("Digite um número inteiro de pontos.") from erro

    pontos = []
    print("Exemplo de entrada: 0:1")
    print("Digite cada ponto como x:y, sem espaços.")
    for i in range(quantidade):
        entrada = input(f"Digite o ponto {i + 1} no formato x:y: ").strip()
        x_str, y_str = entrada.split(":")
        pontos.append((converter_numero(x_str), converter_numero(y_str)))
    return pontos


def main() -> None:
    """Menu simples para testar cada método individualmente.

    Este exemplo serve como uma interface prática para o usuário testar rápida-
    mente os métodos sem precisar alterar o código diretamente.
    """
    print("Escolha um método:")
    print("1 - Lagrange")
    print("2 - Newton")
    print("3 - Mínimos quadrados")
    print("4 - Ajuste exponencial")
    opcao = input("Opção: ").strip()

    # Coleta os pontos que serão usados no cálculo.
    pontos = ler_pontos()

    if opcao == "1":
        try:
            x = converter_numero(input("Digite o valor de x para avaliar: "))
            resultado = interpolacao_lagrange(pontos, x)
            print(f"Resultado de Lagrange em x = {x}: {resultado}")

            caminho = input("Deseja salvar o gráfico? Informe o nome do arquivo (ou deixe em branco para não salvar): ").strip()
            if caminho:
                coeficientes = construir_coeficientes_polinomio(pontos, "lagrange")
                plotar_polinomio(
                    pontos,
                    coeficientes,
                    caminho_salvar=caminho,
                    titulo="Interpolação de Lagrange",
                    destacar_x=x,
                )
                print(f"Gráfico salvo em {caminho}")
        except ValueError as erro:
            print(f"Erro: {erro}")
    elif opcao == "2":
        try:
            # O método de Newton também avalia o polinômio em um ponto x.
            x = converter_numero(input("Digite o valor de x para avaliar: "))
            resultado = interpolacao_newton(pontos, x)
            print(f"Resultado de Newton em x = {x}: {resultado}")
        except ValueError as erro:
            print(f"Erro: {erro}")
    elif opcao == "3":
        try:
            # O ajuste por mínimos quadrados precisa do grau do polinômio desejado.
            grau = int(input("Digite o grau do polinômio de ajuste: "))
            coeficientes = ajustar_minimos_quadrados(pontos, grau=grau)
            print("Coeficientes do ajuste:", coeficientes)
        except ValueError as erro:
            print(f"Erro: {erro}")
    elif opcao == "4":
        try:
            x = converter_numero(input("Digite o valor de x para avaliar: "))
            parametros = ajustar_minimos_quadrados_exponencial(pontos)
            resultado = avaliar_modelo_exponencial(parametros, x)
            print(f"Parâmetros do ajuste exponencial: a = {parametros[0]}, b = {parametros[1]}")
            print(f"Resultado exponencial em x = {x}: {resultado}")

            caminho = input("Deseja salvar o gráfico? Informe o nome do arquivo (ou deixe em branco para não salvar): ").strip()
            if caminho:
                plotar_modelo_exponencial(
                    pontos,
                    parametros,
                    caminho_salvar=caminho,
                    titulo="Ajuste Exponencial",
                    destacar_x=x,
                )
                print(f"Gráfico salvo em {caminho}")
        except ValueError as erro:
            print(f"Erro: {erro}")
    else:
        print("Opção inválida.")


if __name__ == "__main__":
    main()
