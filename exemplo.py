from metodos import ajustar_minimos_quadrados, interpolacao_lagrange, interpolacao_newton


def converter_numero(texto: str) -> float:
    """Converte um número aceitando ponto ou vírgula como separador decimal."""
    texto = texto.strip()
    try:
        return float(texto.replace(",", "."))
    except ValueError as erro:
        raise ValueError("Use um número válido, por exemplo 2.5 ou 2,5.") from erro


def ler_pontos() -> list[tuple[float, float]]:
    """Pede ao usuário os pontos no formato x:y."""
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
    """Menu simples para testar cada método individualmente."""
    print("Escolha um método:")
    print("1 - Lagrange")
    print("2 - Newton")
    print("3 - Mínimos quadrados")
    opcao = input("Opção: ").strip()

    pontos = ler_pontos()

    if opcao == "1":
        try:
            x = converter_numero(input("Digite o valor de x para avaliar: "))
            resultado = interpolacao_lagrange(pontos, x)
            print(f"Resultado de Lagrange em x = {x}: {resultado}")
        except ValueError as erro:
            print(f"Erro: {erro}")
    elif opcao == "2":
        try:
            x = converter_numero(input("Digite o valor de x para avaliar: "))
            resultado = interpolacao_newton(pontos, x)
            print(f"Resultado de Newton em x = {x}: {resultado}")
        except ValueError as erro:
            print(f"Erro: {erro}")
    elif opcao == "3":
        try:
            grau = int(input("Digite o grau do polinômio de ajuste: "))
            coeficientes = ajustar_minimos_quadrados(pontos, grau=grau)
            print("Coeficientes do ajuste:", coeficientes)
        except ValueError as erro:
            print(f"Erro: {erro}")
    else:
        print("Opção inválida.")


if __name__ == "__main__":
    main()
