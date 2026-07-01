#Ajuste de curva por MÍNIMOS QUADRADOS (regressão linear simples: y = a*x + b)

import numpy as np                  # cálculos numéricos (vetores, somatórios)
import matplotlib.pyplot as plt     # geração dos gráficos

# DADOS DE ENTRADA

x_corrente = np.array([0.32, 0.38, 0.45, 0.51, 0.57, 0.64, 0.71, 0.77, 0.83, 0.89, 0.96])  # corrente (mA)
y_tensao = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])                              # tensão (V)
n = len(x_corrente)  # número de pontos
assert len(x_corrente) == len(y_tensao)
# -----------------------------------------------------------------------

# CÁLCULO DOS COEFICIENTES a (inclinação) e b (intercepto)

soma_x = np.sum(x_corrente)              # Σx
soma_y = np.sum(y_tensao)                # Σy
soma_xy = np.sum(x_corrente * y_tensao)  # Σ(x*y)
soma_x2 = np.sum(x_corrente ** 2)        # Σ(x²)

  # Coeficiente angular (inclinação da reta)
a = (n * soma_xy - soma_x * soma_y) / (n * soma_x2 - soma_x ** 2)

  # Coeficiente linear (onde a reta cruza o eixo y)
b = (soma_y - a * soma_x) / n

# -----------------------------------------------------------------------

# VALORES PREVISTOS PELA RETA E RESÍDUOS (erro de cada ponto)

y_pred = a * x_corrente + b       # valor que a reta prevê para cada x_corrente
residuos = y_tensao - y_pred      # diferença real entre ponto e reta (distância vertical)

# ERRO PADRÃO DA ESTIMATIVA E COEFICIENTE DE DETERMINAÇÃO (R²)

ss_res = np.sum(residuos ** 2)                          # soma dos quadrados dos resíduos
ss_tot = np.sum((y_tensao - np.mean(y_tensao)) ** 2)     # soma dos quadrados total

erro_padrao = np.sqrt(ss_res / (n - 2))   # dispersão média dos pontos em torno da reta
r2 = 1 - ss_res / ss_tot                  # quão bem a reta explica a variação dos dados (0 a 1)

# -----------------------------------------------------------------------

# RESISTÊNCIA ESTIMADA (LEI DE OHM: V = R*I)
R_estimado = a

# -----------------------------------------------------------------------

# IMPRESSÃO DOS RESULTADOS

print("===== RESULTADOS DO AJUSTE POR MÍNIMOS QUADRADOS =====")

print("\n--- Somatórios utilizados nas fórmulas ---")
print(f"n (número de pontos): {n}")
print(f"Σx  = {soma_x:.6f}")
print(f"Σy  = {soma_y:.6f}")
print(f"Σxy = {soma_xy:.6f}")
print(f"Σx² = {soma_x2:.6f}")

print("\n--- Equação ajustada ---")
print(f"y = {a:.6f}x + {b:.6f}")
print(f"Coeficiente angular (a): {a:.6f}")
print(f"Coeficiente linear  (b): {b:.6f}")
print(f"Erro padrão da estimativa: {erro_padrao:.6f}")
print(f"Coeficiente de determinação (R²): {r2:.6f}")

print("\n--- Lei de Ohm (V = R*I) ---")
print(f"Resistência estimada: R ≈ {R_estimado:.4f} ohms")

print("\nResíduos (distância vertical real de cada ponto até a reta):")
for xi, yi, ri in zip(x_corrente, y_tensao, residuos):
    print(f"  x={xi:>7.3f}  y={yi:>7.3f}  y_previsto={a*xi+b:>7.3f}  resíduo={ri:>7.3f}")

# -----------------------------------------------------------------------

# GRÁFICO

plt.figure(figsize=(9, 6))

# Pontos experimentais (tamanho maior e contorno para destacar
# cada ponto mesmo quando ele fica "em cima" da reta ajustada)
plt.scatter(x_corrente, y_tensao, s=35, color="royalblue", edgecolor="black", linewidth=0.6,
            label="Dados experimentais", zorder=3)

# Reta ajustada (usamos um x contínuo para desenhar a reta suavemente)
x_linha = np.linspace(x_corrente.min(), x_corrente.max(), 200)
y_linha = a * x_linha + b
plt.plot(x_linha, y_linha, color="crimson", linewidth=1.5, label="Ajuste linear", zorder=2)

# Linhas verticais mostrando o resíduo (distância real) de cada ponto até a reta
for xi, yi, ypi in zip(x_corrente, y_tensao, y_pred):
    plt.plot([xi, xi], [yi, ypi], color="gray", linestyle="--", linewidth=0.8, zorder=1)

# Limites dos eixos com folga proporcional aos dados (se adapta a qualquer dataset)
plt.xlim(x_corrente.min() - 0.02, x_corrente.max() + 0.02)
plt.ylim(y_tensao.min() - 0.5, y_tensao.max() + 0.5)

plt.title("Verificação Experimental da Lei de Ohm\nMétodo dos Mínimos Quadrados")
plt.xlabel("I (mA)")
plt.ylabel("V (Volts)")
plt.grid(True, linestyle="--", alpha=0.4)

# Caixa com o resumo dos resultados (equação, resistência e R²)
texto_resultado = (
    f"y = {a:.3f}x + {b:.3f}\n"
    f"R = {R_estimado:.2f} Ω\n"
    f"$R^2$ = {r2:.5f}"
)
plt.text(0.03, 0.97, texto_resultado,
          transform=plt.gca().transAxes,
          verticalalignment="top",
          bbox=dict(facecolor="white", alpha=0.9))

plt.legend(loc="lower right")
plt.tight_layout()

# Salva o gráfico como imagem e também exibe na tela
plt.savefig("ajuste_minimos_quadrados.png", dpi=150)
plt.show()
