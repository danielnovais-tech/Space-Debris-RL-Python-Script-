#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de detecção de falhas e auto-reparação com IA.
Simula um serviço com métricas, detecta anomalias usando Isolation Forest
e executa ações de recuperação.
"""

import time
import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


# ------------------------------
# Simulador do serviço
# ------------------------------
class ServiceSimulator:
    """
    Simula um serviço com três métricas:
    - response_time: tempo de resposta (ms)
    - error_rate: taxa de erros (%)
    - cpu_usage: uso de CPU (%)

    O serviço pode operar normalmente ou entrar em estado de falha.
    Ações de auto-reparação podem ser aplicadas.
    """

    def __init__(self):
        self.state = "normal"  # 'normal' ou 'fault'
        self.fault_type = None
        self.time = 0
        self.metrics_history = {
            "response_time": [],
            "error_rate": [],
            "cpu_usage": [],
        }
        self.faults_injected = []
        self.healing_actions = []

    def step(self):
        """Avança um passo de simulação e retorna as métricas atuais."""
        self.time += 1

        if self.state == "normal":
            # Comportamento normal: métricas estáveis com pequenas variações
            response_time = 50 + np.random.normal(0, 5)  # ~50 ms
            error_rate = max(0, 1 + np.random.normal(0, 0.5))  # ~1%
            cpu_usage = 30 + np.random.normal(0, 3)  # ~30%
        else:
            # Comportamento com falha: métricas degradadas
            if self.fault_type == "high_load":
                response_time = 200 + np.random.normal(0, 30)
                error_rate = 5 + np.random.normal(0, 2)
                cpu_usage = 95 + np.random.normal(0, 5)
            elif self.fault_type == "memory_leak":
                response_time = 150 + np.random.normal(0, 20)
                error_rate = 3 + np.random.normal(0, 1)
                cpu_usage = 80 + np.random.normal(0, 10)
            else:  # falha genérica
                response_time = 300 + np.random.normal(0, 50)
                error_rate = 15 + np.random.normal(0, 5)
                cpu_usage = 70 + np.random.normal(0, 15)

        # Garantir valores não negativos e limites razoáveis
        response_time = max(10, min(500, response_time))
        error_rate = max(0, min(30, error_rate))
        cpu_usage = max(5, min(100, cpu_usage))

        metrics = [response_time, error_rate, cpu_usage]
        self.metrics_history["response_time"].append(response_time)
        self.metrics_history["error_rate"].append(error_rate)
        self.metrics_history["cpu_usage"].append(cpu_usage)
        return metrics

    def inject_fault(self, fault_type="high_load"):
        """Introduz uma falha no serviço."""
        self.state = "fault"
        self.fault_type = fault_type
        self.faults_injected.append((self.time, fault_type))
        print(f"[{self.time}] ⚠️ Falha injetada: {fault_type}")

    def heal(self, action="restart"):
        """
        Executa uma ação de auto-reparação.
        Ação pode ser 'restart', 'scale_up', 'clear_cache', etc.
        """
        print(f"[{self.time}] 🔧 Ação de reparo: {action}")
        self.healing_actions.append((self.time, action))

        if action == "restart":
            # Reiniciar o serviço – volta ao normal, mas pode levar alguns passos
            self.state = "normal"
            self.fault_type = None
            return True
        if action == "scale_up":
            # Simula aumento de recursos – reduz carga imediatamente
            self.state = "normal"  # simplificado
            self.fault_type = None
            return True
        return False


# ------------------------------
# Gerador de dados normais para treino
# ------------------------------
def generate_normal_data(sim: ServiceSimulator, steps: int = 1000) -> np.ndarray:
    """Roda o simulador em modo normal e coleta métricas para treino."""
    data = []
    for _ in range(steps):
        data.append(sim.step())
    return np.array(data)


# ------------------------------
# Loop principal de detecção e reparo
# ------------------------------
def main():
    print("=== Sistema de Detecção de Falhas e Auto-Reparação com IA ===\n")

    # 1. Criar simulador e gerar dados normais para treino
    sim = ServiceSimulator()
    print("Gerando dados de operação normal para treino do modelo...")
    normal_data = generate_normal_data(sim, steps=1500)
    print(f"Dados normais gerados: {normal_data.shape}\n")

    # 2. Treinar modelo de detecção de anomalias (Isolation Forest)
    print("Treinando modelo Isolation Forest...")
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(normal_data)
    print("Modelo treinado.\n")

    # 3. Simular operação contínua com injeção de falhas e auto-reparação
    print("Iniciando simulação em tempo real (pressione Ctrl+C para parar)...\n")
    sim = ServiceSimulator()  # reinicia simulador para teste

    # Configurar visualização ao vivo
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    lines = []
    for ax, metric in zip(axes, ["response_time", "error_rate", "cpu_usage"]):
        line, = ax.plot([], [], lw=1)
        lines.append(line)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(True)
    axes[2].set_xlabel("Tempo (passos)")
    plt.tight_layout()

    # Janela deslizante para pontuação de anomalia
    window_size = 50
    anomaly_scores = deque(maxlen=window_size)

    step = 0
    try:
        while True:
            step += 1
            metrics = sim.step()

            # Calcular score de anomalia (quanto mais negativo, mais anômalo)
            score = model.decision_function([metrics])[0]
            anomaly_scores.append(score)

            # Detectar anomalia (limiar empírico)
            threshold = -0.1  # ajustável
            is_anomaly = score < threshold

            # Se for anomalia e não estivermos já em estado de falha (para evitar loops)
            if is_anomaly and sim.state == "normal":
                print(f"[{step}] 🚨 Anomalia detectada! Score: {score:.3f}")

                # Escolher ação de reparo baseada nas métricas (simples)
                if metrics[2] > 90:  # CPU muito alta
                    action = "scale_up"
                else:
                    action = "restart"
                sim.heal(action)

            # Opcional: injetar falhas aleatórias para teste
            if step % 200 == 0 and step > 0:
                fault = random.choice(["high_load", "memory_leak"])
                sim.inject_fault(fault)

            # Atualizar gráfico a cada 5 passos para não travar
            if step % 5 == 0:
                for i, (metric_name, line) in enumerate(
                    zip(["response_time", "error_rate", "cpu_usage"], lines)
                ):
                    data = sim.metrics_history[metric_name]
                    line.set_xdata(range(len(data)))
                    line.set_ydata(data)
                    axes[i].relim()
                    axes[i].autoscale_view()
                plt.pause(0.01)

            time.sleep(0.05)  # pequeno atraso para visualização

    except KeyboardInterrupt:
        print("\n\nSimulação encerrada pelo usuário.")

    # Relatório final
    print("\n=== Relatório ===")
    print(f"Total de passos: {step}")
    print(f"Falhas injetadas: {len(sim.faults_injected)}")
    print(f"Ações de reparo tomadas: {len(sim.healing_actions)}")
    for t, f in sim.faults_injected:
        print(f"  - Falha '{f}' no tempo {t}")
    for t, a in sim.healing_actions:
        print(f"  - Ação '{a}' no tempo {t}")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
