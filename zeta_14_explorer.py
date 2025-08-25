#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zeta_14_explorer.py - Explorador da conexão entre o primeiro zero e o número 14
Autor: Jefferson M. Okushigue
Data: 2025-08-24
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpmath import mp
import pandas as pd
import pickle
import os
from scipy import stats
from typing import List, Tuple, Dict, Any
import logging

# Configuração
mp.dps = 50
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("plasma")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Zeta14Explorer:
    """Classe para explorar a conexão entre o primeiro zero e o número 14"""
    
    def __init__(self, cache_file: str = None):
        self.cache_file = cache_file or self.find_cache_file()
        self.zeros = []
        self.load_zeros()
        
        # Primeiro zero não trivial conhecido
        self.first_nontrivial_zero = 14.134725142
        
        # Constantes relacionadas a 14
        self.fourteen_related = {
            'first_zero': 14.134725142,
            'exact_14': 14.0,
            'sqrt_196': np.sqrt(196),
            '2_times_7': 2 * 7,
            'e_2_639': np.exp(2.639),
            'pi_2_2': np.pi**2.2,
            'model_parameters': 14,  # Parâmetros do Modelo Padrão
            'generations': 3,  # Mas 14/3 ≈ 4.666...
            'index_ratio': 14.006038  # Razão encontrada nas ressonâncias
        }
    
    def find_cache_file(self):
        """Encontra o arquivo de cache"""
        locations = [
            "zeta_zeros_cache_fundamental.pkl",
            "~/zvt/code/zeta_zeros_cache_fundamental.pkl",
            os.path.expanduser("~/zvt/code/zeta_zeros_cache_fundamental.pkl"),
            "./zeta_zeros_cache_fundamental.pkl"
        ]
        
        for location in locations:
            if os.path.exists(location):
                return location
        return None
    
    def load_zeros(self):
        """Carrega os zeros da zeta"""
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.zeros = pickle.load(f)
                logger.info(f"✅ {len(self.zeros):,} zeros carregados")
            except Exception as e:
                logger.error(f"❌ Erro ao carregar cache: {e}")
    
    def analyze_first_zero_patterns(self):
        """Analisa padrões relacionados ao primeiro zero"""
        logger.info("🔍 Analisando padrões do primeiro zero não trivial...")
        
        # Comparar o primeiro zero com variações de 14
        comparisons = {}
        
        for name, value in self.fourteen_related.items():
            error = abs(self.first_nontrivial_zero - value) / self.first_nontrivial_zero
            comparisons[name] = {
                'value': value,
                'difference': abs(self.first_nontrivial_zero - value),
                'relative_error': error,
                'significance': 'HIGH' if error < 0.01 else 'MEDIUM' if error < 0.1 else 'LOW'
            }
        
        # Criar DataFrame para exibição
        df_data = []
        for name, data in comparisons.items():
            df_data.append({
                'Property': name,
                'Value': data['value'],
                'First Zero': self.first_nontrivial_zero,
                'Difference': data['difference'],
                'Relative Error': f"{data['relative_error']:.2%}",
                'Significance': data['significance']
            })
        
        df = pd.DataFrame(df_data)
        
        print("\n" + "="*80)
        print("ANÁLISE DO PRIMEIRO ZERO NÃO TRIVIAL vs NÚMERO 14")
        print("="*80)
        print(df.to_string(index=False))
        
        return comparisons
    
    def find_zeros_starting_with_14(self):
        """Encontra todos os zeros que começam com 14"""
        if not self.zeros:
            logger.warning("⚠️ Nenhum dado disponível")
            return []
        
        logger.info("🔍 Procurando zeros que começam com 14...")
        
        zeros_starting_with_14 = []
        
        for idx, gamma in self.zeros:
            gamma_str = f"{gamma:.10f}"
            if gamma_str.startswith("14."):
                zeros_starting_with_14.append((idx, gamma))
        
        logger.info(f"✅ Encontrados {len(zeros_starting_with_14)} zeros começando com 14")
        
        # Exibir os primeiros 10
        print("\n" + "="*80)
        print("ZEROS QUE COMEÇAM COM 14 (primeiros 10)")
        print("="*80)
        for i, (idx, gamma) in enumerate(zeros_starting_with_14[:10]):
            print(f"{i+1:2d}. Zero #{idx:8,} → {gamma:.10f}")
        
        if len(zeros_starting_with_14) > 10:
            print(f"... e mais {len(zeros_starting_with_14) - 10} zeros")
        
        return zeros_starting_with_14
    
    def analyze_14_in_resonances(self):
        """Analisa como 14 aparece nas ressonâncias encontradas"""
        logger.info("🔍 Analisando o papel do 14 nas ressonâncias...")
        
        # Ressonâncias conhecidas
        resonances = {
            'fine_structure': {
                'index': 118412,
                'gamma': 87144.853030,
                'constant': 1/137.035999084
            },
            'electron_mass': {
                'index': 1658483,
                'gamma': 953397.367271,
                'constant': 9.1093837015e-31
            }
        }
        
        # Calcular relações com 14
        print("\n" + "="*80)
        print("RELAÇÕES DAS RESSONÂNCIAS COM O NÚMERO 14")
        print("="*80)
        
        for name, data in resonances.items():
            print(f"\n🔬 {name.upper()}:")
            print(f"   Zero index: {data['index']:,}")
            print(f"   Gamma: {data['gamma']:.6f}")
            
            # Relações diretas
            index_div_14 = data['index'] / 14
            gamma_div_14 = data['gamma'] / 14
            
            print(f"   Index / 14 = {index_div_14:.6f}")
            print(f"   Gamma / 14 = {gamma_div_14:.6f}")
            
            # Verificar se são próximos de inteiros ou frações simples
            index_near_int = round(index_div_14)
            gamma_near_int = round(gamma_div_14)
            
            index_error = abs(index_div_14 - index_near_int) / index_div_14
            gamma_error = abs(gamma_div_14 - gamma_near_int) / gamma_div_14
            
            if index_error < 0.01:
                print(f"   ✅ Index/14 ≈ {index_near_int} (erro: {index_error:.2%})")
            
            if gamma_error < 0.01:
                print(f"   ✅ Gamma/14 ≈ {gamma_near_int} (erro: {gamma_error:.2%})")
    
    def analyze_14_multiples_pattern(self):
        """Analisa padrões de múltiplos de 14 nos índices"""
        if not self.zeros:
            return
        
        logger.info("🔍 Analisando padrões de múltiplos de 14...")
        
        # Encontrar zeros cujos índices são múltiplos de 14
        multiples_of_14 = []
        
        for idx, gamma in self.zeros:
            if idx % 14 == 0:
                multiples_of_14.append((idx, gamma))
        
        logger.info(f"✅ Encontrados {len(multiples_of_14)} zeros com índices múltiplos de 14")
        
        # Analisar estatísticas desses zeros
        if multiples_of_14:
            gammas = [g for _, g in multiples_of_14]
            
            print("\n" + "="*80)
            print("ESTATÍSTICAS DOS ZEROS COM ÍNDICES MÚLTIPLOS DE 14")
            print("="*80)
            print(f"Total: {len(multiples_of_14)} zeros")
            print(f"Média dos γ: {np.mean(gammas):.6f}")
            print(f"Desvio padrão: {np.std(gammas):.6f}")
            print(f"Menor γ: {np.min(gammas):.6f}")
            print(f"Maior γ: {np.max(gammas):.6f}")
            
            # Verificar se há padrão nas diferenças
            differences = np.diff(gammas)
            print(f"\nDiferença média entre γ consecutivos: {np.mean(differences):.6f}")
            print(f"Desvio padrão das diferenças: {np.std(differences):.6f}")
            
            # Exibir os primeiros 10
            print(f"\nPrimeiros 10 zeros com índices múltiplos de 14:")
            for i, (idx, gamma) in enumerate(multiples_of_14[:10]):
                print(f"   {i+1:2d}. Zero #{idx:8,} → {gamma:.6f}")
        
        return multiples_of_14
    
    def visualize_14_patterns(self):
        """Cria visualizações dos padrões relacionados a 14"""
        if not self.zeros:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análise de Padrões Relacionados ao Número 14', fontsize=16)
        
        # 1. Distribuição dos primeiros dígitos
        ax = axes[0, 0]
        first_digits = [int(str(gamma).split('.')[0]) for _, gamma in self.zeros[:10000]]
        
        counts = {}
        for digit in range(10, 20):  # 10 a 19
            counts[digit] = first_digits.count(digit)
        
        bars = ax.bar(counts.keys(), counts.values(), alpha=0.7)
        ax.axvline(x=14, color='red', linestyle='--', label='14')
        ax.set_xlabel('Primeiro Dígito')
        ax.set_ylabel('Frequência')
        ax.set_title('Distribuição dos Primeiros Dígitos (primeiros 10k zeros)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Destacar a barra do 14
        for bar in bars:
            if bar.get_x() == 14:
                bar.set_color('red')
                bar.set_alpha(0.8)
        
        # 2. Zeros começando com 14 vs outros
        ax = axes[0, 1]
        zeros_starting_14 = self.find_zeros_starting_with_14()
        
        if zeros_starting_14:
            indices_14 = [idx for idx, _ in zeros_starting_14]
            gammas_14 = [gamma for _, gamma in zeros_starting_14]
            
            # Comparar com zeros aleatórios
            sample_size = min(len(zeros_starting_14), 1000)
            random_indices = np.random.choice(len(self.zeros), sample_size, replace=False)
            random_gammas = [self.zeros[i][1] for i in random_indices]
            
            ax.scatter(range(len(gammas_14)), gammas_14, alpha=0.6, c='red', label='Começam com 14')
            ax.scatter(range(len(random_gammas)), random_gammas, alpha=0.3, c='blue', label='Amostra aleatória')
            
            ax.set_xlabel('Índice na amostra')
            ax.set_ylabel('Valor de γ')
            ax.set_title('Zeros começando com 14 vs amostra aleatória')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 3. Múltiplos de 14 nos índices
        ax = axes[1, 0]
        multiples_14 = self.analyze_14_multiples_pattern()
        
        if multiples_14:
            indices = [idx for idx, _ in multiples_14[:1000]]  # Primeiros 1000
            gammas = [gamma for _, gamma in multiples_14[:1000]]
            
            ax.scatter(indices, gammas, alpha=0.6, c='green')
            ax.set_xlabel('Índice (múltiplo de 14)')
            ax.set_ylabel('Valor de γ')
            ax.set_title('Zeros com índices múltiplos de 14')
            ax.grid(True, alpha=0.3)
        
        # 4. Relação entre índice e gamma (escala log)
        ax = axes[1, 1]
        sample_indices = [idx for idx, _ in self.zeros[::100]]  # Amostra esparsa
        sample_gammas = [gamma for _, gamma in self.zeros[::100]]
        
        ax.scatter(sample_indices, sample_gammas, alpha=0.3, s=1)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Índice (log)')
        ax.set_ylabel('γ (log)')
        ax.set_title('Relação índice-γ (escala log-log)')
        ax.grid(True, alpha=0.3)
        
        # Marcar as ressonâncias conhecidas
        resonance_indices = [118412, 1658483]
        resonance_gammas = [87144.853030, 953397.367271]
        ax.scatter(resonance_indices, resonance_gammas, c='red', s=100, 
                  label='Ressonâncias', marker='*')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('zeta_14_patterns.png', dpi=300, bbox_inches='tight')
        logger.info("📊 Visualização salva: zeta_14_patterns.png")
        plt.show()
    
    def run_analysis(self):
        """Executa a análise completa"""
        logger.info("🚀 Iniciando análise da conexão com o número 14...")
        
        # 1. Analisar o primeiro zero
        self.analyze_first_zero_patterns()
        
        # 2. Encontrar zeros começando com 14
        self.find_zeros_starting_with_14()
        
        # 3. Analisar 14 nas ressonâncias
        self.analyze_14_in_resonances()
        
        # 4. Analisar múltiplos de 14
        self.analyze_14_multiples_pattern()
        
        # 5. Criar visualizações
        self.visualize_14_patterns()
        
        # 6. Conclusões
        print("\n" + "="*80)
        print("CONCLUSÕES DA ANÁLISE DO NÚMERO 14")
        print("="*80)
        print("1. O primeiro zero não trivial começa com 14.1347...")
        print("2. A razão entre os índices das ressonâncias é ~14.006")
        print("3. 14 é o número de parâmetros do Modelo Padrão")
        print("4. Isso sugere uma possível conexão fundamental")
        print("   entre a estrutura dos zeros da zeta e a física")
        print("5. Mais investigações são necessárias para entender")
        print("   se essa é uma coincidência ou um padrão profundo")
        
        logger.info("✅ Análise concluída!")

# Execução principal
if __name__ == "__main__":
    try:
        explorer = Zeta14Explorer()
        explorer.run_analysis()
    except Exception as e:
        logger.error(f"❌ Erro durante a análise: {e}")
        import traceback
        traceback.print_exc()
