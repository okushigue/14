#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resonance_explorer_fixed.py - Versão corrigida do explorador de ressonâncias
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
from scipy import stats, optimize
from typing import List, Tuple, Dict, Any
import logging
from dataclasses import dataclass

# Configuração
mp.dps = 50
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ResonanceData:
    """Dados de uma ressonância específica"""
    zero_index: int
    gamma: float
    constant_name: str
    constant_value: float
    quality: float
    relative_error: float
    tolerance: float
    energy_gev: float

class ResonanceExplorer:
    """Classe para exploração aprofundada de ressonâncias"""
    
    def __init__(self, cache_file: str = None, zeros_file: str = None):
        """
        Inicializa o explorador com busca flexível de arquivos
        
        Args:
            cache_file: Arquivo de cache (opcional)
            zeros_file: Arquivo de zeros original (opcional)
        """
        self.zeros = []
        self.cache_file = cache_file
        self.zeros_file = zeros_file
        self.resonances = []
        
        # Tentar encontrar os arquivos automaticamente
        self.find_data_files()
        self.load_zeros()
        
        # Definir as ressonâncias encontradas anteriormente
        self.target_resonances = [
            ResonanceData(
                zero_index=118412,
                gamma=87144.853030040001613,
                constant_name="fine_structure",
                constant_value=1/137.035999084,
                quality=9.091261e-10,
                relative_error=1.2458301e-5,
                tolerance=1e-4,
                energy_gev=8714.485303
            ),
            ResonanceData(
                zero_index=1658483,
                gamma=953397.367270938004367,
                constant_name="electron_mass",
                constant_value=9.1093837015e-31,
                quality=3.209771e-37,
                relative_error=3.5235878e-5,
                tolerance=1e-30,
                energy_gev=95339.736727
            )
        ]
    
    def find_data_files(self):
        """Procura os arquivos de dados em locais comuns"""
        # Possíveis locais para o cache
        cache_locations = [
            "zeta_zeros_cache_fundamental.pkl",
            "~/zvt/code/zeta_zeros_cache_fundamental.pkl",
            os.path.expanduser("~/zvt/code/zeta_zeros_cache_fundamental.pkl"),
            "./zeta_zeros_cache_fundamental.pkl"
        ]
        
        # Possíveis locais para o arquivo de zeros
        zeros_locations = [
            "zero.txt",
            "~/zeta/zero.txt",
            os.path.expanduser("~/zeta/zero.txt"),
            "./zero.txt"
        ]
        
        # Procurar cache
        for location in cache_locations:
            if os.path.exists(location):
                self.cache_file = location
                logger.info(f"✅ Cache encontrado: {location}")
                break
        
        # Procurar arquivo de zeros
        for location in zeros_locations:
            if os.path.exists(location):
                self.zeros_file = location
                logger.info(f"✅ Arquivo de zeros encontrado: {location}")
                break
        
        if not self.cache_file and not self.zeros_file:
            logger.error("❌ Nenhum arquivo de dados encontrado!")
            logger.info("🔍 Locais verificados:")
            for loc in cache_locations + zeros_locations:
                logger.info(f"   - {loc}")
    
    def load_zeros(self):
        """Carrega os zeros da zeta do cache ou do arquivo original"""
        # Tentar carregar do cache primeiro
        if self.cache_file and os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self.zeros = pickle.load(f)
                logger.info(f"✅ {len(self.zeros):,} zeros carregados do cache")
                return
            except Exception as e:
                logger.error(f"❌ Erro ao carregar cache: {e}")
        
        # Tentar carregar do arquivo original
        if self.zeros_file and os.path.exists(self.zeros_file):
            try:
                logger.info(f"📂 Carregando zeros do arquivo: {self.zeros_file}")
                zeros = []
                with open(self.zeros_file, 'r') as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if line:
                            try:
                                zero = float(line)
                                zeros.append((i+1, zero))
                            except ValueError:
                                continue
                
                self.zeros = zeros
                logger.info(f"✅ {len(self.zeros):,} zeros carregados do arquivo")
                
                # Salvar um cache local para uso futuro
                try:
                    with open("zeta_zeros_cache_fundamental.pkl", 'wb') as f:
                        pickle.dump(self.zeros, f)
                    logger.info("💾 Cache local criado para uso futuro")
                except Exception as e:
                    logger.warning(f"⚠️ Não foi possível criar cache local: {e}")
                
            except Exception as e:
                logger.error(f"❌ Erro ao carregar arquivo: {e}")
        else:
            logger.error("❌ Nenhum arquivo de dados disponível")
    
    def analyze_neighborhood(self, resonance: ResonanceData, window_size: int = 100) -> Dict[str, Any]:
        """Analisa a vizinhança de um zero de ressonância"""
        idx = resonance.zero_index - 1  # Convertendo para índice base zero
        
        if idx < 0 or idx >= len(self.zeros):
            logger.error(f"❌ Índice {resonance.zero_index} fora dos limites (total: {len(self.zeros)})")
            return {}
        
        start_idx = max(0, idx - window_size)
        end_idx = min(len(self.zeros), idx + window_size + 1)
        neighborhood = self.zeros[start_idx:end_idx]
        
        # Extrair valores de gamma
        gamma_values = [z[1] for z in neighborhood]
        indices = [z[0] for z in neighborhood]
        
        # Análise estatística da vizinhança
        results = {
            'resonance': resonance,
            'neighborhood_size': len(neighborhood),
            'gamma_values': gamma_values,
            'indices': indices,
            'local_stats': {
                'mean': np.mean(gamma_values),
                'std': np.std(gamma_values),
                'min': np.min(gamma_values),
                'max': np.max(gamma_values),
                'median': np.median(gamma_values)
            },
            'differences': np.diff(gamma_values),
            'normalized_distances': []
        }
        
        # Calcular distâncias normalizadas para a constante
        for i, gamma in enumerate(gamma_values):
            mod_val = gamma % resonance.constant_value
            min_dist = min(mod_val, resonance.constant_value - mod_val)  # CORRIGIDO: mod_val em vez de mod_dist
            results['normalized_distances'].append(min_dist)
        
        # Encontrar o mínimo local na vizinhança
        min_dist_idx = np.argmin(results['normalized_distances'])
        results['local_minimum'] = {
            'index': indices[min_dist_idx],
            'gamma': gamma_values[min_dist_idx],
            'distance': results['normalized_distances'][min_dist_idx]
        }
        
        return results
    
    def test_mathematical_relations(self, resonance: ResonanceData) -> Dict[str, Any]:
        """Testa relações matemáticas adicionais para a ressonância"""
        results = {}
        
        # Testar relação com números conhecidos
        known_numbers = {
            'pi': np.pi,
            'e': np.e,
            'golden_ratio': (1 + np.sqrt(5)) / 2,
            'sqrt(2)': np.sqrt(2),
            'sqrt(3)': np.sqrt(3),
            'sqrt(5)': np.sqrt(5),
            'ln(2)': np.log(2),
            'ln(3)': np.log(3),
            'gamma_euler': 0.5772156649
        }
        
        gamma = resonance.gamma
        constant = resonance.constant_value
        
        # Testar relações diretas
        for name, value in known_numbers.items():
            # Testar γ / constante ≈ número
            ratio = gamma / constant
            error = abs(ratio - value) / value
            results[f'gamma/constant_{name}'] = {
                'value': ratio,
                'target': value,
                'error': error,
                'significant': error < 0.01
            }
            
            # Testar constante / γ ≈ número
            ratio_inv = constant / gamma
            error_inv = abs(ratio_inv - value) / value
            results[f'constant/gamma_{name}'] = {
                'value': ratio_inv,
                'target': value,
                'error': error_inv,
                'significant': error_inv < 0.01
            }
        
        # Testar relações especiais para as constantes físicas
        if resonance.constant_name == "fine_structure":
            # Testar relação com 137
            alpha_inv = 1 / constant
            results['alpha_inverse'] = {
                'value': alpha_inv,
                'target': 137.035999084,
                'error': abs(alpha_inv - 137.035999084) / 137.035999084,
                'significant': True
            }
            
            # Testar γ ≈ alpha_inv * fator
            for factor in [635.8, 636, 1000/1.57]:  # Fatores interessantes
                test_value = alpha_inv * factor
                error = abs(gamma - test_value) / gamma
                results[f'alpha_x_{factor}'] = {
                    'value': test_value,
                    'target': gamma,
                    'error': error,
                    'significant': error < 0.01
                }
        
        elif resonance.constant_name == "electron_mass":
            # Testar relações com unidades naturais
            # Converter para unidades naturais (onde ħ = c = 1)
            # Massa do elétron em eV: 510998.9461 eV
            mev_mass = 0.5109989461  # MeV
            
            # Testar γ ≈ massa_em_eV * fator
            for factor in [1.7e11, 1.87e11, 2e11]:  # Fatores de escala
                test_value = mev_mass * factor
                error = abs(gamma - test_value) / gamma
                results[f'mass_eV_x_{factor}'] = {
                    'value': test_value,
                    'target': gamma,
                    'error': error,
                    'significant': error < 0.01
                }
        
        return results
    
    def create_synthetic_analysis(self, resonance: ResonanceData):
        """Cria uma análise sintética quando não temos os zeros reais"""
        logger.info(f"🔬 Criando análise sintética para {resonance.constant_name}")
        
        # Gerar dados sintéticos baseados nas propriedades conhecidas
        gamma = resonance.gamma
        
        # Criar uma vizinhança sintética
        synthetic_gammas = []
        for i in range(-50, 51):
            # Adicionar variação aleatória pequena
            variation = np.random.normal(0, gamma * 0.001)
            synthetic_gammas.append(gamma + i * 10 + variation)
        
        # Calcular distâncias sintéticas
        distances = []
        for g in synthetic_gammas:
            mod_val = g % resonance.constant_value
            min_dist = min(mod_val, resonance.constant_value - mod_val)
            distances.append(min_dist)
        
        # Encontrar o mínimo
        min_idx = np.argmin(distances)
        
        return {
            'synthetic': True,
            'gamma_values': synthetic_gammas,
            'normalized_distances': distances,
            'local_minimum': {
                'index': resonance.zero_index + min_idx - 50,
                'gamma': synthetic_gammas[min_idx],
                'distance': distances[min_idx]
            },
            'local_stats': {
                'mean': np.mean(synthetic_gammas),
                'std': np.std(synthetic_gammas),
                'min': np.min(synthetic_gammas),
                'max': np.max(synthetic_gammas),
                'median': np.median(synthetic_gammas)
            }
        }
    
    def visualize_resonance(self, resonance: ResonanceData, window_size: int = 200):
        """Gera visualizações detalhadas para uma ressonância"""
        
        # Tentar análise real ou usar sintética
        if self.zeros and len(self.zeros) > resonance.zero_index:
            neighborhood_data = self.analyze_neighborhood(resonance, window_size)
        else:
            neighborhood_data = self.create_synthetic_analysis(resonance)
        
        if not neighborhood_data:
            return
        
        # Criar figura com múltiplos subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Análise Detalhada - {resonance.constant_name.upper()}\n'
                    f'Zero #{resonance.zero_index:,} ({"Sintético" if neighborhood_data.get("synthetic") else "Real"})', 
                    fontsize=16)
        
        # 1. Série temporal dos gammas na vizinhança
        ax = axes[0, 0]
        if 'indices' in neighborhood_data:
            indices = neighborhood_data['indices']
        else:
            indices = list(range(resonance.zero_index - 50, resonance.zero_index + 51))
        
        gammas = neighborhood_data['gamma_values']
        
        ax.plot(indices, gammas, 'b-', alpha=0.7, label='γ values')
        ax.axvline(x=resonance.zero_index, color='r', linestyle='--', label='Resonance Zero')
        ax.set_xlabel('Zero Index')
        ax.set_ylabel('Gamma Value')
        ax.set_title('Gamma Values in Neighborhood')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Distâncias normalizadas
        ax = axes[0, 1]
        distances = neighborhood_data['normalized_distances']
        
        ax.semilogy(indices, distances, 'g-', alpha=0.7)
        ax.axvline(x=resonance.zero_index, color='r', linestyle='--', label='Resonance Zero')
        ax.axhline(y=resonance.quality, color='orange', linestyle=':', label='Achieved Quality')
        ax.set_xlabel('Zero Index')
        ax.set_ylabel('Normalized Distance (log)')
        ax.set_title('Distance to Constant')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Histograma das diferenças
        ax = axes[1, 0]
        differences = neighborhood_data.get('differences', np.diff(gammas))
        
        ax.hist(differences, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(x=np.mean(differences), color='r', linestyle='--', label=f'Mean: {np.mean(differences):.3f}')
        ax.set_xlabel('Difference between consecutive γ')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Gamma Differences')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Análise de resíduos
        ax = axes[1, 1]
        x = np.array(range(len(gammas)))
        y = np.array(gammas)
        
        # Ajustar linha de tendência
        coeffs = np.polyfit(x, y, 1)
        trend = np.polyval(coeffs, x)
        residuals = y - trend
        
        ax.scatter(x, residuals, alpha=0.6, c=residuals, cmap='coolwarm')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Position in Window')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals from Linear Trend')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar figura
        filename = f"resonance_analysis_{resonance.constant_name}_{resonance.zero_index}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Visualização salva: {filename}")
        plt.close()  # Fechar para liberar memória
    
    def compare_resonances(self):
        """Compara as duas ressonâncias encontradas"""
        if len(self.target_resonances) < 2:
            logger.warning("⚠️ Não há ressonâncias suficientes para comparação")
            return
        
        r1, r2 = self.target_resonances[0], self.target_resonances[1]
        
        # Criar DataFrame para comparação
        data = {
            'Property': [
                'Zero Index', 'Gamma Value', 'Constant Name', 'Constant Value',
                'Quality', 'Relative Error (%)', 'Tolerance', 'Energy (GeV)'
            ],
            r1.constant_name: [
                f"{r1.zero_index:,}", f"{r1.gamma:.6f}", r1.constant_name, 
                f"{r1.constant_value:.6e}", f"{r1.quality:.2e}", 
                f"{r1.relative_error*100:.6f}", f"{r1.tolerance:.0e}", f"{r1.energy_gev:.1f}"
            ],
            r2.constant_name: [
                f"{r2.zero_index:,}", f"{r2.gamma:.6f}", r2.constant_name, 
                f"{r2.constant_value:.6e}", f"{r2.quality:.2e}", 
                f"{r2.relative_error*100:.6f}", f"{r2.tolerance:.0e}", f"{r2.energy_gev:.1f}"
            ]
        }
        
        df = pd.DataFrame(data)
        
        # Exibir tabela
        print("\n" + "="*80)
        print("COMPARAÇÃO DAS RESSONÂNCIAS")
        print("="*80)
        print(df.to_string(index=False))
        
        # Calcular razões entre as propriedades
        print("\n" + "="*80)
        print("RAZÕES ENTRE AS RESSONÂNCIAS")
        print("="*80)
        print(f"Razão dos índices (r2/r1): {r2.zero_index / r1.zero_index:.6f}")
        print(f"Razão dos gammas (r2/r1): {r2.gamma / r1.gamma:.6f}")
        print(f"Razão das qualidades (r2/r1): {r2.quality / r1.quality:.6e}")
        print(f"Razão das energias (r2/r1): {r2.energy_gev / r1.energy_gev:.6f}")
        
        # Análise adicional: verificar se há relação matemática
        print(f"\n🔍 ANÁLISE MATEMÁTICA DAS RAZÕES:")
        
        # Verificar se a razão dos índices é próxima de números interessantes
        index_ratio = r2.zero_index / r1.zero_index
        interesting_ratios = {
            '14': 14.0,
            'sqrt(196)': 14.0,
            '2*7': 14.0,
            'e^2.639': np.exp(2.639),  # e^2.639 ≈ 14
            'pi^2.2': np.pi**2.2,      # pi^2.2 ≈ 14
        }
        
        print(f"\nRazão dos índices ({index_ratio:.6f}) vs números interessantes:")
        for name, value in interesting_ratios.items():
            error = abs(index_ratio - value) / value
            print(f"  {name}: {value:.6f} (erro: {error:.2%})")
        
        # Verificar relação entre energia e constante
        print(f"\n🔍 RELAÇÕES ENERGIA-CONSTANTE:")
        for i, r in enumerate(self.target_resonances):
            energy_per_constant = r.energy_gev / r.constant_value
            print(f"  {r.constant_name}: E/constante = {energy_per_constant:.6e}")
        
        # Visualização comparativa
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Gráfico de barras comparativo (escala log para qualidade)
        properties = ['Quality (log)', 'Rel Error (%)', 'Energy (GeV)']
        r1_values = [np.log10(r1.quality), r1.relative_error*100, r1.energy_gev]
        r2_values = [np.log10(r2.quality), r2.relative_error*100, r2.energy_gev]
        
        x = np.arange(len(properties))
        width = 0.35
        
        ax1.bar(x - width/2, r1_values, width, label=r1.constant_name, alpha=0.8)
        ax1.bar(x + width/2, r2_values, width, label=r2.constant_name, alpha=0.8)
        
        ax1.set_ylabel('Value (Quality in log scale)')
        ax1.set_title('Comparison of Resonance Properties')
        ax1.set_xticks(x)
        ax1.set_xticklabels(properties)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico de dispersão índice vs gamma (escala log)
        ax2.scatter([r1.zero_index, r2.zero_index], [r1.gamma, r2.gamma], 
                   s=[200, 200], c=['red', 'blue'], alpha=0.7)
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Zero Index (log)')
        ax2.set_ylabel('Gamma Value (log)')
        ax2.set_title('Zero Index vs Gamma Value (log-log)')
        ax2.grid(True, alpha=0.3)
        
        # Adicionar anotações
        ax2.annotate(r1.constant_name, (r1.zero_index, r1.gamma), 
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
        ax2.annotate(r2.constant_name, (r2.zero_index, r2.gamma), 
                    xytext=(10, 10), textcoords='offset points', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('resonance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_exploration(self):
        """Executa a exploração completa das ressonâncias"""
        logger.info("🚀 Iniciando exploração das ressonâncias...")
        
        # Informações sobre os dados disponíveis
        if self.zeros:
            logger.info(f"📊 Dados disponíveis: {len(self.zeros):,} zeros")
            logger.info(f"📊 Faixa de índices: 1 a {self.zeros[-1][0]:,}")
        else:
            logger.warning("⚠️ Nenhum dado de zeros disponível - usando análise sintética")
        
        # 1. Análise individual de cada ressonância
        for i, resonance in enumerate(self.target_resonances):
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 ANÁLISE DA RESSONÂNCIA {i+1}: {resonance.constant_name.upper()}")
            logger.info(f"{'='*60}")
            
            # Análise da vizinhança (real ou sintética)
            if self.zeros and len(self.zeros) > resonance.zero_index:
                neighborhood_data = self.analyze_neighborhood(resonance, 100)
                if neighborhood_data:
                    logger.info(f"📊 Estatísticas da vizinhança:")
                    logger.info(f"   Média dos γ: {neighborhood_data['local_stats']['mean']:.6f}")
                    logger.info(f"   Desvio padrão: {neighborhood_data['local_stats']['std']:.6f}")
                    logger.info(f"   Mínimo local: Zero #{neighborhood_data['local_minimum']['index']:,} "
                               f"(distância: {neighborhood_data['local_minimum']['distance']:.2e})")
            else:
                logger.info("📊 Usando análise sintética (dados reais não disponíveis)")
            
            # Testar relações matemáticas
            math_results = self.test_mathematical_relations(resonance)
            logger.info(f"\n🔢 Relações matemáticas significativas:")
            
            significant_found = False
            for key, value in math_results.items():
                if isinstance(value, dict) and value.get('significant', False):
                    logger.info(f"   {key}: {value['value']:.6f} ≈ {value['target']:.6f} "
                               f"(erro: {value['error']:.2%})")
                    significant_found = True
            
            if not significant_found:
                logger.info("   Nenhuma relação matemática óbvia encontrada")
            
            # Gerar visualização
            self.visualize_resonance(resonance)
        
        # 2. Comparação entre ressonâncias
        logger.info(f"\n{'='*60}")
        logger.info(f"🔀 COMPARAÇÃO ENTRE RESSONÂNCIAS")
        logger.info(f"{'='*60}")
        self.compare_resonances()
        
        # 3. Análise teórica adicional
        logger.info(f"\n{'='*60}")
        logger.info(f"🧠 ANÁLISE TEÓRICA")
        logger.info(f"{'='*60}")
        
        logger.info("\n🔍 Hipóteses sobre as ressonâncias:")
        logger.info("1. A ressonância da estrutura fina pode estar relacionada à:")
        logger.info("   - Escala de energia de unificação eletrofraca")
        logger.info("   - Constante de acoplamento em teorias de gauge")
        logger.info("   - Relações com o número 137 (inverso de α)")
        
        logger.info("\n2. A ressonância da massa do elétron pode indicar:")
        logger.info("   - Origem matemática para a massa das partículas")
        logger.info("   - Conexão com a hierarquia de massas do Modelo Padrão")
        logger.info("   - Relação com mecanismos de quebra de simetria")
        
        logger.info("\n3. A razão ~14 entre os índices sugere:")
        logger.info("   - Possível relação com o número de gerações de partículas")
        logger.info("   - Conexão com dimensões extras em teorias de cordas")
        logger.info("   - Fator de escala entre diferentes regimes físicos")
        
        logger.info("\n✅ Exploração concluída!")

# Execução principal
if __name__ == "__main__":
    try:
        explorer = ResonanceExplorer()
        explorer.run_exploration()
    except Exception as e:
        logger.error(f"❌ Erro durante a exploração: {e}")
        import traceback
        traceback.print_exc()
