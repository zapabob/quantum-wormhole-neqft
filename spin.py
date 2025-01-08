import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# システムサイズ
L = 2  # 2スピンシステム

# 外部磁場の値と点欠陥間の距離の設定
h_values = [0.1, 0.2, 0.3]
distance = 1  # システムサイズ2では距離1のみ有効

# 基本的なスピン演算子
sx = sigmax()
sy = sigmay()
sz = sigmaz()
I = qeye(2)

# 2スピン系の演算子を構築
sx_list = [tensor(sx, I), tensor(I, sx)]
sy_list = [tensor(sy, I), tensor(I, sy)]
sz_list = [tensor(sz, I), tensor(I, sz)]

def create_hamiltonian_naqft(h, wormhole_strength=1.0):
    """2スピンシステム用ハミルトニアンの生成（NAQFTに基づく）"""
    # マヨラナ結合項（計算論的ワームホール）
    H_majorana = 0.5 * wormhole_strength * (sx_list[0] + sx_list[1])
    
    # 外部磁場項（時空の曲率に対応）
    H_field = h * (sz_list[0] + sz_list[1])
    
    # 非局所相互作用項（ワームホール経由の相互作用）
    H_nonlocal = wormhole_strength * tensor(sx, sx)
    
    # 位相因子（幾何学的位相）
    phase = np.exp(1j * np.pi * distance / L)
    H_geometric = 0.1 * (phase * (sx_list[0] + 1j * sy_list[0]) + np.conj(phase) * (sx_list[0] - 1j * sy_list[0]))
    
    return H_majorana + H_field + H_nonlocal + H_geometric

def calculate_entanglement_entropy(state):
    """エンタングルメントエントロピーの計算"""
    rho = ket2dm(state) if isinstance(state, Qobj) and state.type == 'ket' else state
    rho_A = ptrace(rho, 0)
    return entropy_vn(rho_A)

def create_clifford_generators_2spin():
    """2スピンシステム用クリフォード代数の生成子を作成"""
    gamma1 = (sx_list[0] + 1j * sy_list[0]).unit()
    gamma2 = (sx_list[1] + 1j * sy_list[1]).unit()
    gamma3 = sz_list[0].unit()
    gamma4 = sz_list[1].unit()
    gamma5 = (gamma1 * gamma2 * gamma3 * gamma4).unit()
    return [gamma1, gamma2, gamma3, gamma4, gamma5]

def calculate_nonlocal_correlation(h, wormhole_strength=1.0):
    """非局所相関の計算（量子テレポーテーション効率）"""
    # ハミルトニアンの生成
    H = create_hamiltonian_naqft(h, wormhole_strength)
    
    # 初期状態（マクシマリーエンタングルド状態）
    psi0 = (tensor(basis(2,0), basis(2,0)) + tensor(basis(2,1), basis(2,1))).unit()
    
    # 時間発展
    tlist = np.linspace(0, 10, 100)
    options = Options(store_states=True, nsteps=8000)
    
    # スピン相関演算子（正しい次元で構築）
    sz_correlation = tensor(sz, sz)
    
    result = mesolve(H, psi0, tlist, [], [sz_correlation], options=options)
    
    if not hasattr(result, 'expect') or len(result.expect[0]) == 0:
        return 0.0, 0.0
    
    # 相関値の計算
    correlation = np.abs(result.expect[0][-1])
    
    # エンタングルメントエントロピーの計算
    if hasattr(result, 'states') and len(result.states) > 0:
        entropy = calculate_entanglement_entropy(result.states[-1])
    else:
        entropy = 0.0
    
    return correlation, entropy

def calculate_topological_invariant(state, gammas):
    """トポロジカル不変量の計算"""
    inv = 0
    for i in range(len(gammas)-1):
        for j in range(i+1, len(gammas)):
            op = gammas[i] * gammas[j] - gammas[j] * gammas[i]
            inv += np.abs(expect(op, state))
    return np.real(inv) / len(gammas)

# 結果の保存用配列
results = {
    'correlation_naqft': np.zeros(len(h_values)),
    'entropy_naqft': np.zeros(len(h_values)),
    'topological_inv': np.zeros(len(h_values))
}

# クリフォード生成子の作成
gammas = create_clifford_generators_2spin()

# 計算実行
for i, h in enumerate(h_values):
    # NAQFTによる相関とエントロピーの計算
    correlation, entropy = calculate_nonlocal_correlation(h)
    results['correlation_naqft'][i] = correlation
    results['entropy_naqft'][i] = entropy
    
    # トポロジカル不変量の計算
    H = create_hamiltonian_naqft(h)
    eigenvalues, eigenvectors = H.eigenstates()
    psi_final = eigenvectors[0]
    results['topological_inv'][i] = calculate_topological_invariant(psi_final, gammas)

# 結果のプロット
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# 相関のプロット
axes[0].plot(h_values, results['correlation_naqft'], 'bo-', label='NAQFT Correlation')
axes[0].set_xlabel('External Magnetic Field |h|')
axes[0].set_ylabel('Correlation')
axes[0].set_title('Non-local Correlations')
axes[0].legend()
axes[0].grid(True)

# エントロピーのプロット
axes[1].plot(h_values, results['entropy_naqft'], 'ro-', label='NAQFT Entropy')
axes[1].set_xlabel('External Magnetic Field |h|')
axes[1].set_ylabel('Entanglement Entropy')
axes[1].set_title('Entanglement Entropy')
axes[1].legend()
axes[1].grid(True)

# トポロジカル不変量のプロット
axes[2].plot(h_values, results['topological_inv'], 'go-', label='Topological Invariant')
axes[2].set_xlabel('External Magnetic Field |h|')
axes[2].set_ylabel('Topological Invariant')
axes[2].set_title('Topological Structure')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()

# STM測定のシミュレーション
def calculate_ldos(h, energy_range, wormhole_strength=1.0):
    """局所状態密度の計算"""
    H = create_hamiltonian_naqft(h, wormhole_strength)
    energies = np.linspace(-energy_range, energy_range, 100)
    ldos = np.zeros_like(energies)
    
    eigenvalues, eigenvectors = H.eigenstates()
    gamma = 0.1  # ブロードニング因子
    
    for i, E in enumerate(energies):
        for j, eigval in enumerate(eigenvalues):
            psi = eigenvectors[j]
            matrix_element = abs(expect(sx_list[0], psi))**2
            ldos[i] += matrix_element * gamma / ((E - eigval)**2 + gamma**2)
    
    return energies, ldos

# STM結果の表示
energy_range = 2.0
energies, ldos = calculate_ldos(h_values[0], energy_range)

plt.figure(figsize=(10,6))
plt.plot(energies, ldos)
plt.xlabel('Energy (E)')
plt.ylabel('Local Density of States')
plt.title(f'Simulated STM Measurement (h={h_values[0]})')
plt.grid(True)
plt.show()
