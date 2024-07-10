import numpy as np
import tensornetwork as tn
import tensorflow as tf
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ttest_ind

tn.set_default_backend("tensorflow")

class NAQFTTensorNetwork:
    def __init__(self, n_qubits, N):
        self.n_qubits = n_qubits
        self.N = N
        self.dim = 2**n_qubits

    def create_naqft_layer(self):
        nodes = []
        for i in range(self.dim):
            tensor = np.zeros((self.dim, self.dim), dtype=complex)
            for j in range(self.dim):
                phase = 2 * np.pi * i * j / self.dim
                tensor[i, j] = np.exp(1j * phase) * (1 + np.sqrt(2) * np.pi * np.sin(phase) * np.cos(phase))
            nodes.append(tn.Node(tf.convert_to_tensor(tensor)))
        return nodes

    def apply_naqft(self, state_vector):
        state = tn.Node(tf.convert_to_tensor(state_vector, dtype=tf.complex128))
        naqft_layer = self.create_naqft_layer()

        for node in naqft_layer:
            tn.connect(state[0], node[0])
            state = tn.contract_between(state, node, name="naqft_apply")

        return state.tensor

    def compute_path_integral(self, x_initial, x_final, n_paths=100):
        results = []
        for _ in range(n_paths):
            state = np.zeros((self.dim,), dtype=complex)
            state[x_initial] = 1.0

            transformed_state = self.apply_naqft(state)

            if x_final < self.dim:
                action = np.random.uniform(0, 2 * np.pi)
                phase_factor = np.exp(1j * action)
                result = phase_factor * transformed_state[x_final]
                results.append(result)

        return results

def analyze_naqft_wormhole(n_qubits, N, n_samples=50, n_paths=100, x_initial=0):
    naqft_tn = NAQFTTensorNetwork(n_qubits, N)
    x_range = np.linspace(0, naqft_tn.dim - 1, num=n_samples, dtype=int)
    all_results = []

    for x in tqdm(x_range):
        results = naqft_tn.compute_path_integral(x_initial, x, n_paths)
        all_results.append(results)

    return x_range, all_results

def compute_entropy(state):
    probs = np.abs(state) ** 2
    probs = probs[probs > 1e-16]
    return -np.sum(probs * np.log2(probs))

def compute_energy(state):
    return np.sum(np.abs(state)**2)

def compute_correlation(state1, state2):
    return np.abs(np.vdot(state1, state2))

def cohen_d(group1, group2):
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((len(group1) - 1) * std1**2 + (len(group2) - 1) * std2**2) / (len(group1) + len(group2) - 2))
    d = (mean1 - mean2) / pooled_std
    return d

def t_test(group1, group2):
    group1 = np.array(group1)
    group2 = np.array(group2)
    group1 = group1.astype(np.float64)
    group2 = group2.astype(np.float64)
    t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
    return t_stat, p_val

def effect_size_r(group1, group2):
    t_stat, _ = t_test(group1, group2)
    r = np.sqrt(t_stat**2 / (t_stat**2 + len(group1) + len(group2) - 2))
    return r

def analyze_physical_quantities(x_range, all_results):
    mean_results = np.mean(all_results, axis=1)
    phases = np.angle(mean_results)
    phase_changes = np.abs(np.diff(phases))
    entropies = [compute_entropy(result) for result in mean_results]
    amplitudes = np.abs(mean_results)
    energies = [compute_energy(result) for result in mean_results]
    correlations = [compute_correlation(mean_results[0], result) for result in mean_results]

    wormhole_indicator = np.max(phase_changes)
    wormhole_position = x_range[np.argmax(phase_changes) + 1]

    return {
        "phases": phases,
        "phase_changes": phase_changes,
        "entropies": entropies,
        "amplitudes": amplitudes,
        "energies": energies,
        "correlations": correlations,
        "wormhole_indicator": wormhole_indicator,
        "wormhole_position": wormhole_position,
        "all_results": all_results,
    }

def compute_statistics(all_results):
    stats = {
        "cohen_d": [],
        "t_test": [],
        "effect_size_r": []
    }
    for i in range(len(all_results)):
        group_results = [results[i] for results in all_results if i < len(results)]
        if len(group_results) > 1:
            d_value = cohen_d(group_results[:len(group_results)//2], group_results[len(group_results)//2:])
            t_stat, p_val = t_test(group_results[:len(group_results)//2], group_results[len(group_results)//2:])
            r_value = effect_size_r(group_results[:len(group_results)//2], group_results[len(group_results)//2:])
        else:
            d_value, t_stat, p_val, r_value = np.nan, np.nan, np.nan, np.nan
        stats["cohen_d"].append(d_value)
        stats["t_test"].append((t_stat, p_val))
        stats["effect_size_r"].append(r_value)
    return stats

def visualize_results(x_range, quantities, std_results, stats):
    fig, axs = plt.subplots(8, 1, figsize=(12, 40))

    axs[0].errorbar(x_range, quantities["phases"], yerr=std_results, fmt="-o")
    axs[0].set_title("phase")
    axs[0].set_xlabel("position")
    axs[0].set_ylabel("phase")

    axs[1].plot(x_range[1:], quantities["phase_changes"])
    axs[1].set_title("phase change")
    axs[1].set_xlabel("position")
    axs[1].set_ylabel("phase change")
    axs[1].axhline(y=np.pi, color="r", linestyle="--", label="π")
    axs[1].legend()

    axs[2].plot(x_range, quantities["entropies"])
    axs[2].set_title("entropy")
    axs[2].set_xlabel("position")
    axs[2].set_ylabel("entropy")

    axs[3].plot(x_range, quantities["amplitudes"])
    axs[3].set_title("amplitude")
    axs[3].set_xlabel("position")
    axs[3].set_ylabel("amplitude")

    axs[4].plot(x_range, quantities["energies"])
    axs[4].set_title("energy")
    axs[4].set_xlabel("position")
    axs[4].set_ylabel("energy")

    axs[5].plot(x_range, quantities["correlations"])
    axs[5].set_title("correlation")
    axs[5].set_xlabel("position")
    axs[5].set_ylabel("correlation")

    valid_cohen_indices = ~np.isnan(stats["cohen_d"])
    axs[6].plot(
        x_range[valid_cohen_indices],
        np.array(stats["cohen_d"])[valid_cohen_indices],
        label="Cohen's d",
        color="green",
    )
    axs[6].set_title("Cohen's d")
    axs[6].set_xlabel("position")
    axs[6].set_ylabel("Cohen's d")
    axs[6].legend()

    valid_t_indices = ~np.isnan([t_stat for t_stat, _ in stats["t_test"]])
    axs[7].plot(
        x_range[valid_t_indices],
        np.array([t_stat for t_stat, _ in stats["t_test"]])[valid_t_indices],
        label="t-statistic",
        color="blue",
    )
    axs[7].set_title("t-test statistic")
    axs[7].set_xlabel("position")
    axs[7].set_ylabel("t-statistic")
    axs[7].legend()

    plt.tight_layout()
    plt.show()

def analyze_with_different_initial_states(
    n_qubits, N, initial_states, n_samples=50, n_paths=100
):
    results = []

    for init_state in initial_states:
        x_range, all_results = analyze_naqft_wormhole(
            n_qubits, N, n_samples, n_paths, init_state
        )
        quantities = analyze_physical_quantities(x_range, all_results)
        std_results = np.std(all_results, axis=1)
        stats = compute_statistics(all_results)
        results.append(
            (init_state, x_range, quantities, std_results, stats)
        )

    return results

# メイン実行
n_qubits = 8
N = 21
initial_states = [0, 1, 2, 3]  # 初期状態の例
n_samples = 50
n_paths = 50  # 経路積分の回数を増やす

results = analyze_with_different_initial_states(
    n_qubits,N, initial_states, n_samples, n_paths
)

for init_state, x_range, quantities, std_results, stats in results:
    print(f"initial state: {init_state}")
    print(f"calculational wormhole indicator: {quantities['wormhole_indicator']}")
    print(f"wormhole position: {quantities['wormhole_position']}")

    # 各グループのサイズを出力してデバッグ
    group_sizes = [len(results) for results in quantities["all_results"]]
    print(f"group sizes: {group_sizes}")

    visualize_results(x_range, quantities, std_results, stats)
