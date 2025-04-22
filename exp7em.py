import numpy as np

# Given binary sequences (1 = heads, 0 = tails)
data = [
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
    [1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 1, 0]
]

# Initial parameter estimates
theta_A, theta_B = 0.6, 0.5
threshold = 1e-4
max_iterations = 5

iteration = 0
while iteration < max_iterations:
    iteration += 1
    prev_theta_A, prev_theta_B = theta_A, theta_B

    # E-step: Estimate responsibilities (probability of each coin generating each sequence)
    P_A, P_B = [], []
    for seq in data:
        h = sum(seq)
        t = len(seq) - h

        # Likelihood of sequence under each coin
        L_A = (theta_A ** h) * ((1 - theta_A) ** t)
        L_B = (theta_B ** h) * ((1 - theta_B) ** t)

        # Normalize
        denom = L_A + L_B
        P_A_i = L_A / denom if denom > 0 else 0.5
        P_B_i = L_B / denom if denom > 0 else 0.5

        P_A.append(P_A_i)
        P_B.append(P_B_i)

    # M-step: Update theta_A and theta_B using responsibilities
    total_heads_A = sum(P_A[i] * sum(data[i]) for i in range(len(data)))
    total_flips_A = sum(P_A[i] * len(data[i]) for i in range(len(data)))
    theta_A = total_heads_A / total_flips_A if total_flips_A else prev_theta_A

    total_heads_B = sum(P_B[i] * sum(data[i]) for i in range(len(data)))
    total_flips_B = sum(P_B[i] * len(data[i]) for i in range(len(data)))
    theta_B = total_heads_B / total_flips_B if total_flips_B else prev_theta_B

    # Convergence check
    if abs(theta_A - prev_theta_A) < threshold and abs(theta_B - prev_theta_B) < threshold:
        break

# Output
print(f"\n✨ EM Algorithm Complete After {iteration} Iterations ✨")
print(f"Estimated θ_A (Coin A Bias): {theta_A:.6f}")
print(f"Estimated θ_B (Coin B Bias): {theta_B:.6f}")
