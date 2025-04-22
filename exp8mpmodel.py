import numpy as np

# Define the dataset for an AND gate
x1 = np.array([0, 0, 1, 1])
x2 = np.array([0, 1, 0, 1])
y  = np.array([0, 0, 0, 1])

# Prompt user
w1 = float(input("Enter weight w1: "))
w2 = float(input("Enter weight w2: "))
theta = float(input("Enter threshold θ: "))

# Compute the linear combination and predictions
f = w1 * x1 + w2 * x2
y_pred = (f >= theta).astype(int)

# Display results
print("\nInputs  x1 x2 |  w1·x1 + w2·x2  |  Prediction")
print("--------+--------------------+-------------")
for xi1, xi2, fi, pi in zip(x1, x2, f, y_pred):
    print(f"        {xi1}  {xi2}   |     {fi:4.1f}          |     {pi}")

# Check correctness
if np.array_equal(y, y_pred):
    print("\n✅ Correct! (This implements AND.)")
    print(f"  w1 = {w1}, w2 = {w2}, θ = {theta}")
else:
    print("\n❌ Incorrect for AND. Try again!")


# 1 1 2
# 0.5 0.5 1
# 1 1 1