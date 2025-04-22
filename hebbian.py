def hebbian_learning(samples):
    print(f'{"Input":^15} {"Target":^8} {"Weight Changes":^25} {"Updated Weights":^25}')
    w1, w2, b = 0, 0, 0
    print(' ' * 55 + f'({w1:4},{w2:4},{b:4})')
    
    for x1, x2, y in samples:
        delta_w1 = x1 * y
        delta_w2 = x2 * y
        delta_b = y

        w1 += delta_w1
        w2 += delta_w2
        b += delta_b

        print(f'({x1:2},{x2:2})       {y:4}     '
              f'({delta_w1:4},{delta_w2:4},{delta_b:4})     '
              f'({w1:4},{w2:4},{b:4})')

# Define all sample types
AND_samples = {
    'binary_input_binary_output': [
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ],
    'binary_input_bipolar_output': [
        [1, 1, 1],
        [1, 0, -1],
        [0, 1, -1],
        [0, 0, -1]
    ],
    'bipolar_input_bipolar_output': [
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, -1]
    ]
}


print('\n--- AND with Binary Input & Binary Output ---\n')
hebbian_learning(AND_samples['binary_input_binary_output'])

print('\n--- AND with Binary Input & Bipolar Output ---\n')
hebbian_learning(AND_samples['binary_input_bipolar_output'])

print('\n--- AND with Bipolar Input & Bipolar Output ---\n')
hebbian_learning(AND_samples['bipolar_input_bipolar_output'])
