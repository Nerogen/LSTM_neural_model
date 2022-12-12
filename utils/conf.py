from math import factorial, sin, cos, tan

length_of_sequence = 50
start_value = 0
end_value = 100
steps_on_range = 1000
number_of_epochs = 10
batch_size_of_sequence = 10
train_size_of_sequence = 0.8
activation_function = "relu"
sequences = {
    "1/(2^x)": lambda x: 1 / (2 ** x),
    "x": lambda x: x,
    "(-1)^(x-1)*x": lambda x: (-1) ** ((x - 1) * x),
    "x!": lambda x: factorial(x),
    "sin(x)": lambda x: sin(x),
    "cos(x)": lambda x: cos(x),
    "tan(x)": lambda x: tan(x)
}
