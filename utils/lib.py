from tqdm import tqdm

from utils.conf import length_of_sequence, sequences


def get_input():
    while True:
        seq = input("Input function: ").strip().lower()
        if sequences.get(seq) is None:
            print("This function doesn't exist!")
        else:
            function = sequences.get(seq)
            break

    return function


def training(model, train, number_of_epochs) -> None:
    for _ in range(number_of_epochs):
        model.fit(train, validation_data=None)


def testing(model, test_sequence) -> list:
    result = [test_sequence[i][0] for i in range(length_of_sequence)]
    start = [test_sequence[:length_of_sequence]]
    length = len(test_sequence) - length_of_sequence
    for _ in tqdm(range(length)):
        next_step = float(model.predict(start, verbose=0)[0][0])
        start[0] = start[0][1:]
        start[0].append([next_step])
        result.append(next_step)

    return result
