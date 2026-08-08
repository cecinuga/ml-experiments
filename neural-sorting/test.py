import numpy as np

def disorder_test(row: int, length_upto: list[int], seed: float) -> None:
    rng = np.random.default_rng(seed)

    for length in range(2, length_upto):
        X = rng.standard_normal((row, length))
        loss_test_dataset = np.stack([X, np.sort(X)])
        loss_test_dataset = loss_test_dataset.swapaxes(0, 1)

        for i, (x, y) in enumerate(loss_test_dataset):
            random_disorder = round(disorder_v1(x), 5)
            sorted_disorder = round(disorder_v1(y), 5)

            try:
                assert sorted_disorder <= random_disorder
            except Exception as e:
                print(f'iter={i}, lenght={length}')
                print(f"random disorder={random_disorder}")
                print(f"sorted disorder={sorted_disorder}")
                print('-----------------------------------------')
                print(f"random={x}")
                print(f"sorted={y}")
                raise e