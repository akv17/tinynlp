from sklearn.model_selection import train_test_split


def split_val(objects, size, seed=0):
    train, val = train_test_split(objects, test_size=size, random_state=seed)
    return train, val
