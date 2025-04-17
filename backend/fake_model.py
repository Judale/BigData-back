import random

WORDS = ["cat", "bicycle", "banana", "tree", "book"]

def predict(drawing_data):
    predicted_label = random.choice(WORDS)
    probability = round(random.uniform(0.5, 1.0), 2)
    return predicted_label, probability

if __name__ == "__main__":
    test = [[[1, 2, 3], [4, 5, 6]]]
    print(predict(test))
