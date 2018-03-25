import math
from src.dataloader.config import M, F

class NaiveBayesClassfier:

    def __init__(self):
        print("NaiveBayesClassfier instantiated...")


    def fit(self, X, Y):
        print("Training the classifier...")
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = set()

        n = len(X)
        self.log_class_priors[M] = math.log(sum(1 for label in Y if label == 1) / n)
        self.log_class_priors[F] = math.log(sum(1 for label in Y if label == 0) / n)

        self.word_counts[M] = {}
        self.word_counts[F] = {}

        for x, y in zip(X, Y):
            c = y

            for word, count in x.items():
                if word not in self.vocab:
                    self.vocab.add(word)
                if word not in self.word_counts[c]:
                    self.word_counts[c][word] = 0.0

                self.word_counts[c][word] += count
        print("Training completed")

    def predict(self, X):
        print("Predicting class labels...")

        result = []
        for x in X:

            male_score = 0
            female_score = 0
            for word, _ in x.items():
                if word not in self.vocab: continue

                # add Laplace smoothing
                log_w_given_male = math.log((self.word_counts[M].get(word, 0.0) + 1) / (
                    sum(self.word_counts[M].values()) + len(self.vocab)))
                log_w_given_female = math.log((self.word_counts[F].get(word, 0.0) + 1) / (
                    sum(self.word_counts[F].values()) + len(self.vocab)))

                male_score += log_w_given_male
                female_score += log_w_given_female

            male_score += self.log_class_priors[M]
            female_score += self.log_class_priors[F]

            if male_score > female_score:
                result.append(M)
            else:
                result.append(F)
        return result