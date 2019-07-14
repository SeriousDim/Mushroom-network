from definer import init, classify, get_scores, print_scores
import inception_model as inc

if __name__ == "__main__":
    init()
    pred = classify("cut/right/1.jpg")
    print_scores(pred=pred, k=3)
    print(get_scores(pred, k=3))