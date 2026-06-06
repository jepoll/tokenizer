from basic_tokenizer import BasicTokenizer


def main():
    txt = open("tests/taylorswift.txt", "r", encoding="utf-8").read()
    tokenizer = BasicTokenizer()

    tokenizer.train(txt, 300, verbose=True)

    test_string = "hello world,  😄🙂"

    print(tokenizer.decode(tokenizer.encode(test_string)))

if __name__ == '__main__':
    main()
