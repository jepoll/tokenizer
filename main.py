from base_tokenizer import print_tokens
from basic_tokenizer import BasicTokenizer
from regex_tokenizer import RegexTokenizer

def main():
    txt = open("tests/taylorswift.txt", "r", encoding="utf-8").read()
    basic = BasicTokenizer()
    basic.train(txt, 300)

    regex = RegexTokenizer()
    regex.train(txt, 300)

    test_string = "taylor. taylor! taylor?"

    print_tokens(basic, test_string, "BasicTokenizer")
    print_tokens(regex, test_string, "RegexTokenizer")

if __name__ == '__main__':
    main()
