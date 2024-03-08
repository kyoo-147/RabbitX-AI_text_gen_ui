"""
Mô-đun này chứa các tiện ích để xử lý trước văn bản trước khi chuyển đổi nó thành phần nhúng.

- TextPreprocessorBuilder xử lý trước các chuỗi riêng lẻ.
    * trường hợp hạ thấp
    * chuyển đổi số thành từ hoặc ký tự
    * sáp nhập và tước không gian
    *bỏ dấu câu
    * loại bỏ các từ dừng
    * bổ ngữ
    * loại bỏ các phần cụ thể của lời nói (trạng từ và xen kẽ)
- TextSummarizer trích xuất các câu quan trọng nhất từ một chuỗi dài bằng cách sử dụng tính năng xếp hạng văn bản.
"""
import pytextrank
import string
import spacy
import math
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from num2words import num2words


class TextPreprocessorBuilder:
     # Define class variables as None initially
    _stop_words = set(stopwords.words('english'))
    _lemmatizer = WordNetLemmatizer()
    
    # Some of the functions are expensive. We cache the results.
    _lemmatizer_cache = {}
    _pos_remove_cache = {}


    def __init__(self, text: str):
        self.text = text


    def to_lower(self):
        # Match both words and non-word characters
        tokens = re.findall(r'\b\w+\b|\W+', self.text)
        for i, token in enumerate(tokens):
            # Check if token is a word
            if re.match(r'^\w+$', token):
                # Check if token is not an abbreviation or constant
                if not re.match(r'^[A-Z]+$', token) and not re.match(r'^[A-Z_]+$', token):
                    tokens[i] = token.lower()
        self.text = "".join(tokens)
        return self


    def num_to_word(self, min_len: int = 1):
        # Match both words and non-word characters
        tokens = re.findall(r'\b\w+\b|\W+', self.text)
        for i, token in enumerate(tokens):
            # Check if token is a number of length `min_len` or more
            if token.isdigit() and len(token) >= min_len:
                # This is done to pay better attention to numbers (e.g. ticket numbers, thread numbers, post numbers)
                # 740700 will become "seven hundred and forty thousand seven hundred".
                tokens[i] = num2words(int(token)).replace(",","") # Remove commas from num2words.
        self.text = "".join(tokens)
        return self


    def num_to_char_long(self, min_len: int = 1):
        # Match both words and non-word characters
        tokens = re.findall(r'\b\w+\b|\W+', self.text)
        for i, token in enumerate(tokens):
            # Check if token is a number of length `min_len` or more
            if token.isdigit() and len(token) >= min_len:
                # This is done to pay better attention to numbers (e.g. ticket numbers, thread numbers, post numbers)
                # 740700 will become HHHHHHEEEEEAAAAHHHAAA
                convert_token = lambda token: ''.join((chr(int(digit) + 65) * (i + 1)) for i, digit in enumerate(token[::-1]))[::-1]
                tokens[i] = convert_token(tokens[i])
        self.text = "".join(tokens)
        return self
    
    def num_to_char(self, min_len: int = 1):
        # Match both words and non-word characters
        tokens = re.findall(r'\b\w+\b|\W+', self.text)
        for i, token in enumerate(tokens):
            # Check if token is a number of length `min_len` or more
            if token.isdigit() and len(token) >= min_len:
                # This is done to pay better attention to numbers (e.g. ticket numbers, thread numbers, post numbers)
                # 740700 will become HEAHAA
                tokens[i] = ''.join(chr(int(digit) + 65) for digit in token)
        self.text = "".join(tokens)
        return self
    
    def merge_spaces(self):
        self.text = re.sub(' +', ' ', self.text)
        return self
    
    def strip(self):
        self.text = self.text.strip()
        return self
        
    def remove_punctuation(self):
        self.text = self.text.translate(str.maketrans('', '', string.punctuation))
        return self

    def remove_stopwords(self):
        self.text = "".join([word for word in re.findall(r'\b\w+\b|\W+', self.text) if word not in TextPreprocessorBuilder._stop_words])
        return self
    
    def remove_specific_pos(self):
        """
        Trong tiếng Anh, trạng từ và thán từ hiếm khi cung cấp thông tin có ý nghĩa.
        Việc loại bỏ chúng sẽ cải thiện độ chính xác khi nhúng. Tuy nhiên, đừng nói với JK Rowling.
        """
        processed_text = TextPreprocessorBuilder._pos_remove_cache.get(self.text)
        if processed_text:
            self.text = processed_text
            return self

        # Match both words and non-word characters
        tokens = re.findall(r'\b\w+\b|\W+', self.text)

        # Exclude adverbs and interjections
        excluded_tags = ['RB', 'RBR', 'RBS', 'UH']

        for i, token in enumerate(tokens):
            # Check if token is a word
            if re.match(r'^\w+$', token):
                # Part-of-speech tag the word
                pos = nltk.pos_tag([token])[0][1]
                # If the word's POS tag is in the excluded list, remove the word
                if pos in excluded_tags:
                    tokens[i] = ''

        new_text = "".join(tokens)
        TextPreprocessorBuilder._pos_remove_cache[self.text] = new_text
        self.text = new_text

        return self

    def lemmatize(self):
        processed_text = TextPreprocessorBuilder._lemmatizer_cache.get(self.text)
        if processed_text:
            self.text = processed_text
            return self
        
        new_text = "".join([TextPreprocessorBuilder._lemmatizer.lemmatize(word) for word in re.findall(r'\b\w+\b|\W+', self.text)])
        TextPreprocessorBuilder._lemmatizer_cache[self.text] = new_text
        self.text = new_text

        return self

    def build(self):
        return self.text

class TextSummarizer:
    _nlp_pipeline = None
    _cache = {}

    @staticmethod
    def _load_nlp_pipeline():
        # Lazy-load it.
        if TextSummarizer._nlp_pipeline is None:
            TextSummarizer._nlp_pipeline = spacy.load('en_core_web_sm')
            TextSummarizer._nlp_pipeline.add_pipe("textrank", last=True)
        return TextSummarizer._nlp_pipeline

    @staticmethod
    def process_long_text(text: str, min_num_sent: int) -> list[str]:
        """
        Hàm này áp dụng quy trình tóm tắt văn bản trên một chuỗi văn bản nhất định, trích xuất
        những câu quan trọng nhất dựa trên nguyên tắc 20% nội dung chịu trách nhiệm
        cho 80% ý nghĩa (Nguyên tắc Pareto).

        Trả về:
        list: Danh sách các câu quan trọng nhất
        """

        # Attempt to get the result from cache
        cache_key = (text, min_num_sent)
        cached_result = TextSummarizer._cache.get(cache_key, None)
        if cached_result is not None:
            return cached_result

        nlp_pipeline = TextSummarizer._load_nlp_pipeline()
        doc = nlp_pipeline(text)

        num_sent = len(list(doc.sents))
        result = []

        if num_sent >= min_num_sent:

            limit_phrases = math.ceil(len(doc._.phrases) * 0.20)  # 20% of the phrases, rounded up
            limit_sentences = math.ceil(num_sent * 0.20)  # 20% of the sentences, rounded up
            result = [str(sent) for sent in doc._.textrank.summary(limit_phrases=limit_phrases, limit_sentences=limit_sentences)]

        else:
            result = [text]
        
        # Store the result in cache before returning it
        TextSummarizer._cache[cache_key] = result
        return result