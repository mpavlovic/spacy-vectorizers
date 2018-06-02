import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, VectorizerMixin
from sklearn.base import TransformerMixin, BaseEstimator
from scipy.sparse import csr_matrix

class DenseToSparseTransformer(TransformerMixin, BaseEstimator):    
    def __init__(self, input_type='list'):
        self.input_type = input_type
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, array_like):
        ndarray = array_like
        if self.input_type == 'input':
            ndarray = np.array(array_like)
        return csr_matrix(ndarray)
    
class SparseToDenseTransformer(TransformerMixin, BaseEstimator):    
    def fit(self, x, y=None):
        return self
    
    def transform(self, sparse_matrix):
        return sparse_matrix.toarray()

class SpacySimpleTokenizer:
    def __init__(self, nlp):
        self.nlp = nlp
    
    def __call__(self, raw_document):
        return [token.text for token in self.nlp(raw_document)]

class SpacyPipeInitializer(object):
    def __init__(self, nlp, join_str=" ", batch_size=10000, n_threads=2):
        self.nlp = nlp
        self.join_str = join_str
        self.batch_size = batch_size
        self.n_threads = n_threads
        
class SpacyPipeProcessor(SpacyPipeInitializer):
    def __init__(self, nlp, multi_iters=False, join_str=" ", batch_size=10000, n_threads=2):
        super(SpacyPipeProcessor, self).__init__(nlp, join_str, batch_size, n_threads)
        self.multi_iters = multi_iters
    
    def __call__(self, raw_documents):
        docs_generator = self.nlp.pipe(raw_documents, batch_size=self.batch_size, n_threads=self.n_threads)
        return docs_generator if self.multi_iters == False else list(docs_generator)
    
class SpacyTokenizer(SpacyPipeInitializer):
    def __init__(self, nlp, join_str=" ", ignore_chars='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~', batch_size=10000, n_threads=2):
        super(SpacyTokenizer, self).__init__(nlp, join_str, batch_size, n_threads)
        self.ignore_chars = ignore_chars
        self.translate_table = dict((ord(char), None) for char in self.ignore_chars)
        
    def tokenize_from_docs(self, docs):
        for doc in docs:
            tokens_gen = (token.text.translate(self.translate_table) for token in doc)  # generator expression
            yield self.join_str.join(tokens_gen) if self.join_str is not None else [token for token in tokens_gen]
    
    def __call__(self, raw_documents):
        docs_generator = self.nlp.pipe(raw_documents, batch_size=self.batch_size, n_threads=self.n_threads)
        return self.tokenize_from_docs(docs_generator)

class SpacyLemmatizer(SpacyPipeInitializer): # PRON
    def __init__(self, nlp, join_str=" ", ignore_chars='!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~', use_pron=False, batch_size=10000, n_threads=2):
        super(SpacyLemmatizer, self).__init__(nlp, join_str, batch_size, n_threads)
        self.use_pron = use_pron
        self.ignore_chars = ignore_chars
        self.translate_table = dict((ord(char), None) for char in self.ignore_chars)
        
    def lemmatize_from_docs(self, docs):
        for doc in docs:
            lemmas_gen = (token.lemma_.translate(self.translate_table) if self.use_pron or token.lemma_!='-PRON-' else token.lower_.translate(self.translate_table) for token in doc)  # generator expression
            yield self.join_str.join(lemmas_gen) if self.join_str is not None else [lemma for lemma in lemmas_gen]
    
    def __call__(self, raw_documents):
        docs_generator = self.nlp.pipe(raw_documents, batch_size=self.batch_size, n_threads=self.n_threads)
        return self.lemmatize_from_docs(docs_generator)

class SpacyTokenCountVectorizer(CountVectorizer):
    
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)[^\r\n ]+",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64, 
                 nlp=None, join_str=' ', ignore_chars='!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'):
        
        super().__init__(input, encoding, decode_error, strip_accents, 
                                                   lowercase, preprocessor, tokenizer,
                                                   stop_words, token_pattern, ngram_range, 
                                                   analyzer, max_df, min_df, max_features,
                                                   vocabulary, binary, dtype)

        self.join_str = ' ' # tokens have to be joined for splitting
        self.ignore_chars = ignore_chars
        self.translate_table = dict((ord(char), None) for char in self.ignore_chars)
        
    def tokenize_from_docs(self, docs):
        for doc in docs:
            tokens_gen = (token.text.translate(self.translate_table) for token in doc)  # generator expression
            yield self.join_str.join(tokens_gen) if self.join_str is not None else [token for token in tokens_gen]
    
    def build_tokenizer(self):
        return lambda doc: doc.split()

    def fit_transform(self, spacy_docs, y=None):
        raw_documents = self.tokenize_from_docs(spacy_docs)
        return super(SpacyTokenCountVectorizer, self).fit_transform(raw_documents, y)

    def transform(self, spacy_docs):
        raw_documents = self.tokenize_from_docs(spacy_docs)
        return super(SpacyTokenCountVectorizer, self).transform(raw_documents)
    
class SpacyLemmaCountVectorizer(CountVectorizer):
    
    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)[^\r\n ]+",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64, 
                 nlp=None, ignore_chars='!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~', 
                 join_str=" ", use_pron=False):
        
        super().__init__(input, encoding, decode_error, strip_accents, 
                                                   lowercase, preprocessor, tokenizer,
                                                   stop_words, token_pattern, ngram_range, 
                                                   analyzer, max_df, min_df, max_features,
                                                   vocabulary, binary, dtype)
        self.ignore_chars = ignore_chars
        self.join_str = ' ' # lemmas have to be joined for splitting
        self.use_pron = use_pron
        self.translate_table = dict((ord(char), None) for char in self.ignore_chars)
        
    def lemmatize_from_docs(self, docs):
        for doc in docs:
            lemmas_gen = (token.lemma_.translate(self.translate_table) if self.use_pron or token.lemma_!='-PRON-' else token.lower_.translate(self.translate_table) for token in doc)  # generator expression
            yield self.join_str.join(lemmas_gen) if self.join_str is not None else [lemma for lemma in lemmas_gen]
    
    def build_tokenizer(self):
        return lambda doc: doc.split()
    
    def transform(self, spacy_docs):
        raw_documents = self.lemmatize_from_docs(spacy_docs)
        return super(SpacyLemmaCountVectorizer, self).transform(raw_documents)
    
    def fit_transform(self, spacy_docs, y=None):
        raw_documents = self.lemmatize_from_docs(spacy_docs)
        return super(SpacyLemmaCountVectorizer, self).fit_transform(raw_documents, y)
    
class SpacyWord2VecVectorizer(BaseEstimator, VectorizerMixin):
    
    def __init__(self, sparsify=True):
        self.sparsify = sparsify

    def fit(self, spacy_docs, y=None):
        return self
    
    def fit_transform(self, spacy_docs, y=None):
        # TODO this method is not thread safe!
        X = np.array([doc.vector for doc in spacy_docs])
        return csr_matrix(X) if self.sparsify else X
    
    def transform(self, spacy_docs):
        return self.fit_transform(spacy_docs, None)
    