from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import norm, eigh
import plotly_express as px
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
from IPython.display import display_html 
import seaborn as sns
import matplotlib.pyplot as plt
from gensim.models import word2vec
from sklearn.manifold import TSNE

def get_links(url):
    # Connect 
    driver = webdriver.Chrome()

    # Clear cache
    driver.delete_all_cookies()

    # Full screen
    driver.maximize_window()

    # Wait for it to load
    time.sleep(5)

    # Open website 
    driver.get(url)

    # Wait for it to load
    time.sleep(5)

    # Pull all "VIEW HTML" related objects
    objects = driver.find_elements(By.LINK_TEXT, 'VIEW HTML')

    # List to hold the links:
    links = []

    # Iterate over list and get the links:
    for obj in objects:
        links.append(obj.get_attribute('href'))
    
    # Close the driver 
    driver.close()
    
    # Return links
    return links 

def get_constitution(url, useragent):
    # Define OHCO
    OHCO = ['article_num', 'para_num', 'sent_num', 'token_num']
    
    # Make request 
    r = requests.get(url, headers = {'User-Agent': useragent})
    soup = BeautifulSoup(r.text, "html.parser")
    
    # Get HTML data
    soup = BeautifulSoup(r.text, "html.parser")
    text = []
    for x in soup.find_all(['h2', 'h3', 'li', 'p']):  
        # h2 = preamble, h3 = sections, li = bullet points, p = main content
        if x.string != None:                          # Get text from tag and remove whitespaces
            text.append(x.string)
            
    # Get all lines
    LINES = pd.DataFrame(text, columns=['line_str'])
    LINES.index.name = 'line_num'
    
    # Chunk by article
    headers = [x.text for x in soup.find_all(['h2', 'h3'])]
    article_lines = LINES[LINES['line_str'].isin(headers)].index.to_list()
    
    # Assign numbers to articles
    LINES.loc[article_lines, 'article_num'] = [i+1 for i in range(LINES.loc[article_lines].shape[0])]
    
    # Forward-fill article numbers to following text lines
    LINES.article_num = LINES.article_num.ffill()
    
    # Clean up 
    LINES = LINES.dropna(subset=['article_num']) 
    LINES = LINES.loc[~LINES.index.isin(article_lines)]
    LINES.article_num = LINES.article_num.astype('int') 
    
    # Group lines into articles
    # Make big string for each article
    ARTICLES = LINES.groupby(OHCO[:1])\
                 .line_str.apply(lambda x: '\n'.join(x))\
                 .to_frame('article_str') 

    ARTICLES['article_str'] = ARTICLES.article_str.str.strip()
    
    # Split articles into paragraphs
    para_pat = r'\n\n+'

    # Split into paragraphs
    PARAS = ARTICLES['article_str'].str.split(para_pat, expand=True).stack()\
                             .to_frame('para_str').sort_index()
    PARAS.index.names = OHCO[:2]

    # Remove empty paragraphs
    PARAS['para_str'] = PARAS['para_str'].str.replace(r'\n', ' ', regex=True)
    PARAS['para_str'] = PARAS['para_str'].str.strip()
    PARAS = PARAS[~PARAS['para_str'].str.match(r'^\s*$')]
    
    # Split paragraphs into sentences 
    sent_pat = r'[.?!;:]+'
    SENTS = PARAS['para_str'].str.split(sent_pat, expand=True).stack()\
                             .to_frame('sent_str')
    SENTS.index.names = OHCO[:3]

    SENTS = SENTS[~SENTS['sent_str'].str.match(r'^\s*$')] # Remove empty paragraphs
    SENTS.sent_str = SENTS.sent_str.str.strip()           # CRUCIAL TO REMOVE BLANK TOKENS
    
    # Split sentences into tokens
    token_pat = r"[\s',-]+"
    TOKENS = SENTS['sent_str'].str.split(token_pat, expand=True).stack()\
                              .to_frame('token_str')
    TOKENS.index.names = OHCO[:4]
    
    # Convert to terms
    TOKENS['term_str'] = TOKENS.token_str.replace(r'[\W_]+', '', regex=True).str.lower()
    
    # Add country title as "book"
    title = [x.text for x in soup.find_all(['title'])][0]
    country = re.findall('([a-zA-Z ()]*)\d*.*', title)[0][:-1]
    TOKENS.insert(0, 'country', country)
    TOKENS = TOKENS.reset_index().set_index(['country'] + OHCO)
    
    # Output tokens 
    return TOKENS

def get_bow(CORPUS, bag = ['country'], item_type='term_str'):
    BOW = CORPUS.groupby(bag+[item_type])[item_type].count().to_frame('n')
    return BOW

def get_idf(BOW, tf_method='max'): 
    ### Term frequency ###
    DTCM = BOW.n.unstack().fillna(0).astype('int')  # Document-Term Count Matrix 
    
    if tf_method == 'sum':
        TF = DTCM.T / DTCM.T.sum()
    elif tf_method == 'max':
        TF = DTCM.T / DTCM.T.max()
    TF = TF.T
    
    ### IDF (standard) ###
    DF = DTCM.astype('bool').sum()
    N = DTCM.shape[0]
    IDF = np.log2(N / DF)

    # Compute TFIDF and DFIDF
    TFIDF = TF * IDF
    DFIDF = DF * IDF
    return TFIDF, DFIDF

def get_PCA(X, k, norm_docs=True, center_by_mean=False, center_by_variance=False):
    # Normalize doc vector lengths with L2 normalization
    if norm_docs==True:
        X = (X.T / norm(X, 2, axis=1)).T  
    
    # Center the term vectors by mean
    if center_by_mean==True:
        X = X - X.mean()  
        
    # Center the term vectors by variance
    if center_by_variance==True:
        X = X / X.std() 
    
    # Compute covariance matrix 
    COV = X.cov()
    
    # Decompose the matrix 
    eig_vals, eig_vecs = eigh(COV)
    
    # Convert eigen data to dataframes
    EIG_VEC = pd.DataFrame(eig_vecs, index=COV.index, columns=COV.index)
    EIG_VAL = pd.DataFrame(eig_vals, index=COV.index, columns=['eig_val'])
    EIG_VAL.index.name = 'term_str'
    
    # Combine eigenvalues and eigenvectors
    EIG_PAIRS = EIG_VAL.join(EIG_VEC.T)
    
    # Compute explained variance
    EIG_PAIRS['exp_var'] = np.round((EIG_PAIRS.eig_val / EIG_PAIRS.eig_val.sum()) * 100, 2)
    
    # Pick top 'k' components
    COMPS = EIG_PAIRS.sort_values('exp_var', ascending=False).head(k).reset_index(drop=True)
    COMPS.index.name = 'comp_id'
    COMPS.index = ["PC{}".format(i) for i in COMPS.index.tolist()]
    COMPS.index.name = 'pc_id'
    
    # Term-component matrix (LOADINGS)
    LOADINGS = COMPS[COV.index].T
    LOADINGS.index.name = 'term_str'

    # Document-component matrix (DCM)
    DCM = X.dot(LOADINGS) 

    # Component information table (COMPINF)
    top_terms = []
    for i in range(k):
        for j in [0, 1]:
            comp_str = ' '.join(LOADINGS.sort_values(f'PC{i}', ascending=bool(j)).head(5).index.to_list())
            top_terms.append((f"PC{i}", j, comp_str))
    COMPINF = pd.DataFrame(top_terms).set_index([0,1]).unstack()
    COMPINF.index.name = 'comp_id'
    COMPINF.columns = COMPINF.columns.droplevel(0) 
    COMPINF = COMPINF.rename(columns={0:'pos', 1:'neg'})
    COMPINF = COMPS.join(COMPINF)[['pos', 'neg', 'eig_val', 'exp_var']]
    
    return LOADINGS, DCM, COMPINF

class TopicModel: 
    def __init__(self, CORPUS, BAG):
    # Create DOCS table (F1 style corpus with only regular nouns) 
        self.DOCS = CORPUS[CORPUS.pos.str.match(r'^NNS?$')]\
                         .groupby(BAG).term_str\
                         .apply(lambda x: ' '.join(x))\
                         .to_frame()\
                         .rename(columns={'term_str':'doc_str'})
        
    def create_model(self, n_terms, ngram_range, n_topics, max_iter, n_top_terms): 
        # Get Terms
        count_engine = CountVectorizer(max_features = n_terms, ngram_range = ngram_range, stop_words = 'english')
        count_model = count_engine.fit_transform(self.DOCS.doc_str)
        TERMS = count_engine.get_feature_names_out()
        
        # Generate LDA model 
        lda_engine = LDA(n_components = n_topics, max_iter = max_iter, learning_offset = 50., random_state = 0)
        
        # Topic names 
        self.TNAMES = [f"T{str(x).zfill(len(str(n_topics)))}" for x in range(n_topics)]
        
        # Obtain THETA table
        lda_model = lda_engine.fit_transform(count_model)
        self.THETA = pd.DataFrame(lda_model, index = self.DOCS.index)
        self.THETA.columns.name = 'topic_id'
        self.THETA.columns = self.TNAMES
        
        # Obtain PHI table
        self.PHI = pd.DataFrame(lda_engine.components_, columns=TERMS, index=self.TNAMES)
        self.PHI.index.name = 'topic_id'
        self.PHI.columns.name  = 'term_str'
        
        # Obtain TOPICS
        self.TOPICS = self.PHI.stack().to_frame('topic_weight').groupby('topic_id')\
                          .apply(lambda x: x.sort_values('topic_weight', ascending=False)\
                          .head(n_top_terms).reset_index().drop('topic_id', axis=1)['term_str'])
        self.TOPICS['label'] = self.TOPICS.apply(lambda x: x.name + ' ' + ', '.join(x[:n_top_terms]), 1)
        
class WordEmbed: 
    def __init__(self, TOKENS, BAG):
        #######################################################################################################
        ### Convert TOKENS into Gensim corpora for each author ###  
        
        # Filter by only nouns and verbs
        TOKENS = TOKENS[TOKENS.pos.isin(['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])]
        
        # Create docs
        self.DOCS = TOKENS.groupby(BAG)\
                          .term_str.apply(lambda x: x.tolist())\
                          .reset_index()['term_str'].tolist()
        # Lose single word docs
        self.DOCS = [doc for doc in self.DOCS if len(doc) > 1]

        #######################################################################################################
        ### Extract a vocab table and add n and pos_group as features ###
        
        # Handle anomalies
        TOKENS = TOKENS[TOKENS.term_str != '']
        
        # Create vocab
        self.VOCAB = TOKENS.term_str.value_counts().to_frame('n')
        self.VOCAB.index.name = 'term_str'
        self.VOCAB['pos_group'] = TOKENS[['term_str','pos']].value_counts().unstack(fill_value=0).idxmax(1).str[:2]
                         
    #######################################################################################################
    ### Generate a table of word vectors and coordinates ###
    def create_model(self, window, vector_size, min_count):
        self.model = word2vec.Word2Vec(self.DOCS, window = window, vector_size = vector_size, min_count = min_count)
        
    #######################################################################################################
    ### Generate a table of tSNE coordinates ###
    def get_coordinates(self, learning_rate, perplexity, n_components, init, n_iter, random_state):
        
        self.coords = pd.DataFrame(
                    dict(
                        vector = [self.model.wv.get_vector(w) for w in self.model.wv.key_to_index], 
                        term_str = self.model.wv.key_to_index.keys()
                     )).set_index('term_str')
        
        # Use tSNE library 
        tsne_engine = TSNE(learning_rate = learning_rate, 
                           perplexity=perplexity, 
                           n_components=n_components, 
                           init=init, 
                           n_iter=n_iter, 
                           random_state=random_state)
        tsne_model = tsne_engine.fit_transform(np.stack(self.coords.vector.to_list(), axis=0))

        self.coords['x'] = tsne_model[:,0]
        self.coords['y'] = tsne_model[:,1]

        ## Add vocab features
        if self.coords.shape[1] == 3:
            self.coords = self.coords.merge(self.VOCAB.reset_index(), on='term_str')
            self.coords = self.coords.set_index('term_str')
                
    #######################################################################################################
    ### Plot ###     
    def plot(self): 
        plot = px.scatter(self.coords.reset_index(), 'x', 'y', 
                          text='term_str', 
                          color='pos_group', 
                          #hover_name='term_str',
                          size='n', 
                          height=1000).update_traces(mode='markers+text', 
                                                     textfont=dict(color='black', size=10, family='Arial'),
                                                     textposition='top center')   
        #plot.update_layout(yaxis=dict(range=[-13, -37]), xaxis=dict(range=[5, 20]))
        return plot