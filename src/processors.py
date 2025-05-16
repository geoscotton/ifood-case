from  pyspark.sql import functions as F
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path

# current repo path 
REPO_PATH = Path(__file__).parent.parent

class OffersProcessor(BaseEstimator, TransformerMixin):
    """
    Classe que pre-processa os dados de offers com as carateristicas vistas no notebook de EDA.
    Está aqui apenas para placeholder pois não temos muito o que fazer nesta tabela
    """
    
    def __init__(self,):
        pass
        
    def fit(self, df):
        return self
        
    def transform(self, df):
        # nada a fazer
        return df
    
    
class ProfilesProcessor(BaseEstimator, TransformerMixin):
    """
    Classe que pre-processa os dados de profiles com as carateristicas vistas no notebook de EDA.
    """

    def __init__(self):
        pass

    def fit(self, df):
        return self

    def transform(self, df):
        #transformando registered_on em datas
        df = df.withColumn('registered_on', 
                           F.to_date(F.col('registered_on').cast('string'),'yyyyMMdd')
                           )
        #transformando age 118 en nulos e gender Nulo em string
        df = df.withColumn('age', F.when(F.col('age') == 118, None).otherwise(F.col('age')))
        df = df.withColumn('gender', F.when(F.col('gender').isNull(), 'Nulo').otherwise(F.col('gender')))
        return df
    

class TransactionsProcessor(BaseEstimator, TransformerMixin):
    """
    Classe que pre-processa os dados de transactions com as carateristicas vistas no notebook de EDA.
    """

    def __init__(self):
        pass

    def fit(self, df):
        return self

    def transform(self, df):
        #transformando registered_on em datas
        df = (df.withColumn('amount', F.col('value.amount'))
                .withColumn('offer_id', F.coalesce(F.col('value.offer id'), F.col('value.offer_id')))
                .withColumn('reward', F.col('value.reward'))
                .drop('value')
        )
        return df

