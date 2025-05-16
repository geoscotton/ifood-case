from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.pandas import merge_asof
from pyspark.pandas import DataFrame as ps
from pyspark.sql import Window
from pathlib import Path
import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)
import pickle
from src.processors import OffersProcessor, ProfilesProcessor, TransactionsProcessor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# current repo path
REPO_PATH = Path(__file__).parent
# raw data paths
RAW_DATA_PATH = REPO_PATH / "data" / "raw"
PROCESSED_DATA_PATH = REPO_PATH / "data" / "processed"


# carregando spark session
spark = SparkSession.builder.appName("Spark Demo").master("local[*]").getOrCreate()


def preprocessa_input_data():
    logger.info("Começando execução do processo")

    logger.info("Carregando dados brutos")
    offers = spark.read.json((RAW_DATA_PATH / "offers.json").as_posix())
    transactions = spark.read.json((RAW_DATA_PATH / "transactions.json").as_posix())
    profiles = spark.read.json((RAW_DATA_PATH / "profile.json").as_posix())

    logger.info("Processando dados de ofertas")
    offers_processor = OffersProcessor()
    offers_processed = offers_processor.fit_transform(offers)

    logger.info("Processando dados de transações")
    transactions_processor = TransactionsProcessor()
    transactions_processed = transactions_processor.fit_transform(transactions)

    logger.info("Processando dados de perfis")
    profiles_processor = ProfilesProcessor()
    profiles_processed = profiles_processor.fit_transform(profiles)

    logger.info("Salvando dados processados")
    offers_processed.coalesce(1).write.mode("overwrite").json(
        (PROCESSED_DATA_PATH / "offers").as_posix()
    )
    transactions_processed.coalesce(1).write.mode("overwrite").json(
        (PROCESSED_DATA_PATH / "transactions").as_posix()
    )
    profiles_processed.coalesce(1).write.mode("overwrite").json(
        (PROCESSED_DATA_PATH / "profiles").as_posix()
    )


def merge_data():
    logger.info("Carregando dados processados")
    offers = spark.read.json((PROCESSED_DATA_PATH / "offers").as_posix())
    transactions = spark.read.json((PROCESSED_DATA_PATH / "transactions").as_posix())
    profiles = spark.read.json((PROCESSED_DATA_PATH / "profiles").as_posix())

    transactions_full = (
        transactions.join(offers, transactions["offer_id"] == offers["id"], how="left")
        .join(profiles, transactions["account_id"] == profiles["id"], how="left")
        .drop("id")
    )

    transactions_full.coalesce(1).write.mode("overwrite").json(
        (PROCESSED_DATA_PATH / "transactions_full").as_posix()
    )
    logger.info("Salvando dados mergeados")
    return None


def build_dataset():
    transactions_full = spark.read.json(
        (PROCESSED_DATA_PATH / "transactions_full").as_posix()
    )
    transactions = spark.read.json((PROCESSED_DATA_PATH / "transactions").as_posix())

    df = (
        transactions_full.filter('event = "offer received"')
        .select("account_id", "offer_id", "time_since_test_start", "event")
        .distinct()
        .orderBy("time_since_test_start", "account_id")
    )

    target1 = (
        transactions_full.filter('event = "offer completed"')
        .select("account_id", "offer_id", "time_since_test_start")
        .distinct()
        .withColumn("target", F.lit(1))
        .orderBy("time_since_test_start", "account_id")
    )

    dfpd = merge_asof(
        left=ps(df),
        right=ps(target1),
        on="time_since_test_start",
        by=["account_id", "offer_id"],
        direction="forward",
        allow_exact_matches=True,
    )
    dfpd["target"] = dfpd["target"].fillna(0)
    df = dfpd.to_spark()

    informational_offers = (
        spark.read.json((PROCESSED_DATA_PATH / "offers").as_posix())
        .filter('offer_type = "informational"')
        .select("id")
        .distinct()
        .collect()
    )
    informational_offers = [x.id for x in informational_offers]

    window_next = Window.partitionBy("account_id").orderBy("time_since_test_start")

    # pegando time da proxima offer
    next_offers = df.withColumn(
        "next_offer_time", F.lead("time_since_test_start").over(window_next)
    )

    # pegando transações entre uma offer informational e a proxima oferta
    # e vendo se a transação esta entre os dois timestamps (ou nula, ultima oferta)
    informational_success = (
        next_offers.filter(F.col("offer_id").isin(informational_offers))
        .join(
            transactions.filter('event = "transaction"').select(
                "account_id",
                F.col("time_since_test_start").alias(
                    "time_since_test_start_transaction"
                ),
            ),
            on="account_id",
            how="left",
        )
        .filter(
            (
                F.col("time_since_test_start_transaction")
                >= F.col("time_since_test_start")
            )
            & (
                (F.col("time_since_test_start_transaction") < F.col("next_offer_time"))
                | (F.col("next_offer_time").isNull())
            )
        )
        .select("account_id", "offer_id", next_offers.time_since_test_start)
        .distinct()
        .withColumn("target_informational", F.lit(1))
    )

    # atualizando a target do df
    df = (
        df.join(
            informational_success,
            on=["account_id", "offer_id", "time_since_test_start"],
            how="left",
        )
        .withColumn(
            "target",
            F.when(
                F.col("offer_id").isin(informational_offers),
                F.coalesce(F.col("target_informational"), F.lit(0)),
            ).otherwise(F.col("target")),
        )
        .drop("target_informational")
    )

    # Calculando total de ofertas enviadas anterioremente aos clientes
    window = Window.partitionBy("account_id").orderBy("time_since_test_start")

    total_past_offers = (
        transactions.filter('event = "offer received"')
        .withColumn(
            "num_past_offers",
            F.count("offer_id").over(
                window.rangeBetween(Window.unboundedPreceding, -1)
            ),
        )
        .select(
            "account_id",
            "offer_id",
            "event",
            "time_since_test_start",
            "num_past_offers",
        )
    )

    # total de views anteriores da oferta
    total_past_views = transactions.withColumn(
        "num_past_viewed",
        F.sum(F.when(F.col("event") == "offer viewed", 1).otherwise(0)).over(
            window.rangeBetween(Window.unboundedPreceding, -1)
        ),
    ).select(
        "account_id", "offer_id", "time_since_test_start", "event", "num_past_viewed"
    )

    df = df.join(
        total_past_offers,
        on=["account_id", "offer_id", "time_since_test_start", "event"],
        how="left",
    ).join(
        total_past_views,
        on=["account_id", "offer_id", "time_since_test_start", "event"],
        how="left",
    )

    # soma de amounts passados até aquela oferta
    total_past_amount = transactions.withColumn(
        "total_past_amount",
        F.sum(F.col("amount")).over(window.rangeBetween(Window.unboundedPreceding, -1)),
    ).select(
        "account_id", "offer_id", "time_since_test_start", "event", "total_past_amount"
    )
    df = df.join(
        total_past_amount,
        on=["account_id", "offer_id", "time_since_test_start", "event"],
        how="left",
    )

    # soma de rewards passados até aquela oferta
    total_past_reward = transactions.withColumn(
        "total_past_reward",
        F.sum(F.col("reward")).over(window.rangeBetween(Window.unboundedPreceding, -1)),
    ).select(
        "account_id", "offer_id", "time_since_test_start", "event", "total_past_reward"
    )

    df = df.join(
        total_past_reward,
        on=["account_id", "offer_id", "time_since_test_start", "event"],
        how="left",
    )

    # cria feature se ja houve conversão na mesma oferta no passado
    conversoes = df.filter("target = 1").select(
        "account_id",
        "offer_id",
        "time_since_test_start",
        F.lit(1).alias("past_offer_conversion"),
    )

    dfpd2 = merge_asof(
        left=ps(df),
        right=ps(conversoes),
        on="time_since_test_start",
        by=["account_id", "offer_id"],
        direction="backward",
        allow_exact_matches=False,
    )

    df = dfpd2.to_spark()

    # feature que diz quanto tempo passou desde a ultima oferta
    df = df.withColumn(
        "time_since_last_offer",
        F.col("time_since_test_start") - F.lag("time_since_test_start").over(window),
    )

    # adicionando features de ofertas e clientes
    transactions_full_features = transactions_full.select(
        "account_id",  # chaves de cruzamento
        "offer_id",  # chaves de cruzamento
        "event",  # chaves de cruzamento
        "time_since_test_start",  # chaves de cruzamento
        "age",  # features de clientes
        "credit_card_limit",  # features de clientes
        "gender",  # features de clientes
        "registered_on",  # features de clientes
        "discount_value",  # features das offers
        "channels",  # features das offers
        "min_value",  # features das offers
        "offer_type",  # features das offers
        "duration",  # features das offers
    )

    df = df.join(
        transactions_full_features,
        on=["account_id", "offer_id", "event", "time_since_test_start"],
        how="left",
    )

    # Criando features cíclicas do registered_on e outras features de datas
    df = (
        df.withColumn(
            "registered_on_seno",
            F.sin(F.dayofyear("registered_on") * 2 * F.lit(3.14159) / F.lit(365)),
        )
        .withColumn(
            "registered_on_cos",
            F.cos(F.dayofyear("registered_on") * 2 * F.lit(3.14159) / F.lit(365)),
        )
        .withColumn("year_registered", F.year(F.col("registered_on")))
        .withColumn("month_registered", F.month(F.col("registered_on")))
    )

    # criando features de canais
    df = (
        df.withColumn("email", F.array_contains(F.col("channels"), "email"))
        .withColumn("web", F.array_contains(F.col("channels"), "web"))
        .withColumn("mobile", F.array_contains(F.col("channels"), "mobile"))
        .withColumn("social", F.array_contains(F.col("channels"), "social"))
        .withColumn("qtd_canais", F.array_size(F.col("channels")))
    )

    df.write.mode("overwrite").json(
        (PROCESSED_DATA_PATH / "modelling_dataset").as_posix()
    )
    logger.info("Dataset de modelagem salvo com sucesso")
    return None


def train_model():
    logger.info("Carregando dados de modelagem")
    df = spark.read.json(
        (PROCESSED_DATA_PATH / "modelling_dataset").as_posix()
    ).toPandas()

    logger.info("Separando dataset em treino e teste")
    # pegando ultimas offers de cada
    last_offers_idx = (
        df.sort_values(["account_id", "time_since_test_start"])
        .groupby("account_id")
        .tail(1)
        .index
    )

    # criando datasets
    df_test = df.loc[last_offers_idx].copy()
    df_train = df.drop(last_offers_idx).copy()

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_test = df_test.drop(columns=["target"])
    y_test = df_test["target"]

    # Criando transformers que serão usadas no modelo
    simplescaler_features = [
        "discount_value",
        "duration",
        "email",
        "min_value",
        "mobile",
        "qtd_canais",
        "registered_on_cos",
        "registered_on_seno",
        "social",
        "web",
    ]
    simplescaler_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    numeric_median_features = ["age", "credit_card_limit"]
    numeric_median_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    numeric_0_features = [
        "num_past_viewed",
        "past_offer_conversion",
        "total_past_amount",
        "total_past_reward",
    ]
    numeric_0_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )

    numeric_1_features = ["time_since_last_offer"]
    numeric_1_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_features = [
        "gender",
        "offer_type",
        "year_registered",
        "month_registered",
    ]
    categorical_transformer = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", sparse_output=False
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_median", numeric_median_transformer, numeric_median_features),
            ("num_0", numeric_0_transformer, numeric_0_features),
            ("num_1", numeric_1_transformer, numeric_1_features),
            ("cat", categorical_transformer, categorical_features),
            ("simplescaler", simplescaler_transformer, simplescaler_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    logger.info("Treinando modelo HistGradientBoostingClassifier com os hiperparametros já otimizados a seguir:")
    params = {'class_weight': 'balanced', 'l2_regularization': 0.003439284699685042, 'max_depth': 9, 'max_leaf_nodes': 129}
    logger.info(params)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", HistGradientBoostingClassifier(**params)),
             ])
    
    model.fit(X_train, y_train)
    logger.info("Modelo treinado com sucesso")
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_proba = model.predict_proba(X_train)[:,1]
    y_test_proba = model.predict_proba(X_test)[:,1]

    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_roc_auc = roc_auc_score(y_train, y_train_proba)

    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    metrics_df = pd.DataFrame({
        'Train': [train_precision, train_recall, train_f1, train_accuracy, train_roc_auc],
        'Test': [test_precision, test_recall, test_f1, test_accuracy, test_roc_auc]
    }, index=['Precision', 'Recall', 'F1', 'Accuracy', 'ROC AUC'])

    (REPO_PATH / 'models_artefact').mkdir(parents=True, exist_ok=True)
    
    logger.info("\nModel Performance Metrics:")
    logger.info(f"\n{metrics_df.round(3)}")
    metrics_df.to_csv((REPO_PATH / 'models_artefact' / 'metrics.csv').as_posix(), index=True)
    
    logger.info("Salvando modelo")
    

    with open((REPO_PATH / "models_artefact" / "model.pkl").as_posix(), "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Modelo salvo com sucesso em {(REPO_PATH / "models_artefact" / "model.pkl").as_posix()}")


if __name__ == "__main__":
    preprocessa_input_data()
    merge_data()
    build_dataset()
    train_model()
