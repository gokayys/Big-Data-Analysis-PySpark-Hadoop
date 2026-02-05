from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when, count, avg, sum

spark = SparkSession.builder\
	.appName("LojistikRegresyon_Hadoop_Spark_Pipeline_KararAgaci")\
	.getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv(
	"file:///home/hadoopuser/customer.csv",
	header=True,
	inferSchema=True
)
df = df.drop("category_index", "gender_index", "payment_index", "shopping_mall_index")


indexer_gender = StringIndexer(inputCol="gender",
outputCol="gender_index")

indexer_category = StringIndexer(inputCol="category",
outputCol="category_index")

indexer_payment = StringIndexer(inputCol="payment_method",
outputCol="payment_index")

indexer_mall = StringIndexer(inputCol="shopping_mall",
outputCol="shopping_index")

feature_cols=["age", "quantity", "price", "gender_index", "shopping_index", "payment_index"]

assembler=VectorAssembler(
	inputCols=feature_cols,
	outputCol="features_raw"
)

scaler = StandardScaler(
	inputCol="features_raw",
	outputCol="features",
	withMean=True,
	withStd=True
)


pipeline= Pipeline(stages=[indexer_gender, indexer_category, indexer_payment, indexer_mall, assembler, scaler])

train_df,test_df=df.randomSplit([0.8, 0.2], seed=42)

prep_model=pipeline.fit(train_df)

train_prepped=prep_model.transform(train_df)

test_prepped=prep_model.transform(test_df)




lr = LogisticRegression(
featuresCol="features",
labelCol="category_index", maxIter=150, elasticNetParam=0.5, regParam=0.001)
lr_model=lr.fit(train_prepped)
lr_predictions=lr_model.transform(test_prepped)



dt= DecisionTreeClassifier(
featuresCol="features", maxDepth=6, maxBins=32,
labelCol="category_index")
dt_model = dt.fit(train_prepped)
dt_predictions=dt_model.transform(test_prepped)

acc_evaluator = MulticlassClassificationEvaluator(labelCol="category_index", metricName="accuracy")


lr_acc=acc_evaluator.evaluate(lr_predictions)
dt_acc=acc_evaluator.evaluate(dt_predictions)


labels = prep_model.stages[1].labels

print("\n" + "#" * 60)
print("###   MÜŞTERİ DAVRANIŞI ANALİZ RAPORU   ###")
print("###   Hedef: Ürün Kategorisi Tahmini   ###")
print("#" * 60)
print(f"Lojistik Regresyon Doğruluğu: % {lr_acc * 100:.2f}")
print(f"Karar Ağacı Doğruluğu: % {dt_acc * 100:.2f}")
print("-" * 60)



evaluator = MulticlassClassificationEvaluator(labelCol="category_index", predictionCol="prediction")
f1_dt_val = evaluator.setMetricName("f1"). evaluate(dt_predictions) 
precision_dt_val = evaluator.setMetricName("weightedPrecision").evaluate(dt_predictions)
recall_dt_val = evaluator.setMetricName("weightedRecall").evaluate(dt_predictions)

evaluator = MulticlassClassificationEvaluator(labelCol="category_index", predictionCol="prediction")
f1_lr_val = evaluator.setMetricName("f1"). evaluate(lr_predictions) 
precision_lr_val = evaluator.setMetricName("weightedPrecision").evaluate(lr_predictions)
recall_lr_val = evaluator.setMetricName("weightedRecall").evaluate(lr_predictions)

print("DECİSİON TREE DETAYLI METRİKLER (F1, PRECİSİON, RECALL):")
print(f"F1 Score:  {f1_dt_val:.4f}")
print(f"Precision:  {precision_dt_val:.4f}")
print(f"Recall:  {recall_dt_val:.4f}")
print("-" * 60)

print("LOJİSTİK REGRESYON DETAYLI METRİKLER (F1, PRECİSİON, RECALL):")
print(f"F1 Score:  {f1_lr_val:.4f}")
print(f"Precision:  {precision_lr_val:.4f}")
print(f"Recall:  {recall_lr_val:.4f}")
print("-" * 60)


print("Kategorilerin hangi sayısal değere karşılık geldiği:")
print("-" * 60)
labels = prep_model.stages[1].labels
for i, name in enumerate(labels):
	print(f"{float(i)} = {name}")
print("-" * 60)


print("KARAR AĞACI ÖRNEK TAHMİNLERİ:")
print("-" * 60)
dt_predictions.select("age", "gender", "price", "quantity", "category", "prediction").show(10)
print("#" * 60 + "\n")
print("\n" + "="*60)
print("Tablo Açıklaması:")
print("="*60)
print("1. PRICE (Fiyat): Müşterinin harcadığı para.")
print("2. CATEGORY (Gerçek): Harcanan para ile alınan ürünün kategorisi.")
print("3. PREDICTION (Model): Modelin sayısal tahmini.")
print("-" * 60)
print("Category ile Prediction birbirini tutuyorsa model tahmini doğrudur.")
print("-" * 60)

print("--- KARMAŞIKLIK MATRİSİ ---")
dt_predictions.stat.crosstab("category", "prediction").show()


tekrar_edenler = df.groupBy("customer_id").count().filter("count > 1")

toplam_tekrar = tekrar_edenler.count()

print(f"\n--- MÜŞTERİ ANALİZ RAPORU ---")
if toplam_tekrar > 0:
	print(f"EFSANE HABER: Veri setinde {toplam_tekrar} tane tekrar alışveriş yapan sadık müşteri var!")
	print("Örnek Sadık Müşteriler:")
	tekrar_edenler.orderBy("count", ascending=False).show(5)
else:
	print("SONUÇ: Her müşteri sadece 1 kez alışveriş yapmış. (Anlık Bağımsız Fatura Bilgisi)")



from pyspark.sql import Row

test_data = [ Row(customer_id="TEST_01", gender="Male", age=25, category="Clothing", quantity=1.0, price=200.0, payment_method="Cash", shopping_mall="Kanyon"),
	Row(customer_id="TEST_01", gender="Male", age=25, category="Food & Beverage", quantity=2.0, price=50.0, payment_method="Cash", shopping_mall="Kanyon")]
	
sanal_df = spark.createDataFrame(test_data)

sanal_prepped = prep_model.transform(sanal_df)
	
sanal_tahminler = lr_model.transform(sanal_prepped)
	
print("\n" + "="*70)
	
sanal_tahminler.select(
	"customer_id",
	"category",
	"quantity",
	"price",
	"prediction").show()


print("\n" + "="*70)
print("CİNSİYET YÖNELİMİ: HANGİ KATEGORİDE KİM DAHA BASKIN?")
print("="*70)

yonelim_analizi = df.groupBy("category").pivot("gender").count().fillna(0)

yonelim_analizi.show()

print("="*70)


print("\n" + "="*70)
print("AVM BAZLI KATEGORİ YOĞUNLUĞU")
print("="*70)

mall_pivot = df.groupBy("shopping_mall").pivot("category").count().fillna(0)
mall_pivot.show()

print("\n" + "="*70)
print("STRATEJİK BAĞLANTI: KATEGORİ VE ÖDEME YÖNTEMİ ANALİZİ")
print("="*70)

payment_pivot = df.groupBy("category").pivot("payment_method").count().fillna(0)
payment_pivot.show()


from pyspark.sql.functions import floor

df_with_age_bins = df.withColumn("Age_Group", (floor(df["age"] / 10) * 10))
age_analysis = df_with_age_bins.groupBy("Age_Group").pivot("category").count().fillna(0).orderBy("Age_Group")
print("YAŞ GRUPLARININ KATEGORİ TERCİHLERİ")
age_analysis.show()



df_risk = train_prepped.withColumn("risk_label", when((col("price")>800) & (col("quantity") > 2) & (col("age") > 35), 1).otherwise(0))

feature_cols_risk = ["age", "gender_index", "shopping_index", "category_index"]
assembler_risk = VectorAssembler(inputCols=feature_cols_risk, outputCol="risk_features")

data_risk = assembler_risk.transform(df_risk) 
train_r, test_r = data_risk.randomSplit([0.8, 0.2], seed=42)

dt_risk = DecisionTreeClassifier(featuresCol="risk_features", labelCol="risk_label", maxDepth=10, maxBins=64)
model_risk = dt_risk.fit(train_r)
risk_predictions = model_risk.transform(test_r)


print("\n" + "="*50)
print(">>> SATIŞ KAYBI VE İADE RİSK TAHMİNİ")
print("="*50)
risk_predictions.select("age", "shopping_mall", "category", "price","risk_label", "prediction").show(20)

evaluator_r = MulticlassClassificationEvaluator(labelCol="risk_label", predictionCol="prediction", metricName="accuracy")
print(f"RİSK ANALİZİ MODELİ DOĞRULUĞU: % {evaluator_r.evaluate(risk_predictions)*100:.2f}")
