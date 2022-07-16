from pyspark.sql import DataFrame

# {{{ Column classification
IDENTIFIER = ['MaTin']

CONTINUOUS_COLUMNS = [
    'TongGia',
    'Gia/m2',
    'DienTich',
    'DienTichDat',
    'ChieuDai',
    'ChieuRong',
    'ChieuSau',
    'DuongVao',
    'NamXayDung'
]

STRUCTURED_COLUMNS = [
    'TienIchToaNha',
    'TienIchLanCan',
    'NoiThat,TienNghi',
    'HangXom',
    'AnNinh',
    'DuongVaoBds',
    'TienIchGanDat'
]

CATEGORICAL_COLUMNS = [
    'Id_NguoiDangban',
    'LoaiBDS',
    'Tinh',
    'Xa',
    'Huyen',
    'Huong',
    'PhongNgu',
    'PhongTam',
    'GiayToPhapLy',
    'SoTang',
    'ViTri',
    'HienTrangNha',
    'TrangThaiSuDung',
    'HuongBanCong',
    'KetCau'
]
# }}}

# Count null value
def countNull(df: DataFrame) -> DataFrame:
    import plotly.express as px
    from pyspark.sql.functions import col, when, count

    df_count = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
    pdfDF = df_count.toPandas().T
    pdfDF.columns = ['No. null value']
    px.bar(pdfDF, text_auto=True, title="Count number of null value each features").show()
    
    return df_count

# Fill, replace empty value with None/null
def fillEmptyValue(df: DataFrame) -> DataFrame:
    from pyspark.sql.functions import col, when

    for c in df.columns:
        if c not in STRUCTURED_COLUMNS:
            df = df.withColumn(c, when(col(c)=='--', None).otherwise(col(c)).alias(c))\
                .withColumn(c, when(col(c)=='', None).otherwise(col(c)).alias(c))

    return df

# {{{ Remove unit, convert string to castable
def rmU_TongGia(TongGia):
    if TongGia is not None:
        gia = TongGia.split(' ')
        if gia[1] == 'tỷ':
            return float(gia[0])*1000.0
        elif gia[1] == 'triệu':
            return float(gia[0])
        elif gia[1] == 'nghìn':
            return round(float(gia[0])*0.001, 3)
        else:
            return TongGia
    else: 
        return None

def rmU_GiaM2(GiaM2):
    if GiaM2 is not None:
        gia = GiaM2.split(' ')
        if gia[1] == 'tỷ/m²':
            return float(gia[0])*1000.0
        elif gia[1] == 'triệu/m²':
            return float(gia[0])
        elif gia[1] == 'nghìn/m²':
            return round(float(gia[0])*0.001, 3)
    else:
        return None

def removeUnitString(df: DataFrame) -> DataFrame:
    from pyspark.sql.functions import udf

    udf_TongGia= udf(rmU_TongGia)
    udf_GiaM2= udf(rmU_GiaM2)
    udf_DienTich = udf(lambda S: S[:-3] if S is not None else None)
    udf_Dist = udf(lambda D: D[:-2] if D is not None else None)

    for col in df.columns:
        if col == 'TongGia':
            df = df.withColumn(col, udf_TongGia(col))
        if col == 'Gia/m2':
            df = df.withColumn(col, udf_GiaM2(col))
        if col in ['DienTich','DienTichDat']:
            df = df.withColumn(col, udf_DienTich(col))
        if col in ['ChieuDai','ChieuRong','ChieuSau','DuongVao']:
            df = df.withColumn(col, udf_Dist(col))

    return df
# }}}

# {{{ Missing value imputed
def fillNullValue(df: DataFrame) -> DataFrame:
    from pyspark.sql.functions import coalesce, col, avg, array

    if ('DienTich' in df.columns and 'DienTichDat' in df.columns):
        df = df.withColumn('DienTich', coalesce('DienTich', 'DienTichDat'))

    for column in df.columns: 

        if column in ['NamXayDung']:
            mode = df.select(column).where(col(column).isNotNull())\
                .groupby(column).count()\
                .orderBy("count", ascending=False).first()[0]
            df = df.fillna(value=mode, subset=[column])

        if column in ['ChieuDai', 'ChieuRong', 'DuongVao','DienTich']:
            mean = df.agg(avg(column)).first()[0]
            df = df.fillna(value=str(mean), subset=[column])

        if column in ['PhongNgu', 'PhongTam']:
            df = df.fillna(value='0', subset=[column])

        if column in ['SoTang']:
            df = df.fillna(value='1', subset=[column])

        if column in CATEGORICAL_COLUMNS:
            df = df.fillna(value='Unknowns', subset=[column])

        if column in STRUCTURED_COLUMNS:
            df = df.withColumn(column, coalesce(column, array()))


    return df
# }}}

# {{{ Removing useless records, rare features and type casting
def dropData(df: DataFrame, isTrain=True) -> DataFrame:
    if isTrain:
        df = df.dropna(subset=['MaTin','NgayDangBan','TongGia'], how='any')
    else:
        df = df.dropna(subset=['MaTin','NgayDangBan'], how='any')

    df = df.drop(*['DienTichDat','ChieuSau'])
    return df

def typeCasting(df: DataFrame) -> DataFrame:
    from pyspark.sql.functions import col
    
    int_columns = ['NamXayDung']
    float_columns = [
        c for c in CONTINUOUS_COLUMNS
        if c not in int_columns
        and c not in ['ChieuSau','DienTichDat']]

    for column in int_columns:
        df = df.withColumn(column, col(column).cast('int'))
    
    for column in float_columns:
        df = df.withColumn(column, col(column).cast('float'))
    return df
# }}}

def cleanRawData(df: DataFrame, isTrain=True) -> DataFrame:
    df1 = fillEmptyValue(df)
    df2 = removeUnitString(df1)
    df3 = fillNullValue(df2)
    df4 = dropData(df3, isTrain)
    df5 = typeCasting(df4)
    return df5

'''
def getDummy(df: DataFrame, keepInput=False, keepOutput=False, vectorize=True, outputCol='features_idx') -> Tuple[DataFrame, PipelineModel]:
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, VectorAssembler

    idx_columns = [c for c in CATEGORICAL_COLUMNS if not c in ['Tinh','Huyen','Xa']]
    indexers = [ StringIndexer(inputCol=c, outputCol="{0}_idx".format(c))
                 for c in df.columns if c in idx_columns]

    pipeline1 = Pipeline(stages=indexers)
    models_stringIndex = pipeline1.fit(df)
    data = models_stringIndex.transform(df)

    if keepInput:
        pass
    else:
        data = data.drop(*idx_columns)

    if vectorize:
        assembler = VectorAssembler(
            inputCols=[indexer.getOutputCol() for indexer in indexers],
            outputCol=outputCol)
        data = assembler.transform(df)

        if keepOutput:
            pass
        else:
            data = data.drop(*['{0}_idx'.format(c) for c in idx_columns])
    else:
        pass
    return data, models_stringIndex

def getAdministrative(df: DataFrame, vectorize=True, outputCol='features_adm') -> DataFrame:
    from pyspark.sql.functions import col, when
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml import Pipeline
    from utils import _initialize_spark

    spark, _ = _initialize_spark()

    import pandas as pd
    provinces_tier = pd.read_csv('data/provinces_tier.csv')
    provinces_tier = spark.createDataFrame(provinces_tier)
    df = df.join(provinces_tier, ['Tinh'], how='left')

    data = df\
        .withColumn('Tinh_idx',
                    when(col('PhanLoaiTinh').contains('Đặc biệt'), 4)\
                    .when(col('PhanLoaiTinh').contains('III'), 3)\
                    .when(col('PhanLoaiTinh').contains('II'), 2)\
                    .when(col('PhanLoaiTinh').contains('I'), 1))\
        .withColumn('Huyen_idx',
                    when(col('Huyen').contains('TP.'),3)\
                    .when(col('Huyen').contains('Quận'),3)\
                    .when(col('Huyen').contains('Thị xã'),2)\
                    .when(col('Huyen').contains('Huyện'),1))\
        .withColumn('Xa_idx',
                    when(col('Xa').contains('Phường'), 3)\
                    .when(col('Xa').contains('Thị trấn'), 2)\
                    .when(col('Xa').contains('Xã'), 1))
    
    if vectorize:
        assembler = VectorAssembler(
            inputCols=['Tinh_idx','Huyen_idx','Xa_idx'],
            outputCol=outputCol)
        data = assembler.transform(data)
    else:
        pass

    return data.drop(*['Tinh','PhanLoaiTinh','Huyen','Xa'])

class OHE():
    
    Sử dụng One Hot Encoder để transfom dữ liệu từ Array<value> -> Vector[value].
    * categories : {'auto', array<>}, default='auto'.
        - Tự động xác định danh sách giá trị phân loại hoặc sử dungk danh sách giá trị
        phân loại được cấp từ array 
    * dropInput : bool, defauly=False: Bỏ cột thuộc tính đầu vào)
    
    def __init__(self, categories='auto', dropInput=False):
        self.categories = categories
        self.dropInput = dropInput

    def transform(self, df, feature):
        from pyspark.sql.functions import monotonically_increasing_id, regexp_replace, explode_outer, col, lit

        self.feature = feature
        df = df.withColumn("_id", monotonically_increasing_id() )

        if self.categories == 'auto':
            _idPrefix = '{0}_{1}'.format('_id', feature)

            explode_df = df\
                .withColumn(self.feature, explode_outer(self.feature))\
                .withColumn(feature, regexp_replace(feature, r'\.',''))
            crosstab_df = explode_df.crosstab('_id', self.feature).drop('null')

            cats_df = crosstab_df.drop(_idPrefix)
            cats = cats_df.columns

            if 'Other' in cats:
                self.categories = cats
            else:
                self.categories = cats + ['Other']
                cats_df = cats_df.withColumn('Other', lit(0))

            self.categories.sort()
            cats_df = cats_df.select(self.categories)

            categories_order = [col(col_name).alias("{0}_".format(feature) + col_name) 
                                      for col_name in cats_df.columns]
            cats_df = cats_df.select(*categories_order)

            self.categories_prefix = cats_df.columns

            cats_df = cats_df.withColumn("_id", monotonically_increasing_id() )
            df = df.join(cats_df,'_id').drop('_id')

        else:
            _idPrefix = '{0}_{1}'.format('_id', feature)

            explode_df = df\
                .withColumn(self.feature, explode_outer(self.feature))\
                .withColumn(feature, regexp_replace(feature, r'\.',''))
            crosstab_df = explode_df.crosstab('_id', self.feature).drop('null')

            cats_df = crosstab_df.drop(_idPrefix)

            new_cats = [c for c in cats_df.columns if c not in self.categories]
            miss_cats = [c for c in self.categories if c not in cats_df.columns]
 
            cats = cats_df.columns

            if 'Other' in cats:
                cats_df = cats_df.withColumn('Other', sum([col('Other')] + [col(cate) for cate in new_cats])).drop(*new_cats)
            else:
                miss_cats.remove('Other')
                cats_df = cats_df.withColumn('Other', sum([col(cate)] for cate in new_cats)).drop(*new_cats)

            for c in miss_cats:
                cats_df = cats_df.withColumn(c, lit(0))

            self.categories.sort()
            cats_df = cats_df.select(self.categories)

            categories_order = [col(col_name).alias("{0}_".format(feature) + col_name) 
                                      for col_name in cats_df.columns]
            cats_df = cats_df.select(*categories_order)
            self.categories_prefix = cats_df.columns

            cats_df = cats_df.withColumn("_id", monotonically_increasing_id() )
            df = df.join(cats_df,'_id').drop('_id')

        if self.dropInput:
            return df.drop(feature)
        else:
            return df

def featureExtraction(df: DataFrame) -> DataFrame:
    df1 = getAdministrative(df)
    df2, _ = getDummy(df1, keepInput=False)
    return df

def OHEtransform(df: DataFrame, keepInput=False, keepOutput=False, vectorize=True) -> Tuple[DataFrame, List[PipelineModel]]:
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import StringIndexer, VectorAssembler

    df_trans = df
    ohe_models = []
    
    for feature in df.columns:
        if feature in STRUCTURED_COLUMNS:
            encoder = OHE()
            df_trans = encoder.transform(df_trans, feature)
            ohe_models.append(encoder)

            if vectorize:
                vec_name_prefix = '{0}_ohe'.format(feature)
                assembler = VectorAssembler(inputCols = encoder.categories_prefix,
                                            outputCol = vec_name_prefix)
                df_trans = assembler.transform(df_trans)

                if keepOutput:
                    pass
                else:
                    df_trans = df_trans.drop(*encoder.categories_prefix)
            else:
                pass

            if keepInput:
                pass
            else:
                df_trans = df_trans.drop(feature)

    return df_trans, ohe_models'''

if __name__ == '__main__':
    try:
        df
    except NameError:
        df = spark.read.option("header",True)\
            .option("multiline",True)\
            .json("./data/raw/raw_0.json")
    else:
        df_clean = clean_raw_data(df)
        df_featurize = feature_extraction(df_clean)