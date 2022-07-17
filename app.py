import streamlit as st
import numpy as np
import pandas as pd
from utils import _initialize_spark
from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql.functions import udf, col
from pyspark.ml.regression import LinearRegressionModel, RandomForestRegressionModel, GBTRegressionModel, DecisionTreeRegressionModel, IsotonicRegressionModel, FMRegressionModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator
import plotly.express as px


def tranformFetures(X, assembler):
    # Tạo bản sao để tránh ảnh hưởng dữ liệu gốc
    X_ = X.copy()
    ###########################



    ###########################
    st.write("tranform")
    return X_tranform

def prediction(samples, model):
    st.write("predict")
    # Encode dữ liệu
    X = tranformFetures(samples, assembler)
    # Predict
    return model.predict(X)

def model_page(model_name, model):
    option_list = ['Dữ liệu mẩu',
                    'Tự nhập dữ liệu']
    
    choice_input = st.sidebar.selectbox('Cách nhập dữ liệu', option_list)    
    st.subheader(model_name)
    if choice_input == 'Dữ liệu mẩu':
        st.write('#### Sample dataset', pd_df)

        # Chọn dữ liệu từ mẫu
        selected_indices = st.multiselect('Chọn mẫu từ bảng dữ liệu:', pd_df.index)
        selected_rows = pd_df.loc[selected_indices]
        st.write('#### Kết quả')

        if st.button('Dự đoán'):
            if not selected_rows.empty:
                X = selected_rows.iloc[:, :-1]
                pred = prediction(X, model)

                # Xuất ra màn hình
                st.write("predict", pred)
                results = pd.DataFrame({'Giá dự đoán': pred,
                                            'Giá thực tế': selected_rows.TongGia})
                st.write(results)
            else:
                st.error('Hãy chọn dữ liệu trước')

    elif choice_input == 'Tự nhập dữ liệu':
        with st.form("Nhập dữ liệu"):

            feature1 = st.text_input("Feature 1")
            feature2 = st.text_input("feature 2")
            feature3 = st.text_input("Feature 3")

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")
            if submitted:
                data_submitted = {'feature 1' : feature1,
                                    'feature 2' : feature2,
                                    'feature 3': feature3}
                X = pd.DataFrame(data_submitted, index=[0])
                pred = prediction(X, model)

                # Xuất ra màn hình
                st.write("predict", pred)
                results = pd.DataFrame({'Giá dự đoán': pred,
                                            'Giá thực tế': selected_rows.TongGia})
                st.write(results)

def create_dashboard(df):
    st.subheader('Dashboard')

    col1, col2 = st.columns(2)
    col1.metric(label="Số lượng dự án", value=df.shape[0])
    col2.metric(label="Giá tiền trung bình mỗi dự án", value=round(df['TongGia'].mean()))

    fig1 = px.histogram(pd_df, x="Tinh", color="LoaiBDS", labels={
                     "Tinh": "Tỉnh(Thành phố)",
                     "LoaiBDS": "Loại BDS"
                 },)
    st.plotly_chart(fig1, use_container_width=True)
    fig_col2, fig_col3 = st.columns(2)

    fig2 = px.histogram(pd_df, x="LoaiBDS", y="TongGia", histfunc='avg', labels = {
            "LoaiBDS": "Loại BDS",
            "TongGia": "price"
        })

    pd_df2 = pd_df.groupby('LoaiBDS').size().reset_index(name='Observation')
    fig3 = px.pie(pd_df2, values='Observation', names='LoaiBDS', title = 'Tỷ lệ các loại BDS')

    fig_col2.plotly_chart(fig2)
    fig_col3.plotly_chart(fig3)

def main():
    st.set_page_config(layout="wide")
    st.title('Dự đoán giá bất động sản')
    model_list = ['Dashboard',
                    'Mô hình Linear Regression',
                    'Mô hình Random Forest',
                    'Mô hình Gradient Boosting',
                    'Mô hình Decision Tree',
                    'Mô hình Isotonic Regression']
    choice_model = st.sidebar.selectbox('Mô hình huấn luyện trên:', model_list)


    if choice_model =='Dashboard':
        create_dashboard(pd_df)
    elif choice_model == 'Mô hình Linear Regression':
        model_page(choice_model, model_lr)

    elif choice_model == 'Mô hình Random Forest':
        model_page(choice_model, model_rf)

    elif choice_model == 'Mô hình Gradient Boosting':
        model_page(choice_model, model_gbt)

    elif choice_model == 'Mô hình Decision Tree':
        model_page(choice_model, model_dt)

    elif choice_model == 'Mô hình Isotonic Regression':
        model_page(choice_model, model_ir)


if __name__ == '__main__':
    spark, sc = _initialize_spark()
    ## Load dataset
    df = spark.read.format('org.apache.spark.sql.json').load("./data/clean/clean.json")
    data = df.drop(*['id','NgayDangBan', 'MoTa'])
    #st.write("data ready")
    data = data.fillna(0)
    pd_df = data.toPandas()

    ## Load model
    model_lr = LinearRegressionModel.load("./model/linear_regression/lr_basic")
    model_rf = RandomForestRegressionModel.load("./model/random_forest/rf_basic")
    model_gbt = GBTRegressionModel.load("./model/gradient_boosted/gbt_basic")
    model_dt = DecisionTreeRegressionModel.load("./model/decision_tree/dt_basic")
    model_ir = IsotonicRegressionModel.load("./model/isotonic_regression/ir_basic")

    main()
