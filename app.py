from platform import processor
import streamlit as st
from load_data import candidate_labels
import numpy as np
from load_data import *
import pickle
import torch
from BART_utils import get_taggs
from stqdm import stqdm
import pandas as pd

def transform_data(data, filetype = True):
    if filetype:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df
    
def convert_df(df):
   return df.to_csv().encode('utf-8')

stqdm.pandas()

st.title("Domain and Usage tagger")
st.subheader("문장을 입력하면 주제 / 용도 태그를 생성합니다 (EN지원)")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    processor = "🖥️"
else:
    processor = "💽"

st.subheader("Running on {}".format(device + processor))

bulk = st.checkbox("파일을 업로드하시겠어요?")
if not bulk:
    user_input = st.text_area(
    "👇태그를 생성할 문장을 입력하세요 - 현재 영문만 지원됩니다.", """NLI-based Zero Shot Text Classification
Yin et al. proposed a method for using pre-trained NLI models as a ready-made zero-shot sequence classifiers. The method works by posing the sequence to be classified as the NLI premise and to construct a hypothesis from each candidate label. The probabilities for entailment and contradiction are then converted to label probabilities."""
)

    thred = st.slider(
        "👇태그 생성 thredhold 설정. 결과가 나오지 않을경우, threshold를 0에 가깝게 낮추세요!",
        0.0,
        1.0,
        0.5,
        step=0.01,
    )
    if thred:
        st.write(thred, " 이상의 confidence level인 태그만 생성합니다.")

    maximum = st.number_input("👇최대 태그 갯수 설정", 0, 10, 5, step=1)
    st.write("최대 {} 개의 태그 생성".format(maximum))

    check_source = st.checkbox("🏷️용처 / 출처 태그 생성")
    submit = st.button("👈클릭해서 태그 생성")
    if submit:

        with st.spinner("⌛태그를 생성하는 중입니다..."):
            result = get_taggs(user_input, candidate_labels, thred)
            result = result[:maximum]
        st.subheader("🔍혹시 이런 주제의 문장인가요? : ")
        if len(result) == 0:
            st.write("😢저런..결과가 없습니다. Threshold를 낮춰보세요!")
        for i in result:
            st.write("➡️ " + i[0], "{}%".format(int(i[1] * 100)))

        if check_source:
            with st.spinner("⌛사용 목적 태그 생성중..."):
                source_result = get_taggs(user_input, source, thred=0)
            st.subheader("🔍혹시 이 사용목적의 문장인가요? : ")
            for i in source_result[:3]:
                st.write("🏷️ " + i[0], "{}%".format(int(i[1] * 100)))


else:
    st.write("🔍컬럼명을 'text'로 설정해, 파일을 업로드해주세요!")
    filetype = st.checkbox("👈Using CSV? (체크하지 않으면 xlsx 사용): ")
    uploaded_file = st.file_uploader("Choose an csv file")
    if uploaded_file is not None:
        df = transform_data(uploaded_file, filetype)
        st.write(df)
        thred = st.slider(
            "👇태그 생성 thredhold 설정. 결과가 나오지 않을경우, threshold를 0에 가깝게 낮추세요!",
            0.0,
            1.0,
            0.5,
            step=0.01,
        )
        if thred:
            st.write(thred, " 이상의 confidence level인 태그만 생성합니다.")

        maximum = st.number_input("👇최대 태그 갯수 설정", 0, 10, 5, step=1)
        st.write("최대 {} 개의 태그 생성".format(maximum))

        check_source = st.checkbox("🏷️용처 / 출처 태그 생성")
        submit = st.button("👈클릭해서 태그 생성")

        if submit:
            with st.spinner("⌛태그를 생성하는 중입니다..."):
                df["generated_tag"] = df["text"].progress_apply(lambda x : get_taggs(x, candidate_labels, thred)[:maximum])
                
            if check_source:
                with st.spinner("⌛사용 목적 태그 생성중..."):
                    df["source"] = df["text"].progress_apply(lambda x : get_taggs(x, source, thred=0))

            csv = convert_df(df)
            
            to_json = {}
            for idx, row in df.iterrows():
                to_json[row.text] = {}
                to_json[row.text]["generated_tag"] = row.generated_tag
                to_json[row.text]["source"] = row.source
            
            st.download_button(
               "Press to Download",
               csv,
               "file.csv",
               "text/csv",
               key='download-csv'
            )
            st.write("🔔Outcome: ")
            st.write(to_json)