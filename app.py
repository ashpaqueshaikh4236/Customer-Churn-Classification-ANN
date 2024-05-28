import streamlit as st
import pickle
import numpy as np
import joblib
from keras.models import load_model
from sklearn.pipeline import Pipeline


df = pickle.load(open('clean_data.pkl','rb'))
preprocessor= joblib.load('preprocessor.pkl')
model= load_model('model.h5')
model = Pipeline([('preprocessor', preprocessor),('model', model)])


st.title('Customer Churn Classifier')


categorical = df.select_dtypes(include='object')
numerical = df.select_dtypes(exclude='object')

unique_values_dict = {}
for col in df.columns:
    unique_values = df[col].unique().tolist()
    unique_values.sort()
    unique_values.insert(0, 'select')
    unique_values_dict[col] = unique_values


join_data = []
for col, values in unique_values_dict.items():
    if col in categorical:
        come_cat_values= st.selectbox(col, values)
        join_data.append(come_cat_values)
    elif col in numerical:
         come_num_values = st.text_input(col)
         join_data.append(come_num_values) 


try:
    if st.button('predict'):
        reshaped_data = np.asarray(join_data).reshape(1, -1)
        
        if 'select' in reshaped_data or '' in reshaped_data:
            st.warning('Please fill all values')
        else:
            prediction = model.predict(reshaped_data)
            if prediction > 0.5:
                st.warning('yes')
            else:
                st.success('NO')
except Exception as e:
    st.warning(f'An error occurred: {e}')












# try:
#     if st.button('predict'):
#         st.write(join_data)
#         reshaped_data = np.asarray(join_data).reshape(1,-1)
#     try:
#         if reshaped_data == 'select':
#             except:
#                 st.warning('Please Fill all values')
#                 prediction = model.predict(reshaped_data)
#                 if prediction > 0.5:
#                     st.warning('yes')
#                 else:
#                     st.success('NO')
# except:
#     st.warning('Please Fill all values')

