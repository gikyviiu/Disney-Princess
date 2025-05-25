import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import plotly.express as px

#
@st.cache_data
def load_data():
    return pd.read_csv('disney_princess.csv')

# Обучение модели
@st.cache_data
def train_model(df):

    features = ['HasAnimalSidekick', 'IsRoyalByBirth', 'HairColor', 'SidekickType', 
               'EyeColor', 'OutfitStyleEra', 'SpeaksToAnimals']
    X = pd.get_dummies(df[features])
    y = df['HasMagicalPowers'].map({'Yes': 1, 'No': 0})
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy, X.columns

#
def main():
    st.title('👑 Disney Princess Magic Ability Predictor 👑')
    st.write("""
    This app analyzes data about Disney princesses and predicts if a princess has magical powers.
    """)
    st.image("pic.jpeg", width=300)
    
    # Загрузка данных
    df = load_data()
    
    # Сайдбар с контролами
    st.sidebar.header('Settings and filters')
    
    #Фильтр по принцессам
    selected_princesses = st.sidebar.multiselect(
        'Choose the princesses',
        options=df['PrincessName'].unique(),
        default=df['PrincessName'].unique()[:3]
    )
    
    #Фильтр по году выпуска
    year_range = st.sidebar.slider(
        'Year of movie release',
        min_value=int(df['FirstMovieYear'].min()),
        max_value=int(df['FirstMovieYear'].max()),
        value=(int(df['FirstMovieYear'].min()), int(df['FirstMovieYear'].max()))
    )
    
    #Фильтр по магическим способностям
    magic_filter = st.sidebar.radio(
        'Magical powers',
        options=['All', 'With magic', 'No magic']
    )
    
    
    #Фильтр по королевскому происхождению
    royal_filter = st.sidebar.selectbox(
        'Royal descent',
        options=['All', 'Yes', 'No']
    )
    
    
    #Фильтр по рейтингу IMDB
    imdb_rating = st.sidebar.slider(
        'Minimum IMDB rating',
        min_value=float(df['IMDB_Rating'].min()),
        max_value=float(df['IMDB_Rating'].max()),
        value=float(df['IMDB_Rating'].min())
    )
    
    
    # Применение фильтров
    filtered_df = df.copy()
    if selected_princesses:
        filtered_df = filtered_df[filtered_df['PrincessName'].isin(selected_princesses)]
    
    filtered_df = filtered_df[
        (filtered_df['FirstMovieYear'] >= year_range[0]) & 
        (filtered_df['FirstMovieYear'] <= year_range[1]) &
     (filtered_df['IMDB_Rating'] >= imdb_rating)
    ]

    if magic_filter == 'With magic':
        filtered_df = filtered_df[filtered_df['HasMagicalPowers'] == 'Yes']
    elif magic_filter == 'No magic':
        filtered_df = filtered_df[filtered_df['HasMagicalPowers'] == 'No']
    
    if royal_filter == 'Yes':
        filtered_df = filtered_df[filtered_df['IsRoyalByBirth'] == 'Yes']
    elif royal_filter == 'No':
        filtered_df = filtered_df[filtered_df['IsRoyalByBirth'] == 'No']
    
    # 
    st.header('Princess data')
    st.write(f"Records found: {len(filtered_df)}")
    st.dataframe(filtered_df)
    
    # 
    st.header('Data visualization')
    
    # тип графика
    chart_type = st.selectbox(
        'Type of chart',
        options=['Distribution by year', 'The connection between rankings and magic', 'Distribution by eye color']
    )
    
    if chart_type == 'Distribution by year':
        fig = px.histogram(filtered_df, x='FirstMovieYear', nbins=20, 
                          title='Distribution of films by year of release',
                          color='HasMagicalPowers')
        st.plotly_chart(fig)
        
    elif chart_type == 'The connection between rankings and magic':
        fig = px.box(filtered_df, x='HasMagicalPowers', y='IMDB_Rating',
                    title='IMDB rating distribution according to the presence of magic',
                    color='HasMagicalPowers')
        st.plotly_chart(fig)
        
    elif chart_type == 'Distribution by eye color':
        eye_color_counts = filtered_df['EyeColor'].value_counts()
        fig = px.bar(eye_color_counts, x=eye_color_counts.index, y=eye_color_counts.values,
                    title='Distribution of princesses by eye color',
                    labels={'x':'Eye color', 'y':'Quantity'},
                    color=eye_color_counts.index)
        st.plotly_chart(fig)
    
    # Анализ магических способностей
    st.header('Analyzing magical abilities')
    
    # Обучение модели
    model, accuracy, model_columns = train_model(df)
    st.write(f"Accuracy of the magic ability prediction model: {accuracy:.2f}")
    
    # Предсказание
    st.subheader('Predict magical abilities')
    
    col1, col2 = st.columns(2)
    with col1:
        has_sidekick = st.selectbox('Is there an animal sidekick?', ['Yes', 'No'])
        speaks_to_animals = st.selectbox('Can talk to animals?', ['Yes', 'No'])
        is_royal = st.selectbox('Was born into a royal family?', ['Yes', 'No'])
    with col2:
        hair_color = st.selectbox('Hair color', df['HairColor'].unique())
        eye_color = st.selectbox('Eye color', df['EyeColor'].unique())
        outfit_era = st.selectbox('Outfit Style Era', df['OutfitStyleEra'].unique())
    
    
    if st.button('Predict!'):
        input_data = pd.DataFrame({
            'HasAnimalSidekick': [has_sidekick],
            'IsRoyalByBirth': [is_royal],
            'HairColor': [hair_color],
            'EyeColor': [eye_color],
            'OutfitStyleEra': [outfit_era],
            'SpeaksToAnimals': [speaks_to_animals]
        })
        
        
        input_data = pd.get_dummies(input_data)
        
        
        for col in model_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        
        input_data = input_data[model_columns]
        
        # Предсказание
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]
        
        st.write(f"### Result: {'Has magic' if prediction[0] == 1 else 'Does not have magic'}")
        st.write(f"probability: {probability:.2f}")
        

if __name__ == '__main__':
    main()