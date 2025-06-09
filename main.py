import streamlit as st
import pandas as pd
import plotly.express as px
import joblib  

# Загрузка данных
@st.cache_data
def load_data():
    return pd.read_csv('disney_princess.csv')

# Загрузка модели
@st.cache_resource
def load_model():
    model = joblib.load('tmodel.pkl')
    
    return model

def main():
    st.title('👑 Disney Princess Magic Ability Predictor 👑')
    st.write("This app analyzes data about Disney princesses and predicts if a princess has magical powers.")
    st.image("pic.jpeg", width=300)

    df = load_data()
    model = load_model()


    # --- Фильтрация данных ---
    st.sidebar.header('Settings and filters')
    
    selected_princesses = st.sidebar.multiselect(
        'Choose the princesses',
        options=df['PrincessName'].unique(),
        default=df['PrincessName'].unique()[:3]
    )

    year_range = st.sidebar.slider(
        'Year of movie release',
        min_value=int(df['FirstMovieYear'].min()),
        max_value=int(df['FirstMovieYear'].max()),
        value=(int(df['FirstMovieYear'].min()), int(df['FirstMovieYear'].max()))
    )

    magic_filter = st.sidebar.radio(
        'Magical powers',
        options=['All', 'With magic', 'No magic']
    )

    royal_filter = st.sidebar.selectbox(
        'Royal descent',
        options=['All', 'Yes', 'No']
    )

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

    # Отображение отфильтрованных данных
    st.header('Princess data')
    st.write(f"Records found: {len(filtered_df)}")
    st.dataframe(filtered_df)

    # Графики
    st.header('Data visualization')

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
                     labels={'x': 'Eye color', 'y': 'Quantity'},
                     color=eye_color_counts.index)
        st.plotly_chart(fig)

    # Предсказание
    MODEL_COLUMNS = [
    'HairColor_Black', 'HairColor_Blonde', 'HairColor_Brown', 'HairColor_Red', 'HairColor_White',
    'EyeColor_Blue', 'EyeColor_Brown', 'EyeColor_Green', 'EyeColor_Grey', 'EyeColor_Hazel',
    'OutfitStyleEra_Ancient', 'OutfitStyleEra_Medieval', 'OutfitStyleEra_Modern', 'OutfitStyleEra_Victorian',
    'IsRoyalByBirth_No', 'IsRoyalByBirth_Yes',
    'HasAnimalSidekick_No', 'HasAnimalSidekick_Yes',
    'SpeaksToAnimals_No', 'SpeaksToAnimals_Yes']
    
    st.header('Predict magical abilities')
    st.subheader('Predict magical abilities')

    col1, col2 = st.columns(2)
    with col1:
        eye_color = st.selectbox('Eye color', df['EyeColor'].unique())
        outfit_era = st.selectbox('Outfit Style Era', df['OutfitStyleEra'].unique())
        hair_color = st.selectbox('Hair color', df['HairColor'].unique())
        
    with col2:
        speaks_to_animals = st.selectbox('Can talk to animals?', ['Yes', 'No'])
        has_sidekick = st.selectbox('Is there an animal sidekick?', ['Yes', 'No'])
        is_royal = st.selectbox('Was born into a royal family?', ['Yes', 'No'])

    if st.button('Predict!'):
        input_data = pd.DataFrame({
            'EyeColor': [eye_color],
            'HasAnimalSidekick': [has_sidekick],
            'IsRoyalByBirth': [is_royal],
            'HairColor': [hair_color],
            'OutfitStyleEra': [outfit_era],
            'SpeaksToAnimals': [speaks_to_animals]
        })


        # Предсказание
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0][1]

        st.write(f"### Result: {'✅ The princess has magical powers!' if prediction[0] == 'Yes' else '❌ The princess does not have magical powers.'}")
        st.write(f"Probability: {probability:.2f}")
        st.write(f"Raw prediction: {prediction[0]}")

if __name__ == '__main__':
    main()