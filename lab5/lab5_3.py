import numpy as np
import altair as alt
import pandas as pd
import streamlit as st

init_amplitude = 1.0
init_frequency = 0.5
init_phase = 0.0
init_noise_mean = 0.0
init_noise_covariance = 0.1


t = np.linspace(0, 10, 1000) #час від 0 до 10 на 1000 точок
fs = (len(t) - 1) / (t[-1] - t[0]) #частота дискретизації(відображення максимальної частоти без спотворень)


def harmonic(A, f, phi, t):
    return A * np.sin(2 * np.pi * f * t + phi)


def harmonic_with_noise(A, f, phi, t, noise):
    return harmonic(A, f, phi, t) + noise


def moving_average(x, window_size=5): #вираховуємо рухоме середнє по точкам сигналів
    result = np.zeros_like(x) # масив з нулями розміру x
    for i in range(len(x)):
        start = max(0, i - window_size + 1) #початок вікна
        result[i] = np.mean(x[start:i+1]) #середнє значення на відрізку
    return result


def generate_noise(mean, cov, size):
    return np.random.normal(mean, np.sqrt(cov), size)


#streamlit
st.title("Інтерактивна гармоніка + власний фільтр (Altair + Streamlit)")
st.sidebar.header("Налаштування сигналу")
amplitude = st.sidebar.slider('Амплітуда', 0.0, 2.0, init_amplitude, 0.05)
frequency = st.sidebar.slider('Частота (Гц)', 0.1, 5.0, init_frequency, 0.1)
phase = st.sidebar.slider('Фаза (рад)', 0.0, 2*np.pi, init_phase, 0.1)
noise_mean = st.sidebar.slider('Середнє шуму', -1.0, 1.0, init_noise_mean, 0.05)
noise_cov = st.sidebar.slider('Дисперсія шуму', 0.0, 1.0, init_noise_covariance, 0.01)

filter_type = st.sidebar.selectbox('Тип фільтра', 'Рухоме середнє')

#генерація сигналів
noise = generate_noise(noise_mean, noise_cov, len(t))
pure = harmonic(amplitude, frequency, phase, t)
noisy = harmonic_with_noise(amplitude, frequency, phase, t, noise)

if filter_type == 'Рухоме середнє':
    filtered = moving_average(noisy, window_size=15)
else:
    filtered = noisy

#перетворення в DataFrame
data = pd.DataFrame({
    'Час': t,
    'Чиста гармоніка': pure,
    'Шумна гармоніка': noisy,
    'Фільтрована гармоніка': filtered
})

#графіки
st.subheader("чиста + шумна")
fig1 =alt.Chart(data).mark_line(color='orange').encode(
    x='Час',
    y='Шумна гармоніка'
) + alt.Chart(data).mark_line(color='blue').encode(
    x='Час',
    y='Чиста гармоніка'
)

fig1 = fig1.properties(width=700, height=300)
st.altair_chart(fig1, use_container_width=True)

st.subheader("чиста + фільтрована")
fig2 = alt.Chart(data).mark_line().encode(
    x='Час',
    y='Чиста гармоніка',
    color=alt.value('blue')
) + alt.Chart(data).mark_line().encode(
    x='Час',
    y='Фільтрована гармоніка',
    color=alt.value('purple')
)
fig2 = fig2.properties(width=700, height=300)
st.altair_chart(fig2, use_container_width=True)