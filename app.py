#pip install streamlit
#pip install pandas
#pip install sklearn
#pip install plotly
#pip install matplotlib
#pip install numpy
#pip install seaborn


# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns


#Extraccion de data de entrenamiento desde un archivo csv. 
#El CSV es una base de datos de pacientes Diabéticos, contemplando datos como:
#  - Presion Sanguinea
#  - Embarazos
#  - Nivel de Glucosa
#  - Grosor de piel
#  - Insulina
#  - Indice de masa corporal BMI
#  - Edad
#  - Funcion de diabetes pedigri: Función que puntúa la probabilidad de diabetes en función de los antecedentes familiares.



df = pd.read_csv("diabetes.csv") ##Extracion de data de pacientes.

#Datos de estudiantes
st.markdown("<h1 style='text-align: center; color: #0088ff;'>Inteligencia Artificial | UNAD</h1>", unsafe_allow_html=True)
st.header('Participantes:')
st.subheader('Luis Alberto Ceron')
st.subheader('Sandro Aldemar Ceron')
st.subheader('Nayibe Alexandra Ijaji')
st.subheader('Luis Fernando Arciniegas')
st.markdown("<h3 style='text-align: center; color: white;'>Grupo: 18</h3>", unsafe_allow_html=True)

# Encabezados
st.title('Diagnosticador de Diabetes')
st.sidebar.header('Datos del Paciente')
st.subheader('Datos de entrenamiento de IA')
st.write(df.describe()) #Presentacion de datos en la interfaz grafica formato tabla


# Extraccion de datos de la base de datos de entrenamiento
x = df.drop(['Outcome'], axis = 1) #Asignacion de datos en la columna Outcome = 1 a la variable X
y = df.iloc[:, -1] #Indexacion de datos en variable y que relaciona el numero de datos con los valores especificos de la columna Outcome.

#Se divide el conjunto de datos con la libreria sklearn, en subconjuntos que minimizan el potencial de sesgo en el proceso de evaluación y validación.
#La divicion del conjunto en subconjuntos permitira una evaluación imparcial del rendimiento de la predicción.
#De manera que de esta forma se posibilita la implementacion de un aprendizaje automatico supervisado
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0) 


# FUNCTION
def user_report():
  #Controles deslizantes para el ajuste de valores para pacientes determinados
  #De esta manera se pueden evaluar multiples sintomas para arrojar un diagnostico de la diabetes
  pregnancies = st.sidebar.slider('Embarazos', 0,17, 3 ) #Control deslizante embarazos
  glucose = st.sidebar.slider('Nivel de Glucosa', 0,200, 120 ) #Control deslizante glucosa
  bp = st.sidebar.slider('Presion Sanguinea', 0,122, 70 ) #Control deslizante Presion Sanguinea
  skinthickness = st.sidebar.slider('Grosor de piel', 0,100, 20 ) #Control deslizante grosor de piel
  insulin = st.sidebar.slider('Insulina', 0,846, 79 ) #Control deslizante insulina
  bmi = st.sidebar.slider('Indice de masa corporal BMI', 0,67, 20 ) #Control deslizante Indice de masa corporal BMI
  dpf = st.sidebar.slider('Funcion de Diabetes pedigri', 0.0,2.4, 0.47 ) #Control deslizante Funcion de diabetes pedigri
  age = st.sidebar.slider('Edad', 21,88, 33 ) #Control deslizante edad

  #Reporte de datos a presentar en DataFrame o graficos
  user_report_data = {
      'Embarazos':pregnancies,
      'Nivel de Glucosa':glucose,
      'Presion Sanguinea':bp,
      'Grosor de piel':skinthickness,
      'Insulina':insulin,
      'Indice de masa corporal BMI':bmi,
      'Funcion de Diabetes pedigri':dpf,
      'Edad':age
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

# Datos del paciente a presentar en la interface
user_data = user_report()
st.subheader('Datos del Paciente') #Asignacion de datos en header de reporte
st.write(user_data) #Marcado de datos en interfaz


# Modelo de datos
#metaestimador que se ajusta a una serie de clasificadores de árboles de decisión 
#en varias submuestras del conjunto de datos y utiliza promedios para mejorar la precisión predictiva y controlar el sobreajuste.
rf  = RandomForestClassifier()
#Construccion de un bosque de árboles a partir del conjunto de entrenamiento (x_train, y_train).
rf.fit(x_train, y_train)
#Predice una muestra de entrada a partir de los árboles del bosque, ponderado por sus estimaciones de probabilidad. 
#La clase pronosticada es la que tiene la estimación de probabilidad media más alta en todos los árboles.
user_result = rf.predict(user_data) #Prediccion definitiva de si un usuario sufre de diabetes.

#Si el resultado es 0, el usuario esta saludable, si es 1 el usuario es Diabético.

# Titulo de vistas
st.title('Informe de paciente')



# Funcion que cambia el color del punto que esta ubicado en las graficas.
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'

# Edad vs Embarazos
st.header('Gráfico de conteo de embarazos (otros vs paciente)')
fig_preg = plt.figure() #Construccion de grafica
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'Greens')
ax2 = sns.scatterplot(x = user_data['Edad'], y = user_data['Embarazos'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Saludable & 1 - Tiene Diabetes')
st.pyplot(fig_preg)

#Evalua los cambios en el control deslizante de embarazos, edad, BMI, glucosa entre otros campos,
#para predecir la diabetes.

#Edad vs Glucosa
st.header('Gráfica de valores de glucosa (otros vs paciente)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = user_data['Edad'], y = user_data['Nivel de Glucosa'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Saludable & 1 - Tiene Diabetes')
st.pyplot(fig_glucose)



# Edad vs Presion Sanguinea
st.header('Gráfico del valor de la presión arterial (otros vs paciente)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = user_data['Edad'], y = user_data['Presion Sanguinea'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Saludable & 1 - Tiene Diabetes')
st.pyplot(fig_bp)


# Edad vs Grosor de la piel
st.header('Gráfico del valor del grosor de la piel (otros vs paciente)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Blues')
ax8 = sns.scatterplot(x = user_data['Edad'], y = user_data['Grosor de piel'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Saludable & 1 - Tiene Diabetes')
st.pyplot(fig_st)

# Edad vs Insulina
st.header('Gráfico del valor de Insulina (otros vs paciente)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['Edad'], y = user_data['Insulina'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Saludable & 1 - Tiene Diabetes')
st.pyplot(fig_i)


# Edad vs BMI
st.header('Gráfico del valor BMI (otros vs paciente)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
ax12 = sns.scatterplot(x = user_data['Edad'], y = user_data['Indice de masa corporal BMI'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.title('0 - Saludable & 1 - Tiene Diabetes')
st.pyplot(fig_bmi)


# Edad vs Funcion de Diabetes pedigri
st.header('Gráfico del valor de la Funcion de Diabetes pedigri (otros vs paciente)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x = user_data['Edad'], y = user_data['Funcion de Diabetes pedigri'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3,0.2))
plt.title('0 - Saludable & 1 - Tiene Diabetes')
st.pyplot(fig_dpf)



# Salida - Resultado del programa.
st.subheader('Resultado: ')
output=''
if user_result[0]==0: #Evalua si la variable Outcome es 0 o 1 para determinar si el paciente es o no diabetico.
  output = 'El Paciente no es Diabético'
else:
  output = 'El Paciente es Diabético'
diagpercent = str(accuracy_score(y_test, rf.predict(x_test))*100)+'%' #Calcula y presenta el porcentaje de precisión de la estimacion.
st.title(output)
st.subheader('Este Diagnóstico tiene una precisión del: {}'.format(diagpercent))

st.title("Codigo Python utilizado")
codigo = """

# IMPORT STATEMENTS
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns


#Extraccion de data de entrenamiento desde un archivo csv. 
#El CSV es una base de datos de pacientes Diabéticos, contemplando datos como:
#  - Presion Sanguinea
#  - Embarazos
#  - Nivel de Glucosa
#  - Grosor de piel
#  - Insulina
#  - Indice de masa corporal BMI
#  - Edad
#  - Funcion de diabetes pedigri: Función que puntúa la probabilidad de diabetes en función de los antecedentes familiares.



df = pd.read_csv("diabetes.csv") ##Extracion de data de pacientes.

#Datos de estudiantes
st.markdown("<h1 style='text-align: center; color: #0088ff;'>Inteligencia Artificial | UNAD</h1>", unsafe_allow_html=True)
st.header('Participantes:')
st.subheader('Luis Alberto Ceron')
st.subheader('Sandro Aldemar Ceron')
st.subheader('Nayibe Alexandra Ijaji')
st.subheader('Luis Fernando Arciniegas')
st.markdown("<h3 style='text-align: center; color: white;'>Grupo: 18</h3>", unsafe_allow_html=True)

# Encabezados
st.title('Diagnosticador de Diabetes')
st.sidebar.header('Datos del Paciente')
st.subheader('Datos de entrenamiento de IA')
st.write(df.describe()) #Presentacion de datos en la interfaz grafica formato tabla


# Extraccion de datos de la base de datos de entrenamiento
x = df.drop(['Outcome'], axis = 1) #Asignacion de datos en la columna Outcome = 1 a la variable X
y = df.iloc[:, -1] #Indexacion de datos en variable y que relaciona el numero de datos con los valores especificos de la columna Outcome.

#Se divide el conjunto de datos con la libreria sklearn, en subconjuntos que minimizan el potencial de sesgo en el proceso de evaluación y validación.
#La divicion del conjunto en subconjuntos permitira una evaluación imparcial del rendimiento de la predicción.
#De manera que de esta forma se posibilita la implementacion de un aprendizaje automatico supervisado
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0) 


# FUNCTION
def user_report():
  #Controles deslizantes para el ajuste de valores para pacientes determinados
  #De esta manera se pueden evaluar multiples sintomas para arrojar un diagnostico de la diabetes
  pregnancies = st.sidebar.slider('Embarazos', 0,17, 3 ) #Control deslizante embarazos
  glucose = st.sidebar.slider('Nivel de Glucosa', 0,200, 120 ) #Control deslizante glucosa
  bp = st.sidebar.slider('Presion Sanguinea', 0,122, 70 ) #Control deslizante Presion Sanguinea
  skinthickness = st.sidebar.slider('Grosor de piel', 0,100, 20 ) #Control deslizante grosor de piel
  insulin = st.sidebar.slider('Insulina', 0,846, 79 ) #Control deslizante insulina
  bmi = st.sidebar.slider('Indice de masa corporal BMI', 0,67, 20 ) #Control deslizante Indice de masa corporal BMI
  dpf = st.sidebar.slider('Funcion de Diabetes pedigri', 0.0,2.4, 0.47 ) #Control deslizante Funcion de diabetes pedigri
  age = st.sidebar.slider('Edad', 21,88, 33 ) #Control deslizante edad

  #Reporte de datos a presentar en DataFrame o graficos
  user_report_data = {
      'Embarazos':pregnancies,
      'Nivel de Glucosa':glucose,
      'Presion Sanguinea':bp,
      'Grosor de piel':skinthickness,
      'Insulina':insulin,
      'Indice de masa corporal BMI':bmi,
      'Funcion de Diabetes pedigri':dpf,
      'Edad':age
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

# Datos del paciente a presentar en la interface
user_data = user_report()
st.subheader('Datos del Paciente') #Asignacion de datos en header de reporte
st.write(user_data) #Marcado de datos en interfaz


# Modelo de datos
#metaestimador que se ajusta a una serie de clasificadores de árboles de decisión 
#en varias submuestras del conjunto de datos y utiliza promedios para mejorar la precisión predictiva y controlar el sobreajuste.
rf  = RandomForestClassifier()
#Construccion de un bosque de árboles a partir del conjunto de entrenamiento (x_train, y_train).
rf.fit(x_train, y_train)
#Predice una muestra de entrada a partir de los árboles del bosque, ponderado por sus estimaciones de probabilidad. 
#La clase pronosticada es la que tiene la estimación de probabilidad media más alta en todos los árboles.
user_result = rf.predict(user_data) #Prediccion definitiva de si un usuario sufre de diabetes.

#Si el resultado es 0, el usuario esta saludable, si es 1 el usuario es Diabético.

# Titulo de vistas
st.title('Informe de paciente')



# Funcion que cambia el color del punto que esta ubicado en las graficas.
if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'

# Edad vs Embarazos
st.header('Gráfico de conteo de embarazos (otros vs paciente)')
fig_preg = plt.figure() #Construccion de grafica
ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'Greens')
ax2 = sns.scatterplot(x = user_data['Edad'], y = user_data['Embarazos'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title('0 - Saludable & 1 - Tiene Diabetes')
st.pyplot(fig_preg)

#Evalua los cambios en el control deslizante de embarazos, edad, BMI, glucosa entre otros campos,
#para predecir la diabetes.

#Edad vs Glucosa
st.header('Gráfica de valores de glucosa (otros vs paciente)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
ax4 = sns.scatterplot(x = user_data['Edad'], y = user_data['Nivel de Glucosa'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,220,10))
plt.title('0 - Saludable & 1 - Tiene Diabetes')
st.pyplot(fig_glucose)



# Edad vs Presion Sanguinea
st.header('Gráfico del valor de la presión arterial (otros vs paciente)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
ax6 = sns.scatterplot(x = user_data['Edad'], y = user_data['Presion Sanguinea'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,130,10))
plt.title('0 - Saludable & 1 - Tiene Diabetes')
st.pyplot(fig_bp)


# Edad vs Grosor de la piel
st.header('Gráfico del valor del grosor de la piel (otros vs paciente)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Blues')
ax8 = sns.scatterplot(x = user_data['Edad'], y = user_data['Grosor de piel'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,110,10))
plt.title('0 - Saludable & 1 - Tiene Diabetes')
st.pyplot(fig_st)

# Edad vs Insulina
st.header('Gráfico del valor de Insulina (otros vs paciente)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
ax10 = sns.scatterplot(x = user_data['Edad'], y = user_data['Insulina'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,900,50))
plt.title('0 - Saludable & 1 - Tiene Diabetes')
st.pyplot(fig_i)


# Edad vs BMI
st.header('Gráfico del valor BMI (otros vs paciente)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
ax12 = sns.scatterplot(x = user_data['Edad'], y = user_data['Indice de masa corporal BMI'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,70,5))
plt.title('0 - Saludable & 1 - Tiene Diabetes')
st.pyplot(fig_bmi)


# Edad vs Funcion de Diabetes pedigri
st.header('Gráfico del valor de la Funcion de Diabetes pedigri (otros vs paciente)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x = user_data['Edad'], y = user_data['Funcion de Diabetes pedigri'], s = 150, color = color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,3,0.2))
plt.title('0 - Saludable & 1 - Tiene Diabetes')
st.pyplot(fig_dpf)



# Salida - Resultado del programa.
st.subheader('Resultado: ')
output=''
if user_result[0]==0: #Evalua si la variable Outcome es 0 o 1 para determinar si el paciente es o no diabetico.
  output = 'El Paciente no es Diabético'
else:
  output = 'El Paciente es Diabético'
diagpercent = str(accuracy_score(y_test, rf.predict(x_test))*100)+'%' #Calcula y presenta el porcentaje de precisión de la estimacion.
st.title(output)
st.subheader('Este Diagnóstico tiene una precisión del: {}'.format(diagpercent))



"""

st.subheader(codigo)