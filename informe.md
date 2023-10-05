---
title: "Tecnología Digital VI: Inteligencia Artificial - Trabajo Práctico 2"
author: [Federico Giorgi, Tomás Curzio, Gastón Loza Montaña]
date: "04/10/23"
lang: "es"
...

# Análisis exploratorio de los datos

# Variables adicionales

# Conjunto de validación

En las clases vimos principalmente tres maneras de realizar conjuntos de validación: Holdout Set, LOOCV, K-Fold CV.

En nuestro modelo decidimos utilizar Holdout Set, con una proporción de 70% para entrenar y 30% para validación, pensamos que, al querer un validation set similar al leaderboard de kaggle, hacer una partición del mismo tamaño podría tener sentido, y además se condice con los estandares vistos en clase.

¿Por qué decidimos hacer Holdout Set?

La principal respuesta es tiempo de cómputo. Leave-one-out Cross Validation lo descartamos de entrada pues era inviable con una cantidad de datos tan grande como la del dataset que estamos utilizando. K-fold CV podría ser usado con valores chicos de k, pero de igual manera cada entrenamiento lleva su tiempo, y nos parecía mas útil usar ese tiempo en pensar y aplicar posibles mejoras a nuestro modelo, con una validación que tarde menos pero igualmente sea de calidad, pues hay muchos datos para validar en un 30% de un dataset tan grande, y a su vez no duele tanto perderlos para entrenar como si podría pasar en un dataset pequeño.

Buscabamos además que el validation set no nos arroje resultados muy optimistas por sobre a lo obtenido en el leaderboard y experimentalmente logramos ese efecto de esta manera, por lo que nos pareció razonable utilizarlo de brújula para la construcción de nuestro modelo.

# Modelo predictivo

## Hiperparámetros

# Análisis Final