############################# CLASIFICACIÓN ORDINAL ############################# 
# Cargamos las librerías necesarias para este trabajo
library(tidyverse)
library(RWeka)
library(caret)
# Establecemos una semilla para que los resultados sean reproducibles 
set.seed(26)

# Función que divide un dataset proporcionado en dos subconjuntos para el 
# entrenamiento y testeo de modelos predictivos a partir del porcentaje proporcionado.
# Devuelve una lista con los conjuntos de entrenamiento y test resultantes
create_train_test <- function(dataset, class_index, perc) {
  # Renombramos la variable a predecir 
  names(dataset)[class_index] <- "target"
  # Dividimos el conjunto de datos según el porcentaje proporcionado
  train_index <- createDataPartition(dataset[, class_index], p=perc, list=FALSE)
  return(list(train=dataset[train_index, ], test=dataset[-train_index, ]))
}

# Función que permite transformar un problema de clasificación ordinal en K-1
# problemas de clasificación binaria a partir de las K clases existentes. Para 
# ello se deberá proporcionar un dataset de entrenamiento con el que realizar
# las particiones y entrenar un modelo para cada una de ellas utilizando el 
# algoritmo RPART.
# Devuelve una lista con los clasificadores generados.
ordinal_train <- function(dataset) {
  # Lista para almacenar los modelos generados por cada dataset
  models <- list()
  # Obtenemos las clases en orden
  classes <- as.integer(unique(dataset$target))
  # Vector con las muestras para cada clase 
  indexes <- c()
  # Generamos los datasets binarios para K-1 clases
  for (i in 1:(length(classes)-1)) {
    # Seleccionamos las instancias de una clase
    indexes <- c(indexes, which(dataset$target==classes[i]))
    # Convertimos las etiquetas numéricas asignando 0 a la clase actual y
    # 1 a las clases restantes
    y <- as.integer(dataset$target)
    y[indexes] <- 0
    y <- ifelse(y==0, 0, 1)
    # Unificamos la nueva variable binaria con el resto del dataset 
    binary_dataset <- cbind(dataset[,1:(ncol(dataset)-1)], target=as.factor(y))
    sapply(binary_dataset, class)
    # Generamos un clasificador para el dataset resultante
    models[[i]] <- PART(target~., data=binary_dataset)
  }
  return(models)
}

# Función que recibe como parámetros un conjunto de modelos predictivos
# procedentes de los K-1 problemas de clasificación binaria, junto al dataset
# de test para calcular la probabilidad de cada muestra con respecto cada clase
# y asignar la categoría cuya probabilidad sea máxima.
ordinal_test <- function(models, dataset) {
  # Obtenemos la lista de clases ordenada del dataset proporcionado
  unique_classes <- as.integer(unique(dataset$target))
  # Eliminamos la variable dependiente del dataset de test
  test_dataset <- dataset %>% select(-target)
  # Vector para almacenar las predicciones sobre test
  preds <- c()
  # Recorremos cada una de las muestras de test para elegir su clase
  for (index_test in 1:nrow(test_dataset)) {
    # Vector para almacenar las probabilidades de la muestra actual para cada clase
    # Calculamos la probabilidad para la primera clase
    probs <- c(predict(models[[1]], test_dataset[index_test, ], type="prob")[,1])
    # Calculamos las probabilidades para las clases intermedias
    for(i in 2:length(models)) {
      probs <- c(probs, predict(models[[i-1]], test_dataset[index_test, ], type="prob")[,2] * 
        predict(models[[i]], test_dataset[index_test, ], type="prob")[,1])
    }
    # Calculamos la probabilidad para la última clase
    probs <- c(probs, predict(models[[length(models)]], test_dataset[index_test, ], type="prob")[,2])
    # Seleccionamos la clase con mayor probabilidad y la asignamos a la muestra de test
    preds <- c(preds, unique_classes[which(probs == max(probs))])
  }
  return (preds)
}

# Probamos la clasificación ordinal generalizada con todos los datasets disponibles
# c("era.arff", "esl.arff", "lev.arff", "swd.arff")
for (filename in c("era.arff", "esl.arff", "lev.arff", "swd.arff")) {
  print(filename)
  # Cargamos el dataset
  df <- read.arff(filename)
  # Ordenamos el dataset según las clases
  df <- df[order(df[, ncol(df)]), ]
  # Generamos un conjunto de entrenamiento y otro de test
  df.partitions <- create_train_test(df, ncol(df), 0.75)
  # Generamos K-1 problemas de clasificación binaria con el conjunto de entrenamiento
  df.models <- ordinal_train(df.partitions$train)
  # Evaluamos los modelos conseguidos calculando las probabilidades y seleccionando
  # la clase de la mayor
  df.test_preds <- ordinal_test(df.models, df.partitions$test)
  # Porcentaje de precisión
  print(sum(df.test_preds==df.partitions$test$target)/length(df.partitions$test$target))
}
