########################### CLASIFICACIÓN MONOTÓNICA ########################### 
# Cargamos las librerías necesarias para este trabajo
library(RWeka)
library(tidyverse)
library(caret)
library(xgboost)
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
# las particiones y entrenar un modelo para cada una de ellas utilizando XGBoost
# aplicando la restricción de monotonicidad.
# Devuelve una lista con los clasificadores generados.
monotonic_train <- function(dataset) {
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
    binary_dataset <- cbind(dataset[,1:(ncol(dataset)-1)], target=y)
    sapply(binary_dataset, class)
    # Convertimos el dataset binario a una matriz numérica con las etiquetas
    # de entrenamiento también numéricas
    binary_matrix <- xgb.DMatrix(data=data.matrix(binary_dataset[,-ncol(binary_dataset)]),
                                  label=binary_dataset[,ncol(binary_dataset)])
    # Generamos un clasificador para el problema binario con la matriz anterior
    # aplicando la restricción de monotonicidad durante 100 iteraciones
    models[[i]] <- xgboost(data=binary_matrix, nrounds=100, verbose=FALSE,
     eval_metric='logloss', monotone_constraints=1, objective = "binary:logistic")
  }
  return(models)
}

# Función que recibe como parámetros un conjunto de modelos predictivos
# procedentes de los K-1 problemas de clasificación binaria, junto al dataset
# de test para combinar las probabilidades de todas las clases con las que 
# decidir la categoría de cada una de las muestras de test.
# Devuelve una lista con las predicciones sobre el conjunto de test.
monotonic_test <- function(models, dataset) {
  # Convertimos el dataset de test en una matriz para validar los modelos
  test_matrix <- xgb.DMatrix(data=data.matrix(dataset[, -ncol(dataset)]))
  # Calculamos la predicción para cada modelo con todas las muestras de test
  # aplicando la fórmula vista en la diapositiva 92 de teoría
  preds <- sapply(1:length(models), 
    function(x) ifelse(as.numeric(predict(models[[x]], test_matrix) > 0.50), 1, 0))
  return(rowSums(preds)+1)
}

# Probamos la clasificación ordinal generalizada con todos los datasets disponibles
for (filename in c("era.arff", "esl.arff", "lev.arff", "swd.arff")) {
  print(filename)
  # Cargamos el dataset
  df <- read.arff(filename)
  # Ordenamos el dataset según las clases
  df <- df[order(df[, ncol(df)]), ]
  # Generamos un conjunto de entrenamiento y otro de test
  df.partitions <- create_train_test(df, ncol(df), 0.75)
  # Generamos K-1 problemas de clasificación binaria con el conjunto de entrenamiento
  df.models <- monotonic_train(df.partitions$train)
  # Evaluamos los modelos conseguidos calculando las predicciones según la
  # formulación de la diapositiva 92
  df.test_preds <- monotonic_test(df.models, df.partitions$test)
  # Porcentaje de precisión
  print(sum(df.test_preds==df.partitions$test$target)/length(df.partitions$test$target))
}
