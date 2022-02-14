########################## CLASIFICACIÓN ORDINAL ########################## 
## Generalización del proceso de modelos múltiples
# Cargamos las librerías necesarias para este trabajo
library(tidyverse)
library(RWeka)
library(caret)
# Establecemos una semilla para que los resultados sean reproducibles 
set.seed(2022)

# Función que divide un dataset proporcionado en dos subconjuntos para el 
# entrenamiento y testeo de modelos predictivos a partir del porcentaje proporcionado.
# Devuelve una lista con los conjuntos de entrenamiento y test resultantes
create_train_test <- function(dataset, class_index, perc) {
  # Renombramos la variable a predecir 
  names(dataset)[class_index] <- "target"
  # Dividimos el conjunto de datos según el porcentaje proporcionado
  train_index <- createDataPartition(dataset[, class_index], p=perc, list=FALSE)
  # Retorna el conjunto de entrenamiento y de test
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
  # Eliminamos la variable dependiente del dataset
  test_dataset <- dataset %>% select(-target)
  # Calculamos la probabilidad de que sea la primera clase
  prob1 <- predict(models[[1]], test_dataset, type="prob")[,1]
  # Matriz para almacenar las probabilidades por muestra para cada clase
  prob_matrix <- data.frame(matrix(0, ncol = 0, nrow = nrow(dataset)))
  prob_matrix <- cbind(prob_matrix, prob1)
  # Calculamos las restantes probabilidades excepto la última para las clases
  # intermedias
  for(i in 2:length(models)) {
    probi <- predict(models[[i-1]], test_dataset, type="prob")[,2] * 
      predict(models[[i]], test_dataset, type="prob")[,1]
    prob_matrix <- cbind(prob_matrix, probi)
  }
  # Calculamos la probabilidad con el último dataset
  probk <- predict(models[[length(models)]], test_dataset, type="prob")[,2]
  prob_matrix <- cbind(prob_matrix, probk)
  # Asignamos las clases de las muestras de test a partir de la probabilidad
  # máxima de cada clase
  test_preds <- apply(prob_matrix, MARGIN = 1, 
                      function(x){dataset$target[which.max(x)]})
  return(test_preds)
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
  df.models <- ordinal_train(df.partitions$train)
  # Evaluamos los modelos conseguidos calculando las probabilidades y seleccionando
  # la clase de la mayor
  df.test_preds <- ordinal_test(df.models, df.partitions$test)
  # Porcentaje de precisión
  print(sum(df.test_preds==df.partitions$test$target)/length(df.partitions$test$target))
}
