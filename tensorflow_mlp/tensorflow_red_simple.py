#!/usr/bin/env python
# coding: utf-8

# <img src="imagenes/rn3.png" width="200">
# <img src="http://www.identidadbuho.uson.mx/assets/letragrama-rgb-150.jpg" width="200">

# # [Curso de Redes Neuronales](https://curso-redes-neuronales-unison.github.io/Temario/)
# 
# # Una red neuronal multicapa simple usando TensorFlow
# 
# 
# [**Julio Waissman Vilanova**](http://mat.uson.mx/~juliowaissman/), 27 de septiembre de 2017.
# 
# 
# 
# En esta libreta se muestra el ejemplo básico para una red multicapa sencilla
# aplicada al conjunto de datos [MNIST](http://yann.lecun.com/exdb/mnist/).
# 
# Esta libreta es básicamente una traducción del ejemplo
# desarrollado por [Aymeric Damien](https://github.com/aymericdamien/TensorFlow-Examples/)
# 

# In[ ]:


import tensorflow as tf


# ## 1. Cargar datos
# 
# Primero cargamos los archivos que se utilizan para el aprendizaje. Para otro tipo de problemas, es necesario hacer un proceso conocido como *Data Wrangling*, que normalmente se realiza con la ayuda de *Pandas*. 

# In[ ]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Para que un aprendizaje tenga sentido es necesario tener bien separado un conjunto de datos de aprendizaje y otro de prueba (en caso de grandes conjuntos de datos es la opción). Como vemos tanto las imágenes como las etiquetas están separados en archivos de datos y de aprendizaje.
# 
# El objeto `mnist` es un objeto tensorflow que contiene 3 objetos tipo tensorflow: *test*, *train* y *validation*, los cuales a su vez contienen *ndarrays* de *numpy*. La estructura es la misma para cada conjunto de datos. Veamos su estructura:
# 

# In[ ]:


print("Tipo de images: {}".format(type(mnist.train.images)))
print("Tipo de epochs_completed: {}".format(type(mnist.train.epochs_completed)))
print("Tipo de labels: {}".format(type(mnist.train.labels)))
print("Tipo de nest_batch: {}".format(type(mnist.train.next_batch)))
print("Tipo de num_examples: {}".format(type(mnist.train.num_examples)))


# Como generar el conjunto de datos para ser utilizado dentro de TensorFlow es objeto de otra libreta. Por el momento concentremonos en como hacer una red neuronal rápido y sin dolor.
# 
# Sin embargo, vamos a ver unos cuantos datos que nos pueden ser de útilidad para la construcción de la red neuronal.

# In[ ]:


print("Forma del ndarray con las imágenes: {}".format(mnist.train.images.shape))
print("Forma del ndarray con las etiquetas: {}".format(mnist.train.labels.shape))
print("-" * 79)
print("Número de imagenes de entrenamiento: {}".format(mnist.train.images.shape[0]))
print("Tamaño de las imagenes: {}".format(mnist.train.images.shape[1]))
print("Clases diferentes: {}".format(mnist.train.labels.shape[1]))


# ## 2. Construcción de la red neuronal

# Para hacer una red neuronal lo más genérica posible y que pdamos reutilizar en otros proyectos, vamos a establecer los parámetros base independientemente de la inicialización de la red, independientemente de la forma en que construimos la red. 
# 
# Comencemos por establecer una función genérica que nos forme una red neuronal con dos capas ocultas. No agrego más comentarios porque, con la experiencia de las libretas anteriores, la construcción de la red neuronal se explica sola.

# In[ ]:


def red_neuronal_dos_capas_ocultas(x, pesos, sesgos):
    """
    Genera una red neuronal de dos capas para usar en TensorFlow
    
    Parámetros
    ----------
    pesos: un diccionario con tres etiquetas: 'h1', 'h2' y 'ho'
           en donde cada una es una tf.Variable conteniendo una 
           matriz de dimensión [num_neuronas_capa_anterior, num_neuronas_capa]
                  
    sesgos: un diccionario con tres etiquetas: 'b1', 'b2' y 'bo'
            en donde cada una es una tf.Variable conteniendo un
            vector de dimensión [numero_de_neuronas_capa]
                   
    Devuelve
    --------
    Un ops de tensorflow que calcula la salida de una red neuronal
    con dos capas ocultas, y activaciones RELU.
    
    """
    # Primera capa oculta con activación ReLU
    capa_1 = tf.matmul(x, pesos['h1'])
    capa_1 = tf.add(capa_1, sesgos['b1'])
    capa_1 = tf.nn.relu(capa_1)
    
    # Segunda capa oculta con activación ReLU
    capa_2 = tf.matmul(capa_1, pesos['h2'])
    capa_2 = tf.add(capa_2, sesgos['b2'])
    capa_2 = tf.nn.relu(capa_2)
    
    # Capa de salida con activación lineal
    # En Tensorflow la salida es siempre lineal, y luego se especifica
    # la función de salida a la hora de calcularla como vamos a ver 
    # más adelante
    capa_salida = tf.matmul(capa_2, pesos['ho']) + sesgos['bo']
    return capa_salida


# Y ahora necesitamos poder generar los datos de entrada a la red neuronal de
# alguna manera posible. Afortunadamente sabemos exactamente que necesitaos, así
# que vamos a hacer una función que nos genere las variables de peso y sesgo.
# 
# Por el momento, y muy a la brava, solo vamos a generarlas con números aletorios con una 
# distribución $\mathcal{N}(0, 1)$.

# In[ ]:


def inicializa_pesos(entradas, n1, n2, salidas):
    """
    Genera un diccionario con pesos  
    para ser utilizado en la función red_neuronal_dos_capas_ocultas
    
    Parámetros
    ----------
    entradas: Número de neuronas en la capa de entrada
    
    n1: Número de neuronas en la primer capa oculta
    
    n2: Número de neuronas en la segunda capa oculta
    
    salidas: Número de neuronas de salida
    
    Devuelve
    --------
    Dos diccionarios, uno con los pesos por capa y otro con los sesgos por capa
    
    """
    pesos = {
        'h1': tf.Variable(tf.random_normal([entradas, n1])),
        'h2': tf.Variable(tf.random_normal([n1, n2])),
        'ho': tf.Variable(tf.random_normal([n2, salidas]))
    }
    
    sesgos = {
        'b1': tf.Variable(tf.random_normal([n1])),
        'b2': tf.Variable(tf.random_normal([n2])),
        'bo': tf.Variable(tf.random_normal([salidas]))
    }
    
    return pesos, sesgos


# Ahora necesitamos establecer los parámetros de la topología de la red neuronal. 
# Tomemos en cuenta que estos prámetros los podríamos haber establecido desde
# la primer celda, si el fin es estar variando los parámetros para escoger los que 
# ofrezcan mejor desempeño.

# In[ ]:


num_entradas = 784  #  Lo sabemos por la inspección que hicimos a mnist
num_salidas = 10    # Ídem

# Aqui es donde podemos jugar
num_neuronas_capa_1 = 256
num_neuronas_capa_2 = 256


# ¡A construir la red! Para esto vamos a necesitar crear las entradas
# con un placeholder, y crear nuestra topología de red neuronal.
# 
# Observa que la dimensión de x será [None, num_entradas], lo que significa que 
# la cantidad de renglones es desconocida (o variable).

# In[ ]:


# La entrada a la red neuronal
x = tf.placeholder("float", [None, num_entradas])

# Los pesos y los sesgos
w, b = inicializa_pesos(num_entradas, num_neuronas_capa_1, num_neuronas_capa_2, num_salidas)

# Crea la red neuronal
estimado = red_neuronal_dos_capas_ocultas(x, w, b)


# Parecería que ya está todo listo. Sin ambargo falta algo muy importante: No hemos explicado
# ni cual es el criterio de error (loss) que vamos a utilizar, ni cual va a ser el método de
# optimización (aprendizaje) que hemos decidido aplicar.
# 
# Primero definamos el costo que queremos minimizar, y ese costo va a estar en función de lo
# estimado con lo real, por lo que necesitamos otra entrada de datos para los datos de salida.
# 
# Sin ningun lugar a dudas, el costo que mejor describe este problema es el de *softmax*
# 

# In[ ]:


#  Creamos la variable de datos de salida conocidos
y = tf.placeholder("float", [None, num_salidas])

#  Definimos la función de costo
costo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=estimado, labels=y))


# Y ahora definimos que función de aprendizaje vamos a utilizar. Existen muchas funciones
# de aprendizaje en tensorflow, las cuales se pueden consultar en `tf.train.`. Entre las
# existentes podemos ver algunas conocidas del curso como descenso de gradiente simple,
# momento, rprop, rmsprop entre otras. Casi todas las funciones de optimización (aprendizaje)
# acaban su nombre con `Optimize`.
# 
# En este caso vamos a usar un método comocido como el *algoritmo de Adam* el cual 
# se puede consultar [aqui](http://arxiv.org/pdf/1412.6980.pdf). El metodo utiliza dos calculos
# de momentos diferentes, y por lo visto genera resultados muy interesantes desde el punto 
# de vista práctico.
# 
# ¿Cual es el mejor método? Pues esto es en función de tu problema y de la cantidad de datos que tengas.
# Lo mejor es practicar con varios métodos para entender sus ventajas y desventajas.
# 
# En todo caso el método de optimización requiere que se le inicialice con una tasa de aprendizaje.

# In[ ]:


alfa = 0.001
optimizador = tf.train.AdamOptimizer(learning_rate=alfa)
paso_entrenamiento = optimizador.minimize(costo)


# ## 3. Ejecutar la sesión usando mini-batches
# 
# Ahora, ya que la red neuronal está lista vamos a ejecutar la red utilizando el algoritmo de
# Adam pero en forma de mini-batches. Con el fin de tener control sobre el problema, vamos a establecer un número máximo de epoch (ciclos de aprendizaje), el tamaño de los mini-batches, y cada cuandos epoch 
# quisieramos ver como está evolucionando la red neuronal.
# 
# Como entrenar una red neuronal no tiene sentido, si no es porque la queremos usar para reconocer,
# no tendría sentido entrenarla y luego perderla y tener que reentrenar en cada ocasión. Recuerda que cuando
# se cierra la sesión se borra todo lo que se tenía en memoria. 
# 
# Para esto vamos a usar una ops especial llamada `Saver`, que permite guardar en un archivo la red neuronal y 
# después utilizarla en otra sesión (en otro script, computadora, ....).
# 

# In[ ]:


archivo_modelo = "rnn2.ckpt"
saver = tf.train.Saver()


# Como todo se ejecuta dentro de una sesión, no es posible hacerlo por partes (si usamos el 
# `with` que debería ser la única forma en la que iniciaramos una sesión). Por lo tanto procuraré dejar comentado el código.

# In[ ]:


numero_epochs = 30
tamano_minibatch = 100
display_step = 1

# Muy importante la primera vez que se ejecuta inicializar todas las variables
init = tf.global_variables_initializer()

# La manera correcta de iniciar una sesión y realizar calculos
with tf.Session() as sess:
    sess.run(init)

    # Ciclos de entrenamiento
    for epoch in range(numero_epochs):

        #  Inicializa el costo promedio de todos los minibatches en 0
        avg_cost = 0.
        
        #  Calcula el número de minibatches que se pueden usar 
        total_batch = int(mnist.train.num_examples/tamano_minibatch)

        #  Por cada minibatch
        for i in range(total_batch):
            
            #  Utiliza un generador incluido en mnist que obtiene 
            #  tamano_minibatch ejemplos selecionados aleatoriamente del total
            batch_x, batch_y = mnist.train.next_batch(tamano_minibatch)
            
            #  Ejecuta la ops del paso_entrenamiento para aprender 
            #  y la del costo, con el fin de mostrar el aprendizaje
            _, c = sess.run([paso_entrenamiento, costo], feed_dict={x: batch_x, y: batch_y})
            
            #  Calcula el costo del minibatch y lo agrega al costo total
            avg_cost += c / total_batch
        
        # Muestra los resultados
        if epoch % display_step == 0:
            print (("Epoch: " + str(epoch)).ljust(20)
                   + ("Costo: " + str(avg_cost)))
    
    #  Guarda la sesión en el archivo rnn2.cptk
    saver.save(sess, archivo_modelo)
    
    print("Se acabaron los epochs, saliendo de la sesión de tensorflow.")


# Ahora vamos a revisar que tan bien realizó el aprendizaje cuando se aplica la red adatos que
# no se usaron para entrenamiento. Para esto vamos a utilizar dos ops extas: una 
# para definir la operaración de datos bien estimados o mal estimados, y otra para
# calcular el promedio de datos bien estimados. Para calcular los datos bien estimados vamos a utilizar `tf.cast` que permite ajustar los tipos
# al tipo tensor. 
# 
# Recuerda que, al haber definido `prediccion_correcta` y `precision` como ops de tensorflow *después* de la ultima vez que usaste dentro de una sesión la ops `saver`, estas operaciones no se guardaron en memoria. Si quieres conservarlas, debes de definirlas *antes* de ejecutar el aprendizaje.

# In[ ]:


prediction_correcta = tf.equal(tf.argmax(estimado, 1), tf.argmax(y, 1))

precision = tf.reduce_mean(tf.cast(prediction_correcta, "float"))


# Ahora si, vamos a abrir una nueva sesión, vamos a restaurar los valores de la sesión anterior,
# y vamos a ejecutar el grafo con el fin de evaluar la ops precision, pero ahora con el
# diccionario de alimentación con los datos de prueba.

# In[ ]:


with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, archivo_modelo)
    porcentaje_acierto = sess.run(precision, feed_dict={x: mnist.test.images,
                                                        y: mnist.test.labels})
    print("Precisión: {}".format(porcentaje_acierto))


# ## 4. Contesta las siguientes preguntas
# 
# 1. ¿Que pasa si aumenta el número de epochs? ¿Cuando deja de ser util aumentar los epoch?
# 
# 2. ¿Que pasa si aumentas o disminuyes la tasa de aprendizaje?
# 
# 3. Utiliza al menos otros 2 métodos de optimización (existentes en Tensorflow), ajústalos y compáralos. ¿Cual de los métodos te gusta más y porqué preferirías uno sobre los otros?
# 
# 4. ¿Que pasa si cambias el tamaño de los minibatches?
# 
# 5. ¿Como harías si dejaste a medias un proceso de aprendizaje (en 10 epochs por ejemplo) y quisieras entrenar la red 10 epoch más, y mañana quisieras entrenarla otros 10 epoch más?
# 
# **Para contestar las preguntas, agrega cuantas celdas con comentarios y con códgo sean necesarias.** Aprovecha que las libretas de *Jupyter* te permite hacerte una especie de tutorial personalizado.
