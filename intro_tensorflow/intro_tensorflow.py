#!/usr/bin/env python
# coding: utf-8

# <img src="imagenes/rn3.png" width="200">
# <img src="http://www.identidadbuho.uson.mx/assets/letragrama-rgb-150.jpg" width="200">

# 
# # [Curso de Redes Neuronales](https://curso-redes-neuronales-unison.github.io/Temario/)
# 
# # Operaciones básicas en TensorFlow
# 
# [**Julio Waissman Vilanova**](http://mat.uson.mx/~juliowaissman/), 27 de septiembre de 2017.
# 
# 
# 
# TensorFlow es un sistema de programación para representar calculos como grafos. Los nodos en el grafo son llamados *ops* (por operaciones). Una op toma de cero a muchos tensores, realiza algunos cálculos y produce cero o muchos tensores. Un tensor es un arreglo multidimensional con tipo fijo.
# 
# Así, un grafo de TensorFlow no es más que una descripción de cálculos. El grafo que describe los calculos se realiza en una fase llamada de *construcción del grafo*. Pero para calcular cualquier cosa, es necesario ejecutar el grafo en una *Session*. Una Session coloca las ops del grafo en los *Devices* (como CPUs, GPUs), y provée métodos para ejecutarlos. Los métodos devuelven los tensores producidos por las ops como un `ndarray` de *numpy* en *Python* (o como un objeto de la clase `tensorflow::Tensor` en *C++*).  
# 
# Para una revisión completa de tensorflow, recomiendo tener como referencia genera la página del [white paper on TensorFlow](https://github.com/samjabrahams/tensorflow-white-paper-notes), la cual contiene una explicación muy completa sobre la forma en que opera TensorFlow.
# 
# Para una serie de ejemplos y proyectos realizados en TensorFlow, se puede revisar el proyecto de *GitHub* llamado [Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow).
# 
# 
# ## Anatomia de un grafo extremadamente simple
# 

# In[ ]:


import tensorflow as tf


# Tensoflow tiene un grafo por default en el cual las ops generadas se agregan como nodos. Este grafo por default es suficiente para la mayoría de las aplicaciones y no esperamos utilizar otro. 
# 
# Lo que sigue no es necesario realizarlo, solo lo haremos con el fin de explorar como funciona TensorFlow:
# 

# In[ ]:


graph = tf.get_default_graph()
graph.get_operations()


# Como vemos, en el grafo por default no se encuentra ningun nodo todavía. Vamos a agregar el nodo más sencillo posible: Una constante escalar.

# In[ ]:


entrada = tf.constant(1.0)


# Y ahora vamos a ver que se agregó en el grafo

# In[ ]:


ops = graph.get_operations()
print("Operaciones en el grafo por default de Tensorflow:")
print(ops)
print("Definición de la primer operación")
print(ops[0].node_def)


# TensorFlow utiliza un protocol interno basado en JSON. ¿Porqué hacer una versión propia de
# la definición de cada cosa y no usar la que existe en *Python*? ¿Porqué no usar las variables que provée *Python* o *Numpy*?
# 
# 
# > To do efficient numerical computing in Python, we typically use libraries like NumPy that do expensive operations such as matrix multiplication outside Python, using highly efficient code implemented in another language. Unfortunately, there can still be a lot of overhead from switching back to Python every operation. This overhead is especially bad if you want to run computations on GPUs or in a distributed manner, where there can be a high cost to transferring data.
# 
# > TensorFlow also does its heavy lifting outside Python, but it takes things a step further to avoid this overhead. Instead of running a single expensive operation independently from Python, TensorFlow lets us describe a graph of interacting operations that run entirely outside Python. This approach is similar to that used in Theano or Torch.
# 
# Tensorflow no hace nada que no le indiques explicitamente, así sea asignar una constante. Más aun:

# In[ ]:


entrada


# Podemos ver la definición y el tipo, pero no sabemos el valor. Para esto hay que ejecutar una Session.

# In[ ]:


sess = tf.Session()
print(sess.run(entrada))
sess.close()


# Al principio puede resultar cansado, ... y con el tiempo sigue siendo cansado, pero tiene su razón de ser cuando se trabaja con grandes volumenes de datos o grandes volumenes de operaciones.
# 
# Ahora agreguemos una variable y veamos que pasa en el grafo.

# In[ ]:


x = tf.Variable(0.8)

print("Operaciones en el grafo:")
for op in graph.get_operations(): 
    print(op.name)
print(op.node_def)


# ¡Una variable agrega 4 operaciones al grafo, no solo una!
# 
# Vamos ahora a agregar una operacion entre la constante y la variable.

# In[ ]:


y = tf.multiply(entrada, x)

print("Operaciones en el grafo:")
for op in graph.get_operations(): 
    print(op.name)

print("\nEntradas para la multiplicación")
for op_input in op.inputs: 
    print(op_input)
    


# Y como vemos, la multiplicación reconoce cuales ops hay que realizar para poder calcular a su vez esta op. Por supuesto que revisar un grafo de esta manera solo es posible si es así de sencillo, y para eso no requeriríamos usar TensorFlow. Más adelante lo veremos.
# 
# Para calcular esto necesitamos asegurar que las variables se encuentran correctamente inicializadas (en este caso es solo una, pero podría haber muchas). Por esto, antes de ejecutar la Session, es necesario inicializar *todas* las variables.

# In[ ]:


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y))

sess.close()


# Lo que es, de acuerdo a una multiplicacion de flotantes de 32 bits el resultado de 1.0 * 8.0. Una sesión debe siempre de cerrarse para liberar los recursos.
# 
# Ahora, si construimos un grafo y queremos estar seguros que lo que hicimos es lo que queremos, es conveniente poder observar el grafo de mejor manera. Para eso vamos a utilizar *TensorBoard*. 
# 
# Reiniciemos el grafo de mejor manera.

# In[ ]:


tf.reset_default_graph()
sess = tf.Session()

x = tf.constant(1.0, name='entrada')
w = tf.Variable(0.8, name='peso')
y = tf.multiply(w, x, name='salida')


# Y ahora guardemos este grafo en un conjunto de archivos dentro del directorio `ejemplo_simple` (si el directorio no existe, lo crea el comando).

# In[ ]:


summary_writer = tf.summary.FileWriter('ejemplo_simple', sess.graph)
sess.close()


# El grafo lo podemos visualizar ejecutando en la terminal
# 
# ```
# tensorboard --logdir=ejemplo_simple
# ```
# 
# Y buscarlo en el navegador web en la dirección `localhost:6006`.

# In[ ]:


get_ipython().system(' tensorboard --logdir=ejemplo_simple')


# ### Ejercicio: Explica que pasa aqui.

# ## Construcción de un grafo más elaborado

# In[ ]:


tf.reset_default_graph()

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product1 = tf.matmul(matrix1, matrix2)
product2 = tf.matmul(matrix2, matrix1)


# ## Ejecutando la sesion

# In[ ]:


with tf.Session() as sess:
    # To run the matmul op we call the session 'run()' method, passing 'product'
    # which represents the output of the matmul op.  This indicates to the call
    # that we want to get the output of the matmul op back.
    #
    # All inputs needed by the op are run automatically by the session.  They
    # typically are run in parallel.
    result = sess.run([product1, product2])
    print("product1 =  \n{}".format(result[0]))
    print("product2 = \n{}".format(result[1]))


# ## Uso interactivo
# 
# Con el fin de utilizar Tensorflow dentro de un entorno ipython con el fin de realizar prototipos, el modulo viene con clases que pueden ser utilizadas dentro del REPL.

# In[ ]:


# Enter an interactive TensorFlow Session.
sess = tf.InteractiveSession()

x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])

# Initialize 'x' using the run() method of its initializer op.
x.initializer.run()

# Add an op to subtract 'a' from 'x'.  Run it and print the result
sub = tf.subtract(x, a)
print(sub.eval())
# ==> [-2. -1.]

# Close the Session when we're done.
sess.close()


# ## Manejo de variables y su actualización
# 
# Las variables mantienen su valor durante la ejecución, y únicamente cambian su valor a través de ops bien establecidas, cuyo fin es modificar el valor de las variables. La ops más directa es `assign` que se usa como se muestra a continuación.

# In[ ]:


# Create a Variable, that will be initialized to the scalar value 0.
state = tf.Variable(0, name="counter")

# Create an Op to add one to `state`.

one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# Variables must be initialized by running an `init` Op after having
# launched the graph.  We first have to add the `init` Op to the graph.
init_op = tf.global_variables_initializer()

# Launch the graph and run the ops.
with tf.Session() as sess:
  # Run the 'init' op
  sess.run(init_op)
  # Print the initial value of 'state'
  print(sess.run(state))
  # Run the op that updates 'state' and print 'state'.
  for _ in range(5):
    sess.run(update)
    print(sess.run(state))


# ## Alimentación (Feeds)
# 
# Hasta ahorita, en los ejemplos no utilizamos datos para las entradas, solamente valores constantes. 
# TensorFlow permite (por supuesto) el uso de valores de entrada que modifican un tensor directamente en el grafo de operaciones. 
# 
# Una alimentación (feed) reemplaza temporalmente la salida de una operación con el valor de un tensor. Por cada llamada a `run()` los datos se envían como argumento al grafo. La alimentación es usada *únicamente* en dicha llamada. La forma típica de agregar feeds a un grafo, es utilizando las operaciones asociadas a `tf.placeholder()` como se muestra en el ejemplo:
# 

# In[ ]:


input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)

with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))


# Si no se alimenta con datos un feed en una ejecución, TensorFlow genera un error.
