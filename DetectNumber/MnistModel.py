import tensorflow as tf
import tfcoreml

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.save('NumberDetectorModel.h5')

input_name = model.inputs[0].name.split(':')[0]
output_node_name = model.outputs[0].name.split(':')[0]
graph_output_node_name = output_node_name.split('/')[-1]

model = tfcoreml.convert(tf_model_path='NumberDetectorModel.h5',
                         input_name_shape_dict={
                             input_name: (1, 28, 28)},
                         output_feature_names=[
                             graph_output_node_name],
                         minimum_ios_deployment_target='13')

model.save('NumberDetectorModel.mlmodel')
