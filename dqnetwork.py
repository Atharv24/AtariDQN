from tensorflow import keras
import tensorflow as tf

def make_dqnet():
    net = keras.Sequential()
    net.add(keras.layers.Conv2D(32, kernel_size=8, strides=4, activation='relu', input_shape=[84, 84, 4]))
    net.add(keras.layers.BatchNormalization())

    net.add(keras.layers.Conv2D(64, kernel_size=4, strides=2, activation='relu'))
    net.add(keras.layers.BatchNormalization())

    net.add(keras.layers.Conv2D(64, kernel_size=4, strides=1, activation='relu'))
    net.add(keras.layers.BatchNormalization())

    net.add(keras.layers.Flatten())
    net.add(keras.layers.Dense(512, activation='relu'))
    net.add(keras.layers.Dense(6, activation='softmax'))

    return net

opt = tf.keras.optimizers.Adam(5e-5)

@tf.function
def optimize(moving_net, target_net, minibatch, gamma):
    states, actions, rewards, next_states = minibatch
    next_state_action_values = keras.backend.max(target_net(next_states, training=True), axis=-1)
    expected_state_action_values = rewards + gamma*next_state_action_values

    with tf.GradientTape() as tape:
        state_action_values = moving_net(states, training=True)
        state_action_values_used = tf.squeeze(tf.gather_nd(state_action_values,tf.stack([tf.range(actions.shape[0])[...,tf.newaxis], actions[...,tf.newaxis]], axis=2)))
        loss = tf.keras.losses.MSE(state_action_values_used, expected_state_action_values)
    
    grads = tape.gradient(loss, moving_net.trainable_variables)
    opt.apply_gradients(zip(grads, moving_net.trainable_variables))

    return loss
