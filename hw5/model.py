import numpy as np
from keras.layers import Input, Embedding, Dot, Add, Flatten
from keras.layers import Reshape, Merge, Dropout, Dense
from keras.models import Model, Sequential

def mf_build(n_users, m_items, k_factors, bias=False):
    n_input = Input(shape=[1])
    n_embed = Embedding(n_users, k_factors, input_length=1)(n_input)
    n_flat = Flatten()(n_embed)
    
    m_input = Input(shape=[1])
    m_embed = Embedding(m_items, k_factors, input_length=1)(m_input)
    m_flat = Flatten()(m_embed)

    dot = Dot(-1)([n_flat, m_flat])
    if bias:
        user_bias = Embedding(n_users, 1, input_length=1, embeddings_initializer='zeros', trainable=True)(n_input)
        user_bias = Flatten()(user_bias)
        item_bias = Embedding(m_items, 1, input_length=1, embeddings_initializer='zeros', trainable=True)(m_input)
        item_bias = Flatten()(item_bias)
        out = Add()([dot, user_bias, item_bias])
    else:
        out = dot

    model = Model([n_input, m_input], out)
    model.summary()
    return model


class CFModel(Sequential):

    def __init__(self, n_users, m_items, k_factors, **kwargs):
        P = Sequential()
        P.add(Embedding(n_users, k_factors, input_length=1))
        P.add(Flatten())
        Q = Sequential()
        Q.add(Embedding(m_items, k_factors, input_length=1))
        Q.add(Flatten())
        super().__init__(**kwargs)
        #self.add(Dot(1))([P, Q])
        self.add(Merge([P, Q], mode='dot', dot_axes=1))

    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0][0]

class DeepModel(Sequential):

    def __init__(self, n_users, m_items, k_factors, p_dropout=0.1, **kwargs):
        P = Sequential()
        P.add(Embedding(n_users, k_factors, input_length=1))
        P.add(Reshape((k_factors,)))
        Q = Sequential()
        Q.add(Embedding(m_items, k_factors, input_length=1))
        Q.add(Reshape((k_factors,)))
        super(DeepModel, self).__init__(**kwargs)
        self.add(Merge([P, Q], mode='concat'))
        self.add(Dropout(p_dropout))
        self.add(Dense(k_factors, activation='relu'))
        self.add(Dropout(p_dropout))
        self.add(Dense(1, activation='linear'))

    def rate(self, user_id, item_id):
        return self.predict([np.array([user_id]), np.array([item_id])])[0][0]
