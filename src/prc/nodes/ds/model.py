"""Modeling related files."""
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def build_embedding_dic(df, embedding_cols):
    """Build embedding mapping dict."""
    embed_lookup = {}
    for embed_col in embedding_cols:
        embed_dic = {val: idx for idx, val in enumerate(df[embed_col].unique())}
        embed_lookup[embed_col] = embed_dic

    x_embed_raw = []
    for embed_col in embedding_cols:
        x_embed_raw.append(
            df[embed_col].map(lambda x: embed_lookup[embed_col][x]).values
        )
    # x_embed_raw = np.array(x_embed_raw)
    return embed_lookup, x_embed_raw


def build_model(num_dimensions, num_samples, embed_lookup, lr=0.001):
    """Based model."""
    input_layer = tf.keras.layers.Input(shape=num_dimensions)
    # add embedding layer for neighbourhood
    neighbourhood_group_input = tf.keras.layers.Input(shape=1)
    neighbourhood_group_embed_layer = tf.keras.layers.Embedding(
        input_dim=len(embed_lookup["neighbourhood_group"]), output_dim=5
    )(neighbourhood_group_input)
    neighbourhood_flatten = tf.keras.layers.Flatten()(neighbourhood_group_embed_layer)

    # add embedding layer for room
    room_type_input = tf.keras.layers.Input(shape=1)
    room_type_embed_layer = tf.keras.layers.Embedding(
        input_dim=len(embed_lookup["room_type"]), output_dim=5
    )(room_type_input)
    room_type_flatten = tf.keras.layers.Flatten()(room_type_embed_layer)

    # add embedding layer for month
    month_input = tf.keras.layers.Input(shape=1)
    month_embed_layer = tf.keras.layers.Embedding(
        input_dim=len(embed_lookup["month"]), output_dim=12
    )(month_input)
    month_flattern = tf.keras.layers.Flatten()(month_embed_layer)

    concat_layer = tf.keras.layers.concatenate(
        [input_layer, neighbourhood_flatten, room_type_flatten, month_flattern]
    )

    def kl_divergence_function(q, p, _):
        """Helper func."""
        return tfd.kl_divergence(q, p) / tf.cast(num_samples, dtype=tf.float32)

    dense_layer = tfp.layers.DenseFlipout(
        units=1,
        activation="sigmoid",
        kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
        bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(),
        kernel_divergence_fn=kl_divergence_function,
    )(concat_layer)

    model = tf.keras.Model(
        inputs=[input_layer, neighbourhood_group_input, room_type_input, month_input],
        outputs=dense_layer,
    )

    optimizer = tf.keras.optimizers.Adam(lr=lr)
    model.compile(optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model
