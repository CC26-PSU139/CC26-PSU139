import tensorflow as tf
from tensorflow.keras.layers import Layer
import pickle
import numpy as np
import pandas as pd
import re


# ── Definisi ulang semua custom class (wajib sebelum load_model) ──────────────
try:
    register_serializable = tf.keras.saving.register_keras_serializable
except AttributeError:
    register_serializable = tf.keras.utils.register_keras_serializable

@register_serializable()
class MyCustomDense(Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units      = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.W = self.add_weight(name='W', shape=(input_shape[-1], self.units), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='b', shape=(self.units,), initializer='zeros', trainable=True)
        super().build(input_shape)

    def call(self, x, training=None):
        out = tf.matmul(x, self.W) + self.b
        return self.activation(out) if self.activation else out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'units': self.units,
                    'activation': tf.keras.activations.serialize(self.activation)})
        return cfg


@register_serializable()
class MyCustomLeakyReLU(Layer):
    def __init__(self, alpha=0.01, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    def call(self, x, training=None):
        return tf.nn.leaky_relu(x, alpha=self.alpha)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'alpha': self.alpha})
        return cfg


@register_serializable()
class MyCustomDropout(Layer):
    def __init__(self, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate

    def call(self, x, training=None):
        if training:
            return tf.nn.dropout(x, rate=self.rate)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'rate': self.rate})
        return cfg


@register_serializable()
class MyCustomAttention(Layer):
    def __init__(self, num_heads=4, key_dim=32, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim   = key_dim
        self.drop_rate = dropout
        self.mha  = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=key_dim, dropout=dropout)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=None):
        attn = self.mha(x, x, x, training=training)
        return self.norm(x + attn)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'num_heads': self.num_heads, 'key_dim': self.key_dim, 'dropout': self.drop_rate})
        return cfg


@register_serializable()
class MyTransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.drop_rate = dropout
        self.attention  = MyCustomAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout)
        self.ff1        = MyCustomDense(ff_dim)
        self.act        = MyCustomLeakyReLU(alpha=0.01)
        self.ff2        = MyCustomDense(embed_dim)
        self.dropout_ff = MyCustomDropout(rate=dropout)
        self.norm       = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=None):
        x   = self.attention(x, training=training)
        ffn = self.ff1(x)
        ffn = self.act(ffn)
        ffn = self.ff2(ffn)
        ffn = self.dropout_ff(ffn, training=training)
        return self.norm(x + ffn)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads, 'ff_dim': self.ff_dim, 'dropout': self.drop_rate})
        return cfg


@register_serializable()
class ContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, margin=1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        y_true   = tf.cast(y_true, tf.float32)
        pos_loss = y_true * tf.square(y_pred)
        neg_loss = (1.0 - y_true) * tf.square(tf.maximum(self.margin - y_pred, 0.0))
        return tf.reduce_mean(pos_loss + neg_loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'margin': self.margin})
        return cfg


@register_serializable()
class CareerPathAIModel(tf.keras.Model):
    def __init__(self, vocab_size_cv=20000, vocab_size_job=20000, embed_dim=128, num_heads=4, ff_dim=256, num_transformer_blocks=2, proj_dim=64, dropout=0.1, **kwargs):
        super().__init__(**kwargs)

        self.embed_dim              = embed_dim
        self.num_transformer_blocks = num_transformer_blocks

        # ── CV Tower ──────────────────────────────────────────────────────────
        self.cv_embedding = tf.keras.layers.Embedding(
            vocab_size_cv, embed_dim, mask_zero=True, name='cv_embedding')
        self.cv_pos_dropout = MyCustomDropout(rate=dropout, name='cv_pos_dropout')
        self.cv_transformer_blocks = [
            MyTransformerBlock(embed_dim, num_heads, ff_dim, dropout, name=f'cv_transformer_{i}')
            for i in range(num_transformer_blocks)
        ]
        self.cv_proj = MyCustomDense(proj_dim, name='cv_proj')
        self.cv_act  = MyCustomLeakyReLU(alpha=0.01, name='cv_act')
        self.cv_drop = MyCustomDropout(rate=dropout, name='cv_drop')

        # ── Job Tower ─────────────────────────────────────────────────────────
        self.job_embedding = tf.keras.layers.Embedding(
            vocab_size_job, embed_dim, mask_zero=True, name='job_embedding')
        self.job_pos_dropout = MyCustomDropout(rate=dropout, name='job_pos_dropout')
        self.job_transformer_blocks = [
            MyTransformerBlock(embed_dim, num_heads, ff_dim, dropout, name=f'job_transformer_{i}')
            for i in range(num_transformer_blocks)
        ]
        self.job_proj = MyCustomDense(proj_dim, name='job_proj')
        self.job_act  = MyCustomLeakyReLU(alpha=0.01, name='job_act')
        self.job_drop = MyCustomDropout(rate=dropout, name='job_drop')

    def _encode_cv(self, token_ids, training=None):
        x = self.cv_embedding(token_ids)
        x = self.cv_pos_dropout(x, training=training)
        for block in self.cv_transformer_blocks:
            x = block(x, training=training)
        x = tf.reduce_mean(x, axis=1)
        x = self.cv_proj(x)
        x = self.cv_act(x)
        x = self.cv_drop(x, training=training)
        return tf.math.l2_normalize(x, axis=-1)

    def _encode_job(self, token_ids, training=None):
        x = self.job_embedding(token_ids)
        x = self.job_pos_dropout(x, training=training)
        for block in self.job_transformer_blocks:
            x = block(x, training=training)
        x = tf.reduce_mean(x, axis=1)
        x = self.job_proj(x)
        x = self.job_act(x)
        x = self.job_drop(x, training=training)
        return tf.math.l2_normalize(x, axis=-1)

    def call(self, inputs, training=None):
        cv_ids, job_ids = inputs
        cv_vec  = self._encode_cv(cv_ids,  training=training)
        job_vec = self._encode_job(job_ids, training=training)
        return tf.norm(cv_vec - job_vec, axis=-1)

    def get_similarity(self, cv_ids, job_ids):
        cv_vec  = self._encode_cv(cv_ids,  training=False)
        job_vec = self._encode_job(job_ids, training=False)
        return tf.reduce_sum(cv_vec * job_vec, axis=-1)


# ── Load assets (dipanggil sekali saat startup) ───────────────────────────────

def load_assets():
    model = CareerPathAIModel(
        vocab_size_cv          = 20000,
        vocab_size_job         = 20000,
        embed_dim              = 128,
        num_heads              = 4,
        ff_dim                 = 256,
        num_transformer_blocks = 3,
        proj_dim               = 64,
        dropout                = 0.1,
    )

    dummy_cv  = tf.zeros((1, 256), dtype=tf.int64)
    dummy_job = tf.zeros((1, 512), dtype=tf.int64)
    _         = model([dummy_cv, dummy_job], training=False)

    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore('models/career_ckpt').expect_partial()

    with open('models/vectorizer_config.pkl', 'rb') as f:
        vocab_config = pickle.load(f)

    vec_cv = tf.keras.layers.TextVectorization(
        max_tokens=vocab_config['max_tokens_cv'],
        output_sequence_length=vocab_config['seq_len_cv'])
    vec_cv.set_vocabulary(vocab_config['vocab_cv'])

    vec_job = tf.keras.layers.TextVectorization(
        max_tokens=vocab_config['max_tokens_job'],
        output_sequence_length=vocab_config['seq_len_job'])
    vec_job.set_vocabulary(vocab_config['vocab_job'])

    df_jobs = pd.read_csv('data/job_postings_final_cleaned.csv')

    # ── Pre-compute semua job vectors saat startup ────────────────────────────
    # Dilakukan sekali di sini, bukan setiap request
    print("Pre-computing job vectors...")
    job_texts        = df_jobs['cleaned_description'].fillna('').tolist()
    BATCH_SIZE       = 64
    job_vectors_list = []

    for i in range(0, len(job_texts), BATCH_SIZE):
        batch     = job_texts[i:i + BATCH_SIZE]
        batch_vec = vec_job(tf.constant(batch))
        encoded   = model._encode_job(batch_vec, training=False)
        job_vectors_list.append(encoded.numpy())

    job_vectors = np.vstack(job_vectors_list)  # shape: (total_jobs, proj_dim)
    print(f"Job vectors ready: {job_vectors.shape}")

    return model, vec_cv, vec_job, df_jobs, job_vectors


# ── Preprocessing — identik dengan saat training ─────────────────────────────

def preprocess_cv(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Fungsi predict utama — batch cosine similarity ───────────────────────────

def predict_top_jobs(cv_text: str, model, vec_cv, job_vectors, df_jobs, top_k=10):
    clean_cv = preprocess_cv(cv_text)
    cv_vec   = vec_cv(tf.constant([clean_cv]))                     # (1, seq_len_cv)
    cv_enc   = model._encode_cv(cv_vec, training=False).numpy()    # (1, proj_dim)

    # Cosine similarity antara 1 CV vector dan semua job vectors sekaligus
    similarities = np.dot(job_vectors, cv_enc[0])                  # (total_jobs,)

    # Ambil top_k index
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        row = df_jobs.iloc[idx]
        results.append({
            'job_id'           : str(row.get('job_id', '')),
            'title'            : str(row.get('title', '')),
            'location'         : str(row.get('location', '')),
            'experience_level' : str(row.get('formatted_experience_level', '')),
            'work_type'        : str(row.get('formatted_work_type', '')),
            'job_posting_url'  : str(row.get('job_posting_url', '')),
            'application_url'  : str(row.get('application_url', '')),
            'similarity_score' : round(float(similarities[idx]), 4),
        })

    return results