import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, LayerNormalization, MultiHeadAttention, Reshape
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow as tf
import matplotlib.pyplot as plt
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

#--------------------------------------------------------------------------------------------------------------------------------------------------

"""
Description-- Latest

"""

#--------------------------------------------------------------------------------------------------------------------------------------------------
# PREPROCESSING & ENCODING

# Load the training dataset
file_path = '/Users/sohamsane/Documents/Coding Projects/DualDynamicTransformer/Data/Model1_Full/Model1_Full.csv'
data = pd.read_csv(file_path)

# Drop rows with missing values for the output column
data = data.dropna(subset=['Price'])

# Identify relevant features and target
target_trim = 'Price'
input_columns = ['Brand', 'Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']  

# Split the training data into features and target
X = data[input_columns]
y_trim = data[target_trim]

# Define preprocessing for numerical and categorical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create a preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Fit and transform the features for the training dataset
X_preprocessed = preprocessor.fit_transform(X)

# Encode the target variable
trim_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
y_trim_encoded = trim_encoder.fit_transform(y_trim.values.reshape(-1, 1))

# Split the data into training and test sets
X_train, X_test, y_trim_train, y_trim_test = train_test_split(
    X_preprocessed, y_trim_encoded, test_size=0.2, random_state=42)

# Load the unseen test dataset and handle missing values by filling them with zero
file_path_unseen = '/Users/sohamsane/Documents/Coding Projects/DualDynamicTransformer/HP_Dell_laptops.csv'
unseen_data = pd.read_csv(file_path_unseen)
unseen_data.fillna(0, inplace=True)

# Transform the unseen test data with the same preprocessing pipeline
unseen_preprocessed = preprocessor.transform(unseen_data[input_columns])

# Ensure there are no NaNs in the input data
X_train[np.isnan(X_train)] = 0
X_test[np.isnan(X_test)] = 0

#--------------------------------------------------------------------------------------------------------------------------------------------------
# TRANSFORMER ARCHITECTURE 

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"), 
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)  # Assumes self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config

def build_transformer_model(input_shape, embed_dim, num_heads, ff_dim, num_transformer_blocks, dropout_rate):
    inputs = Input(shape=(input_shape,))
    x = Dense(embed_dim, kernel_regularizer=l2(0.05))(inputs)
    x = Reshape((1, embed_dim))(x)  # Ensures embedding dimension is properly set

    for _ in range(num_transformer_blocks):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)
    
    x = Flatten()(x)  # This should create a 2D tensor [batch_size, features]
    x = Dropout(dropout_rate)(x)  # Adds dropout layer

    num_classes = y_trim_encoded.shape[1]
    trim_output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))(x)  # Applies softmax directly to the 2D tensor

    model = Model(inputs=inputs, outputs=trim_output)
    return model

# Build the transformer model
input_shape = X_train.shape[1]
embed_dim = 64  # Embedding size for each token
num_heads = 4  # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer
num_transformer_blocks = 2  # Number of transformer blocks
dropout_rate = 0.1  # Increased dropout rate

model = build_transformer_model(input_shape, embed_dim, num_heads, ff_dim, num_transformer_blocks, dropout_rate)
model.summary()

#--------------------------------------------------------------------------------------------------------------------------------------------------
# TRANSFORMER TRAINING & BUILD

# Compile the model with the specified optimizer and loss functions
model.compile(
    optimizer=Adam(learning_rate=ExponentialDecay(
        initial_learning_rate=0.0025, 
        decay_steps=100000,
        decay_rate=0.98,
        staircase=True
    )), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)  
history = model.fit(X_train, y_trim_train, 
                    epochs=125, 
                    batch_size=64,  
                    validation_split=0.05,
                    callbacks=[early_stopping])

# Save the model
model.save('transformer_trim_model.keras')

# Load the saved model
loaded_model = tf.keras.models.load_model('transformer_trim_model.keras', custom_objects={'TransformerBlock': TransformerBlock})

#------------------------------------------------------------------------------------------------------------------------------------
# DEBUGGING OUTPUT 

# Get predictions
y_trim_pred = loaded_model.predict(X_test)

# Decode the trim predictions
y_trim_pred_decoded = trim_encoder.inverse_transform(y_trim_pred)

# Print the true and predicted trims for the first 10 inputs
true_trim_decoded = trim_encoder.inverse_transform(y_trim_test[:10])
print("True Price:", true_trim_decoded.flatten())
print("Predicted Price:", y_trim_pred_decoded[:10].flatten())

# Visualize the training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------
# TESTING UNSEEN DATA 

# Get predictions
y_trim_pred = loaded_model.predict(unseen_preprocessed)

# Decode the trim predictions
y_trim_pred_decoded = trim_encoder.inverse_transform(y_trim_pred)

# Extract true prices from the last column of the unseen data
true_prices = unseen_data.iloc[:, -1].values.reshape(-1, 1)

# Print the true and predicted trims for the first 10 inputs
print("UNSEEN DATA SET REGULAR TRANSFORMER PREDICTED PRICES: ")
print("True Price:", true_prices[:10])
print("Predicted Price:", y_trim_pred_decoded[:10].flatten())

#--------------------------------------------------------------------------------------------------------------------------------------------------
# SECOND MODEL BUILDING ADJUSTMENT FUNCTIONS 

# Step 1: Freeze the first transformer model
def freeze_model(model):
    for layer in model.layers:
        layer.trainable = False

# Function to calculate the total number of parameters in the first transformer
def calculate_total_params(model):
    total_params = 0
    for layer in model.layers:
        if isinstance(layer, layers.MultiHeadAttention):
            total_params += np.prod(layer.query_dense.kernel.shape)
            total_params += np.prod(layer.key_dense.kernel.shape)
            total_params += np.prod(layer.value_dense.kernel.shape)
            total_params += np.prod(layer.output_dense.kernel.shape)
    return total_params

# Function to apply weight adjustments
def apply_weight_adjustments(model, adjustments):
    offset = 0
    for layer in model.layers:
        if isinstance(layer, layers.MultiHeadAttention):
            q_shape = layer.query_dense.kernel.shape
            k_shape = layer.key_dense.kernel.shape
            v_shape = layer.value_dense.kernel.shape
            o_shape = layer.output_dense.kernel.shape

            num_params_q = np.prod(q_shape)
            num_params_k = np.prod(k_shape)
            num_params_v = np.prod(v_shape)
            num_params_o = np.prod(o_shape)

            adjustment_q = adjustments[:, offset:offset + num_params_q]
            offset += num_params_q
            adjustment_k = adjustments[:, offset:offset + num_params_k]
            offset += num_params_k
            adjustment_v = adjustments[:, offset:offset + num_params_v]
            offset += num_params_v
            adjustment_o = adjustments[:, offset:offset + num_params_o]
            offset += num_params_o

            layer.query_dense.kernel.assign_add(tf.reshape(adjustment_q, q_shape))
            layer.key_dense.kernel.assign_add(tf.reshape(adjustment_k, k_shape))
            layer.value_dense.kernel.assign_add(tf.reshape(adjustment_v, v_shape))
            layer.output_dense.kernel.assign_add(tf.reshape(adjustment_o, o_shape))
            
    return model

# Save and restore functions for the model weights
def save_model_weights(model):
    return [layer.get_weights() for layer in model.layers]

def restore_model_weights(model, weights):
    for layer, weight in zip(model.layers, weights):
        layer.set_weights(weight)

def compute_loss(adjusted_predictions, y_true):
    # Ensure adjusted_predictions and y_true have the same shape
    adjusted_predictions = tf.reshape(adjusted_predictions, (-1, 1))
    y_true = tf.reshape(y_true, (-1, 1))
    
    # Calculate the loss
    return tf.reduce_mean(tf.square(adjusted_predictions - y_true))

# Define the Transformer Block
class TransformerBlock2(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock2, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"), 
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)  # Assume self-attention
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock2, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config

#--------------------------------------------------------------------------------------------------------------------------------------------------
# SECOND MODEL DATA PREPROCESSING

# Load the training dataset for the second model
file_path = '/Users/sohamsane/Documents/Coding Projects/DualDynamicTransformer/Data/Model1_Full/Model1_Full.csv'
data2 = pd.read_csv(file_path)

# Drop rows with missing values for the output column
data2 = data2.dropna(subset=['Price'])

# Identify relevant features and target
input_columns = ['Brand', 'Processor_Speed', 'RAM_Size', 'Storage_Capacity', 'Screen_Size', 'Weight']  

# Split the training data into features and target
X2 = data2[input_columns]
y_trim2 = data2['Price']

# Define preprocessing for numerical and categorical features
categorical_features = X2.select_dtypes(include=['object']).columns
numerical_features = X2.select_dtypes(include=['int64', 'float64']).columns

# Fit and transform the features for the training dataset
X_preprocessed2 = preprocessor.fit_transform(X2)

# Encode the target variable
trim_encoder2 = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
y_trim_encoded2 = trim_encoder2.fit_transform(y_trim2.values.reshape(-1, 1))

# Split the data into training and test sets
X_train2, X_test2, y_trim_train2, y_trim_test2 = train_test_split(
    X_preprocessed2, y_trim_encoded2, test_size=0.2, random_state=42)

# Ensure there are no NaNs in the input data
X_train2[np.isnan(X_train2)] = 0
X_test2[np.isnan(X_test2)] = 0

#--------------------------------------------------------------------------------------------------------------------------------------------------
# SECOND MODEL ARCHITECTURE & TRAINING

def build_transformer_model2(input_shape, embed_dim, num_heads, ff_dim, num_transformer_blocks, dropout_rate):
    combined_input_shape = input_shape + 1
    inputs = Input(shape=(combined_input_shape,))
    x = Dense(embed_dim, kernel_regularizer=l2(0.05))(inputs)
    x = Reshape((1, embed_dim))(x)  # Ensure embedding dimension is properly set

    for _ in range(num_transformer_blocks):
        x = TransformerBlock2(embed_dim, num_heads, ff_dim, dropout_rate)(x)
    
    x = Flatten()(x)  # This should create a 2D tensor [batch_size, features]
    x = Dropout(dropout_rate)(x)  # Add dropout layer

    adjustment_output = Dense(input_shape, activation='linear', kernel_regularizer=l2(0.01))(x)  # Linear activation for weight adjustments

    model = Model(inputs=inputs, outputs=adjustment_output)
    return model

ff_dim = 256
num_transformer_blocks = 12
num_heads = 48
model2 = build_transformer_model2(input_shape, embed_dim, num_heads, ff_dim, num_transformer_blocks, dropout_rate)
model2.summary()

# Compile the second model with the specified optimizer and loss functions
model2.compile(
    optimizer=Adam(learning_rate=ExponentialDecay(
        initial_learning_rate=0.0075, 
        decay_steps=1000000,
        decay_rate=0.98,
        staircase=True
    )), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

def train_second_transformer_model(first_model, model2, X_train, y_trim_train, num_epochs, batch_size, trim_encoder):
    dataset = tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), y_trim_train.astype(np.float32)))
    dataset = dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for step, (X_batch_train, y_batch_train) in enumerate(dataset):
            X_batch_train = tf.reshape(X_batch_train, (-1, X_train.shape[1]))
            y_batch_train = tf.reshape(y_batch_train, (-1, y_trim_train.shape[1]))
            
            original_weights = save_model_weights(first_model)
            with tf.GradientTape() as tape:
                try:
                    # Get the first model's prediction for the batch
                    first_prediction = first_model(X_batch_train, training=False)
                    
                    # Decode the first model's prediction
                    first_prediction_decoded = trim_encoder.inverse_transform(first_prediction)
                except Exception as e:
                    print(f"Error during first model prediction: {e}")
                    return

                # Ensure the batch dimensions align and types match
                X_batch_train = tf.cast(X_batch_train, tf.float32)
                first_prediction_decoded = tf.cast(first_prediction_decoded, tf.float32)
                
                # Combine input and first model's decoded prediction
                combined_input = tf.concat([X_batch_train, first_prediction_decoded], axis=1)
                combined_input = tf.reshape(combined_input, (combined_input.shape[0], -1))

                # Get the weight adjustments from the second model for the entire batch
                adjustments = model2(combined_input, training=True)

                # Apply the weight adjustments to the first model
                first_model = apply_weight_adjustments(first_model, adjustments)

                # Get the adjusted prediction from the first model for the entire batch
                adjusted_prediction = first_model(X_batch_train, training=False)

                # Decode the adjusted first model's prediction
                adjusted_prediction_decoded = trim_encoder.inverse_transform(adjusted_prediction)

                # Ensure adjusted_prediction and y_batch_train are the same size for loss calculation
                y_true = tf.reshape(y_batch_train, adjusted_prediction.shape)

                # Calculate the lxxoss
                loss = compute_loss(adjusted_prediction, y_true)

            # Get gradients and apply them to the second model's trainable variables
            #gradients = tape.gradient(adjustments, model2.trainable_variables) #ORIGiNAL
            gradients = tape.gradient({'adjustments' : adjustments, 'loss' : loss}, model2.trainable_variables)


            # --->>>> The commented statement is a true calculation of the gradient. I am not sure how the test cases are resulting in better values
            #         However, the issue relies that when setting adjustments to zero, the graph becomes disconnected. Therefore, a split gradient calculation is needed
            #         Currently, the gradient is a calculation of both values
            
            #START RESEARCHING HERE NEXT (LOOK ABOVE)


            #model2.optimizer.apply_gradients(zip(gradients, model2.trainable_variables))
            model2.optimizer.apply(gradients, model2.trainable_variables)
           
            restore_model_weights(first_model, original_weights)

            if step % 50 == 0:
                print(f"Step {step}: Loss = {loss.numpy()}")

        # Shuffle the dataset at the end of each epoch
        dataset = dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)

    return model2

#--------------------------------------------------------------------------------------------------------------------------------------------------
# SECOND MODEL CALL

# Freeze the first transformer model
freeze_model(model)

# Train the second transformer model
second_model = train_second_transformer_model(model, model2, X_train, y_trim_train, num_epochs=5, batch_size=1, trim_encoder=trim_encoder)

print("Second transformer training complete.")

#--------------------------------------------------------------------------------------------------------------------------------------------------
# TESTING DUAL DYNAMIC TRANSFORMER (DDT)

def test_ddt(first_model, second_model, unseen_preprocessed, trim_encoder):
    # Ensure unseen_preprocessed is of type float32
    unseen_preprocessed = tf.cast(unseen_preprocessed, tf.float32)
    
    # Load original weights to restore later
    original_weights = save_model_weights(first_model)
    
    # Initialize lists to store true and predicted values for comparison
    true_values = []
    predicted_values = []

    for i in range(unseen_preprocessed.shape[0]):
        X_batch = unseen_preprocessed[i:i+1]
        
        with tf.GradientTape() as tape:
            # Get the first model's prediction
            first_prediction = first_model.predict(X_batch)
            first_prediction_decoded = trim_encoder.inverse_transform(first_prediction)

            # Combine input and first model's decoded prediction
            combined_input = tf.concat([X_batch, first_prediction_decoded], axis=1)

            # Get the weight adjustments from the second model
            adjustments = second_model(combined_input, training=False)

            # Apply the weight adjustments to the first model
            first_model = apply_weight_adjustments(first_model, adjustments)

            # Get the adjusted prediction from the first model
            adjusted_prediction = first_model.predict(X_batch)

            # Decode the adjusted prediction
            adjusted_prediction_decoded = trim_encoder.inverse_transform(adjusted_prediction)

            # Append predicted values for comparison
            predicted_values.append(adjusted_prediction_decoded)
            
            # Restore the original weights of the first model
            restore_model_weights(first_model, original_weights)
    
    # Print predicted values for the first 10 samples
    print("DDT Unseen Predicted Price:", np.array(predicted_values).flatten()[:10])

# Example usage
test_ddt(model, second_model, unseen_preprocessed, trim_encoder=trim_encoder)
