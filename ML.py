import gradio as gr
from sklearn import tree

# Hypothetical dataset
# Features: [money loss, victims, tools and techniques]
# Labels: 0 for cyber crime, 1 for Through cyber crime
features = [
    [254, 400],  # cyber crime
    [200, 340],  # cyber crime
    [500, 200],  # Through cyber crime
    [400, 220],  # Through cyber crime
    [180, 430],  # cyber crime
    [450, 210],  # Through cyber crime
]
labels = [0, 0, 1, 1, 0, 1]

# Define the classifier
CrimeClassifier = tree.DecisionTreeClassifier()

# Train the model
CrimeModel = CrimeClassifier.fit(features, labels)

# Function to make prediction
def predict_crime(money_loss, victims):
    test_feature = [money_loss, victims]
    prediction = CrimeClassifier.predict([test_feature])
    return "Through cyber crime" if prediction[0] == 1 else "Cyber crime"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_crime,
    inputs=[gr.Number(label="Money Loss"), gr.Number(label="Number of Victims")],
    outputs="text",
    title="Crime Classifier",
    description="Predict whether an incident is a cyber crime or through cyber crime based on money loss and number of victims."
)

# Launch the interface
iface.launch()