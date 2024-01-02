import os
from flask import Flask, request, jsonify, render_template
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegressionModel

app = Flask(__name__)

# Set Java home
java_home = "/usr/lib/jvm/java-17-openjdk-amd64"
os.environ["JAVA_HOME"] = java_home

# Set Spark home and configurations
spark_home = "/opt/spark"
os.environ["SPARK_HOME"] = spark_home

# Create Spark session
spark = SparkSession.builder.appName("Flask-Spark-Integration").getOrCreate()

# Load the saved Logistic Regression model
model_save_path = "new_logistic_regression_model_path"  # Change this to your saved model path
lr_model = LogisticRegressionModel.load(model_save_path)

@app.route('/')
def index():
    app_dir = os.path.dirname(os.path.abspath(__file__))
    return render_template('index.html')

@app.route('/model', methods=['POST'])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json(force=True)
        
        # Assuming the input data is in 'url' field, adjust as needed
        url = data['url']
        
        # Create a Spark DataFrame from the input data
        test_data = spark.createDataFrame([(url,)], ["url"])
        
        # Make predictions using the loaded model
        predictions = lr_model.transform(test_data)
        
        # Extracting prediction result
        result = predictions.select("url", "prediction").collect()[0].asDict()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000,debug=True)
