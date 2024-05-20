from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('data/tiger_population_india_2023_dataset.csv')
df.dropna(inplace=True)

# Train the model
X = df.iloc[:, 1:-1]
y = df.iloc[:, -1]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Get the list of regions
regions = df['State'].unique()

def predict_tiger_population(region, year):
    region_data = df[df['State'] == region].iloc[:, 1:-1]
    return model.predict(region_data)[0]

def plot_tiger_distribution(predicted_population):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Total length of Transect(km)'], df['Camera trap numbers'], color='indigo', edgecolors='black', label='Actual Population')
    plt.scatter(df['Total length of Transect(km)'], predicted_population, color='thistle', edgecolors='black', label='Predicted Population')
    plt.xlabel('Total length of Transect (km)')
    plt.ylabel('Number of Tigers (Camera trap numbers)')
    plt.title('Distribution of Tigers in India')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    canvas = FigureCanvas(plt.gcf())
    img = io.BytesIO()
    canvas.print_png(img)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

def evaluate_tiger_extinction(year, predicted_population):
    total_population = df['Camera trap numbers'].sum()
    extinction_percentage = ((total_population - predicted_population.sum()) / total_population) * 100
    if predicted_population.sum() < total_population:
        return f"There is a {extinction_percentage:.2f}% chance of tiger extinction in {year}."
    else:
        return f"There is no chance of tiger extinction in {year}."

@app.route('/')
def index():
    return render_template('layout.html')

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        year = int(request.form['year'])
        region = request.form['region']

        predicted_population = predict_tiger_population(region, year)
        total_predicted_population = model.predict(df.iloc[:, 1:-1])
        total_population_in_year = total_predicted_population.sum()
        plot_url = plot_tiger_distribution(total_predicted_population)
        extinction_message = evaluate_tiger_extinction(year, total_predicted_population)
        population_difference = df['Camera trap numbers'].sum() - total_population_in_year

        return render_template('prediction.html',
                               region=region,
                               year=year,
                               predicted_population=predicted_population,
                               total_population_in_year=total_population_in_year,
                               extinction_message=extinction_message,
                               population_difference=population_difference,
                               plot_url=plot_url,
                               regions=regions)
    return render_template('prediction.html', regions=regions)

@app.route('/awareness')
def awareness():
    return render_template('awareness.html')

@app.route('/about_project')
def about_project():
    return render_template('about_project.html')

@app.route('/recent_data')
def recent_data():
    recent_data_df = pd.read_csv('data/recent_data.csv')
    recent_data = recent_data_df.values.tolist()
    return render_template('recent_data.html', recent_data=recent_data)


if __name__ == '__main__':
    app.run(debug=True)
