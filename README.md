# Food Recognition System

A comprehensive food recognition and dietary analysis system built with Python, Flask, and YOLOv8.

## Features

- **Food Recognition**: Uses YOLOv8 model to identify food items from images
- **Dietary Analysis**: Provides nutritional information and dietary recommendations
- **Web Interface**: User-friendly Flask web application
- **Database Integration**: SQLite database for storing food and user data
- **AI Agent**: Intelligent agent for dietary advice and recommendations

## Project Structure

```
├── app.py                 # Main Flask application
├── ai_agent.py           # AI agent for dietary recommendations
├── requirements.txt      # Python dependencies
├── templates/           # HTML templates
├── static/             # CSS, JS, and static assets
├── Test Images/        # Test images for food recognition
├── *.ipynb             # Jupyter notebooks for model training
├── *.csv               # Dataset files
└── *.pt                # Trained model files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/roht94/dessertation.git
cd dessertation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database:
```bash
python init_database.py
```

4. Run the application:
```bash
python app.py
```

## Usage

1. Open your web browser and navigate to `http://localhost:5000`
2. Upload an image of food
3. The system will identify the food items and provide nutritional information
4. Get personalized dietary recommendations from the AI agent

## Technologies Used

- **Backend**: Python, Flask
- **AI/ML**: YOLOv8, TensorFlow/PyTorch
- **Database**: SQLite
- **Frontend**: HTML, CSS, JavaScript
- **Model Training**: Jupyter Notebooks

## Model Training

The project includes Jupyter notebooks for training the YOLOv8 model:
- `train_Using_YOLOV8.ipynb`: Main training notebook
- `decisiontreemodel.ipynb`: Decision tree model for dietary analysis

## Database Schema

The system uses a SQLite database with the following main tables:
- Food items and nutritional information
- User data and preferences
- Dietary recommendations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is part of an M.Tech dissertation.

## Author

- GitHub: [@roht94](https://github.com/roht94) 