# 🍳 Meal Meet AI - Recipe Generator API

An intelligent recipe generation API powered by Google's Gemini AI. This FastAPI backend generates recipes from text prompts, images, URLs, and creates personalized weekly meal plans.

## ✨ Features

- **Multi-Input Recipe Generation**: Generate recipes from:
  - Text descriptions (e.g., "healthy breakfast recipe")
  - Uploaded images of food
  - Image URLs
  - Recipe webpage URLs (with automatic extraction)
  
- **AI-Powered Meal Planning**: Create comprehensive weekly meal plans based on your favorite recipes

- **Structured Output**: Get well-formatted recipes with:
  - Recipe name
  - Ingredients with measurements
  - Step-by-step instructions
  - Cooking duration

- **Smart Recipe Extraction**: Automatically extracts structured recipe data from popular recipe websites

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- A Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd recipebackendaiapi
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**
   
   Create a `.env` file in the project root and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

### Running the Server

After activating your virtual environment, start the server with:

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

## 📚 API Documentation

Once the server is running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

### Endpoints

#### 1. Health Check
```http
GET /
```
Returns API status.

#### 2. Generate Recipe
```http
POST /generate-recipe
```

**Parameters:**
- `prompt` (optional): Text description, image URL, or webpage URL
- `image` (optional): Uploaded image file

**Example using cURL:**
```bash
# Text prompt
curl -X POST "http://localhost:8000/generate-recipe" \
  -F "prompt=healthy pasta recipe"

# Image upload
curl -X POST "http://localhost:8000/generate-recipe" \
  -F "image=@path/to/food-image.jpg"

# Image URL
curl -X POST "http://localhost:8000/generate-recipe" \
  -F "prompt=https://example.com/food-image.jpg"

# Recipe webpage URL
curl -X POST "http://localhost:8000/generate-recipe" \
  -F "prompt=https://www.example.com/recipe-page"
```

**Response:**
```json
{
  "recipe_name": "Healthy Pasta Primavera",
  "ingredients": [
    "200g whole wheat pasta",
    "2 cups mixed vegetables",
    "2 tbsp olive oil",
    "3 cloves garlic, minced"
  ],
  "steps": [
    "Boil pasta according to package instructions",
    "Sauté vegetables in olive oil",
    "Combine pasta with vegetables and serve"
  ],
  "duration": "25 minutes"
}
```

#### 3. Generate Meal Plan
```http
POST /generate-meal-plan
```

**Parameters:**
- `recipes` (required): Comma-separated list of recipe names

**Example:**
```bash
curl -X POST "http://localhost:8000/generate-meal-plan" \
  -F "recipes=biryani, pasta, oatmeal, grilled chicken"
```

**Response:**
```json
{
  "meal_plan": "Detailed weekly meal plan with breakfast, lunch, dinner, snacks, shopping list, and meal prep tips..."
}
```

## 🛠️ Technology Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **Google Gemini AI**: Advanced AI model for recipe generation
- **LangChain**: Framework for building LLM applications
- **Pydantic**: Data validation using Python type annotations
- **BeautifulSoup4**: Web scraping for recipe extraction
- **Pillow**: Image processing
- **Uvicorn**: ASGI server for running the application

## 📁 Project Structure

```
recipebackendaiapi/
├── app.py                    # Main FastAPI application
├── requirements.txt          # Python dependencies
├── .env                      # Environment variables (not in git)
├── test_recipe_api.html      # Frontend test interface
├── venv/                     # Virtual environment (not in git)
└── README.md                 # This file
```

## 🔒 Security Notes

- Never commit your `.env` file or expose your API keys
- The `.env` file contains sensitive credentials
- In production, use proper secrets management
- Configure CORS settings appropriately for your frontend domain

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 🐛 Troubleshooting

### Common Issues

1. **Import errors**: Make sure you've activated the virtual environment and installed all dependencies
2. **API key errors**: Verify your `GEMINI_API_KEY` is correctly set in the `.env` file
3. **Port already in use**: Change the port with `uvicorn app:app --reload --port 8001`

## 📧 Support

For issues and questions, please open an issue on GitHub.

---

Made with ❤️ using Google Gemini AI
