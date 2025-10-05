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
   git clone git@github.com:sibbi16/Meal-mate-ai-endpoints.git
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
