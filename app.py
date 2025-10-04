import os
import io
import requests
import json
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
from bs4 import BeautifulSoup

# LangChain & Google imports
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load environment
load_dotenv()

# Always use browser-like headers to avoid 403 errors
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/129.0.0.0 Safari/537.36"
    )
}

# Recipe Schema
class Recipe(BaseModel):
    recipe_name: str
    ingredients: List[str]
    steps: List[str]
    duration: str


class GeminiRecipeGenerator:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        
        # LangChain setup
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3,
            max_output_tokens=2048,
        )
        self.parser = JsonOutputParser(pydantic_object=Recipe)

    def is_url(self, text: str) -> bool:
        return text.startswith(("http://", "https://"))

    def is_image_file(self, path: str) -> bool:
        return any(path.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"])

    def extract_recipe_from_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Try extracting recipe from JSON-LD metadata in webpage"""
        try:
            response = requests.get(url, headers=DEFAULT_HEADERS, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(script.string)
                    # Handle both dict and list cases
                    if isinstance(data, dict) and data.get("@type") == "Recipe":
                        return self._convert_jsonld_to_recipe(data)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and item.get("@type") == "Recipe":
                                return self._convert_jsonld_to_recipe(item)
                except Exception:
                    continue
        except Exception as e:
            print(f"❌ Recipe extraction failed: {e}")
        return None

    def _convert_jsonld_to_recipe(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert JSON-LD Recipe format to our Recipe schema"""
        name = data.get("name", "Unknown Recipe")
        ingredients = data.get("recipeIngredient", [])
        instructions = data.get("recipeInstructions", [])
        steps = []
        if isinstance(instructions, list):
            for instr in instructions:
                if isinstance(instr, dict):
                    steps.append(instr.get("text", ""))
                elif isinstance(instr, str):
                    steps.append(instr)
        elif isinstance(instructions, str):
            steps = [instructions]

        return {
            "recipe_name": name,
            "ingredients": ingredients,
            "steps": steps,
            "duration": data.get("totalTime", "Not specified"),
        }

    def generate_recipe(self, prompt: str = None, image: Image.Image = None) -> Dict[str, Any]:
        """Generate recipe from text, image, or URL"""
        try:
            if image:
                # Handle uploaded image - use direct Gemini
                content = [image, self._get_recipe_prompt()]
                response = self.model.generate_content(content)
                response_text = self._extract_text_from_response(response)
                return self._parse_recipe_response(response_text)
                
            elif prompt and self.is_url(prompt):
                if self.is_image_file(prompt):
                    # Handle image URL - use direct Gemini
                    response = requests.get(prompt, headers=DEFAULT_HEADERS)
                    response.raise_for_status()
                    image = Image.open(io.BytesIO(response.content))
                    content = [image, self._get_recipe_prompt()]
                    response = self.model.generate_content(content)
                    response_text = self._extract_text_from_response(response)
                    return self._parse_recipe_response(response_text)
                else:
                    # Handle webpage URL - try to extract recipe first
                    extracted = self.extract_recipe_from_url(prompt)
                    if extracted:
                        return extracted
                    
                    # If extraction fails, use LangChain with webpage content
                    response = requests.get(prompt, headers=DEFAULT_HEADERS)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, "html.parser")
                    text_content = soup.get_text(separator="\n")
                    input_text = f"Webpage Content: {text_content[:4000]}"
                    return self._generate_with_langchain(input_text)
                    
            elif prompt and os.path.exists(prompt) and self.is_image_file(prompt):
                # Handle local image file - use direct Gemini
                image = Image.open(prompt)
                content = [image, self._get_recipe_prompt()]
                response = self.model.generate_content(content)
                response_text = self._extract_text_from_response(response)
                return self._parse_recipe_response(response_text)
                
            elif prompt:
                # Handle text prompt - use LangChain for better structured output
                return self._generate_with_langchain(prompt)
            else:
                return {
                    "recipe_name": "Error",
                    "ingredients": [],
                    "steps": [],
                    "duration": "Not specified",
                    "error": "No input provided"
                }

        except Exception as e:
            print(f"Recipe generation failed: {e}")
            return {
                "recipe_name": "Error",
                "ingredients": [],
                "steps": [],
                "duration": "Not specified",
                "error": str(e)
            }
    
    def _generate_with_langchain(self, prompt: str) -> Dict[str, Any]:
        """Generate recipe using LangChain for structured output"""
        try:
            # Create LangChain prompt
            recipe_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a professional chef. Generate a detailed recipe in JSON format.
                
                Return a valid JSON object with these exact fields:
                - recipe_name: string (name of the recipe)
                - ingredients: array of strings (each ingredient with measurements)
                - steps: array of strings (numbered cooking instructions)
                - duration: string (total cooking time like "30 minutes" or "1 hour")
                
                Make sure the JSON is valid and complete."""),
                ("human", "{input}")
            ])
            
            # Create the chain
            chain = recipe_prompt | self.llm | self.parser
            
            # Generate recipe
            result = chain.invoke({"input": prompt})
            
            # Ensure we have a dict
            if isinstance(result, dict):
                return {
                    "recipe_name": result.get("recipe_name", "Generated Recipe"),
                    "ingredients": result.get("ingredients", []),
                    "steps": result.get("steps", []),
                    "duration": result.get("duration", "Not specified"),
                }
            else:
                # Fallback to direct Gemini
                return self._generate_with_direct_gemini(prompt)
                
        except Exception as e:
            print(f"LangChain generation failed: {e}")
            # Fallback to direct Gemini
            return self._generate_with_direct_gemini(prompt)
    
    def _generate_with_direct_gemini(self, prompt: str) -> Dict[str, Any]:
        """Fallback to direct Gemini generation"""
        try:
            content = f"User Request: {prompt}\n\n{self._get_recipe_prompt()}"
            response = self.model.generate_content(content)
            response_text = self._extract_text_from_response(response)
            return self._parse_recipe_response(response_text)
        except Exception as e:
            print(f"Direct Gemini generation failed: {e}")
            return {
                "recipe_name": "Error",
                "ingredients": [],
                "steps": [],
                "duration": "Not specified",
                "error": str(e)
            }
    
    def _extract_text_from_response(self, response) -> str:
        """Safely extract text from Gemini response"""
        try:
            return response.text
        except Exception:
            # Fallback methods
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    for part in parts:
                        if hasattr(part, 'text') and part.text:
                            return part.text
            return str(response)
    
    def generate_meal_plan_from_recipes(self, recipes_text: str) -> Dict[str, Any]:
        """Generate a weekly meal plan based on multiple recipe names"""
        try:
            # Parse the recipes from the input text
            recipes = [recipe.strip() for recipe in recipes_text.split(',') if recipe.strip()]
            
            if not recipes:
                return {"error": "No valid recipes found in input"}
            
            # Create the prompt for meal plan generation
            prompt = f"""Create a comprehensive weekly meal plan using these recipes: {', '.join(recipes)}

Please create a detailed 7-day meal plan that includes:

1. RECIPE LIST: {', '.join(recipes)}

2. WEEKLY MEAL PLAN:
   - Monday: Breakfast, Lunch, Dinner, Snacks
   - Tuesday: Breakfast, Lunch, Dinner, Snacks  
   - Wednesday: Breakfast, Lunch, Dinner, Snacks
   - Thursday: Breakfast, Lunch, Dinner, Snacks
   - Friday: Breakfast, Lunch, Dinner, Snacks
   - Saturday: Breakfast, Lunch, Dinner, Snacks
   - Sunday: Breakfast, Lunch, Dinner, Snacks

3. MEAL PREP TIPS:
   - Suggestions for preparing meals in advance
   - Storage recommendations
   - Time-saving strategies

4. SHOPPING LIST:
   - Organized by food categories
   - Quantities for the entire week

5. NUTRITIONAL NOTES:
   - Key nutrients and health benefits
   - Portion size recommendations
   - Dietary considerations

Make sure to incorporate the provided recipes ({', '.join(recipes)}) throughout the week in a balanced and practical way."""

            # Use LangChain for better structured output
            return self._generate_meal_plan_with_langchain(prompt)
            
        except Exception as e:
            print(f"Meal plan generation from recipes failed: {e}")
            return {"error": str(e)}
    
    def _generate_meal_plan_with_langchain(self, prompt: str) -> Dict[str, Any]:
        """Generate meal plan using direct Gemini (simplified approach)"""
        try:
            # Use direct Gemini instead of LangChain to avoid parsing issues
            response = self.model.generate_content(prompt)
            response_text = self._extract_text_from_response(response)
            return {"meal_plan": response_text}
                
        except Exception as e:
            print(f"Meal plan generation failed: {e}")
            return {"error": str(e)}
    
    def _generate_meal_plan_with_direct_gemini(self, prompt: str) -> Dict[str, Any]:
        """Fallback to direct Gemini generation for meal plan"""
        try:
            response = self.model.generate_content(prompt)
            response_text = self._extract_text_from_response(response)
            return {"meal_plan": response_text}
        except Exception as e:
            print(f"Direct Gemini meal plan generation failed: {e}")
            return {"error": str(e)}

    
    def _get_recipe_prompt(self) -> str:
        """Get the standardized prompt for recipe generation"""
        return """You are a professional chef. Generate a detailed recipe and format it EXACTLY as shown below. Do not deviate from this format:

RECIPE NAME: [Name of the recipe]

DURATION: [Total time like "30 minutes" or "1 hour 30 minutes"]

INGREDIENTS:
- [First ingredient with exact measurements]
- [Second ingredient with exact measurements]
- [Third ingredient with exact measurements]
- [Continue with each ingredient on a separate line starting with dash]

STEPS:
1. [First step with detailed instructions]
2. [Second step with detailed instructions]
3. [Third step with detailed instructions]
4. [Continue with numbered steps]

CRITICAL: Follow this exact format. Do not add extra text, explanations, or formatting outside these sections."""
    
    def _parse_recipe_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the Gemini response into structured format"""
        try:
            print(f"Raw response text: {response_text[:500]}...")
            
            # Initialize default values
            recipe_name = "Generated Recipe"
            duration = "Not specified"
            ingredients = []
            steps = []
            
            import re
            
            # Extract recipe name - try multiple patterns
            name_patterns = [
                r'RECIPE NAME:\s*(.+)',
                r'^(.+?)(?:\n|$)',
                r'#\s*(.+)',
                r'\*\*(.+?)\*\*'
            ]
            
            for pattern in name_patterns:
                name_match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
                if name_match:
                    potential_name = name_match.group(1).strip()
                    if len(potential_name) < 100 and not potential_name.upper().startswith(('INGREDIENTS', 'STEPS', 'DURATION')):
                        recipe_name = potential_name
                        print(f"Found recipe name: {recipe_name}")
                        break
            
            # Extract duration - try multiple patterns
            duration_patterns = [
                r'DURATION:\s*(.+)',
                r'TIME:\s*(.+)',
                r'COOKING TIME:\s*(.+)',
                r'TOTAL TIME:\s*(.+)'
            ]
            
            for pattern in duration_patterns:
                duration_match = re.search(pattern, response_text, re.IGNORECASE)
                if duration_match:
                    duration = duration_match.group(1).strip()
                    print(f"Found duration: {duration}")
                    break
            
            # Extract ingredients section - more flexible approach
            ingredients_section = None
            ingredients_patterns = [
                r'INGREDIENTS:\s*(.*?)(?=STEPS:|INSTRUCTIONS:|$)',  # Original
                r'INGREDIENTS\s*(.*?)(?=STEPS:|INSTRUCTIONS:|$)',   # Without colon
                r'🥘\s*INGREDIENTS\s*(.*?)(?=STEPS:|INSTRUCTIONS:|📋|$)',  # With emoji
                r'INGREDIENTS\s*(.*?)(?=\d+\.|STEPS|INSTRUCTIONS|$)'  # More flexible
            ]
            
            for pattern in ingredients_patterns:
                ingredients_match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if ingredients_match:
                    ingredients_section = ingredients_match.group(1).strip()
                    print(f"Found ingredients section: {ingredients_section[:200]}...")
                    break
            
            if ingredients_section:
                # Parse individual ingredients - more flexible
                for line in ingredients_section.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Handle different ingredient formats
                    if line.startswith(('-', '•', '*', '+')):
                        ingredient = line[1:].strip()
                    elif re.match(r'^\d+\.', line):  # Numbered ingredients
                        ingredient = re.sub(r'^\d+\.\s*', '', line)
                    elif any(unit in line.lower() for unit in ['cup', 'tbsp', 'tsp', 'oz', 'lb', 'g', 'ml', 'kg', 'gram', 'liter', 'piece', 'clove', 'bunch', 'teaspoon', 'tablespoon']):
                        ingredient = line
                    else:
                        continue
                    
                    if ingredient and len(ingredient) > 2 and len(ingredient) < 200:
                        ingredients.append(ingredient)
                        print(f"Added ingredient: {ingredient}")
            
            # Extract steps section - more flexible approach
            steps_section = None
            steps_patterns = [
                r'STEPS:\s*(.*?)$',
                r'INSTRUCTIONS:\s*(.*?)$',
                r'📋\s*INSTRUCTIONS\s*(.*?)$',
                r'STEPS\s*(.*?)$',
                r'INSTRUCTIONS\s*(.*?)$'
            ]
            
            for pattern in steps_patterns:
                steps_match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
                if steps_match:
                    steps_section = steps_match.group(1).strip()
                    print(f"Found steps section: {steps_section[:200]}...")
                    break
            
            if steps_section:
                # Parse individual steps - more flexible
                for line in steps_section.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Handle different step formats
                    if line.startswith('**') and line.endswith('**'):
                        step_text = line[2:-2].strip()
                    elif re.match(r'^\d+\.', line):  # Numbered steps
                        step_text = re.sub(r'^\d+\.\s*', '', line)
                    elif line.startswith(('•', '*', '-', '+')):
                        step_text = line[1:].strip()
                    else:
                        # Check if it's a continuation or standalone step
                        if len(line) > 10 and len(line) < 500:
                            step_text = line
                        else:
                            continue
                    
                    if step_text and len(step_text) > 5:
                        steps.append(step_text)
                        print(f"Added step: {step_text[:50]}...")
            
            print(f"Final results - Name: {recipe_name}, Duration: {duration}, Ingredients: {len(ingredients)}, Steps: {len(steps)}")
            
            return {
                "recipe_name": recipe_name,
                "ingredients": ingredients if ingredients else ["No ingredients found"],
                "steps": steps if steps else ["No steps found"],
                "duration": duration,
            }
            
        except Exception as e:
            print(f"Error parsing recipe response: {str(e)}")
            return {
                "recipe_name": "Generated Recipe",
                "ingredients": ["Error parsing ingredients"],
                "steps": ["Error parsing steps"],
                "duration": "Not specified",
                "error": f"Failed to parse response: {str(e)}"
            }
    
    def _extract_ingredients_from_text(self, text: str) -> List[str]:
        """Extract ingredients from raw text when structured parsing fails"""
        ingredients = []
        lines = text.split('\n')
        in_ingredients = False
        
        for line in lines:
            line = line.strip()
            if 'ingredient' in line.lower() and ':' in line:
                in_ingredients = True
                continue
            elif in_ingredients and ('step' in line.lower() or 'instruction' in line.lower()):
                break
            elif in_ingredients and line:
                if any(char in line for char in ['cup', 'tbsp', 'tsp', 'oz', 'lb', 'g', 'ml', 'kg']):
                    ingredients.append(line)
                elif line.startswith(('-', '•', '*')):
                    ingredients.append(line[1:].strip())
        
        return ingredients
    
    def _extract_steps_from_text(self, text: str) -> List[str]:
        """Extract steps from raw text when structured parsing fails"""
        steps = []
        lines = text.split('\n')
        in_steps = False
        
        for line in lines:
            line = line.strip()
            if 'step' in line.lower() and ':' in line:
                in_steps = True
                continue
            elif in_steps and line:
                if line[0].isdigit() or line.startswith(('-', '•', '*')):
                    if line[0].isdigit():
                        step_text = line.split('.', 1)[1].strip() if '.' in line else line
                    else:
                        step_text = line[1:].strip()
                    steps.append(step_text)
                elif line.startswith('**') and line.endswith('**'):
                    steps.append(line)
        
        return steps
    


# --- FastAPI App ---
app = FastAPI(title="Gemini Recipe Generator")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

generator = GeminiRecipeGenerator()


@app.post("/generate-recipe")
async def generate_recipe(
    prompt: str = Form(None),
    image: UploadFile = None
):
    """Generate a recipe from text, image upload, image URL, or webpage URL"""
    try:
        # Handle image upload
        if image:
            # Read and process uploaded image
            image_data = await image.read()
            pil_image = Image.open(io.BytesIO(image_data))
            result = generator.generate_recipe(image=pil_image)
        elif prompt:
            # Handle text prompt, URL, or local file path
            result = generator.generate_recipe(prompt=prompt)
        else:
            result = {
                "recipe_name": "Error",
                "ingredients": [],
                "steps": [],
                "duration": "Not specified",
                "error": "No input provided. Please provide either text prompt or upload an image."
            }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(
            content={
                "recipe_name": "Error",
                "ingredients": [],
                "steps": [],
                "duration": "Not specified",
                "error": str(e)
            },
            status_code=500
        )




@app.post("/generate-meal-plan")
async def generate_meal_plan(
    recipes: str = Form(...)
):
    """Generate a weekly meal plan based on multiple recipe names provided as text"""
    try:
        if not recipes or not recipes.strip():
            result = {
                "error": "No recipes provided. Please provide recipe names separated by commas (e.g., 'biryani, pasta, oatmeal')."
            }
        else:
            result = generator.generate_meal_plan_from_recipes(recipes.strip())
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(
            content={
                "error": str(e)
            },
            status_code=500
        )


@app.get("/")
async def root():
    """API Health Check"""
    return JSONResponse(content={"status": "ok", "message": "Gemini Recipe Generator API is running"})