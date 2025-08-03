from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from typing import List, Union, Dict, Any
import re
import json
import os
import requests
from datetime import datetime, timedelta
import sqlite3
import dotenv

dotenv.load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

def get_db_path():
    """Get the absolute path to the database file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(script_dir), 'diet_recommendation.db')

def get_user_profile(user_id: str) -> dict:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute("SELECT * FROM user WHERE u_id=?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if row:
        return {
            'id': row[0], 'username': row[1], 'password': row[2], 'gender': row[3], 'age': row[4],
            'email': row[5], 'weight': row[6], 'feet': row[7], 'inches': row[8], 'bodyfat': row[9],
            'status': row[10], 'journey': row[11], 'vegan': row[12], 'allergy': row[13],
            'protein_goal': row[14], 'carb_goal': row[15], 'fat_goal': row[16], 'calorie_goal': row[17],
            'bmi': row[18], 'fiber_goal': row[19], 'activity_level': row[20], 'startdate': row[21], 'goal': row[22]
        }
    return {}

def get_notes(user_id: str) -> str:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute("SELECT note FROM user_notes WHERE u_id=? ORDER BY created_at DESC", (user_id,))
    notes = [row[0] for row in cur.fetchall()]
    conn.close()
    return "; ".join(notes)

def get_feedback(user_id: str) -> str:
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute("SELECT feedback_text FROM user_feedback WHERE u_id=? ORDER BY created_at DESC", (user_id,))
    feedbacks = [row[0] for row in cur.fetchall()]
    conn.close()
    return "; ".join(feedbacks)

def get_exercise_history(user_id: str) -> str:
    """Get user's exercise history and performance data"""
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    
    # Get recent workout plans and their completion status
    cur.execute("""
        SELECT plan_text, created_at, completed 
        FROM workout_plans 
        WHERE u_id=? 
        ORDER BY created_at DESC 
        LIMIT 5
    """, (user_id,))
    workout_history = cur.fetchall()
    
    # Get exercise performance data if available
    cur.execute("""
        SELECT exercise_name, sets, reps, weight, date 
        FROM exercise_log 
        WHERE u_id=? 
        ORDER BY date DESC 
        LIMIT 10
    """, (user_id,))
    performance_data = cur.fetchall()
    
    conn.close()
    
    history_summary = []
    if workout_history:
        history_summary.append("Recent Workout Plans:")
        for plan, date, completed in workout_history:
            status = "Completed" if completed else "Not completed"
            history_summary.append(f"- {date}: {status}")
    
    if performance_data:
        history_summary.append("Recent Exercise Performance:")
        for exercise, sets, reps, weight, date in performance_data:
            history_summary.append(f"- {exercise}: {sets} sets x {reps} reps @ {weight}kg on {date}")
    
    return "; ".join(history_summary) if history_summary else "No exercise history available"

def adapt_plan_based_on_feedback(plan: str, feedback: str) -> str:
    # Simple adaptation: if feedback contains 'hard', reduce intensity, etc.
    if 'hard' in feedback.lower():
        return plan + "\n[Adapted: Reduced intensity due to feedback]"
    if 'injury' in feedback.lower():
        return plan + "\n[Adapted: Avoid exercises that may cause injury]"
    return plan

def check_feasibility(plan: str) -> bool:
    # Dummy feasibility check: just check plan length for now
    return len(plan) < 2000

def save_workout_plan(user_id: str, plan_text: str) -> bool:
    """Save the generated workout plan to the database"""
    try:
        conn = sqlite3.connect(get_db_path())
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO workout_plans (u_id, plan_text, created_at, completed) 
            VALUES (?, ?, ?, ?)
        """, (user_id, plan_text, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), False))
        conn.commit()
        conn.close()
        print(f"[DEBUG] Workout plan saved for user {user_id}")
        return True
    except Exception as e:
        print(f"[DEBUG] Error saving workout plan: {e}")
        return False

def generate_fallback_plan(user: dict) -> str:
    """Generate a workout plan without using OpenAI API"""
    print(f"[DEBUG] ai_agent.py - Generating fallback plan for user: {user.get('username', 'Unknown')}")
    
    # Get user data
    age = user.get('age', 30)
    gender = user.get('gender', 'male')
    weight = user.get('weight', 70)
    goal = user.get('goal', 'fitness')
    activity_level = user.get('activity_level', 'moderate')
    
    # Base workout templates
    cardio_exercises = [
        "Walking (30 minutes)",
        "Jogging (20 minutes)", 
        "Cycling (25 minutes)",
        "Swimming (20 minutes)",
        "Jump rope (15 minutes)"
    ]
    
    strength_exercises = {
        'upper_body': [
            "Push-ups (3 sets of 10-15 reps)",
            "Dumbbell rows (3 sets of 12 reps)",
            "Shoulder press (3 sets of 10 reps)",
            "Bicep curls (3 sets of 12 reps)",
            "Tricep dips (3 sets of 10 reps)"
        ],
        'lower_body': [
            "Squats (3 sets of 15 reps)",
            "Lunges (3 sets of 10 reps each leg)",
            "Calf raises (3 sets of 20 reps)",
            "Glute bridges (3 sets of 15 reps)",
            "Wall sits (3 sets of 30 seconds)"
        ],
        'core': [
            "Plank (3 sets of 30 seconds)",
            "Crunches (3 sets of 15 reps)",
            "Russian twists (3 sets of 20 reps)",
            "Leg raises (3 sets of 10 reps)",
            "Mountain climbers (3 sets of 20 reps)"
        ]
    }
    
    # Generate plan based on goal
    if goal.lower() in ['weight loss', 'fat loss']:
        plan = f"""🏃‍♂️ **Weight Loss Workout Plan for {user.get('username', 'you')}**

**Warm-up (10 minutes):**
• Light stretching
• 5 minutes of brisk walking

**Cardio Session (30 minutes):**
• {cardio_exercises[0] if 'no' in activity_level.lower() else cardio_exercises[1]}

**Strength Training (20 minutes):**
• {strength_exercises['upper_body'][0]}
• {strength_exercises['lower_body'][0]}
• {strength_exercises['core'][0]}

**Cool-down (5 minutes):**
• Gentle stretching

**Frequency:** 4-5 times per week
**Rest Days:** 2-3 days per week

💡 **Tips:** Stay hydrated, maintain a calorie deficit, and be consistent!"""
    
    elif goal.lower() in ['muscle gain', 'strength']:
        plan = f"""💪 **Muscle Building Workout Plan for {user.get('username', 'you')}**

**Warm-up (10 minutes):**
• Dynamic stretching
• Light cardio (5 minutes)

**Strength Training (45 minutes):**
**Upper Body:**
• {strength_exercises['upper_body'][0]}
• {strength_exercises['upper_body'][1]}
• {strength_exercises['upper_body'][2]}

**Lower Body:**
• {strength_exercises['lower_body'][0]}
• {strength_exercises['lower_body'][1]}

**Core:**
• {strength_exercises['core'][0]}
• {strength_exercises['core'][1]}

**Frequency:** 3-4 times per week
**Rest Days:** 3-4 days per week

💡 **Tips:** Focus on proper form, progressive overload, and adequate protein intake!"""
    
    else:  # General fitness
        plan = f"""🌟 **General Fitness Workout Plan for {user.get('username', 'you')}**

**Warm-up (10 minutes):**
• Light stretching
• 5 minutes of moderate cardio

**Mixed Workout (30 minutes):**
**Cardio (15 minutes):**
• {cardio_exercises[2] if 'active' in activity_level.lower() else cardio_exercises[0]}

**Strength (15 minutes):**
• {strength_exercises['upper_body'][0]}
• {strength_exercises['lower_body'][0]}
• {strength_exercises['core'][0]}

**Frequency:** 3-4 times per week
**Rest Days:** 3-4 days per week

💡 **Tips:** Listen to your body, stay consistent, and enjoy the process!"""

    # Add notes if available
    notes = get_notes(user.get('id', ''))
    if notes:
        plan += f"\n\n📝 **Your Notes:** {notes}"
        plan += "\n\n[AI-Generated Exercise Plan - Please review and adjust based on your fitness level and capabilities]"
    
    return plan

def regenerate_plan(user_id: str) -> str:
    print(f"[DEBUG] ai_agent.py regenerate_plan called with user_id: {user_id}")
    user = get_user_profile(user_id)
    notes = get_notes(user_id)
    feedback = get_feedback(user_id)
    exercise_history = get_exercise_history(user_id)
    
    # Create a comprehensive, personalized prompt with all user data
    prompt = f"""Create a personalized {user.get('goal', 'fitness')} workout plan for:
- Age: {user.get('age')} years old
- Gender: {user.get('gender')}
- Weight: {user.get('weight')} lbs
- Height: {user.get('feet')}'{user.get('inches')}"
- Activity Level: {user.get('activity_level')}
- BMI: {user.get('bmi', 'N/A')}
- Body Fat: {user.get('bodyfat', 'N/A')}%
- Fitness Journey: {user.get('journey', 'N/A')}
- Protein Goal: {user.get('protein_goal', 'N/A')}g
- Calorie Goal: {user.get('calorie_goal', 'N/A')} calories

User Notes & Constraints: {notes if notes else 'None'}
Previous Feedback: {feedback if feedback else 'None'}
Exercise History: {exercise_history if exercise_history else 'None'}

Create a detailed, personalized workout plan that:
1. Matches their fitness goal and current fitness level
2. Considers their activity level and available time
3. Avoids any equipment or exercises mentioned in their notes/constraints
4. Builds upon their exercise history and previous performance
5. Addresses any issues mentioned in their feedback
6. Includes warm-up, cardio, strength training, and cool-down
7. Provides specific exercises, sets, reps, and duration
8. Adapts to their physical limitations or preferences
9. Includes progression recommendations based on their history
10. Considers their current fitness journey stage

Format the response clearly with sections and bullet points."""
    
    # Check which LLM to use: Google AI Studio or fallback
    use_google_ai = os.getenv('USE_GOOGLE_AI', 'true').lower() == 'true'
    
    if use_google_ai:
        # Use Google AI Studio API for plan generation
        google_api_key = os.getenv('GOOGLE_AI_API_KEY')
        google_model = os.getenv('GOOGLE_AI_MODEL', 'gemini-1.5-flash')
        
        print(f"[DEBUG] ai_agent.py - Google AI API Key: {google_api_key[:20]}..." if google_api_key else "None")
        if not google_api_key or google_api_key == 'your_google_ai_api_key_here':
            print("[DEBUG] ai_agent.py - Google AI API key check failed, using fallback")
            return generate_fallback_plan(user)
        
        print(f"[DEBUG] ai_agent.py - API key check passed, proceeding with Google AI call using {google_model}")
        try:
            print("[DEBUG] ai_agent.py - Making direct Google AI API call...")
            
            # Use Google AI Studio API
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{google_model}:generateContent"
            
            headers = {
                "Content-Type": "application/json"
            }
            
            data = {
                "contents": [
                    {
                        "parts": [
                            {
                                "text": prompt
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 2000
                }
            }
            
            print(f"[DEBUG] ai_agent.py - Prompt length: {len(prompt)} characters")
            print(f"[DEBUG] ai_agent.py - Using model: {google_model}")
            
            response = requests.post(
                f"{url}?key={google_api_key}",
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                plan = result['candidates'][0]['content']['parts'][0]['text']
                print("[DEBUG] Google AI API response successful")
                print(f"[DEBUG] ai_agent.py - Response length: {len(plan)} characters")
            else:
                print(f"[DEBUG] ai_agent.py - Google AI API error: {response.status_code}")
                print(f"[DEBUG] ai_agent.py - Error response: {response.text}")
                raise Exception(f"Google AI API error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"[DEBUG] ai_agent.py - Error during Google AI call: {str(e)}")
            print("[DEBUG] ai_agent.py - Falling back to local plan generator")
            return generate_fallback_plan(user)
    else:
        print("[DEBUG] ai_agent.py - Using fallback plan generator (Google AI disabled)")
        return generate_fallback_plan(user)
    
    if not check_feasibility(plan):
        plan += "\n[AI-Generated Exercise Plan - Please review and adjust based on your fitness level and capabilities]"
    
    # Save the generated plan to the database
    save_workout_plan(user_id, plan)
    
    return plan

# Register tools for LangChain agent
registered_tools = [
    Tool(name="get_user_profile", func=get_user_profile, description="Get user profile from DB"),
    Tool(name="get_notes", func=get_notes, description="Get user notes/constraints from DB"),
    Tool(name="adapt_plan_based_on_feedback", func=adapt_plan_based_on_feedback, description="Adapt plan based on feedback"),
    Tool(name="regenerate_plan", func=regenerate_plan, description="Regenerate plan with latest user and notes info"),
    Tool(name="check_feasibility", func=check_feasibility, description="Check if plan is feasible"),
]

class FitnessAgent:
    def __init__(self, openai_api_key: str):
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="gpt-3.5-turbo-16k",
            openai_api_key=openai_api_key
        )
        self.memory = ConversationBufferWindowMemory(k=10)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.vector_store = None
        self.tools = registered_tools
        self.setup_agent()
    
    def setup_tools(self):
        """Setup specialized tools for the fitness agent"""
        self.tools = [
            Tool(
                name="database_query",
                func=self.database_query_tool,
                description="Fetches user profile and progress data from the database. Input: user_id (str)."
            ),
            Tool(
                name="feedback_adaptation",
                func=self.feedback_adaptation_tool,
                description="Adapts the workout plan based on user feedback. Input: JSON string with 'plan' and 'feedback'."
            ),
            Tool(
                name="calculate_bmr",
                func=self.calculate_bmr_tool,
                description="Calculate Basal Metabolic Rate (BMR) using Mifflin-St Jeor equation"
            ),
            Tool(
                name="calculate_tdee",
                func=self.calculate_tdee_tool,
                description="Calculate Total Daily Energy Expenditure (TDEE) based on activity level"
            ),
            Tool(
                name="get_user_history",
                func=self.get_user_history_tool,
                description="Retrieve user's previous workout plans and feedback"
            ),
            Tool(
                name="save_workout_plan",
                func=self.save_workout_plan_tool,
                description="Save generated workout plan to database"
            ),
            Tool(
                name="analyze_feedback",
                func=self.analyze_feedback_tool,
                description="Analyze user feedback to determine plan adjustments needed"
            ),
            Tool(
                name="generate_progressive_plan",
                func=self.generate_progressive_plan_tool,
                description="Generate a progressive workout plan based on user's current fitness level"
            ),
            Tool(
                name="get_user_profile",
                func=self.get_user_profile_tool,
                description="Fetch user profile from the database."
            ),
            Tool(
                name="adapt_plan_based_on_feedback",
                func=lambda args: self.adapt_plan_based_on_feedback(args['original_plan'], args['feedback'], args['user_data']),
                description="Adapt workout plan based on feedback."
            ),
            Tool(
                name="save_feedback",
                func=self.save_feedback_tool,
                description="Save user feedback to the database."
            )
        ]
    
    def setup_agent(self):
        """Setup the LangChain agent with custom prompt template"""
        prompt = FitnessPromptTemplate(tools=self.tools, template=self.get_agent_prompt_template())
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        tool_names = [tool.name for tool in self.tools]
        self.agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=FitnessOutputParser(),
            stop=["\nObservation:"],
            allowed_tools=tool_names
        )
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True
        )
    
    def get_agent_prompt_template(self):
        return """You are an expert AI fitness trainer with deep knowledge of exercise science, nutrition, and personalized training.

Your capabilities include:
- Creating personalized workout plans based on user profiles
- Calculating BMR and TDEE
- Analyzing user feedback and adapting plans
- Progressive overload principles
- Injury prevention and modification
- Nutrition guidance integration

Available tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Previous conversation history:
{history}

Question: {input}
{agent_scratchpad}"""
    
    # Tool implementations
    def calculate_bmr_tool(self, user_data: str) -> str:
        """Calculate BMR using Mifflin-St Jeor equation"""
        try:
            data = json.loads(user_data)
            weight_kg = float(data['weight']) * 0.453592
            height_cm = (float(data['feet']) * 12 + float(data['inches'])) * 2.54
            age = int(data['age'])
            gender = data['gender'].lower()
            
            if gender == "male":
                bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
            else:
                bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
            
            return f"BMR calculated: {int(bmr)} calories/day"
        except Exception as e:
            return f"Error calculating BMR: {str(e)}"
    
    def calculate_tdee_tool(self, user_data: str) -> str:
        """Calculate TDEE based on activity level"""
        try:
            data = json.loads(user_data)
            bmr = float(data['bmr'])
            activity_level = data['activity_level'].lower()
            
            multipliers = {
                "sedentary": 1.2,
                "lightly active": 1.375,
                "moderately active": 1.55,
                "very active": 1.725,
                "extra active": 1.9
            }
            
            multiplier = multipliers.get(activity_level, 1.2)
            tdee = bmr * multiplier
            
            return f"TDEE calculated: {int(tdee)} calories/day"
        except Exception as e:
            return f"Error calculating TDEE: {str(e)}"
    
    def get_user_profile_tool(self, user_id: str) -> str:
        """Fetch the latest user profile from the user table."""
        try:
            conn = sqlite3.connect(get_db_path())
            cur = conn.cursor()
            cur.execute("SELECT * FROM user WHERE u_id=?", (user_id,))
            row = cur.fetchone()
            conn.close()
            if row:
                profile = {
                    'id': row[0],
                    'username': row[1],
                    'email': row[2],
                    'age': row[4],
                    'gender': row[5],
                    'vegan': row[6],
                    'allergy': row[7],
                    'weight': row[8],
                    'feet': row[9],
                    'inches': row[10],
                    'bmi': row[11],
                    'activity_level': row[12],
                    'protein_goal': row[13],
                    'carb_goal': row[14],
                    'fat_goal': row[15],
                    'fiber_goal': row[16],
                    'calorie_goal': row[17],
                    'bodyfat': row[18],
                    'status': row[19],
                    'journey': row[20],
                    'startdate': row[21],
                    'goal': row[22]
                }
                return f"User profile: {json.dumps(profile)}"
            else:
                return "User profile not found."
        except Exception as e:
            return f"Error fetching user profile: {str(e)}"
    
    def get_user_history_tool(self, user_id: str) -> str:
        """Retrieve user's workout history (last 5 plans)."""
        try:
            conn = sqlite3.connect(get_db_path())
            cur = conn.cursor()
            cur.execute("""
                SELECT plan_text, feedback, status, created_at 
                FROM exercise_history 
                WHERE u_id = ? 
                ORDER BY created_at DESC 
                LIMIT 5
            """, (user_id,))
            history = cur.fetchall()
            conn.close()
            if history:
                return f"User has {len(history)} previous plans. Latest plan: {history[0][0][:200]}..."
            else:
                return "No previous workout plans found."
        except Exception as e:
            return f"Error retrieving history: {str(e)}"
    
    def save_workout_plan_tool(self, plan_data: str) -> str:
        """Save workout plan to database"""
        try:
            data = json.loads(plan_data)
            conn = sqlite3.connect(get_db_path())
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO exercise_history (u_id, user_data, plan_text, feedback, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                data['u_id'], json.dumps(data['user_data']), data['plan_text'], data.get('feedback', ''), data.get('status', 'active'), datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            conn.commit()
            conn.close()
            return "Workout plan saved successfully."
        except Exception as e:
            return f"Error saving workout plan: {str(e)}"
    
    def analyze_feedback_tool(self, feedback: str) -> str:
        """Analyze user feedback to determine plan adjustments needed."""
        try:
            # Simple analysis: look for keywords
            feedback_lower = feedback.lower()
            if 'hard' in feedback_lower:
                return "User found the plan too hard. Consider reducing intensity."
            elif 'easy' in feedback_lower:
                return "User found the plan too easy. Consider increasing intensity."
            elif 'injury' in feedback_lower:
                return "User mentioned injury. Suggest modifications."
            else:
                return "Feedback analyzed. No major issues detected."
        except Exception as e:
            return f"Error analyzing feedback: {str(e)}"
    
    def generate_progressive_plan_tool(self, user_data: str) -> str:
        """Generate progressive workout plan"""
        try:
            data = json.loads(user_data)
            
            progressive_prompt = f"""
            Create a progressive 7-day workout plan for:
            - Age: {data['age']}
            - Gender: {data['gender']}
            - Weight: {data['weight']} lbs
            - Height: {data['feet']}'{data['inches']}"
            - Activity Level: {data['activity_level']}
            - Goal: {data['goal']}
            - Experience: {data.get('experience', 'beginner')}
            - Equipment: {data.get('equipment', 'minimal')}
            - Time Available: {data.get('time_available', '30-45 minutes')}
            
            Include:
            1. Progressive overload principles
            2. Proper warm-up and cool-down
            3. Rest day recommendations
            4. Exercise modifications for different fitness levels
            5. Safety considerations
            """
            
            response = self.llm.predict(progressive_prompt)
            return f"Progressive plan generated: {response[:500]}..."
        except Exception as e:
            return f"Error generating progressive plan: {str(e)}"
    
    def adapt_plan_based_on_feedback(self, original_plan: str, feedback: str, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt the workout plan based on user feedback."""
        try:
            # For demonstration, just append feedback to the plan
            adapted_plan = original_plan + "\n\n[Adapted based on feedback: " + feedback + "]"
            return {'success': True, 'adapted_plan': adapted_plan, 'feedback_analysis': feedback}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def save_feedback_tool(self, feedback_data: str) -> str:
        """Save user feedback to the database."""
        try:
            data = json.loads(feedback_data)
            conn = sqlite3.connect(get_db_path())
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO user_feedback (u_id, plan_id, feedback_text, rating, difficulty, time_spent, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                data['u_id'], data['plan_id'], data['feedback_text'], data.get('rating', ''), data.get('difficulty', ''), data.get('time_spent', ''), datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            conn.commit()
            conn.close()
            return "Feedback saved successfully."
        except Exception as e:
            return f"Error saving feedback: {str(e)}"

    def database_query_tool(self, user_id: str) -> str:
        """Fetch user profile and progress data from the database."""
        try:
            conn = sqlite3.connect(get_db_path())
            cur = conn.cursor()
            cur.execute("SELECT * FROM user WHERE u_id=?", (user_id,))
            user = cur.fetchone()
            cur.execute("SELECT * FROM progress WHERE u_id=? ORDER BY p_date DESC LIMIT 7", (user_id,))
            progress = cur.fetchall()
            conn.close()
            return json.dumps({
                'user': user,
                'progress': progress
            }, default=str)
        except Exception as e:
            return f"Error fetching user data: {str(e)}"

    def feedback_adaptation_tool(self, input_json: str) -> str:
        """Adapt the workout plan based on user feedback."""
        try:
            data = json.loads(input_json)
            plan = data['plan']
            feedback = data['feedback']
            # Placeholder: In production, call an LLM or use rules to adapt the plan
            return f"Adapted plan based on feedback: {feedback}\nOriginal plan: {plan}"
        except Exception as e:
            return f"Error adapting plan: {str(e)}"

class FitnessPromptTemplate(StringPromptTemplate):
    """Custom prompt template for the fitness agent"""
    def __init__(self, tools, template: str):
        super().__init__(template=template, input_variables=["input", "agent_scratchpad"])
        self.tools = tools

    def format(self, **kwargs) -> str:
        tools_str = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        tool_names = ", ".join([tool.name for tool in self.tools])
        return self.template.format(
            tools=tools_str,
            tool_names=tool_names,
            **kwargs
        )

class FitnessOutputParser:
    """Custom output parser for the fitness agent"""
    
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text
            )
        
        # Parse action and input
        action_match = re.search(r"Action: (.*?)\n", text, re.DOTALL)
        action_input_match = re.search(r"Action Input: (.*?)(?:\n|$)", text, re.DOTALL)
        
        if action_match and action_input_match:
            action = action_match.group(1).strip()
            action_input = action_input_match.group(1).strip()
            
            return AgentAction(
                tool=action,
                tool_input=action_input,
                log=text
            )
        
        raise ValueError(f"Could not parse LLM output: {text}")

# Utility functions for the main app
def get_fitness_agent():
    """Get or create the fitness agent instance"""
    if not hasattr(get_fitness_agent, 'agent'):
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key or openai_api_key == 'your_openai_api_key_here':
            raise ValueError("OPENAI_API_KEY environment variable not set or is placeholder. Please set your actual OpenAI API key in the .env file.")
        get_fitness_agent.agent = FitnessAgent(openai_api_key)
    return get_fitness_agent.agent 