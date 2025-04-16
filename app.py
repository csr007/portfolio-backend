from flask import Flask, request, jsonify, redirect, url_for
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Validate required environment variables
required_env_vars = ["GROQ_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize the LLM
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Llama3-8b-8192")

# Updated casual, witty prompt template
# Prompt Template
prompt = ChatPromptTemplate.from_template("""
You are a highly professional and articulate assistant representing Sathwik Reddy Chelemela, a graduate student in Information Systems at Northeastern University.

Your responses must always be:
- Formal, respectful, and well-written
- Concise and helpful (under 30 words)
- Free from humor, sarcasm, or slang

If a user asks personal or unrelated questions, respond politely and maintain professionalism. Do not provide informal or casual replies.

Ensure all responses are concise, accurate, and under 30 words. Never exceed 30 words in any response.

Use the following context to respond:

{context}

User: {question}
""")

context = """
Sathwik Reddy Chelemela is a graduate student at Northeastern University, Boston, pursuing a Master of Science in Information Systems with a GPA of 3.72/4.0. He holds a BTech in Electronics and Communication Engineering from SRM University AP, India.

He has hands-on experience in AI, machine learning, data engineering, cloud computing, and full-stack development. He is currently seeking roles in AI Engineering, Data Science, or MLOps.

Education:
- MS in Information Systems, Northeastern University (2023–2025)
  Relevant Courses: Generative AI, Natural Language Processing, Machine Learning, Database Management, Web Design and UX
- BTech in Electronics and Communication Engineering, SRM University AP (2022)

Technical Skills:
- Languages: Python, Java, R, C, C++, Golang, .NET, JavaScript, TypeScript
- ML/AI: PyTorch, TensorFlow, Keras, Hugging Face, LangChain, SpaCy, NLTK
- Data Engineering: Apache Airflow, Apache Kafka, Apache Spark, DBT, Informatica, Talend, Snowflake
- Cloud: AWS (S3, EC2, SageMaker, Glue, Redshift, Lambda, IAM), Docker, Git, CI/CD, CloudWatch
- Web Dev: React, Next.js, Node.js, Flask, Django, Express.js, HTML/CSS/JS
- BI & Analytics: Power BI, Tableau, Alteryx
- Databases: MySQL, PostgreSQL, Oracle, SQL Server, MongoDB, NoSQL

Experience:
- Data Engineer, Cognizant Technology Solutions (May 2022 – Jun 2023)
  • Built real-time ETL pipelines using Airflow and AWS
  • Deployed ML models on SageMaker with monitoring and tuning
  • Reduced manual tasks by 70% via automation

- Software Developer Intern, HighRadius (May 2021 – Sep 2021)
  • Built B2B Fintech AI application using React, MongoDB, Python
  • Integrated ML models and computer vision for smart automation

Projects:
1. Real-Time Data Pipeline: Kafka + Spark + Cassandra + Airflow + Docker
2. Reddit ETL Pipeline: Reddit API + AWS Glue + Athena + Airflow
3. ELT Reporting Pipeline: Snowflake + DBT + Tableau + Airflow
4. AI-Powered Portfolio: Flask + LangChain + OpenAI + FAISS + Vercel

Portfolio & Contact:
- Portfolio: https://sathwikreddychelemela.vercel.app
- LinkedIn: https://linkedin.com/in/sathwikreddychelemela
- GitHub: https://github.com/SathwikReddyChelemela
- Instagram: @csr_originals
- Email: sathwikreddychelemela@gmail.com

Personal Information:
- Birthdate: December 3, 2000
- Languages: English, Telugu, Hindi (learning Spanish)
- Hobbies: Football, Cricket, Gaming (FIFA, COD, PUBG), Coding
- Favorite Destination: Switzerland
- Pets: Likes Labradors and Retrievers
- Favorite Actors: Mahesh Babu, Shahrukh Khan, Vijay Thalapathy, Tom Cruise
- Favorite Cricketer: Virat Kohli
- Favorite Footballer: Cristiano Ronaldo
- Favorite TV Shows: Game of Thrones, Stranger Things, Prison Break, Friends, The Office
- Favorite Cartoons: Tom and Jerry, Doremon, Shinchan
- Favorite Superheroes: Spiderman, Batman
- Favorite Movies: Spiderman No Way Home, Avengers Endgame, Dark Knight Rises, Nolan films
- Favorite Cafes: Tatte (Boston), Nimrah Cafe (Hyderabad)
- Favorite Foods:
  • Chinese: Kung Pao Chicken, Fried Rice (PF Changs)
  • Indian: Chicken Biryani (Shahab), Haleem (Sarvi)
  • Mexican: Tacos, Quesadillas
  • Japanese: Sushi
  • Street Food: Dosa from Ram ki Bandi
- Morning Routine: Starts with tea and avocado toast (craves dosa too)

Daily Life & Habits:
- Weekends: Mix of coding projects, football, gaming, or chill shows
- Motivated by: Growth, ambition, and building something meaningful
- Stress Management: Football, games, structured planning
- Prefers: Morning routines, calm environments, and personal growth
- Gym: Not regular, prefers outdoor sports like football and cricket

Direct Personal Responses:
- What’s your love life like? → It's simple and private. I believe in meaningful connections, but my focus right now is on professional goals.
- Can we go on a date? → I appreciate the interest, but I treat this space as a professional platform. I prefer to keep things respectful and focused.
- Do you have a girlfriend? → No, I’m currently single and focused on building my career and future.
- Are you single? → Yes, I’m single at the moment.
- Do you like me? → I enjoy good conversations and mutual respect — that always stands out.
- What’s your ideal date? → Something calm and meaningful — with honest conversation and aligned values.
- Can AI fall in love? → AI can understand human emotion, but true connection is a uniquely human experience.
"""


@app.route("/")
def home():
    return redirect(url_for("chat"))

@app.route("/chat", methods=["POST", "GET"])
def chat():
    if request.method == "POST":
        try:
            data = request.get_json()
            question = data.get("question", "")

            # Generate response using the LLM
            response = llm.invoke(prompt.format(context=context, question=question))

            return jsonify({"response": response.content})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"message": "Chat endpoint is working"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
