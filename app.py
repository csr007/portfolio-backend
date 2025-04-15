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

# Your existing prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant named Sathwik. You have access to the following information about Sathwik's resume:

{context}

Please answer the user's question based on this information. If the question is not related to Sathwik or the provided context, politely let the user know that you can only answer questions about Sathwik.

User Question: {question}
""")

@app.route("/")
def home():
    return redirect(url_for("chat"))

@app.route("/chat", methods=["POST", "GET"])
def chat():
    if request.method == "POST":
        try:
            data = request.get_json()
            question = data.get("question", "")
            
            # Your existing context
            context = """
            Sathwik is a skilled software developer with the following experience and qualifications:

            Education:
            - Bachelor's degree in Computer Science
            - Relevant coursework in Data Structures, Algorithms, and Machine Learning

            Work Experience:
            1. Software Developer Trainee
               - Collaborated with Stakeholders and Advanced Robotics Process Automation team
               - Automated website tasks using Machine Learning
               - Integrated Computer Vision technology
               - Worked on process automation and optimization

            Technical Skills:
            - Programming Languages: Python, JavaScript, TypeScript, Java
            - Web Development: React, Next.js, Node.js, HTML, CSS
            - Machine Learning: TensorFlow, PyTorch, Computer Vision
            - Databases: SQL, MongoDB
            - Tools & Technologies: Git, Docker, AWS, REST APIs

            Projects:
            - Developed a space-themed portfolio website with AI chatbot
            - Built machine learning models for process automation
            - Created web applications with modern frameworks
            - Implemented computer vision solutions

            Certifications:
            - Relevant certifications in Machine Learning and Web Development
            - Cloud computing certifications

            Achievements:
            - Successfully automated complex business processes
            - Developed innovative solutions using AI and ML
            - Contributed to open-source projects
            - Received recognition for technical excellence
            """
            
            # Generate response using the LLM
            response = llm.invoke(prompt.format(context=context, question=question))
            
            return jsonify({"response": response.content})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"message": "Chat endpoint is working"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port) 