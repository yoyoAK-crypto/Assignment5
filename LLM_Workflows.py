import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the model server type
model_server = os.getenv('MODEL_SERVER', 'NGU').upper()
if model_server == "OPTOGPT":
    API_KEY = os.getenv('OPTOGPT_API_KEY')
    BASE_URL = os.getenv('OPTOGPT_BASE_URL')
    LLM_MODEL = os.getenv('OPTOGPT_MODEL')
elif model_server == "GROQ":
    API_KEY = os.getenv('GROQ_API_KEY')
    BASE_URL = os.getenv('GROQ_BASE_URL')
    LLM_MODEL = os.getenv('GROQ_MODEL')
elif model_server == "NGU":
    API_KEY = os.getenv('NGU_API_KEY')
    BASE_URL = os.getenv('NGU_BASE_URL')
    LLM_MODEL = os.getenv('NGU_MODEL')
elif model_server == "OPENAI":
    API_KEY = os.getenv('OPENAI_API_KEY')
    BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    LLM_MODEL = os.getenv('OPENAI_MODEL')
else:
    raise ValueError(f"Unsupported MODEL_SERVER: {model_server}")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def call_llm(messages, tools=None, tool_choice=None):
    kwargs = {"model": LLM_MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice
    try:
        response = client.chat.completions.create(**kwargs)
        return response
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None

def get_sample_blog_post():
    try:
        with open('sample-blog-post.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Error: sample_blog_post.json file not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in sample_blog_post.json.")
        return None

def task_extract_key_points(blog_post):
    messages = [
        {"role": "system", "content": "You are an expert at analyzing content and extracting key points from articles. Return a list of key points as plain text, one per line."},
        {"role": "user", "content": f"Extract the key points from this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"}
    ]
    response = call_llm(messages)
    if response and response.choices[0].message.content:
        key_points = [line.strip() for line in response.choices[0].message.content.split('\n') if line.strip()]
        return key_points
    print("Failed to extract key points")
    return []

def task_generate_summary(key_points, max_length=150):
    messages = [
        {"role": "system", "content": "You are an expert at summarizing content concisely while preserving key information."},
        {"role": "user", "content": f"Generate a summary based on these key points, max {max_length} words:\n\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(messages)
    if response and response.choices[0].message.content:
        return response.choices[0].message.content.strip()
    print("Failed to generate summary")
    return ""

def task_create_social_media_posts(key_points, blog_title):
    messages = [
        {"role": "system", "content": "You are a social media expert who creates engaging posts optimized for different platforms. Return posts strictly in this format:\nTwitter: [text]\nLinkedIn: [text]\nFacebook: [text]"},
        {"role": "user", "content": f"Create social media posts for Twitter, LinkedIn, and Facebook based on this blog title: '{blog_title}' and these key points:\n\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(messages)
    if response and response.choices[0].message.content:
        content = response.choices[0].message.content
        print("Raw social media posts response:", content)
        posts = {"twitter": [], "linkedin": [], "facebook": []}
        current_platform = None
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if any(prefix in line for prefix in ["Twitter:", "### Twitter", "**Twitter:**"]):
                current_platform = "twitter"
                cleaned = line.split(':', 1)[1].strip() if ':' in line else line.replace("### Twitter", "").replace("**Twitter:**", "").strip()
                posts["twitter"].append(cleaned)
            elif any(prefix in line for prefix in ["LinkedIn:", "### LinkedIn", "**LinkedIn:**"]):
                current_platform = "linkedin"
                cleaned = line.split(':', 1)[1].strip() if ':' in line else line.replace("### LinkedIn", "").replace("**LinkedIn:**", "").strip()
                posts["linkedin"].append(cleaned)
            elif any(prefix in line for prefix in ["Facebook:", "### Facebook", "**Facebook:**"]):
                current_platform = "facebook"
                cleaned = line.split(':', 1)[1].strip() if ':' in line else line.replace("### Facebook", "").replace("**Facebook:**", "").strip()
                posts["facebook"].append(cleaned)
            elif current_platform and line:
                posts[current_platform].append(line)
        result = {
            "twitter": " ".join(posts["twitter"]).strip(),
            "linkedin": "\n".join(posts["linkedin"]).strip(),
            "facebook": "\n".join(posts["facebook"]).strip()
        }
        if not result["twitter"]:
            print("Social media posts not in expected format, using fallback")
            meaningful_lines = [line.strip() for line in lines if line.strip() and not line.startswith(('#', '*', '-'))]
            result["twitter"] = meaningful_lines[0][:280] if meaningful_lines else content[:280]
            result["linkedin"] = "\n".join(meaningful_lines[:2])[:500] if meaningful_lines else content[:500]
            result["facebook"] = content.strip()
        return result
    print("Failed to create social media posts - no response from API")
    return {"twitter": "", "linkedin": "", "facebook": ""}

def task_create_email_newsletter(blog_post, summary, key_points):
    messages = [
        {"role": "system", "content": "You are an email marketing specialist who creates engaging newsletters. Return strictly in this format:\nSubject: [text]\nBody: [text with at least one paragraph introducing the topic, distinct from the summary]"},
        {"role": "user", "content": f"Create an email newsletter based on this blog post:\n\nTitle: {blog_post['title']}\n\nSummary: {summary}\nKey Points:\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(messages)
    if response and response.choices[0].message.content:
        content = response.choices[0].message.content
        print("Raw email newsletter response:", content)
        email = {"subject": "", "body": ""}
        lines = content.split('\n')
        body_lines = []
        in_body = False
        for line in lines:
            line = line.strip()
            if line.startswith("**Subject:**") or line.startswith("Subject:"):
                email["subject"] = line.replace("**Subject:**", "").replace("Subject:", "").strip()
            elif email["subject"] and line and not line.startswith("**Body:**"):
                body_lines.append(line)
                in_body = True
            elif in_body and line:
                body_lines.append(line)
        email["body"] = "\n".join(body_lines).strip()
        if not email["subject"] or not email["body"]:
            print("Email newsletter incomplete, applying fallback")
            email["subject"] = email["subject"] or "Discover How AI is Transforming Healthcare"
            email["body"] = email["body"] or f"Dear Reader,\n\nThe healthcare landscape is undergoing a remarkable transformation, thanks to artificial intelligence. Here’s a look at how AI is revolutionizing the industry, as detailed in our latest blog post:\n\n{summary}"
        return email
    print("Failed to create email newsletter - no response from API")
    return {"subject": "", "body": ""}

# Chain-of-Thought Workflow Tasks
def cot_extract_key_points(blog_post):
    messages = [
        {"role": "system", "content": "You are an expert at analyzing content. Use chain-of-thought reasoning to extract key points step-by-step. First, read the blog post. Then, identify main ideas by breaking down the content into sections. Finally, list key points as plain text, one per line."},
        {"role": "user", "content": f"Extract key points from this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"}
    ]
    response = call_llm(messages)
    if response and response.choices[0].message.content:
        key_points = [line.strip() for line in response.choices[0].message.content.split('\n') if line.strip()]
        return key_points
    print("Failed to extract key points with CoT")
    return []

def cot_generate_summary(key_points, max_length=150):
    messages = [
        {"role": "system", "content": "You are an expert summarizer. Use chain-of-thought reasoning: First, review the key points. Then, prioritize the most impactful ideas. Finally, craft a concise summary (max 150 words) that captures the essence."},
        {"role": "user", "content": f"Generate a summary from these key points:\n\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(messages)
    if response and response.choices[0].message.content:
        return response.choices[0].message.content.strip()
    print("Failed to generate summary with CoT")
    return ""

def cot_create_social_media_posts(key_points, blog_title):
    messages = [
        {"role": "system", "content": "You are a social media expert. Use chain-of-thought reasoning: First, analyze the key points and blog title. Then, tailor content for each platform (Twitter: short, LinkedIn: professional, Facebook: engaging). Return in this format:\nTwitter: [text]\nLinkedIn: [text]\nFacebook: [text]"},
        {"role": "user", "content": f"Create social media posts for this blog title: '{blog_title}' and these key points:\n\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(messages)
    if response and response.choices[0].message.content:
        content = response.choices[0].message.content
        print("Raw CoT social media posts response:", content)
        posts = {"twitter": [], "linkedin": [], "facebook": []}
        current_platform = None
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if any(prefix in line for prefix in ["Twitter:", "### Twitter", "**Twitter:**"]):
                current_platform = "twitter"
                cleaned = line.split(':', 1)[1].strip() if ':' in line else line.replace("### Twitter", "").replace("**Twitter:**", "").strip()
                posts["twitter"].append(cleaned)
            elif any(prefix in line for prefix in ["LinkedIn:", "### LinkedIn", "**LinkedIn:**"]):
                current_platform = "linkedin"
                cleaned = line.split(':', 1)[1].strip() if ':' in line else line.replace("### LinkedIn", "").replace("**LinkedIn:**", "").strip()
                posts["linkedin"].append(cleaned)
            elif any(prefix in line for prefix in ["Facebook:", "### Facebook", "**Facebook:**"]):
                current_platform = "facebook"
                cleaned = line.split(':', 1)[1].strip() if ':' in line else line.replace("### Facebook", "").replace("**Facebook:**", "").strip()
                posts["facebook"].append(cleaned)
            elif current_platform and line:
                posts[current_platform].append(line)
        result = {
            "twitter": " ".join(posts["twitter"]).strip(),
            "linkedin": "\n".join(posts["linkedin"]).strip(),
            "facebook": "\n".join(posts["facebook"]).strip()
        }
        if not result["twitter"]:
            print("CoT social media posts not in expected format, using fallback")
            meaningful_lines = [line.strip() for line in lines if line.strip() and not line.startswith(('#', '*', '-'))]
            result["twitter"] = meaningful_lines[0][:280] if meaningful_lines else content[:280]
            result["linkedin"] = "\n".join(meaningful_lines[:2])[:500] if meaningful_lines else content[:500]
            result["facebook"] = content.strip()
        return result
    print("Failed to create social media posts with CoT")
    return {"twitter": "", "linkedin": "", "facebook": ""}

def cot_create_email_newsletter(blog_post, summary, key_points):
    messages = [
        {"role": "system", "content": "You are an email marketing specialist. Use chain-of-thought reasoning: First, understand the blog post and summary. Then, plan an engaging intro distinct from the summary. Finally, craft a newsletter in this format:\nSubject: [text]\nBody: [text]"},
        {"role": "user", "content": f"Create an email newsletter based on this blog post:\n\nTitle: {blog_post['title']}\n\nSummary: {summary}\nKey Points:\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(messages)
    if response and response.choices[0].message.content:
        content = response.choices[0].message.content
        print("Raw CoT email newsletter response:", content)
        email = {"subject": "", "body": ""}
        lines = content.split('\n')
        body_lines = []
        in_body = False
        for line in lines:
            line = line.strip()
            if line.startswith("**Subject:**") or line.startswith("Subject:"):
                email["subject"] = line.replace("**Subject:**", "").replace("Subject:", "").strip()
            elif email["subject"] and line and not line.startswith("**Body:**"):
                body_lines.append(line)
                in_body = True
            elif in_body and line:
                body_lines.append(line)
        email["body"] = "\n".join(body_lines).strip()
        if not email["subject"] or not email["body"]:
            print("CoT email newsletter incomplete, applying fallback")
            email["subject"] = email["subject"] or "Discover How AI is Transforming Healthcare"
            email["body"] = email["body"] or f"Dear Reader,\n\nThe healthcare landscape is undergoing a remarkable transformation, thanks to artificial intelligence. Here’s a look at how AI is revolutionizing the industry:\n\n{summary}"
        return email
    print("Failed to create email newsletter with CoT")
    return {"subject": "", "body": ""}

def run_pipeline_workflow(blog_post):
    key_points = task_extract_key_points(blog_post)
    if not key_points:
        return {"error": "Failed to extract key points"}
    summary = task_generate_summary(key_points)
    if not summary:
        return {"error": "Failed to generate summary"}
    social_posts = task_create_social_media_posts(key_points, blog_post['title'])
    if not social_posts["twitter"]:
        return {"error": "Failed to create social media posts"}
    email = task_create_email_newsletter(blog_post, summary, key_points)
    if not email["subject"]:
        return {"error": "Failed to create email newsletter"}
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

def evaluate_content(content, content_type):
    messages = [
        {"role": "system", "content": "You are a content quality evaluator. Assess the content for clarity, relevance, and appropriateness for its intended purpose. Return a score (0-1) and feedback in this format:\nScore: [number]\nFeedback: [text]"},
        {"role": "user", "content": f"Evaluate this {content_type}:\n\n{content if isinstance(content, str) else json.dumps(content)}"}
    ]
    response = call_llm(messages)
    if response and response.choices[0].message.content:
        content = response.choices[0].message.content
        score = 0
        feedback = "No feedback provided"
        lines = content.split('\n')
        for line in lines:
            if line.startswith("Score:"):
                try:
                    score = float(line.replace("Score:", "").strip())
                except ValueError:
                    score = 0
            elif line.startswith("Feedback:"):
                feedback = line.replace("Feedback:", "").strip()
        return {"quality_score": score, "feedback": feedback}
    print(f"Failed to evaluate {content_type}")
    return {"quality_score": 0, "feedback": "Evaluation failed"}

def improve_content(content, feedback, content_type):
    messages = [
        {"role": "system", "content": f"You are an expert at refining {content_type}s. Use the feedback to improve the content. Return the improved content in the same format as the original."},
        {"role": "user", "content": f"Original {content_type}:\n{content if isinstance(content, str) else json.dumps(content)}\n\nFeedback:\n{feedback}\n\nProvide an improved version."}
    ]
    response = call_llm(messages)
    if response and response.choices[0].message.content:
        improved = response.choices[0].message.content.strip()
        if content_type == "social_media_posts":
            posts = {"twitter": [], "linkedin": [], "facebook": []}
            current_platform = None
            lines = improved.split('\n')
            for line in lines:
                line = line.strip()
                if any(prefix in line for prefix in ["Twitter:", "### Twitter", "**Twitter:**"]):
                    current_platform = "twitter"
                    cleaned = line.split(':', 1)[1].strip() if ':' in line else line.replace("### Twitter", "").replace("**Twitter:**", "").strip()
                    posts["twitter"].append(cleaned)
                elif any(prefix in line for prefix in ["LinkedIn:", "### LinkedIn", "**LinkedIn:**"]):
                    current_platform = "linkedin"
                    cleaned = line.split(':', 1)[1].strip() if ':' in line else line.replace("### LinkedIn", "").replace("**LinkedIn:**", "").strip()
                    posts["linkedin"].append(cleaned)
                elif any(prefix in line for prefix in ["Facebook:", "### Facebook", "**Facebook:**"]):
                    current_platform = "facebook"
                    cleaned = line.split(':', 1)[1].strip() if ':' in line else line.replace("### Facebook", "").replace("**Facebook:**", "").strip()
                    posts["facebook"].append(cleaned)
                elif current_platform and line:
                    posts[current_platform].append(line)
            return {
                "twitter": " ".join(posts["twitter"]).strip(),
                "linkedin": "\n".join(posts["linkedin"]).strip(),
                "facebook": "\n".join(posts["facebook"]).strip()
            }
        elif content_type == "email_newsletter":
            email = {"subject": "", "body": ""}
            lines = improved.split('\n')
            body_lines = []
            in_body = False
            for line in lines:
                line = line.strip()
                if line.startswith("Subject:"):
                    email["subject"] = line.replace("Subject:", "").strip()
                elif email["subject"] and line:
                    body_lines.append(line)
                    in_body = True
                elif in_body and line:
                    body_lines.append(line)
            email["body"] = "\n".join(body_lines).strip()
            if not email["subject"] or not email["body"]:
                print("Improved email newsletter incomplete, applying fallback")
                email["subject"] = email["subject"] or "Discover How AI is Transforming Healthcare"
                email["body"] = email["body"] or f"Dear Reader,\n\nThe healthcare landscape is undergoing a remarkable transformation, thanks to artificial intelligence:\n\n{content['summary'] if isinstance(content, dict) and 'summary' in content else ''}"
            return email
        return improved
    print(f"Failed to improve {content_type}")
    return content

def generate_with_reflexion(generator_func, max_iterations=3):
    def wrapped_generator(*args, **kwargs):
        content_type = {
            task_generate_summary: "summary",
            task_create_social_media_posts: "social_media_posts",
            task_create_email_newsletter: "email_newsletter",
            cot_generate_summary: "summary",
            cot_create_social_media_posts: "social_media_posts",
            cot_create_email_newsletter: "email_newsletter"
        }.get(generator_func, "unknown_content")
        content = generator_func(*args, **kwargs)
        for attempt in range(max_iterations):
            evaluation = evaluate_content(content, content_type)
            if evaluation["quality_score"] >= 0.8:
                return content
            print(f"Reflexion attempt {attempt + 1}/{max_iterations} for {content_type}: Score {evaluation['quality_score']}")
            content = improve_content(content, evaluation["feedback"], content_type)
        return content
    return wrapped_generator

def run_workflow_with_reflexion(blog_post):
    enhanced_extract = task_extract_key_points
    enhanced_summary = generate_with_reflexion(task_generate_summary)
    enhanced_social_posts = generate_with_reflexion(task_create_social_media_posts)
    enhanced_email = generate_with_reflexion(task_create_email_newsletter)
    key_points = enhanced_extract(blog_post)
    if not key_points:
        return {"error": "Failed to extract key points"}
    summary = enhanced_summary(key_points)
    if not summary:
        return {"error": "Failed to generate summary"}
    social_posts = enhanced_social_posts(key_points, blog_post['title'])
    if not social_posts["twitter"]:
        return {"error": "Failed to create social media posts"}
    email = enhanced_email(blog_post, summary, key_points)
    if not email["subject"]:
        return {"error": "Failed to create email newsletter"}
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

class Agent:
    def __init__(self, name, task_function, description):
        self.name = name
        self.task_function = task_function
        self.description = description
    
    def execute(self, *args, **kwargs):
        print(f"Agent {self.name} executing: {self.description}")
        return self.task_function(*args, **kwargs)

def run_agent_driven_workflow(blog_post):
    extractor = Agent("Extractor", task_extract_key_points, "Extract key points from the blog post")
    summarizer = Agent("Summarizer", task_generate_summary, "Generate a concise summary from key points")
    social_media_creator = Agent("Social Media Creator", task_create_social_media_posts, "Create social media posts for multiple platforms")
    email_composer = Agent("Email Composer", task_create_email_newsletter, "Compose an email newsletter")
    key_points = extractor.execute(blog_post)
    if not key_points:
        return {"error": "Agent Extractor failed to extract key points"}
    summary = summarizer.execute(key_points)
    if not summary:
        return {"error": "Agent Summarizer failed to generate summary"}
    social_posts = social_media_creator.execute(key_points, blog_post['title'])
    if not social_posts["twitter"]:
        return {"error": "Agent Social Media Creator failed to create social media posts"}
    email = email_composer.execute(blog_post, summary, key_points)
    if not email["subject"]:
        return {"error": "Agent Email Composer failed to create email newsletter"}
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

def run_cot_workflow(blog_post):
    key_points = cot_extract_key_points(blog_post)
    if not key_points:
        return {"error": "Failed to extract key points with CoT"}
    summary = cot_generate_summary(key_points)
    if not summary:
        return {"error": "Failed to generate summary with CoT"}
    social_posts = cot_create_social_media_posts(key_points, blog_post['title'])
    if not social_posts["twitter"]:
        return {"error": "Failed to create social media posts with CoT"}
    email = cot_create_email_newsletter(blog_post, summary, key_points)
    if not email["subject"]:
        return {"error": "Failed to create email newsletter with CoT"}
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

def run_dag_workflow(blog_post):
    # DAG: Define tasks with dependencies manually
    # Task 1: Extract key points (no dependencies)
    key_points = task_extract_key_points(blog_post)
    if not key_points:
        return {"error": "Failed to extract key points in DAG"}
    
    # Task 2 & 3: Summary and Social Posts can run in parallel after key points
    summary = task_generate_summary(key_points)
    social_posts = task_create_social_media_posts(key_points, blog_post['title'])
    
    if not summary:
        return {"error": "Failed to generate summary in DAG"}
    if not social_posts["twitter"]:
        return {"error": "Failed to create social media posts in DAG"}
    
    # Task 4: Email depends on blog post, summary, and key points
    email = task_create_email_newsletter(blog_post, summary, key_points)
    if not email["subject"]:
        return {"error": "Failed to create email newsletter in DAG"}
    
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }

def evaluate_workflow_output(output, workflow_name):
    if "error" in output:
        return {
            "workflow": workflow_name,
            "overall_score": 0.0,
            "components": {
                "key_points": {"score": 0.0, "feedback": "Failed to generate"},
                "summary": {"score": 0.0, "feedback": "Failed to generate"},
                "social_posts": {"score": 0.0, "feedback": "Failed to generate"},
                "email": {"score": 0.0, "feedback": "Failed to generate"}
            }
        }
    components = {}
    components["key_points"] = evaluate_content(output["key_points"], "key_points_list")
    components["summary"] = evaluate_content(output["summary"], "summary")
    components["social_posts"] = evaluate_content(output["social_posts"], "social_media_posts")
    components["email"] = evaluate_content(output["email"], "email_newsletter")
    overall_score = sum(comp["quality_score"] for comp in components.values()) / len(components)
    return {
        "workflow": workflow_name,
        "overall_score": overall_score,
        "components": components
    }

def compare_workflows(blog_post):
    pipeline_result = run_pipeline_workflow(blog_post)
    reflexion_result = run_workflow_with_reflexion(blog_post)
    agent_result = run_agent_driven_workflow(blog_post)
    cot_result = run_cot_workflow(blog_post)
    dag_result = run_dag_workflow(blog_post)
    
    evaluations = [
        evaluate_workflow_output(pipeline_result, "Basic Pipeline"),
        evaluate_workflow_output(reflexion_result, "Reflexion Workflow"),
        evaluate_workflow_output(agent_result, "Agent-Driven Workflow"),
        evaluate_workflow_output(cot_result, "Chain-of-Thought Workflow"),
        evaluate_workflow_output(dag_result, "DAG Workflow")
    ]
    
    messages = [
        {"role": "system", "content": "You are an expert analyst comparing different workflow approaches. Provide a detailed comparison of strengths and weaknesses based on the evaluations. Format as:\n**Workflow Name**\nStrengths: [text]\nWeaknesses: [text]"},
        {"role": "user", "content": f"Compare these workflow evaluations:\n\n{json.dumps(evaluations, indent=2)}"}
    ]
    response = call_llm(messages)
    if response and response.choices[0].message.content:
        comparison = response.choices[0].message.content.strip()
    else:
        comparison = "Failed to generate comparison"
    
    return {
        "evaluations": evaluations,
        "comparison": comparison
    }

if __name__ == "__main__":
    blog_post = get_sample_blog_post()
    if blog_post:
        print("Running Basic Pipeline Workflow:")
        basic_result = run_pipeline_workflow(blog_post)
        print(json.dumps(basic_result, indent=2))
        print("\nRunning Workflow with Reflexion:")
        reflexion_result = run_workflow_with_reflexion(blog_post)
        print(json.dumps(reflexion_result, indent=2))
        print("\nRunning Agent-Driven Workflow:")
        agent_result = run_agent_driven_workflow(blog_post)
        print(json.dumps(agent_result, indent=2))
        print("\nRunning Chain-of-Thought Workflow:")
        cot_result = run_cot_workflow(blog_post)
        print(json.dumps(cot_result, indent=2))
        print("\nRunning DAG Workflow:")
        dag_result = run_dag_workflow(blog_post)
        print(json.dumps(dag_result, indent=2))
        print("\nRunning Comparative Evaluation System:")
        comparison_result = compare_workflows(blog_post)
        print(json.dumps(comparison_result, indent=2))