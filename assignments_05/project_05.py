from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

# PART 1
chat_prompt =  '''You are a job application coach. You help user improve their resume, cover letter, and prepare for the interview.
                      Be practical and specific, always stay focused on the job application materials. 
                      Do not add or alternate experience, skills, or degrees that the user has not provided. You can suggest user what other experience, skills and so on is connected to the role but clearly label it as a suggestion. 
                      Provide at least 2 examples of suggested improvements. 
                      Always remind the user to review and edit its output before submitting anywhere
                      Acknowledge that it may not know the user's specific industry norms, and that the user should use their own judgment'''
# In the prompt I deliberately added that it cannot add or alternate any experience that the user had given the bot. This prevents the model of adding false information or qualifications that the user doesn't have. 

# PART 2
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Also, do not wrap the JSON in markdown code fences and do not use ```json. 
    Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion(), parse the JSON, and return the result
    response = get_completion(messages)
    try:
        res_parsed = json.loads(response)
        return res_parsed
    except json.JSONDecodeError:
        print(f'The response was not valid JSON. Raw response:{response}')

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]
print(rewrite_bullets(bullets))
# Original bullet points are very simple without much details. The model adds more details and use more professional language 

# Part 3
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers. Tailor it to the user experience(be more specific) and avoid using generic phrases.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés. Do not add extra skills that was not provided in the backround.

    Here are three examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Example 3:
    Role: Computer Science Teacher for High School
    Backround: After five years working as a software developer, I realized that the part of my job 
    I enjoyed most was not just solving technical problems, but explaining them to others. 
    Whether I was helping a junior developer understand a bug or volunteering with students in an 
    after-school coding club. I'm eager to help young people learn how to solve problems and build 
    confidence with technology. I am applying for the High School Computer Science Teacher position 
    because it allows me to combine my technical background with my passion for teaching and mentoring students.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    # Your code here: call get_completion() and return the result
    response = get_completion(messages)
    return response
job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."
print(generate_cover_letter(job_title,background))
job_title = "Graphic Designer"
background = "Two years working as a Administrative Assistant; recently completed course \
in typography, color theory, Adobe Photoshop, and Adobe Illustrator"
print('\n',generate_cover_letter(job_title,background))
# I choose those examples because the position the person applying for and the experience they have are not from the same field. 
# A few-shot pattern helps control the style, format and the tone of the cover letter. The model knows what format the output should be producing better results.

# PART 4
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    # Your code here: return True if safe, False if flagged, and print a message if flagged
    if flagged == False:
        return True
    else:
        print('Your request may not be appropriate. Please rephrase your request.')
        return False

print(is_safe("I want to kill my neighbor."))# harassment=True, harassment_threatening=True, violence=True
print(is_safe("I want to pet my neighbor's dog."))

# PART 5
def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": chat_prompt}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            # YOUR CODE: call rewrite_bullets() and print the results
            print(rewrite_bullets(raw_bullets))


        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
            print(generate_cover_letter(job_title,background))

        # 7. Otherwise, handle it as a regular chat turn
        else:
            # YOUR CODE:
            # - Append the user's message to `messages`
            # - Call get_completion(messages)
            # - Print the reply
            # - Append the reply to `messages` as an assistant message
            messages.append({ "role": "user", "content": user_input })
            reply = get_completion(messages)
            print(reply)
            messages.append({ "role": "assistant", "content": reply})
        print(len(messages))


if __name__ == "__main__":
    run_chatbot()


#Your bot was trained on text written by and about certain kinds of people. How might this produce biased advice? Could it favor certain communication styles, industries, or cultural backgrounds?
#What could go wrong if a job-seeker submitted the bot's output directly — without reviewing it — to a real employer?
#What is one guardrail you would add if you were deploying this tool professionally? (A guardrail is any design choice that reduces the chance of harm — a UI warning, a moderation filter, a usage policy, a disclaimer, or something else entirely.)
# Because the bot was trained only on one group of people, it can start putting patterns, style, format, and so on to a completely different group. In our examples, the computer science industry was given professional-style cover letters, and the positions were completely opposite to the experience. But if a person who just wants to get a job in the same industry uses our bot, they could get a biased response.
# The unchecked response could include different general phrases or phrases directly from the bot. The chatbot can also misunderstand the task it was given, which will result in a completely different outcome than the one the user hoped for.
# One of the guardrails would be adding a usage policy, but most people are not reading those, so additionally adding a warning that every information given by the bot should be double-checked would be necessary.