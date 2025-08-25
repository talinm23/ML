# In this code, we use an API to connect to LinkedIn to search jobs,
# then scrape the job URLs to feed into Openai
# (since Openai doesn't support web browsing through the API even with Openai Plus).
# Then we display the organized job summary and description so that we spend less time
# on making a decision to apply to the job or not.

# Library imports:
from openai import OpenAI
from requests import get
import requests
from bs4 import BeautifulSoup
import pandas as pd
import feedparser
import http.client
import json
import os
import pickle
from dotenv import load_dotenv
import time
from pathlib import Path, PurePath
import pickle
# pip install openai pandas
import os, json
import pandas as pd
from openai import OpenAI
load_dotenv()

start_time = time.time()
# Use your Openai API key.
OPENAI_API_KEY = ("YOUR OPENAI API KEY")
client = OpenAI(api_key =  OPENAI_API_KEY)

# Create a connection to the rapid API, send the request, and get a response:
# https://rapidapi.com/rockapis-rockapis-default/api/rapid-linkedin-jobs-api
conn = http.client.HTTPSConnection("rapid-linkedin-jobs-api.p.rapidapi.com")
headers = {
    'x-rapidapi-key': "rapid api key",
    'x-rapidapi-host': "rapid-linkedin-jobs-api.p.rapidapi.com"
}
conn.request("GET", "/search-jobs-v2?keywords=data%20scientist&locationId=103644278&datePosted=pastMonth&salary=100k%2B&jobType=fullTime&experienceLevel=not%20senior&onsiteRemote=remote&sort=mostRecent", headers=headers)
res = conn.getresponse()
#Reading the data from the API, decoding and parsing the response, getting the job URLs.
data = res.read()
data_decoded = data.decode("utf-8")
data_dict = json.loads(data_decoded)
#Finding the right tag to extract the job URLs:
data_list = data_dict["data"]
df = pd.DataFrame(data_list)
links = df.url

# Function to scrape the URLs (links series from above):
def scrape_feed(url):
    rss_feed = get(url).text
    feed = feedparser.parse(rss_feed)
    text = ''
    for post in feed['entries']:
        text = f'{text} {post["title"]} - {post["description"]}'
    return text

# Function to send prompts to Openai, defining the roles, providing the prompt and the URL's text.
def ai(query,text):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a data analyst summarizing information given to you."},
            {"role": "user", "content": f"provide answer from this document {text}"},
            {"role": "user", "content": query}
        ]
    )
    response = completion.choices[0].message.content
    return response

# First query/prompt to send to Openai to get a dictionary of the contents of the job posting.
query = ('Do the following step by step.'
         'Do not go to a new line. Put all code parts in one line.'
         'Make a table of the job posting. Include as much information as possible.'
         'Summarize using tags and similar to the format of a python dictionary.'
         'Display the python dictionary code.'
         'The code should be runnable in python.'
         'Do not explain anything. Only display a dictionary of the created dictionary in python formatting.'
         'This is not an external link. Use the information given in the prompt. This information is all you need.'
         'Check your written dictionary code and remove all the backslash-n in the dictionary code.'
          )

# A second query became necessary since Chatgpt does not return a clean dictionary output
# so have to instruct it to clean the outputted dictionary string.


# Saving the contents in a list of dictionaries:
data = []
for url in links:
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'html.parser')
    # First and second prompt
    result_ai = ai(query, soup.get_text())
    query2 = (
        "Extract a single JSON object for the job posting from the text below.\n"
        "RULES:\n"
        "• Return ONLY a single-line JSON object. No backticks, no code fences, no language tags, no explanations.\n"
        "• Keys to include if present: Job Title, Company, Location, Posted, URL or links, Applicants, Responsibilities,\n"
        "• Keep lists as JSON arrays. Omit keys that aren’t present.\n"
        "TEXT:\n"
        f"{result_ai}"
    )
    result_ai2 = ai(query2, result_ai)
    # eval creates a dictionary out of the outputted string
    my_dict = eval(result_ai2)
    if isinstance(my_dict, dict):
        data.append(my_dict)

# Save the list of dictionaries in a pickle file
here = Path(__file__).resolve().parent    # script’s directory
with open(here / 'saved_list.pkl', 'wb') as f:
    pickle.dump(data, f)

# Access te pickle file (that is a list of dictionaries to use in the same or another code)
with open('saved_list.pkl', 'rb') as f:
    loaded_list_of_dicts = pickle.load(f)

SYSTEM_INSTRUCTIONS = (
    "You are a data wrangler. Given a JSON array of heterogeneous records, "
    "infer a normalized set of column names. Merge clear synonyms (e.g., "
    "'Title' ~ 'Job Title'; 'Reports To' ~ 'Reports to'; 'Seniority level' ~ 'Seniority Level'). "
    "Pick up to 15 columns that best cover the data; if needed, put overflow into an 'Other' column. "
    "Flatten nested lists as '; ' joined and dicts as 'key: value' pairs joined by '; '. "
    "Return ONLY valid JSON with this shape: "
    "{columns:[...], rows:[[...],[...],...]} where rows align exactly to the columns and preserve input order."
)

def list_to_table_with_openai(records, model="gpt-4o-mini", max_items=25):
    # Keep prompt short to avoid truncation; sample first N items if huge
    sample = records[:max_items]

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": json.dumps(sample, ensure_ascii=False)}
        ],
    )

    payload = json.loads(resp.choices[0].message.content)
    columns = payload["columns"]
    rows = payload["rows"]

    # Markdown table
    def md_row(cells): return "| " + " | ".join(str(c or "") for c in cells) + " |"
    md = [md_row(columns), "| " + " | ".join("---" for _ in columns) + " |"] + [md_row(r) for r in rows]
    print("\n".join(md))

    # Optional: DataFrame if you prefer
    df = pd.DataFrame(rows, columns=columns)
    return {"columns": columns, "rows": rows, "df": df}

records = loaded_list_of_dicts  # your big list of dicts
result = list_to_table_with_openai(records)
df = result["df"]

from pathlib import Path
out = Path.cwd() / 'table.csv'
df.to_csv(out, index=False)
print('Saved to:', out.resolve())

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time (docs = loader.load_and_split()): {elapsed_time:.4f} seconds")



''' 
#the prompt that didn't work properly:
query2 = ('Do these step by step: '
          'Take this text, put everything in one line'
          'remove the following: '
          'the triple quotes in the beginning and end of the string,'
          'the word python in the beginning of the string,'
          'the job_posting=.'
          'Start the string with curly braces.'
          'Do not explain anything. Just show the output inside the curly braces'
          'Double check that you have removed the triple quotes in the beginning and end of the string')
'''






















