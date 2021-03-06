import time
import json
from googleapiclient.discovery import build
from tqdm import tqdm

api_key = #API KEY
cse_key = #The search-engine-ID you created

resource = build("customsearch", 'v1', developerKey=api_key).cse()

results = {}
input_file_name = # input_file
output_file_name = # output_file

input_file = json.load(open(input_file_name))
for q_id, item in tqdm(input_file.items()):
    query = item['predicted_q']
    try:
        time.sleep(3)
        search_results = resource.list(q=query, cx=cse_key).execute()
        l = []
        for v in search_results['items']:
            try:
                inf = {}
                inf['url'] = v['link']
                inf['name'] = v['title']
                inf['context'] = v['snippet']
                l.append(inf)
            except:
                pass
        results[q_id] = l
    except Exception as ex:
        continue

with open(output_file_name, 'w') as h:
    json.dump(results, h)                          
