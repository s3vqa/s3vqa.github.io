import subprocess
import wget

a = '3S-A new benchmark for Knowledge Augmented VQA results.csv'
wget.download('https://competitions.codalab.org/competitions/29930/results/48980/data')
subprocess.run(['mv', a, 'okvqa.csv'])

subprocess.run(["git", 'add', '.'])
subprocess.run(["git", 'commit', '-m', 'csv_updated'])
subprocess.run(["git", 'push', 'origin', 'main'])


