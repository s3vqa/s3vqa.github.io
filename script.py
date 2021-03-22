from git import Repo

repo_dir = 's3vqa.github.io'
repo = Repo(repo_dir)
file_list = [
'1.txt'
]
commit_message = 'fix'
repo.index.add(file_list)
repo.index.commit(commit_message)
origin = repo.remote('origin')
origin.push()
